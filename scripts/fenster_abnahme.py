#!/usr/bin/env python3
"""Abnahme und Messung für das Wartungsfenster der Sparks (Plan 13-11-2026, Variante A). Nur Standardbibliothek.

Läuft auf Spark1 gegen den lokalen Endpunkt. Der Schlüssel wird zur Laufzeit aus der Schlüsseldatei gelesen und
nie ausgegeben.

  python3 fenster_abnahme.py pruefen                      # /health, Werkzeugaufruf, striktes JSON-Schema, Deutsch
  python3 fenster_abnahme.py messen --stufen 1,2,4 --name start1   # Tokens/s je Stufe, Ergebnis als JSON
  python3 fenster_abnahme.py vergleich ausgang.json start1.json    # Abnahmegrenzen aus dem Plan prüfen
  python3 fenster_abnahme.py vorrang                      # nimmt der Server das Feld «priority» an?

Rückgabe 0 = bestanden, 1 = nicht bestanden, 2 = Aufruf- oder Verbindungsfehler.
Messverfahren wie bench/throughput_probe.py des Rezepts: feste Ausgabelänge (min_tokens, ignore_eos), Temperatur 0.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

FRAGE = ("Explain how a transformer language model generates text, step by step, "
         "covering tokenization, attention, and decoding. Be thorough and concrete.")


def kopf(args):
    k = {"Content-Type": "application/json"}
    pfad = os.path.expanduser(args.schluesseldatei)
    if os.path.exists(pfad):
        with open(pfad, encoding="utf-8") as f:
            wert = f.read().strip()
        if wert:
            k["Authorization"] = "Bearer " + wert
    return k


def sende(args, pfad, koerper=None, frist=300):
    daten = json.dumps(koerper).encode() if koerper is not None else None
    anfrage = urllib.request.Request(args.basis + pfad, daten, kopf(args))
    with urllib.request.urlopen(anfrage, timeout=frist) as r:
        return r.status, r.read().decode("utf-8", "replace")


def chat(args, koerper, frist=300):
    koerper = dict(koerper, model=args.modell)
    _, text = sende(args, "/v1/chat/completions", koerper, frist)
    return json.loads(text)


# ---------------------------------------------------------------- pruefen

def p_gesund(args):
    status, _ = sende(args, "/health", None, 10)
    return status == 200, f"/health {status}"


def p_modell(args):
    _, text = sende(args, "/v1/models", None, 20)
    namen = [m.get("id") for m in json.loads(text).get("data", [])]
    return args.modell in namen, f"/v1/models nennt {namen}"


def p_werkzeug(args):
    werkzeug = {"type": "function", "function": {
        "name": "wetter_abfragen", "description": "Liefert das Wetter für einen Ort.",
        "parameters": {"type": "object", "properties": {"ort": {"type": "string"}},
                       "required": ["ort"], "additionalProperties": False}}}
    a = chat(args, {"messages": [{"role": "user", "content": "Wie ist das Wetter in Basel? Nutze das Werkzeug."}],
                    "tools": [werkzeug], "tool_choice": "auto", "max_tokens": 4000, "temperature": 0})
    rufe = (a["choices"][0]["message"].get("tool_calls") or [])
    if not rufe:
        return False, "kein tool_calls in der Antwort"
    f = rufe[0]["function"]
    arg = json.loads(f.get("arguments") or "{}")
    gut = f.get("name") == "wetter_abfragen" and "basel" in str(arg.get("ort", "")).lower()
    return gut, f"Werkzeugaufruf {f.get('name')} {arg}"


def p_schema(args):
    schema = {"type": "object", "additionalProperties": False, "required": ["geraet", "anzahl", "tags"],
              "properties": {"geraet": {"type": "string"}, "anzahl": {"type": "integer"},
                             "tags": {"type": "array", "items": {"type": "string"}}}}
    a = chat(args, {"messages": [{"role": "user", "content":
                                  "Extrahiere als JSON: «Auf dem Server laufen 3 Dienste: nginx, postgres, redis.» "
                                  "geraet = Name des Geräts, anzahl = Zahl der Dienste, tags = Namen der Dienste."}],
                    "response_format": {"type": "json_schema",
                                        "json_schema": {"name": "fakt", "schema": schema, "strict": True}},
                    "max_tokens": 4000, "temperature": 0})
    inhalt = a["choices"][0]["message"].get("content") or ""
    try:
        d = json.loads(inhalt)
    except json.JSONDecodeError as e:
        return False, f"kein gültiges JSON ({e}): {inhalt[:120]!r}"
    gut = (set(d) == {"geraet", "anzahl", "tags"} and isinstance(d["geraet"], str) and d["anzahl"] == 3
           and isinstance(d["tags"], list) and len(d["tags"]) == 3 and all(isinstance(t, str) for t in d["tags"]))
    return gut, f"Schema-Antwort {d}"


def p_deutsch(args):
    a = chat(args, {"messages": [{"role": "user", "content":
                                  "Antworte in genau einem deutschen Satz mit den Wörtern «Übergrösse», «Äpfel» und «Zürich»."}],
                    "max_tokens": 4000, "temperature": 0})
    inhalt = a["choices"][0]["message"].get("content") or ""
    gut = bool(inhalt.strip()) and "�" not in inhalt and "Zürich" in inhalt and "Äpfel" in inhalt
    return gut, f"Deutsch: {inhalt.strip()[:160]!r}"


def pruefen(args):
    alle_gut = True
    for name, probe in [("gesund", p_gesund), ("modell", p_modell), ("werkzeug", p_werkzeug),
                        ("schema", p_schema), ("deutsch", p_deutsch)]:
        t0 = time.perf_counter()
        try:
            gut, text = probe(args)
        except Exception as e:  # jede Probe einzeln melden, die anderen laufen weiter
            gut, text = False, f"Fehler {type(e).__name__}: {e}"
        alle_gut &= gut
        print(f"{'BESTANDEN' if gut else 'NICHT BESTANDEN':16} {name:9} {time.perf_counter() - t0:6.1f} s  {text}")
    return 0 if alle_gut else 1


# ---------------------------------------------------------------- messen

def ein_strom(args, ergebnis, i):
    koerper = {"model": args.modell, "messages": [{"role": "user", "content": FRAGE}],
               "max_tokens": args.tokens, "min_tokens": args.tokens, "ignore_eos": True, "temperature": 0.0,
               "stream": True, "stream_options": {"include_usage": True}}
    anfrage = urllib.request.Request(args.basis + "/v1/chat/completions", json.dumps(koerper).encode(), kopf(args))
    t0 = time.perf_counter()
    t_erst = None
    nutzung = None
    n = 0
    try:
        with urllib.request.urlopen(anfrage, timeout=3600) as r:
            for roh in r:
                zeile = roh.decode("utf-8", "replace").strip()
                if not zeile.startswith("data:"):
                    continue
                p = zeile[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    st = json.loads(p)
                except json.JSONDecodeError:
                    continue
                if st.get("usage"):
                    nutzung = st["usage"]
                for c in st.get("choices") or []:
                    d = c.get("delta") or {}
                    if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                        if t_erst is None:
                            t_erst = time.perf_counter()
                        n += 1
    except Exception as e:
        ergebnis[i] = {"fehler": f"{type(e).__name__}: {e}"}
        return
    t_ende = time.perf_counter()
    t_erst = t_erst or t_ende
    tokens = (nutzung or {}).get("completion_tokens") or n
    dauer = t_ende - t_erst
    ergebnis[i] = {"erstes_token_s": t_erst - t0, "tokens": tokens, "start": t0, "ende": t_ende,
                   "tok_s": (tokens - 1) / dauer if dauer > 0 and tokens > 1 else 0.0}


def stufe(args, c):
    ergebnis = [None] * c
    faeden = [threading.Thread(target=ein_strom, args=(args, ergebnis, i)) for i in range(c)]
    for f in faeden:
        f.start()
    for f in faeden:
        f.join()
    gut = [e for e in ergebnis if e and "fehler" not in e]
    fehler = [e["fehler"] for e in ergebnis if e and "fehler" in e]
    if not gut:
        return {"parallel": c, "ok": 0, "fehler": fehler}
    wand = max(e["ende"] for e in gut) - min(e["start"] for e in gut)
    return {"parallel": c, "ok": len(gut), "fehler": fehler,
            "je_anfrage_tok_s": round(statistics.median(e["tok_s"] for e in gut), 2),
            "gesamt_tok_s": round(sum(e["tokens"] for e in gut) / wand, 2),
            "erstes_token_s": round(statistics.median(e["erstes_token_s"] for e in gut), 3),
            "wand_s": round(wand, 1)}


def speicher_frei_gib():
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for z in f:
                if z.startswith("MemAvailable:"):
                    return round(int(z.split()[1]) / 1048576, 1)
    except OSError:
        return None
    return None


def messen(args):
    stufen = [int(s) for s in args.stufen.split(",")]
    aus = {"name": args.name, "zeit": time.strftime("%d-%m-%Y %H:%M:%S"), "modell": args.modell,
           "tokens_je_anfrage": args.tokens, "speicher_frei_gib": speicher_frei_gib(), "stufen": []}
    ein_strom(args, [None], 0)  # Aufwärmen, nicht gewertet
    for c in stufen:
        if c == 1:
            laeufe = [stufe(args, 1) for _ in range(args.wiederholungen)]
            gute = [l for l in laeufe if l["ok"]]
            if gute:
                gute.sort(key=lambda l: l["je_anfrage_tok_s"])
                e = gute[len(gute) // 2]
            else:
                e = laeufe[0]
        else:
            e = stufe(args, c)
        aus["stufen"].append(e)
        print(f"parallel {c}: je Anfrage {e.get('je_anfrage_tok_s', '-')} Tokens/s, gesamt "
              f"{e.get('gesamt_tok_s', '-')} Tokens/s, erstes Token {e.get('erstes_token_s', '-')} s, "
              f"ok {e['ok']}/{c}" + (f", Fehler {e['fehler'][:1]}" if e.get("fehler") else ""))
    ziel = os.path.join(args.ablage, f"fenster-{args.name}.json")
    os.makedirs(args.ablage, exist_ok=True)
    with open(ziel, "w", encoding="utf-8") as f:
        json.dump(aus, f, indent=1, ensure_ascii=False)
    print(f"verfügbarer Speicher {aus['speicher_frei_gib']} GiB; gespeichert: {ziel}")
    return 0 if all(s["ok"] == s["parallel"] for s in aus["stufen"]) else 1


# ---------------------------------------------------------------- vergleich

def vergleich(args):
    with open(args.ausgang, encoding="utf-8") as f:
        a = json.load(f)
    with open(args.neu, encoding="utf-8") as f:
        n = json.load(f)
    st_a = {s["parallel"]: s for s in a["stufen"]}
    st_n = {s["parallel"]: s for s in n["stufen"]}
    gut = True
    if 1 in st_a and 1 in st_n:
        q = st_n[1]["je_anfrage_tok_s"] / st_a[1]["je_anfrage_tok_s"]
        ok = q >= 0.90
        gut &= ok
        print(f"{'BESTANDEN' if ok else 'NICHT BESTANDEN':16} einzeln {st_a[1]['je_anfrage_tok_s']} -> "
              f"{st_n[1]['je_anfrage_tok_s']} Tokens/s ({q:.0%}, Grenze 90 %)")
    hoch = max(set(st_a) & set(st_n))
    if hoch > 1:
        q = st_n[hoch]["gesamt_tok_s"] / st_a[hoch]["gesamt_tok_s"]
        ok = q >= 0.90
        gut &= ok
        print(f"{'BESTANDEN' if ok else 'NICHT BESTANDEN':16} gesamt bei {hoch} parallel {st_a[hoch]['gesamt_tok_s']} -> "
              f"{st_n[hoch]['gesamt_tok_s']} Tokens/s ({q:.0%}, Grenze 90 %)")
    frei = n.get("speicher_frei_gib")
    if frei is not None:
        ok = frei >= args.speicher_min
        gut &= ok
        print(f"{'BESTANDEN' if ok else 'NICHT BESTANDEN':16} verfügbarer Speicher {a.get('speicher_frei_gib')} -> "
              f"{frei} GiB (Grenze {args.speicher_min} GiB)")
    return 0 if gut else 1


# ---------------------------------------------------------------- vorrang

def vorrang(args):
    try:
        a = chat(args, {"messages": [{"role": "user", "content": "Antworte nur mit: ok"}],
                        "max_tokens": 200, "temperature": 0, "priority": -1}, 120)
        print("BESTANDEN        Feld «priority» angenommen:", (a["choices"][0]["message"].get("content") or "")[:40])
        return 0
    except urllib.error.HTTPError as e:
        print(f"NICHT BESTANDEN  Feld «priority» abgelehnt: HTTP {e.code} {e.read().decode('utf-8', 'replace')[:200]}")
        return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basis", default="http://127.0.0.1:8000")
    ap.add_argument("--modell", default="lokalspark")
    ap.add_argument("--schluesseldatei", default="~/.config/llm-api-key")
    sub = ap.add_subparsers(dest="befehl", required=True)
    sub.add_parser("pruefen")
    m = sub.add_parser("messen")
    m.add_argument("--stufen", default="1,2,4")
    m.add_argument("--tokens", type=int, default=256)
    m.add_argument("--wiederholungen", type=int, default=3, help="Läufe bei parallel 1, gewertet wird der mittlere")
    m.add_argument("--name", required=True)
    m.add_argument("--ablage", default=os.path.expanduser("~/glm53-cluster/results"))
    v = sub.add_parser("vergleich")
    v.add_argument("ausgang")
    v.add_argument("neu")
    v.add_argument("--speicher-min", type=float, default=14.0)
    sub.add_parser("vorrang")
    args = ap.parse_args()
    try:
        return {"pruefen": pruefen, "messen": messen, "vergleich": vergleich, "vorrang": vorrang}[args.befehl](args)
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        print(f"Fehler: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
