#!/usr/bin/env python3
"""Gütetest für den A/B-Vergleich der Engines auf den Sparks (16-11-2026). Nur Standardbibliothek.

  python3 guetetest.py --basis http://127.0.0.1:8000 --modell lokalspark --name vllm
  python3 guetetest.py --basis http://127.0.0.1:8888 --modell glm --name tf1104

Prüft:
  kopie     exaktes Kopieren von JSON mit langen Ziffernfolgen (Fehlerbericht #9/#59 des Rezepts), 12 Fälle
  werkzeug  Werkzeugaufrufe mit drei angebotenen Werkzeugen, Argumente als JSON (8 Fälle)
  rechnen   Rechenaufgaben mit eindeutiger Lösung (8 Fälle), Denken an
  prefill   kalter Prompt (zufälliger Inhalt) 8k und 32k Tokens: Zeit bis erstes Token
Ergebnis als JSON nach --ablage/<name>-guete.json; Schlüssel aus --schluesseldatei, nie ausgegeben.
"""
import argparse, json, os, random, re, string, time, urllib.request


def kopf(a):
    k = {"Content-Type": "application/json"}
    p = os.path.expanduser(a.schluesseldatei)
    if os.path.exists(p):
        w = open(p).read().strip()
        if w:
            k["Authorization"] = "Bearer " + w
    return k


def chat(a, koerper, frist=900):
    koerper = dict(koerper, model=a.modell)
    r = urllib.request.Request(a.basis + "/v1/chat/completions", json.dumps(koerper).encode(), kopf(a))
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=frist) as h:
        d = json.load(h)
    return d, time.time() - t0


def zufall_json(rng):
    return {
        "kunde": "".join(rng.choices(string.ascii_uppercase, k=3)) + "-" + str(rng.randrange(10**11, 10**12)),
        "iban": "CH" + "".join(rng.choices(string.digits, k=19)),
        "betraege": [round(rng.uniform(0, 99999), 2) for _ in range(5)],
        "uuid": "-".join("".join(rng.choices("0123456789abcdef", k=n)) for n in (8, 4, 4, 4, 12)),
        "zaehler": rng.randrange(10**15, 10**16),
        "codes": ["".join(rng.choices(string.digits, k=rng.randint(9, 14))) for _ in range(4)],
    }


def t_kopie(a, rng):
    ok, faelle = 0, []
    for i in range(12):
        obj = zufall_json(rng)
        quelle = json.dumps(obj, ensure_ascii=False)
        d, s = chat(a, {"messages": [
            {"role": "system", "content": "Gib ausschliesslich das JSON-Objekt zurück, Zeichen für Zeichen unverändert, ohne Codeblock."},
            {"role": "user", "content": "Kopiere exakt:\n" + quelle}],
            "temperature": 0, "max_tokens": 6000})
        text = (d["choices"][0]["message"].get("content") or "").strip()
        text = re.sub(r"^```(json)?\s*|\s*```$", "", text)
        if "{" in text:  # Denktext ohne Trennung: das letzte Objekt nehmen
            text = text[text.rfind("{\"kunde\""):] if "{\"kunde\"" in text else text
            text = text[:text.rfind("}") + 1]
        try:
            gleich = json.loads(text) == obj
        except ValueError:
            gleich = False
        ok += gleich
        if not gleich:
            faelle.append({"soll": quelle, "ist": text[:600]})
    return {"bestanden": ok, "von": 12, "fehler": faelle[:4]}


WERKZEUGE = [
    {"type": "function", "function": {"name": "wetter", "description": "Wetter für einen Ort",
     "parameters": {"type": "object", "properties": {"ort": {"type": "string"}, "tage": {"type": "integer"}},
                    "required": ["ort", "tage"]}}},
    {"type": "function", "function": {"name": "datei_suchen", "description": "Dateien nach Mustern suchen",
     "parameters": {"type": "object", "properties": {"muster": {"type": "array", "items": {"type": "string"}},
                    "ordner": {"type": "string"}}, "required": ["muster", "ordner"]}}},
    {"type": "function", "function": {"name": "rechnen", "description": "Ausdruck berechnen",
     "parameters": {"type": "object", "properties": {"ausdruck": {"type": "string"}}, "required": ["ausdruck"]}}},
]
AUFGABEN_W = [
    ("Wie wird das Wetter in Basel die nächsten 3 Tage?", "wetter", {"ort": "Basel", "tage": 3}),
    ("Suche im Ordner /daten nach *.pdf und *.docx.", "datei_suchen", {"ordner": "/daten", "muster": ["*.pdf", "*.docx"]}),
    ("Berechne 17*23+5 mit dem Werkzeug.", "rechnen", None),
    ("Wetter in Bern für 7 Tage bitte.", "wetter", {"ort": "Bern", "tage": 7}),
    ("Finde in /home/user alle Dateien mit *.log, *.txt und *.md.", "datei_suchen", {"ordner": "/home/user", "muster": ["*.log", "*.txt", "*.md"]}),
    ("Wie ist das Wetter morgen in Zürich? Nur 1 Tag.", "wetter", {"ort": "Zürich", "tage": 1}),
    ("Rechne (1234-567)/3 mit dem Rechenwerkzeug aus.", "rechnen", None),
    ("Durchsuche /var/log nach *.gz.", "datei_suchen", {"ordner": "/var/log", "muster": ["*.gz"]}),
]


def t_werkzeug(a):
    ok, fehler = 0, []
    for frage, name, args_soll in AUFGABEN_W:
        d, s = chat(a, {"messages": [{"role": "user", "content": frage}], "tools": WERKZEUGE,
                        "tool_choice": "auto", "temperature": 0, "max_tokens": 2000})
        m = d["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        gut = False
        if calls and calls[0]["function"]["name"] == name:
            try:
                ist = json.loads(calls[0]["function"]["arguments"])
                gut = args_soll is None or all(ist.get(k) == v for k, v in args_soll.items())
            except ValueError:
                pass
        ok += gut
        if not gut:
            fehler.append({"frage": frage, "calls": calls[:1], "content": (m.get("content") or "")[:300]})
    return {"bestanden": ok, "von": len(AUFGABEN_W), "fehler": fehler[:4]}


AUFGABEN_R = [
    ("Wie viel ist 48'219 mal 37?", "1784103"),
    ("Ein Zug fährt 2 h 15 min mit 84 km/h. Wie viele km fährt er?", "189"),
    ("Was ist die Summe aller ganzen Zahlen von 1 bis 250?", "31375"),
    ("Wie viele Primzahlen gibt es zwischen 1 und 100?", "25"),
    ("Ein Artikel kostet 80 CHF, wird um 25 % erhöht und dann um 20 % gesenkt. Neuer Preis in CHF?", "80"),
    ("Wie viele Sekunden hat eine Woche?", "604800"),
    ("2 hoch 20?", "1048576"),
    ("Löse 3x + 17 = 101. x = ?", "28"),
]


def t_rechnen(a):
    ok, fehler, toks, zeit = 0, [], 0, 0.0
    for frage, soll in AUFGABEN_R:
        d, s = chat(a, {"messages": [{"role": "user", "content": frage + " Antworte am Schluss mit «ERGEBNIS: <Zahl>»."}],
                        "temperature": 0.6, "max_tokens": 6000})
        text = d["choices"][0]["message"].get("content") or ""
        z = re.findall(r"ERGEBNIS:\s*\**\s*([0-9'’., ]+)", text)
        ist = re.sub(r"[^0-9]", "", z[-1].split(".")[0].split(",")[0]) if z else ""
        gut = ist == soll
        ok += gut
        toks += d.get("usage", {}).get("completion_tokens", 0)
        zeit += s
        if not gut:
            fehler.append({"frage": frage, "soll": soll, "ist": ist, "ende": text[-200:]})
    return {"bestanden": ok, "von": len(AUFGABEN_R), "ausgabe_tokens": toks, "sekunden": round(zeit, 1), "fehler": fehler}


def t_prefill(a, rng):
    aus = {}
    for ziel in (8000, 32000):
        woerter = [ "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9))) for _ in range(int(ziel * 0.75)) ]
        prompt = "Zähle nicht, antworte nur mit OK.\n" + " ".join(woerter)
        koerper = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False}}
        d, s = chat(a, koerper, 1800)
        pt = d.get("usage", {}).get("prompt_tokens", 0)
        aus[str(ziel)] = {"prompt_tokens": pt, "sekunden": round(s, 2), "tokens_s": round(pt / s) if s else 0,
                          "cached": (d.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens")}
    return aus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="http://127.0.0.1:8000")
    ap.add_argument("--modell", default="lokalspark")
    ap.add_argument("--schluesseldatei", default="~/.config/llm-api-key")
    ap.add_argument("--name", required=True)
    ap.add_argument("--ablage", default=os.path.expanduser("~/glm53-cluster/results"))
    ap.add_argument("--nur", default="kopie,werkzeug,rechnen,prefill")
    a = ap.parse_args()
    rng = random.Random(16112026)
    erg = {"name": a.name, "zeit": time.strftime("%d-%m-%Y %H:%M:%S"), "basis": a.basis}
    for teil in a.nur.split(","):
        t0 = time.time()
        try:
            erg[teil] = {"kopie": lambda: t_kopie(a, rng), "werkzeug": lambda: t_werkzeug(a),
                         "rechnen": lambda: t_rechnen(a), "prefill": lambda: t_prefill(a, random.Random(time.time_ns()))}[teil]()
        except Exception as e:  # Fehler je Teil festhalten, weitermachen
            erg[teil] = {"fehler": repr(e)[:400]}
        erg[teil]["dauer_s"] = round(time.time() - t0, 1)
        kurz = {k: v for k, v in erg[teil].items() if k != "fehler"} if isinstance(erg[teil], dict) else erg[teil]
        print(teil, json.dumps(kurz, ensure_ascii=False), flush=True)
    os.makedirs(a.ablage, exist_ok=True)
    pfad = os.path.join(a.ablage, a.name + "-guete.json")
    json.dump(erg, open(pfad, "w"), ensure_ascii=False, indent=1)
    print("abgelegt", pfad)


if __name__ == "__main__":
    main()
