#!/usr/bin/env python3
"""Durchsatz mit verschiedenen Fragen je Anfrage und Temperatur 0.7 (keine gleichen Prompts, keine Kopie-Hilfe).
  python3 bench_gemischt.py --basis http://127.0.0.1:8888 --name tf1104 [--stufen 1,2,4,8]
"""
import argparse, json, os, threading, time, urllib.request

FRAGEN = [
    "Erkläre ausführlich, wie eine Wärmepumpe im Winter funktioniert und worauf man bei der Auslegung achten muss.",
    "Write a detailed Python tutorial on asyncio with three worked examples and explanations.",
    "Beschreibe die Geschichte der Stadt Basel vom Mittelalter bis heute in mehreren Abschnitten.",
    "Explain the differences between TCP congestion control algorithms Reno, CUBIC and BBR in depth.",
    "Schreibe eine Kurzgeschichte über einen Leuchtturmwärter, der ein Funksignal aus der Zukunft empfängt.",
    "Give a thorough comparison of PostgreSQL and SQLite for a small home server, with concrete recommendations.",
    "Erkläre Schritt für Schritt, wie man ein Solardach plant: Ausrichtung, Ertrag, Wechselrichter, Speicher.",
    "Describe how a modern CPU executes instructions out of order, including reorder buffer and branch prediction.",
]


def kopf(a):
    k = {"Content-Type": "application/json"}
    p = os.path.expanduser(a.schluesseldatei)
    if os.path.exists(p) and open(p).read().strip():
        k["Authorization"] = "Bearer " + open(p).read().strip()
    return k


def eine(a, i, aus):
    koerper = {"model": a.modell, "messages": [{"role": "user", "content": FRAGEN[i % len(FRAGEN)]}],
               "temperature": 0.7, "max_tokens": a.tokens, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}
    r = urllib.request.Request(a.basis + "/v1/chat/completions", json.dumps(koerper).encode(), kopf(a))
    t0 = time.time(); erst = None; n = 0; usage = None
    with urllib.request.urlopen(r, timeout=1800) as h:
        for zeile in h:
            zeile = zeile.decode().strip()
            if not zeile.startswith("data:") or zeile.endswith("[DONE]"):
                continue
            d = json.loads(zeile[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning"):
                    if erst is None:
                        erst = time.time()
                    n += 1
    ende = time.time()
    tok = (usage or {}).get("completion_tokens") or n
    aus[i] = {"tokens": tok, "ttft": round((erst or ende) - t0, 3), "decode_s": round(ende - (erst or t0), 2),
              "ende": ende, "start": t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="http://127.0.0.1:8000"); ap.add_argument("--modell", default="lokalspark")
    ap.add_argument("--schluesseldatei", default="~/.config/llm-api-key"); ap.add_argument("--name", required=True)
    ap.add_argument("--stufen", default="1,2,4,8"); ap.add_argument("--tokens", type=int, default=600)
    a = ap.parse_args()
    erg = {"name": a.name, "zeit": time.strftime("%d-%m-%Y %H:%M:%S"), "stufen": {}}
    for c in [int(x) for x in a.stufen.split(",")]:
        aus = {}
        th = [threading.Thread(target=eine, args=(a, i, aus)) for i in range(c)]
        t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
        wand = max(v["ende"] for v in aus.values()) - t0
        ges = sum(v["tokens"] for v in aus.values())
        je = [v["tokens"] / v["decode_s"] for v in aus.values() if v["decode_s"] > 0]
        s = {"gesamt_tok_s": round(ges / wand, 1), "je_anfrage_tok_s": round(sum(je) / len(je), 1),
             "ttft_mittel": round(sum(v["ttft"] for v in aus.values()) / c, 3), "tokens": ges}
        erg["stufen"][str(c)] = s
        print(f"parallel {c}: gesamt {s['gesamt_tok_s']} tok/s, je Anfrage {s['je_anfrage_tok_s']}, TTFT {s['ttft_mittel']} s, {ges} Tokens", flush=True)
    p = os.path.expanduser(f"~/glm53-cluster/results/gemischt-{a.name}.json")
    json.dump(erg, open(p, "w"), indent=1); print("abgelegt", p)


if __name__ == "__main__":
    main()
