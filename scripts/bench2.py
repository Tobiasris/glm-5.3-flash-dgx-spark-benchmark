#!/usr/bin/env python3
"""Second-round benchmark for an OpenAI-compatible endpoint. Standard library only.

  python3 bench2.py --base http://127.0.0.1:8888 --model NAME --name RUN \
      --parts mixed,code,interleave,longctx --levels 1,2,4 --runs 5

Parts:
  mixed       prose/Q&A prompts, a different one per stream and per run, T=0.7, thinking off
  code        code-generation prompts, same procedure
  interleave  one stream decodes while a cold 50k-token prompt is prefilled next to it
  longctx     one cold ~150k-token prompt, then a second turn on the same conversation (cache resume)
Results: JSON in --out (default ~/glm53-cluster/results/r2-RUN.json), rewritten after every part.
"""
import argparse, json, os, random, statistics, string, threading, time, urllib.request

MIXED = [
    "Erkläre ausführlich, wie eine Wärmepumpe im Winter funktioniert und worauf man bei der Auslegung achten muss.",
    "Explain in depth how TCP congestion control differs between Reno, CUBIC and BBR.",
    "Beschreibe die Geschichte der Stadt Basel vom Mittelalter bis heute in mehreren Abschnitten.",
    "Give a thorough comparison of PostgreSQL and SQLite for a small home server, with concrete recommendations.",
    "Schreibe eine Kurzgeschichte über einen Leuchtturmwärter, der ein Funksignal aus der Zukunft empfängt.",
    "Describe how a modern CPU executes instructions out of order, including the reorder buffer and branch prediction.",
    "Erkläre Schritt für Schritt, wie man ein Solardach plant: Ausrichtung, Ertrag, Wechselrichter, Speicher.",
    "Explain how vaccines train the immune system, from antigen presentation to memory cells.",
    "Beschreibe, wie ein Käse wie Gruyère hergestellt wird, von der Milch bis zur Reifung.",
    "Write a detailed guide to planning a week-long hiking trip in the Alps, including gear and safety.",
    "Erkläre die Unterschiede zwischen Wechselstrom und Gleichstrom und warum das Netz Wechselstrom nutzt.",
    "Explain how public-key cryptography works and why RSA and elliptic curves differ in key size.",
    "Schreibe einen Aufsatz über die Folgen der Erfindung des Buchdrucks für Europa.",
    "Describe the life cycle of a star from molecular cloud to white dwarf, neutron star or black hole.",
    "Erkläre, wie ein Verbrennungsmotor und ein Elektromotor Drehmoment erzeugen, und vergleiche beide.",
    "Explain what causes inflation and which tools a central bank has against it.",
]
CODE = [
    "Write a Python module implementing an LRU cache with TTL support, full type hints, and unit tests using pytest.",
    "Implement a thread-safe bounded blocking queue in C++20 with condition variables, including a usage example.",
    "Write a Rust command-line tool that walks a directory tree and reports duplicate files by SHA-256 hash.",
    "Write a TypeScript Express server with JWT authentication, a users router, input validation and error middleware.",
    "Implement Dijkstra's algorithm and A* in Go on a weighted grid, with a small benchmark in main().",
    "Write a Python asyncio web crawler with a concurrency limit, retries with backoff, and robots.txt handling.",
    "Write a SQL schema for a bookstore (PostgreSQL) with indexes, plus ten non-trivial queries with comments.",
    "Implement a recursive-descent parser and evaluator for arithmetic expressions with variables in Java.",
    "Write a Bash script that rotates and compresses log files, with a dry-run flag, locking, and usage help.",
    "Implement a minimal HTTP/1.1 server in C using epoll that serves static files, with comments.",
    "Write a Python class for a persistent key-value store using an append-only log and compaction.",
    "Write a React component in TypeScript for a sortable, filterable, paginated data table with hooks.",
    "Implement a rate limiter (token bucket and sliding window) in Go with tests.",
    "Write a Python script that parses a large CSV in streaming fashion and computes grouped statistics.",
    "Implement a trie with insert, delete, prefix search and autocomplete in Kotlin, with tests.",
    "Write a Dockerfile and docker-compose.yml for a Django app with PostgreSQL, Redis and Celery, with comments.",
]
ESSAY = ("Write a very long, detailed essay of at least 4000 words on the history of computing, "
         "from mechanical calculators to modern GPUs. Do not stop early.")


def headers(a):
    h = {"Content-Type": "application/json"}
    p = os.path.expanduser(a.keyfile)
    if os.path.exists(p):
        k = open(p).read().strip()
        if k:
            h["Authorization"] = "Bearer " + k
    return h


def stream(a, messages, max_tokens, temperature=0.7, out=None, key=None, trace=False):
    body = {"model": a.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(), headers(a))
    t0 = time.time(); first = None; usage = None; chars = 0; marks = []
    with urllib.request.urlopen(req, timeout=3600) as h:
        for line in h:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                txt = (dl.get("content") or "") + (dl.get("reasoning_content") or "") + (dl.get("reasoning") or "")
                if txt:
                    now = time.time()
                    if first is None:
                        first = now
                    chars += len(txt)
                    if trace:
                        marks.append((now, len(txt)))
    end = time.time()
    r = {"t0": t0, "first": first or end, "end": end, "chars": chars,
         "completion_tokens": (usage or {}).get("completion_tokens", 0),
         "prompt_tokens": (usage or {}).get("prompt_tokens", 0),
         "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}
    if trace:
        r["marks"] = marks
    if out is not None:
        out[key] = r
    return r


def summary(xs):
    return {"mean": round(statistics.mean(xs), 2), "min": round(min(xs), 2), "max": round(max(xs), 2),
            "stdev": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0, "runs": [round(x, 2) for x in xs]}


def throughput(a, prompts, levels, runs, tokens):
    res = {}; offset = 0
    for c in levels:
        tot, per, ttft = [], [], []
        level_start = time.time()
        for _ in range(runs):
            out = {}
            th = [threading.Thread(target=stream, args=(a, [{"role": "user", "content": prompts[(offset + i) % len(prompts)]}], tokens),
                                   kwargs={"out": out, "key": i}) for i in range(c)]
            offset += c
            t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
            if len(out) < c:
                continue
            wall = max(v["end"] for v in out.values()) - t0
            tot.append(sum(v["completion_tokens"] for v in out.values()) / wall)
            per.append(statistics.mean(v["completion_tokens"] / max(v["end"] - v["first"], 1e-6) for v in out.values()))
            ttft.append(statistics.mean(v["first"] - v["t0"] for v in out.values()))
        res[str(c)] = {"total_tok_s": summary(tot), "per_request_tok_s": summary(per), "ttft_s": summary(ttft),
                       "t_start": round(level_start, 1), "t_end": round(time.time(), 1)}
        print(f"  level {c}: total {res[str(c)]['total_tok_s']['mean']} ±{res[str(c)]['total_tok_s']['stdev']} tok/s "
              f"(min {res[str(c)]['total_tok_s']['min']}, max {res[str(c)]['total_tok_s']['max']}), "
              f"per request {res[str(c)]['per_request_tok_s']['mean']}, ttft {res[str(c)]['ttft_s']['mean']} s", flush=True)
    return res


def random_prompt(words, seed):
    rng = random.Random(seed)
    return " ".join("".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9))) for _ in range(words))


def interleave(a, runs=2):
    """Stream A decodes an essay; 4 s in, request B sends a cold ~50k-token prompt. What does A keep?"""
    res = []
    for r in range(runs):
        out = {}
        ta = threading.Thread(target=stream, args=(a, [{"role": "user", "content": ESSAY}], 1500),
                              kwargs={"out": out, "key": "A", "trace": True})
        ta.start(); time.sleep(4)
        pb = "Antworte nur mit OK.\n" + random_prompt(14600, time.time_ns())
        b = stream(a, [{"role": "user", "content": pb}], 1, temperature=0)
        ta.join()
        A = out["A"]; marks = A["marks"]
        tpc = A["completion_tokens"] / max(A["chars"], 1)
        def rate(lo, hi):
            ch = sum(n for t, n in marks if lo <= t < hi)
            return ch * tpc / max(hi - lo, 1e-6)
        inside = [t for t, _ in marks if b["t0"] <= t <= b["end"]]
        pts = [b["t0"]] + inside + [b["end"]]
        gap = max(y - x for x, y in zip(pts, pts[1:]))
        one = {"prefill_prompt_tokens": b["prompt_tokens"], "prefill_s": round(b["end"] - b["t0"], 2),
               "prefill_tok_s": round(b["prompt_tokens"] / (b["end"] - b["t0"])),
               "decode_before_tok_s": round(rate(A["first"], b["t0"]), 1),
               "decode_during_tok_s": round(rate(b["t0"], b["end"]), 1),
               "decode_after_tok_s": round(rate(b["end"], A["end"]), 1) if A["end"] - b["end"] > 1 else None,
               "worst_pause_s": round(gap, 2), "cached_tokens": b["cached_tokens"]}
        print("  interleave", json.dumps(one), flush=True)
        res.append(one)
    return res


def longctx(a):
    p = "Merke dir das Wort am Ende. Antworte nur mit OK.\n" + random_prompt(44000, time.time_ns()) + "\nDas Wort ist: Murmeltier."
    m = [{"role": "user", "content": p}]
    r1 = stream(a, m, 4, temperature=0)
    m += [{"role": "assistant", "content": "OK"}, {"role": "user", "content": "Welches Wort solltest du dir merken? Antworte mit einem Wort."}]
    body = {"model": a.model, "messages": m, "temperature": 0, "max_tokens": 20,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(), headers(a))
    d = json.load(urllib.request.urlopen(req, timeout=3600)); t2 = time.time() - t0
    ans = (d["choices"][0]["message"].get("content") or "").strip()
    u = d.get("usage", {})
    one = {"prompt_tokens": r1["prompt_tokens"], "turn1_s": round(r1["end"] - r1["t0"], 1),
           "turn1_tok_s": round(r1["prompt_tokens"] / (r1["end"] - r1["t0"])), "turn1_cached": r1["cached_tokens"],
           "turn2_s": round(t2, 2), "turn2_prompt_tokens": u.get("prompt_tokens"),
           "turn2_cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "recall_ok": "murmeltier" in ans.lower(), "answer": ans[:60]}
    print("  longctx", json.dumps(one), flush=True)
    return one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000"); ap.add_argument("--model", default="lokalspark")
    ap.add_argument("--keyfile", default="~/.config/llm-api-key"); ap.add_argument("--name", required=True)
    ap.add_argument("--parts", default="mixed,code,interleave,longctx"); ap.add_argument("--levels", default="1,2,4")
    ap.add_argument("--runs", type=int, default=5); ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--out", default=os.path.expanduser("~/glm53-cluster/results"))
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",")]
    res = {"name": a.name, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "tokens_per_request": a.tokens, "runs": a.runs}
    path = os.path.join(a.out, f"r2-{a.name}.json")
    for part in a.parts.split(","):
        print(part, flush=True); t0 = time.time()
        try:
            if part == "mixed":
                res[part] = throughput(a, MIXED, levels, a.runs, a.tokens)
            elif part == "code":
                res[part] = throughput(a, CODE, [l for l in levels if l in (1, 4)], max(3, a.runs - 2), a.tokens)
            elif part == "interleave":
                res[part] = interleave(a)
            elif part == "longctx":
                res[part] = longctx(a)
        except Exception as e:
            res[part] = {"error": repr(e)[:300]}; print("  ERROR", res[part], flush=True)
        res[part + "_seconds"] = round(time.time() - t0, 1)
        json.dump(res, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
