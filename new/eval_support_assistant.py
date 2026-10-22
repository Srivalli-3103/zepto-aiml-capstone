"""
Evaluates the Zepto support_assistant RAG service against a set of
test questions with known-correct keywords.

USAGE:
  1. Start the server first (in another terminal):
       cd zepto-aiml-capstone
       python support_assistant/main.py
  2. Edit questions.json (next to this script) with your real questions.
  3. Run:
       python eval_support_assistant.py

NOTE: As of this code review, MOCK_LLM defaults to "1" and even
MOCK_LLM=0 does not call a real LLM (see main.py) -- it returns the
top retrieved document snippet or a hardcoded JSON string. So this
script is measuring RETRIEVAL + templated response quality, not
real LLM answer quality. Report it honestly as that.
"""

import json
import time
import statistics
import sys
import urllib.request
import urllib.error

ENDPOINT = "http://localhost:7860/ask"
QUESTIONS_FILE = "questions.json"


def ask(query: str):
    payload = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            elapsed_ms = (time.time() - start) * 1000
            return body, elapsed_ms, None
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.time() - start) * 1000
        return None, elapsed_ms, f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}"
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return None, elapsed_ms, str(e)


def score(answer_text: str, expected_keywords: list):
    """Very simple auto-scorer: fraction of expected keywords found
    (case-insensitive) in the returned answer. You should eyeball
    the printed answers too -- this is a rough signal, not ground truth."""
    if not answer_text:
        return 0.0
    lower = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower)
    return hits / len(expected_keywords) if expected_keywords else 0.0


def main():
    try:
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            questions = json.load(f)
    except FileNotFoundError:
        print(f"Missing {QUESTIONS_FILE}. Create it next to this script -- "
              f"see questions.json template.")
        sys.exit(1)

    results = []
    print(f"{'#':<3} {'Latency(ms)':<12} {'KeywordScore':<13} Question")
    print("-" * 80)

    for i, q in enumerate(questions, 1):
        body, elapsed_ms, err = ask(q["question"])
        if err:
            print(f"{i:<3} {elapsed_ms:<12.0f} {'ERROR':<13} {q['question']!r} -> {err}")
            results.append({"question": q["question"], "ok": False, "latency_ms": elapsed_ms,
                             "score": 0.0, "answer": None, "error": err})
            continue

        answer_text = body.get("answer", "")
        s = score(answer_text, q.get("expected_keywords", []))
        print(f"{i:<3} {elapsed_ms:<12.0f} {s:<13.2f} {q['question']!r}")
        print(f"      -> {answer_text[:200]}")
        results.append({"question": q["question"], "ok": True, "latency_ms": elapsed_ms,
                         "score": s, "answer": answer_text, "error": None})

    ok_results = [r for r in results if r["ok"]]
    if ok_results:
        avg_latency = statistics.mean(r["latency_ms"] for r in ok_results)
        avg_score = statistics.mean(r["score"] for r in ok_results)
        # "Correct" = keyword score >= 0.5 as a simple threshold; adjust to taste
        correct_count = sum(1 for r in ok_results if r["score"] >= 0.5)
        accuracy_pct = 100 * correct_count / len(ok_results)
    else:
        avg_latency = avg_score = accuracy_pct = 0

    print("\n" + "=" * 80)
    print(f"Questions run:            {len(results)}")
    print(f"Successful responses:     {len(ok_results)}")
    print(f"Avg response latency:     {avg_latency:.0f} ms")
    print(f"Avg keyword-match score:  {avg_score:.2f}")
    print(f"Accuracy (>=0.5 match):   {accuracy_pct:.1f}%")
    print("=" * 80)
    print("\nReminder: this measures retrieval + templated response quality")
    print("(MOCK_LLM mode), not a live LLM's answer quality. Report it as such.")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to eval_results.json")


if __name__ == "__main__":
    main()
