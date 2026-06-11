"""
evaluate.py
Phase 6: Evaluasi sistem CBR.
  - Split seed cases 80/20
  - Hitung Precision@K, Recall@K, NDCG@K
Jalankan: python evaluate.py --k 5
"""

import argparse
import json
import math
import sqlite3
import random
import sys

from recommend import CBRRecommender


def dcg(rel: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rel[:k]))


def ndcg(recommended: list[int], relevant: set[int], k: int) -> float:
    rel = [1 if mid in relevant else 0 for mid in recommended[:k]]
    ideal = sorted(rel, reverse=True)
    d = dcg(rel, k)
    id_ = dcg(ideal, k)
    return d / id_ if id_ > 0 else 0.0


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    hits = sum(1 for mid in recommended[:k] if mid in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for mid in recommended[:k] if mid in relevant)
    return hits / len(relevant)


def evaluate(k: int = 5, test_split: float = 0.2):
    print("\n" + "=" * 55)
    print(f" Evaluasi CBR Recommender  |  @K={k}  |  test={test_split:.0%}")
    print("=" * 55)

    # ── Ambil seed cases dari DB ──────────────────────────
    conn = sqlite3.connect("models/cases.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT case_id, query_text, reference_movie, accepted_ids, user_id
        FROM   retained_cases
        WHERE  from_case_id IS NULL
        ORDER  BY case_id
    """).fetchall()
    conn.close()

    if not rows:
        print("[ERROR] Tidak ada seed cases. Jalankan build_casebase.py dulu.")
        sys.exit(1)

    print(f"Total seed cases: {len(rows)}")

    # ── Split ─────────────────────────────────────────────
    random.seed(42)
    shuffled = list(rows)
    random.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_split))
    test_set = shuffled[:n_test]
    train_set = shuffled[n_test:]

    print(f"Train: {len(train_set)} | Test: {len(test_set)}")

    # ── Load recommender ──────────────────────────────────
    rec = CBRRecommender()

    # ── Evaluasi tiap kasus test ──────────────────────────
    p_scores, r_scores, n_scores = [], [], []
    skip = 0

    for test_case in test_set:
        accepted = json.loads(test_case["accepted_ids"] or "[]")
        relevant = set(int(x) for x in accepted)
        if not relevant:
            skip += 1
            continue

        ref_movie = test_case["reference_movie"]
        query = test_case["query_text"]

        result = rec.recommend(
            query, reference_movie=ref_movie, user_id=test_case["user_id"], top_k=k
        )
        recommended = [r["movieId"] for r in result["recommendations"]]

        p_scores.append(precision_at_k(recommended, relevant, k))
        r_scores.append(recall_at_k(recommended, relevant, k))
        n_scores.append(ndcg(recommended, relevant, k))

    evaluated = len(p_scores)
    if evaluated == 0:
        print("[WARN] Tidak ada kasus test yang bisa dievaluasi.")
        return

    p_mean = sum(p_scores) / evaluated
    r_mean = sum(r_scores) / evaluated
    n_mean = sum(n_scores) / evaluated

    conn2 = sqlite3.connect("models/cases.db")
    n_all = conn2.execute("SELECT COUNT(*) FROM retained_cases").fetchone()[0]
    conn2.close()

    print("\n" + "-" * 40)
    print(f"  Precision@{k}   : {p_mean:.4f}")
    print(f"  Recall@{k}      : {r_mean:.4f}")
    print(f"  NDCG@{k}        : {n_mean:.4f}")
    print(f"  Evaluasi       : {evaluated} kasus ({skip} dilewati)")
    print(f"  Case base size : {n_all} kasus")
    print("-" * 40)

    return {
        "precision": p_mean,
        "recall": r_mean,
        "ndcg": n_mean,
        "evaluated": evaluated,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--test_split", type=float, default=0.2)
    args = parser.parse_args()
    evaluate(k=args.k, test_split=args.test_split)
