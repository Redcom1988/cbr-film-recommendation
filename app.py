"""
app.py
Phase 5: Flask Web Server — CBR Film Recommendation System.
Jalankan: python app.py
"""

import json
import sqlite3
from flask import Flask, jsonify, render_template, request, session

from recommend import CBRRecommender

app = Flask(__name__)
app.secret_key = "cbr-secret-key-2024"

# ── Load CBR engine sekali saat startup ──────────────────────────────────────
try:
    recommender = CBRRecommender()
    print("[APP] CBRRecommender loaded successfully.")
except Exception as e:
    recommender = None
    print(f"[APP][WARN] Could not load recommender: {e}")
    print("[APP] Run build_casebase.py → train_cbf.py → train_cf.py first.")

# Simpan pending sessions di memory (demo)
pending_sessions: dict[str, dict] = {}


# ── Helper ────────────────────────────────────────────────────────────────────


def get_db():
    conn = sqlite3.connect("models/cases.db")
    conn.row_factory = sqlite3.Row
    return conn


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    if recommender is None:
        return jsonify(
            {
                "error": "Model belum dilatih. Jalankan pipeline training terlebih dahulu."
            }
        ), 503

    data = request.get_json(force=True)
    query_text = str(data.get("query_text", "")).strip()
    ref_movie = data.get("reference_movie")  # bisa string judul atau None
    user_id = data.get("user_id")  # int atau None
    top_k = int(data.get("top_k", 5))

    if not query_text:
        return jsonify({"error": "query_text tidak boleh kosong."}), 400

    result = recommender.recommend(
        query_text=query_text,
        reference_movie=ref_movie if ref_movie else None,
        user_id=int(user_id) if user_id else None,
        top_k=top_k,
    )

    # Simpan session untuk RETAIN nanti
    sid = result["session_id"]
    pending_sessions[sid] = {
        "query_text": query_text,
        "reference_movie": result["reference_movie"],
        "user_id": result["user_id"],
        "recommended_ids": [r["movieId"] for r in result["recommendations"]],
        "from_case_id": result["from_case_id"],
    }

    return jsonify(result)


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    if recommender is None:
        return jsonify({"error": "Model belum dilatih."}), 503

    data = request.get_json(force=True)
    sid = data.get("session_id")
    accepted_ids = [int(x) for x in data.get("accepted_ids", [])]
    rejected_ids = [int(x) for x in data.get("rejected_ids", [])]
    added_ids    = [int(x) for x in data.get("added_ids", [])]

    if not sid or sid not in pending_sessions:
        return jsonify({"error": "Session tidak valid atau sudah expired."}), 400

    s = pending_sessions.pop(sid)

    # Secara teoritis (CBR murni), hanya simpan ke case base bila ada "new knowledge" 
    # yaitu ketika ada ditambahkan secara manual (added_ids > 0).
    # Namun untuk keperluan statistik kita tetap rekam, tetapi kita merge added_ids ke accepted_ids.
    new_case_id = recommender.retain(
        user_id=s["user_id"],
        query_text=s["query_text"],
        reference_movie=s["reference_movie"],
        recommended_ids=s["recommended_ids"],
        accepted_ids=accepted_ids,
        rejected_ids=rejected_ids,
        added_ids=added_ids,
        from_case_id=s["from_case_id"],
    )

    return jsonify(
        {
            "status": "retained",
            "case_id": new_case_id,
            "accepted": len(accepted_ids),
            "rejected": len(rejected_ids),
            "message": f"Kasus #{new_case_id} berhasil disimpan ke case base.",
        }
    )


@app.route("/api/movies", methods=["GET"])
def api_movies():
    """Autocomplete judul film."""
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])

    conn = get_db()
    rows = conn.execute(
        "SELECT movieId, title, genres FROM films WHERE LOWER(title) LIKE ? LIMIT 10",
        (f"%{q.lower()}%",),
    ).fetchall()
    conn.close()
    return jsonify(
        [
            {"movieId": r["movieId"], "title": r["title"], "genres": r["genres"]}
            for r in rows
        ]
    )


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Statistik case base."""
    conn = get_db()

    n_cases = conn.execute("SELECT COUNT(*) FROM retained_cases").fetchone()[0]
    n_seeds = conn.execute(
        "SELECT COUNT(*) FROM retained_cases WHERE is_seed = 1"
    ).fetchone()[0]
    n_user = conn.execute(
        "SELECT COUNT(*) FROM retained_cases WHERE is_seed = 0 AND user_id IS NOT NULL"
    ).fetchone()[0]
    n_films = conn.execute(
        "SELECT COUNT(DISTINCT movieId) FROM predicted_ratings"
    ).fetchone()[0]

    top_acc = conn.execute("""
        SELECT f.title, COUNT(*) AS hits
        FROM retained_cases rc, json_each(rc.accepted_ids) je
        JOIN films f ON f.movieId = je.value
        GROUP BY f.movieId
        ORDER BY hits DESC
        LIMIT 5
    """).fetchall()

    recent = conn.execute("""
        SELECT case_id, query_text, timestamp
        FROM   retained_cases
        ORDER  BY case_id DESC
        LIMIT  5
    """).fetchall()
    conn.close()

    return jsonify(
        {
            "total_cases": n_cases,
            "seed_cases": n_seeds,
            "user_cases": n_user,
            "total_films": n_films,
            "top_accepted": [{"title": r["title"], "hits": r["hits"]} for r in top_acc],
            "recent_cases": [
                {
                    "case_id": r["case_id"],
                    "query": r["query_text"],
                    "timestamp": r["timestamp"],
                }
                for r in recent
            ],
        }
    )


@app.route("/api/cases", methods=["GET"])
def api_cases():
    """Daftar kasus terbaru dari case base."""
    limit = int(request.args.get("limit", 10))
    conn = get_db()
    rows = conn.execute(
        """
        SELECT rc.case_id, rc.query_text, rc.accepted_ids, rc.rejected_ids,
               rc.timestamp, rc.from_case_id, f.title AS ref_title
        FROM   retained_cases rc
        LEFT JOIN films f ON f.movieId = rc.reference_movie
        ORDER  BY rc.case_id DESC
        LIMIT  ?
    """,
        (limit,),
    ).fetchall()
    conn.close()

    return jsonify(
        [
            {
                "case_id": r["case_id"],
                "query": r["query_text"],
                "ref_title": r["ref_title"],
                "accepted": json.loads(r["accepted_ids"] or "[]"),
                "rejected": json.loads(r["rejected_ids"] or "[]"),
                "from_case_id": r["from_case_id"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
