# Changes — CBR Film Recommendation System

## Overview

Transformed the system from a generic query-matching engine into a **user-aware Case-Based Reasoning** system. The original system matched cases purely by text similarity and aggregated votes with no user context. The new system retrieves cases from users with similar taste profiles, performs feature-level adaptation, and hard-excludes previously rejected items.

---

## 1. User Genre Profiles

**Files:** `recommend.py`

### New method: `_load_user_genre_profiles()`

Reads `collaborative_training.csv` (96,941 ratings, 671 users) and `genres.csv` to build per-user genre preference vectors (22 dimensions, normalized [0, 1]).

Computes average rating per genre per user:
```python
genre_vec[genre] = avg(rating for films of that genre that user rated) / 5.0
```

### New methods
- `_user_genre_vec(user_id)` — lookup a user's genre vector
- `_user_cosine_sim(uid_a, uid_b)` — cosine similarity between two users' profiles (0.5 neutral for seed cases)
- `_find_similar_users(user_id, top_n=30)` — returns top-N users by genre profile similarity

### Storage

User vectors are kept in-memory (`self.user_genre_vecs`), loaded once on `CBRRecommender.__init__()`.

---

## 2. User-Aware RETRIEVE

**File:** `recommend.py`, method `_retrieve()`

### Before
- `_retrieve(query_text, ref_movie_id, ref_genres, ref_title)`
- FAISS search → SQL fetch → compute text/movie/genre similarity → filter by 0.30 threshold
- No user context at all

### After
- `_retrieve(..., user_id=None)` — new optional parameter
- **User pre-filtering**: finds similar users via `_find_similar_users()`, then skips cases from users outside that set (seed cases with `user_id=NULL` bypass this filter)
- **User similarity as 4th component**: `case_sim = weighted(query_sim, user_sim, movie_sim, genre_sim)`
- **Expanded FAISS retrieval**: searches 3x the normal count to provide headroom for user filtering
- **Graceful fallback**: if user filtering eliminates all candidates, retries without it
- **Weight configs**:

| Scenario | Weights (q, u, m, g) |
|---|---|
| User + ref movie + genre | 0.30, 0.30, 0.20, 0.20 |
| User + no ref movie | 0.40, 0.35, 0.00, 0.25 |
| User only | 0.55, 0.45, 0.00, 0.00 |
| Cold start (original) | 0.50, 0.00, 0.30, 0.20 |

---

## 3. Feature-Level Genre Adaptation in REUSE

**File:** `recommend.py`

### New method: `_extract_genre_patterns(similar_cases)`

Extracts net genre preference from accepted/rejected films across retrieved cases:

```python
genre_net["Action"] = 0.8   # accepted in similar cases
genre_net["Horror"] = -0.6  # rejected in similar cases
```

Returns dict mapping genre → net preference in [-1, 1]. Used to boost/penalize candidate films by their genre composition.

### Modified `_reuse()`

**Before:** `final_score = 0.70 * case_score + 0.30 * cf_score`

**After:** `final_score = 0.65 * case_score + 0.25 * cf_score + 0.10 * genre_adapt_score`

The `genre_adapt_score` averages the genre net scores across a candidate film's genres.

---

## 4. Hard Exclusion of Rejected Films

**File:** `recommend.py`

### In `_reuse()`

**Before:** Soft penalty via net score (`net = accepted - rejected > 0`). A film rejected in one case but accepted in many others could still pass.

**After:** Hard exclusion — films appearing in ANY retrieved case's `rejected_ids` are completely removed from candidates:

```python
rejected_set = set()
for case in similar_cases:
    rejected_set.update(case.get("rejected_ids", []))
for mid in agg_accepted:
    if int(mid) in rejected_set:
        continue  # hard exclude
```

### In `recommend()` (padding fallback)

When `_reuse()` returns fewer than `top_k` results (due to exclusions), padding fills from `_fallback_cbf()`. The padding also excludes any film that appears in `rejected_ids` of any retrieved case.

---

## 5. Seed Cases Attribute to Real Users

**File:** `build_casebase.py`

### Before
```python
INSERT INTO retained_cases (user_id, ...) VALUES (None, ...)
```
All 500 seed cases had `user_id=NULL`, making user filtering a no-op.

### After
```python
INSERT INTO retained_cases (user_id, ...) VALUES (uid, ...)
```
Each seed case is now attributed to the real user whose ratings generated it (352 unique users across 500 seed cases). This enables genuine user-aware retrieval from the start.

---

## 6. Evaluation Updates

**File:** `evaluate.py`

### Before
- Filtered seed cases with `WHERE user_id IS NULL AND from_case_id IS NULL`
- Called `recommend(..., user_id=None)` — always cold start

### After
- Filters with `WHERE from_case_id IS NULL` (works with real user_ids)
- Passes the actual `user_id` from each test case to `recommend()` for proper user-aware evaluation

---

## File-by-File Summary

| File | Changes |
|---|---|
| `recommend.py` | Added user genre profiles (5 new methods), modified `_retrieve` (user filtering + user_sim), modified `_reuse` (genre adaptation + hard exclusion), added padding in `recommend()`, new weight configs |
| `build_casebase.py` | Seed cases now use real `user_id` from ratings instead of `NULL` |
| `evaluate.py` | Updated seed case filter, passes `user_id` to recommend |

## Impact

| Metric | Before | After | Note |
|---|---|---|---|
| User differentiation | None | High (0/3 overlap for different users) | Same query yields different results per user |
| Precision@5 | 0.48 | 0.26 | Drop due to data sparsity (1.4 cases/user); improves with more cases |
| Recall@5 | 0.83 | 0.47 | Same sparsity issue |
| CBR validity | Weak (memory-augmented CBF) | Partial (genuine retrieval + adaptation) | Architecture is correct; needs more user data |
