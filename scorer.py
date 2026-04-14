"""
scorer.py
---------
Composite scoring and candidate ranking engine.

Scoring weights:
    Skill match score   → 50%
    TF-IDF similarity   → 35%
    Experience bonus    → 15%
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessor import deep_clean
from src.skill_extractor import (
    skill_match_score,
    experience_bonus_score,
    compute_skill_gap
)

# Scoring weights (must sum to 1.0)
WEIGHTS = {
    "skill_match": 0.50,
    "tfidf_similarity": 0.35,
    "experience_bonus": 0.15
}


# ---------------------------------------------------------------------------
# TF-IDF similarity
# ---------------------------------------------------------------------------

def compute_tfidf_similarity(resume_texts: list, jd_text: str) -> np.ndarray:
    """
    Vectorise resumes + JD with TF-IDF and compute cosine similarity
    of each resume against the job description.

    Parameters
    ----------
    resume_texts : list of str
        Raw or cleaned resume strings.
    jd_text : str
        Job description text.

    Returns
    -------
    np.ndarray of shape (n_resumes,)
        Cosine similarity score (0–1) per resume.
    """
    cleaned_resumes = [deep_clean(r) for r in resume_texts]
    cleaned_jd = deep_clean(jd_text)

    corpus = cleaned_resumes + [cleaned_jd]

    vectoriser = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10_000,
        sublinear_tf=True
    )
    tfidf_matrix = vectoriser.fit_transform(corpus)

    jd_vector = tfidf_matrix[-1]            # last entry is the JD
    resume_vectors = tfidf_matrix[:-1]      # all others are resumes

    similarities = cosine_similarity(resume_vectors, jd_vector).flatten()
    return similarities


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def score_single_resume(resume_text: str, jd_text: str, tfidf_score: float) -> dict:
    """
    Compute all component scores and a weighted composite for one resume.
    """
    s_skill = skill_match_score(resume_text, jd_text)
    s_tfidf = round(tfidf_score * 100, 2)
    s_exp = experience_bonus_score(resume_text)

    composite = round(
        s_skill * WEIGHTS["skill_match"]
        + s_tfidf * WEIGHTS["tfidf_similarity"]
        + s_exp * WEIGHTS["experience_bonus"],
        2
    )

    gap = compute_skill_gap(resume_text, jd_text)

    return {
        "skill_score": s_skill,
        "tfidf_score": s_tfidf,
        "experience_score": s_exp,
        "composite_score": composite,
        "matched_skills": gap["matched_skills"],
        "missing_skills": gap["missing_skills"],
        "extra_skills": gap["extra_skills"],
        "skill_match_rate": gap["match_rate"]
    }


def rank_candidates(df: pd.DataFrame, jd_text: str) -> pd.DataFrame:
    """
    Score and rank all candidates in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least a 'resume_text' column.
        Optional: 'candidate_name', 'category' columns.
    jd_text : str
        The job description to screen against.

    Returns
    -------
    pd.DataFrame
        Original columns + scoring columns, sorted by composite_score desc.
    """
    resume_texts = df["resume_text"].tolist()

    # Compute TF-IDF similarities in batch (more efficient)
    tfidf_scores = compute_tfidf_similarity(resume_texts, jd_text)

    results = []
    for idx, (_, row) in enumerate(df.iterrows()):
        scores = score_single_resume(
            resume_text=row["resume_text"],
            jd_text=jd_text,
            tfidf_score=float(tfidf_scores[idx])
        )
        results.append(scores)

    scores_df = pd.DataFrame(results)
    ranked = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    ranked = ranked.sort_values("composite_score", ascending=False)
    ranked["rank"] = range(1, len(ranked) + 1)

    return ranked.reset_index(drop=True)


def get_top_n(ranked_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top-N ranked candidates."""
    return ranked_df.head(n)


def score_summary(ranked_df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for a ranked candidate set.
    """
    return {
        "total_candidates": len(ranked_df),
        "avg_composite": round(ranked_df["composite_score"].mean(), 2),
        "max_composite": round(ranked_df["composite_score"].max(), 2),
        "min_composite": round(ranked_df["composite_score"].min(), 2),
        "avg_skill_match": round(ranked_df["skill_score"].mean(), 2),
        "avg_tfidf": round(ranked_df["tfidf_score"].mean(), 2),
        "shortlist_threshold": 60.0  # composite score cutoff for shortlisting
    }
