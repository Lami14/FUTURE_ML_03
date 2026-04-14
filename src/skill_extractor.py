"""
skill_extractor.py
------------------
Extracts skills from resume and job description text by matching against
the curated skill taxonomy. Supports both exact and fuzzy matching.
"""

import re
from typing import Dict, List, Set
from src.skill_taxonomy import SKILL_TAXONOMY, ALL_SKILLS
from src.preprocessor import basic_clean


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def extract_skills_from_text(text: str) -> Dict[str, List[str]]:
    """
    Scan cleaned text for skill keywords grouped by taxonomy category.

    Parameters
    ----------
    text : str
        Raw or lightly cleaned resume / JD text.

    Returns
    -------
    dict
        {category: [matched_skill, ...], ...}
        Only categories with at least one match are included.
    """
    cleaned = basic_clean(text)
    found: Dict[str, List[str]] = {}

    for category, skills in SKILL_TAXONOMY.items():
        matched = []
        for skill in skills:
            # Use word-boundary regex so "r" doesn't match inside "render"
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, cleaned):
                matched.append(skill)
        if matched:
            found[category] = matched

    return found


def get_flat_skill_set(text: str) -> Set[str]:
    """
    Return a flat set of all skills found in text (category-agnostic).
    Convenient for set-based gap analysis.
    """
    categorised = extract_skills_from_text(text)
    return {skill for skills in categorised.values() for skill in skills}


def extract_required_skills(jd_text: str) -> Set[str]:
    """
    Parse a job description and return the set of required skills.
    Alias of get_flat_skill_set — kept separate for semantic clarity.
    """
    return get_flat_skill_set(jd_text)


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

def compute_skill_gap(resume_text: str, jd_text: str) -> Dict[str, object]:
    """
    Compare resume skills against job description requirements.

    Returns
    -------
    dict with keys:
        matched_skills  : skills present in both resume and JD
        missing_skills  : skills in JD but absent from resume
        extra_skills    : skills in resume not mentioned in JD
        match_rate      : float 0–1 (matched / required)
    """
    resume_skills = get_flat_skill_set(resume_text)
    required_skills = extract_required_skills(jd_text)

    if not required_skills:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "extra_skills": sorted(resume_skills),
            "match_rate": 0.0
        }

    matched = resume_skills & required_skills
    missing = required_skills - resume_skills
    extra = resume_skills - required_skills

    match_rate = len(matched) / len(required_skills)

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "extra_skills": sorted(extra),
        "match_rate": round(match_rate, 4)
    }


def skill_match_score(resume_text: str, jd_text: str) -> float:
    """
    Return a 0–100 skill match score for a resume against a JD.
    """
    gap = compute_skill_gap(resume_text, jd_text)
    return round(gap["match_rate"] * 100, 2)


# ---------------------------------------------------------------------------
# Experience signal extraction
# ---------------------------------------------------------------------------

_YEAR_PATTERNS = [
    r"(\d+)\+?\s*years?\s+(?:of\s+)?experience",
    r"(\d+)\+?\s*yrs?\s+(?:of\s+)?experience",
    r"experience\s+of\s+(\d+)\+?\s*years?",
]

def extract_experience_years(text: str) -> float:
    """
    Attempt to extract years of experience from resume text.
    Returns the maximum found value, or 0.0 if none detected.
    """
    cleaned = basic_clean(text)
    years_found = []
    for pattern in _YEAR_PATTERNS:
        matches = re.findall(pattern, cleaned)
        years_found.extend(int(m) for m in matches)
    return float(max(years_found)) if years_found else 0.0


def experience_bonus_score(text: str, max_expected: int = 10) -> float:
    """
    Convert years of experience into a 0–100 bonus score,
    capped at max_expected years (default 10).
    """
    years = extract_experience_years(text)
    return round(min(years / max_expected, 1.0) * 100, 2)
