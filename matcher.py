"""
Core scoring logic: given a job description and a resume, produce a
match percentage, verdict, skill gap breakdown, and experience estimate.
"""

import numpy as np
from skills import clean_text, extract_skills, extract_experience_years
from model_utils import predict_category, embed_text


def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def skill_overlap(jd_skills: list, resume_skills: list) -> float:
    """% of JD-required skills that are present in the resume."""
    if not jd_skills:
        return 1.0  # no explicit skills required in JD -> don't penalize
    jd_set = set(s.lower() for s in jd_skills)
    resume_set = set(s.lower() for s in resume_skills)
    matched = jd_set & resume_set
    return len(matched) / len(jd_set)


def verdict_from_score(score_pct: float) -> str:
    if score_pct >= 75:
        return "Strong Match"
    elif score_pct >= 50:
        return "Moderate Match"
    else:
        return "Not Suitable"


def evaluate_resume(jd_text: str, resume_raw_text: str, target_role: str = None) -> dict:
    """
    Main entry point. Compares one resume against a job description
    (and optionally a target role name, e.g. "React Developer").

    Weights: 50% semantic similarity, 30% skill overlap, 20% category match bonus.
    Tune these based on your validation results.
    """
    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_raw_text)

    predicted_category, confidence, resume_embedding = predict_category(resume_clean)
    jd_embedding = embed_text(jd_clean)

    semantic_score = cosine_similarity(resume_embedding, jd_embedding)  # 0-1

    jd_skills = extract_skills(jd_clean)
    resume_skills = extract_skills(resume_clean)
    overlap_score = skill_overlap(jd_skills, resume_skills)  # 0-1

    category_match = False
    if target_role:
        category_match = predicted_category.strip().lower() == target_role.strip().lower()
    category_bonus = 1.0 if category_match else 0.0

    final_score = (0.5 * semantic_score) + (0.3 * overlap_score) + (0.2 * category_bonus)
    final_score_pct = round(final_score * 100, 1)

    experience_years = extract_experience_years(resume_raw_text)

    return {
        "predicted_category": predicted_category,
        "category_confidence": round(confidence, 3) if confidence is not None else None,
        "category_match": category_match,
        "match_score_pct": final_score_pct,
        "semantic_similarity_pct": round(semantic_score * 100, 1),
        "skill_overlap_pct": round(overlap_score * 100, 1),
        "matched_skills": sorted(set(s.lower() for s in jd_skills) & set(s.lower() for s in resume_skills)),
        "missing_skills": sorted(set(s.lower() for s in jd_skills) - set(s.lower() for s in resume_skills)),
        "all_resume_skills": sorted(set(resume_skills)),
        "experience_years": experience_years,
        "verdict": verdict_from_score(final_score_pct),
        # carried through so the frontend can call /hiring-decision, /interview-questions
        # later without re-uploading the file
        "resume_text": resume_raw_text,
    }
