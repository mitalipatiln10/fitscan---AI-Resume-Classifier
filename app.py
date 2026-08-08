"""
Fitscan backend — FastAPI app.

Run with:
    uvicorn app:app --reload --port 8000

Endpoints:
    GET  /roles              -> list of trained role categories
    POST /match-jd            -> JD text + one resume file -> match score
    POST /bulk-rank            -> target role + many resume files -> ranked list
    POST /extract-only         -> one resume file -> skills + category (debug)
    POST /hiring-decision       -> AI-generated hiring recommendation
    POST /interview-questions    -> AI-generated interview questions
    POST /recruiter-chat          -> natural-language Q&A over a candidate batch
    POST /export-csv                -> download ranked results as CSV
"""

import os
import csv
import io
import shutil
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from text_extraction import extract_text_any
from matcher import evaluate_resume
from model_utils import le
from llm_utils import generate_hiring_decision, generate_interview_questions, recruiter_chat

app = FastAPI(title="Fitscan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your actual frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_temp_file(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        return tmp.name


@app.get("/roles")
def get_roles():
    """Returns the list of role categories the model was trained on."""
    return {"roles": list(le.classes_)}


@app.post("/match-jd")
async def match_jd(
    jd_text: str = Form(...),
    target_role: Optional[str] = Form(None),
    resume: UploadFile = File(...),
):
    """Recruiter pastes a JD + uploads one resume -> get a match score."""
    tmp_path = _save_temp_file(resume)
    try:
        resume_text = extract_text_any(tmp_path)
        result = evaluate_resume(jd_text, resume_text, target_role)
        result["filename"] = resume.filename
        return result
    finally:
        os.remove(tmp_path)


@app.post("/bulk-rank")
async def bulk_rank(
    target_role: str = Form(...),
    jd_text: Optional[str] = Form(""),
    resumes: List[UploadFile] = File(...),
):
    """
    Recruiter selects/writes a role, optionally adds a JD, uploads many resumes.
    Returns all candidates ranked by match score, best first.
    """
    comparison_text = jd_text.strip() if jd_text.strip() else target_role

    results = []
    for resume in resumes:
        tmp_path = _save_temp_file(resume)
        try:
            resume_text = extract_text_any(tmp_path)
            result = evaluate_resume(comparison_text, resume_text, target_role)
            result["filename"] = resume.filename
            results.append(result)
        except Exception as e:
            results.append({"filename": resume.filename, "error": str(e)})
        finally:
            os.remove(tmp_path)

    results.sort(key=lambda r: r.get("match_score_pct", -1), reverse=True)

    return {
        "target_role": target_role,
        "jd_text": comparison_text,
        "total_candidates": len(results),
        "results": results,
    }


@app.post("/extract-only")
async def extract_only(resume: UploadFile = File(...)):
    """Debug endpoint: just run extraction + category prediction, no JD needed."""
    from skills import clean_text, extract_skills
    from model_utils import predict_category

    tmp_path = _save_temp_file(resume)
    try:
        raw_text = extract_text_any(tmp_path)
        clean = clean_text(raw_text)
        category, confidence, _ = predict_category(clean)
        skills_found = extract_skills(clean)
        return {
            "filename": resume.filename,
            "predicted_category": category,
            "confidence": round(confidence, 3) if confidence else None,
            "skills": skills_found,
        }
    finally:
        os.remove(tmp_path)


@app.post("/hiring-decision")
async def hiring_decision(payload: dict = Body(...)):
    """
    payload: { "jd_text": str, "resume_text": str, "match_result": {...} }
    match_result should be the object returned from /match-jd or a /bulk-rank entry.
    """
    jd_text = payload.get("jd_text", "")
    resume_text = payload.get("resume_text", "")
    match_result = payload.get("match_result", {})
    decision = generate_hiring_decision(jd_text, resume_text, match_result)
    return {"decision": decision}


@app.post("/interview-questions")
async def interview_questions(payload: dict = Body(...)):
    """payload: { "jd_text": str, "resume_text": str }"""
    jd_text = payload.get("jd_text", "")
    resume_text = payload.get("resume_text", "")
    questions = generate_interview_questions(jd_text, resume_text)
    return {"questions": questions}


@app.post("/recruiter-chat")
async def chat(payload: dict = Body(...)):
    """payload: { "question": str, "candidates": [ {...}, {...} ] }"""
    question = payload.get("question", "")
    candidates = payload.get("candidates", [])
    answer = recruiter_chat(question, candidates)
    return {"answer": answer}


@app.post("/export-csv")
async def export_csv(payload: dict = Body(...)):
    """payload: { "results": [ {...}, {...} ] } -> returns a downloadable CSV"""
    results = payload.get("results", [])

    output = io.StringIO()
    fieldnames = [
        "filename", "predicted_category", "match_score_pct",
        "semantic_similarity_pct", "skill_overlap_pct", "experience_years",
        "verdict", "matched_skills", "missing_skills",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in results:
        row = dict(r)
        row["matched_skills"] = "; ".join(r.get("matched_skills", []))
        row["missing_skills"] = "; ".join(r.get("missing_skills", []))
        writer.writerow(row)

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fitscan_candidates.csv"},
    )
