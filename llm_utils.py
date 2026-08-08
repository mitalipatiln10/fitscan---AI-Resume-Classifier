"""
LLM-powered features: AI Hiring Decision, Interview Question Generator,
and Recruiter Chat. Uses the Google Gemini API (free tier).

IMPORTANT: never paste your actual API key into this file or anywhere in
this repo -- GitHub will detect it and the key will get auto-revoked. Set
it as an environment variable / hosting platform secret instead:
    Windows (this session only):  $env:GEMINI_API_KEY = "your-key-here"
    Windows (permanent):           setx GEMINI_API_KEY "your-key-here"
    Mac/Linux:                      export GEMINI_API_KEY=your-key-here
    Render / HF / other hosts:      add it under Environment Variables / Secrets in their dashboard

Install: pip install google-generativeai
Get a free key (no credit card): https://aistudio.google.com -> Get API Key
"""

import os
import json
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-3.6-flash"  # current free-tier model as of Aug 2026 (2.5-flash was retired for new users)
_model = genai.GenerativeModel(MODEL)


def _call(prompt: str, max_tokens: int = 500) -> str:
    # Disable "thinking" where the SDK supports it, so the full token budget
    # goes to the actual answer instead of internal reasoning tokens (which
    # can otherwise eat the budget and truncate the response, e.g. cutting
    # a 6-item question list down to 1). Falls back gracefully on older SDKs.
    try:
        config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
        )
    except AttributeError:
        config = genai.types.GenerationConfig(max_output_tokens=max_tokens)

    response = _model.generate_content(prompt, generation_config=config)
    return response.text.strip()


def generate_hiring_decision(jd_text: str, resume_text: str, match_result: dict) -> str:
    prompt = f"""You are an experienced technical recruiter assistant. Based on the job
description and candidate resume below, write a short, recruiter-friendly hiring
recommendation (4-6 sentences).

Cover: the candidate's key strengths relevant to this role, their relevant experience,
any notable gaps or missing qualifications, and a clear recommendation on whether they
should proceed to the next stage (e.g. "Recommend for interview", "Consider with
reservations", or "Not recommended").

Job Description:
{jd_text[:3000]}

Candidate Resume:
{resume_text[:3000]}

Quantitative signals already computed (use as supporting context, don't just repeat them):
- Overall match score: {match_result.get('match_score_pct')}%
- Skill overlap: {match_result.get('skill_overlap_pct')}%
- Predicted role category: {match_result.get('predicted_category')}
- Matched skills: {', '.join(match_result.get('matched_skills', [])) or 'none detected'}
- Missing skills: {', '.join(match_result.get('missing_skills', [])) or 'none'}

Write the recommendation now, in plain prose, no headers or bullet points."""
    return _call(prompt, max_tokens=400)


def generate_interview_questions(jd_text: str, resume_text: str) -> list:
    prompt = f"""You are a technical interviewer preparing for a candidate screen.
Based on the job description and candidate resume below, generate 6 interview questions:
2 technical/concept questions, 2 questions about specific projects or experience
mentioned in the resume, and 2 behavioral questions relevant to this role.

Job Description:
{jd_text[:3000]}

Candidate Resume:
{resume_text[:3000]}

Respond ONLY with a JSON array of 6 strings (the questions), no other text, no markdown
fences, no preamble."""
    raw = _call(prompt, max_tokens=1500)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        questions = json.loads(cleaned)
        if isinstance(questions, list):
            return [q for q in questions if isinstance(q, str) and q.strip()]
    except json.JSONDecodeError:
        pass
    # fallback: split by lines, strip stray JSON punctuation/bullets/quotes
    lines = []
    for line in cleaned.split("\n"):
        line = line.strip().strip(",").strip('"').strip("-• ").strip()
        if line and line not in ("[", "]", "{", "}"):
            lines.append(line)
    return lines[:6]


def recruiter_chat(question: str, candidates_context: list) -> str:
    """
    candidates_context: list of dicts summarizing each ranked candidate, e.g.
        {"filename": ..., "predicted_category": ..., "match_score_pct": ...,
         "experience_years": ..., "matched_skills": [...], "missing_skills": [...]}
    """
    context_str = json.dumps(candidates_context[:50], indent=2)  # cap to avoid huge prompts
    prompt = f"""You are a recruiter's AI assistant with access to a batch of already-screened
candidates. Answer the recruiter's question using ONLY the candidate data provided below.
Be concise and reference candidates by filename. If the question asks to compare or filter
candidates, do that directly. If the data doesn't support answering the question, say so.

Candidate data:
{context_str}

Recruiter's question: {question}

Answer:"""
    return _call(prompt, max_tokens=500)
