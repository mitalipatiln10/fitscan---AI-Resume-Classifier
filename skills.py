"""
Cleaning + skill extraction + experience-years extraction.
Model B uses Resume_clean (this clean_text output) for SBERT.
"""

import re
import spacy
from spacy.matcher import PhraseMatcher

# ---------------------------------------------------------------------
# Text cleaning (same as notebook)
# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\+?\d[\d\-\(\) ]{8,}\d', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------
# Skill list — copy this EXACTLY from your notebook. Add more roles/skills
# here as you expand beyond the original 4 categories.
# ---------------------------------------------------------------------

skill_list = [
    # Peoplesoft
    "application server", "peoplesoft", "process scheduler", "oracle", "web server",
    "peoplecode", "component interface", "application engine", "peopletools",
    "integration broker", "app engine", "component",

    # React Developer
    "react", "javascript", "html", "css3", "css", "bootstrap", "ui",
    "redux", "typescript", "node", "node.js", "jquery", "rest api",
    "es6", "json", "npm", "webpack", "responsive design",

    # SQL Developer
    "sql", "sql server", "stored procedure", "query", "index", "table", "bi",
    "t-sql", "ssis", "ssrs", "mysql", "postgresql", "database design",
    "power bi", "etl", "join", "view", "trigger", "database administration",

    # Workday
    "workday", "eib", "hcm", "integration", "studio", "security", "connector", "core",
    "workday hcm", "workday integration", "calculated field", "business process",
    "workday studio", "core connector", "payroll", "benefits", "recruiting",
    "workday report writer", "workday security",
]

_nlp = spacy.load("en_core_web_sm")
_matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
_matcher.add("SKILLS", [_nlp.make_doc(skill) for skill in skill_list])


def extract_skills(text: str) -> list:
    doc = _nlp(text.lower())
    matches = _matcher(doc)
    found = set()
    for match_id, start, end in matches:
        found.add(doc[start:end].text)
    return list(found)


# ---------------------------------------------------------------------
# Experience extraction — heuristic regex, not ML-based.
# Looks for explicit "X years" mentions; falls back to None if not found.
# Good enough for filtering/display, not a substitute for careful review.
# ---------------------------------------------------------------------

_EXPERIENCE_PATTERNS = [
    r'(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s*(?:of)?\s*experience',
    r'experience\s*(?:of)?\s*(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)',
]

def extract_experience_years(text: str):
    text_lower = text.lower()
    candidates = []
    for pattern in _EXPERIENCE_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            try:
                candidates.append(float(match.group(1)))
            except (ValueError, IndexError):
                continue
    if not candidates:
        return None
    return max(candidates)
