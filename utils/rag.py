"""
Simple Retrieval-Augmented Generation helpers for PawPal AI.

This module keeps the RAG pipeline beginner-friendly:
1. Load a local text knowledge base.
2. Retrieve relevant sections using keyword overlap.
3. Use an OpenAI model if an API key is available.
4. Fall back to a context-only answer if no API key is configured.
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path


EMERGENCY_MESSAGE = (
    "This may be an emergency. PawPal cannot diagnose medical problems. "
    "Please contact a veterinarian or emergency animal clinic immediately."
)

EMERGENCY_TERMS = [
    "trouble breathing",
    "breathing",
    "seizure",
    "seizures",
    "collapse",
    "poisoning",
    "blood",
    "bleeding",
    "choking",
    "unconscious",
    "emergency",
    "can't walk",
    "cant walk",
    "cannot walk",
    "repeated vomiting",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "can",
    "do",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "with",
}


def load_knowledge_base(path: str | Path) -> str:
    """Load the local pet-care guide from disk."""
    knowledge_path = Path(path)
    if not knowledge_path.exists():
        return ""
    return knowledge_path.read_text(encoding="utf-8")


def _split_sections(knowledge_text: str) -> list[str]:
    """Split text into sections that start with headings like [Feeding]."""
    sections = []
    current_section = []

    for line in knowledge_text.splitlines():
        if line.strip().startswith("[") and line.strip().endswith("]"):
            if current_section:
                sections.append("\n".join(current_section).strip())
            current_section = [line.strip()]
        elif current_section:
            current_section.append(line)

    if current_section:
        sections.append("\n".join(current_section).strip())

    return [section for section in sections if section]


def _normalize_word(word: str) -> str:
    """Normalize common plural and tense endings for simple matching."""
    if len(word) > 4 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s"):
        word = word[:-1]

    # skipped -> skipp -> skip
    if len(word) > 3 and word[-1] == word[-2]:
        word = word[:-1]

    return word


def _keywords(text: str) -> set[str]:
    """Return simple lowercase keywords, excluding common filler words."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {
        _normalize_word(word)
        for word in words
        if word not in STOPWORDS and len(word) > 1
    }


def retrieve_context(user_question: str, knowledge_text: str, top_k: int = 2) -> str:
    """Retrieve the top matching knowledge-base sections by keyword overlap."""
    sections = _split_sections(knowledge_text)
    question_keywords = _keywords(user_question)

    if not sections:
        return ""

    scored_sections = []
    for section in sections:
        section_keywords = _keywords(section)
        score = len(question_keywords.intersection(section_keywords))
        scored_sections.append((score, section))

    # Prefer matching sections, but still return one general section as a fallback.
    matches = [item for item in scored_sections if item[0] > 0]
    ranked = sorted(matches or scored_sections, key=lambda item: item[0], reverse=True)
    return "\n\n".join(section for _, section in ranked[:top_k])


def calculate_relevance_score(user_question: str, retrieved_context: str) -> float:
    """Estimate confidence from keyword overlap between the question and context."""
    question_keywords = _keywords(user_question)
    context_keywords = _keywords(retrieved_context)

    if not question_keywords or not context_keywords:
        return 0.0

    overlap = question_keywords.intersection(context_keywords)
    return round(len(overlap) / len(question_keywords), 2)


def confidence_label(score: float) -> str:
    """Convert a numeric relevance score into a simple label for users."""
    if score >= 0.8:
        return "strong match"
    if score >= 0.5:
        return "medium match"
    return "low confidence"


def is_emergency_question(user_question: str) -> bool:
    """Check whether the question contains urgent warning terms."""
    question = user_question.lower()
    return any(term in question for term in EMERGENCY_TERMS)


def _get_openai_api_key() -> str | None:
    """Read the API key from environment variables or Streamlit secrets."""
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    try:
        import streamlit as st

        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def _fallback_answer(retrieved_context: str) -> str:
    """Answer from retrieved context when no model key is configured."""
    if not retrieved_context:
        return (
            "I could not find a matching section in PawPal's retrieved pet-care guide. "
            "For serious, unusual, or persistent symptoms, contact a veterinarian."
        )

    return (
        "Based on PawPal's retrieved pet-care guide:\n\n"
        f"{retrieved_context}\n\n"
        "PawPal cannot diagnose medical problems. If symptoms are serious, unusual, "
        "or persist, contact a veterinarian."
    )


def generate_ai_answer_with_error(
    user_question: str,
    retrieved_context: str,
) -> tuple[str, str]:
    """Generate an answer and return any model/API error message for logging."""
    if is_emergency_question(user_question):
        return EMERGENCY_MESSAGE, ""

    api_key = _get_openai_api_key()
    if not api_key:
        return _fallback_answer(retrieved_context), ""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = f"""
You are PawPal AI, a cautious pet-care assistant.
Use only the retrieved context when possible.
Do not diagnose medical problems.
Recommend a veterinarian for serious or persistent symptoms.
Keep the answer clear and practical.
Mention that the answer is based on PawPal's retrieved pet-care guide.

Retrieved context:
{retrieved_context or "No matching context was retrieved."}

User question:
{user_question}
"""
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        return response.output_text, ""
    except Exception as error:
        error_message = str(error)
        return (
            f"{_fallback_answer(retrieved_context)}\n\n"
            f"Note: PawPal could not call the AI model right now ({error_message}).",
            error_message,
        )


def generate_ai_answer(user_question: str, retrieved_context: str) -> str:
    """Generate a cautious answer using retrieved context and the OpenAI SDK."""
    answer, _ = generate_ai_answer_with_error(user_question, retrieved_context)
    return answer


def log_interaction(
    question: str,
    retrieved_context: str,
    answer: str,
    confidence_score: float = 0.0,
    emergency_triggered: bool = False,
    error_message: str = "",
    log_path: str | Path = "logs/rag_logs.csv",
) -> None:
    """Append a RAG interaction to a CSV log file."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "question",
        "retrieved_context",
        "confidence_score",
        "answer",
        "emergency_triggered",
        "error_message",
    ]
    file_exists = path.exists() and path.stat().st_size > 0

    needs_header = True
    if file_exists:
        with path.open("r", newline="", encoding="utf-8") as existing_log:
            first_line = existing_log.readline().strip()
        needs_header = first_line != ",".join(fieldnames)

    with path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "retrieved_context": retrieved_context,
                "confidence_score": confidence_score,
                "answer": answer,
                "emergency_triggered": emergency_triggered,
                "error_message": error_message,
            }
        )
