from pathlib import Path

import utils.rag as rag
from utils.rag import (
    calculate_relevance_score,
    is_emergency_question,
    load_knowledge_base,
    log_interaction,
    retrieve_context,
)


KNOWLEDGE_BASE_PATH = Path("data/pet_care_knowledge_base.txt")


def test_emergency_detection_flags_dangerous_symptoms():
    assert is_emergency_question("My cat is having trouble breathing") is True


def test_normal_question_does_not_trigger_emergency_mode():
    assert is_emergency_question("How often should I walk my dog?") is False


def test_retrieval_returns_exercise_context_for_walking_question():
    knowledge_text = load_knowledge_base(KNOWLEDGE_BASE_PATH)

    context = retrieve_context("How often should I walk my dog?", knowledge_text)

    assert "[Exercise]" in context
    assert calculate_relevance_score("How often should I walk my dog?", context) >= 0.5


def test_retrieval_returns_feeding_context_for_skipped_dinner_question():
    knowledge_text = load_knowledge_base(KNOWLEDGE_BASE_PATH)

    context = retrieve_context("My dog skipped dinner, what should I do?", knowledge_text)

    assert "[Feeding]" in context
    assert calculate_relevance_score(
        "My dog skipped dinner, what should I do?",
        context,
    ) >= 0.5


def test_fallback_answer_is_generated_without_api_key(monkeypatch):
    monkeypatch.setattr(rag, "_get_openai_api_key", lambda: None)
    context = "[Exercise]\nMost dogs benefit from daily walks."

    answer = rag.generate_ai_answer("How often should I walk my dog?", context)

    assert "Based on PawPal's retrieved pet-care guide" in answer
    assert "Most dogs benefit from daily walks" in answer


def test_log_interaction_includes_reliability_fields(tmp_path):
    log_path = tmp_path / "rag_logs.csv"

    log_interaction(
        "How often should I walk my dog?",
        "[Exercise]\nMost dogs benefit from daily walks.",
        "Walks are usually helpful.",
        confidence_score=0.67,
        emergency_triggered=False,
        error_message="",
        log_path=log_path,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert "confidence_score" in log_text
    assert "emergency_triggered" in log_text
    assert "0.67" in log_text
    assert "False" in log_text
