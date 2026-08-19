"""
End-to-end pipeline test using monkeypatched embedder/LLM/store so it
runs in CI without any API keys or network access.
"""
from unittest.mock import patch

from rag import retrieval


@patch("rag.retrieval.get_store")
@patch("rag.retrieval.llm.generate_answer")
@patch("rag.retrieval.embedder.embed_query", return_value=[0.1, 0.2, 0.3])
def test_answer_question_returns_structured_result(mock_embed, mock_generate, mock_store):
    mock_generate.return_value = {"answer": "Paris is the capital of France.", "provider": "groq"}
    mock_store.return_value.query.return_value = [
        {
            "text": "France's capital is Paris.",
            "metadata": {"source": "geo.pdf", "chunk_index": 0},
            "distance": 0.1,
        }
    ]

    result = retrieval.answer_question("What is the capital of France?")

    assert result["answer"] == "Paris is the capital of France."
    assert result["provider"] == "groq"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "geo.pdf"


@patch("rag.retrieval.embedder.embed_texts", return_value=[[0.1, 0.2]])
@patch("rag.retrieval.get_store")
def test_ingest_chunks_empty_list_is_noop(mock_store, mock_embed):
    result = retrieval.ingest_chunks([], [])
    assert result == 0
    mock_store.return_value.add.assert_not_called()
