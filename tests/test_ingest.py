from rag.ingest import chunk_text, extract_text


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_basic():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # every chunk (except possibly the last) should respect the size
    for c in chunks[:-1]:
        assert len(c) == 100


def test_chunk_text_overlap():
    text = "abcdefghij" * 20
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert chunks[0][-10:] == chunks[1][:10]


def test_extract_text_txt():
    result = extract_text("note.txt", b"hello world")
    assert result == "hello world"


def test_extract_text_unsupported():
    import pytest

    with pytest.raises(ValueError):
        extract_text("file.exe", b"binary")
