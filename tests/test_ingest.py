from rag.ingest import DocumentIngester


def test_chunk_text_respects_overlap_and_size():
    ingester = DocumentIngester(chunk_size=50, chunk_overlap=10)
    text = "Sentence one. " * 20  # long enough to force multiple chunks
    chunks = ingester.chunk_text(text)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 60  # allow a little slack for boundary snapping


def test_chunk_text_empty_input():
    ingester = DocumentIngester()
    assert ingester.chunk_text("") == []
    assert ingester.chunk_text("   ") == []


def test_extract_txt():
    ingester = DocumentIngester()
    text = ingester.extract_text(b"hello world", "notes.txt")
    assert text == "hello world"


def test_unsupported_extension_raises():
    ingester = DocumentIngester()
    try:
        ingester.extract_text(b"data", "file.exe")
        assert False, "expected ValueError"
    except ValueError:
        pass
