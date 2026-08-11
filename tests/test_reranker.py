from rag.reranker import Reranker


def test_rerank_prefers_keyword_and_semantic_match():
    reranker = Reranker(keyword_weight=0.6)
    candidates = [
        {"text": "VPN policy requires remote workers to connect securely.", "metadata": {"source": "a.txt"}, "distance": 0.30},
        {"text": "Unrelated cafeteria menu for next week.", "metadata": {"source": "b.txt"}, "distance": 0.28},
    ]
    results = reranker.rerank("What is the VPN policy for remote work?", candidates, top_k=2)

    assert results[0]["metadata"]["source"] == "a.txt"


def test_rerank_empty_candidates():
    reranker = Reranker()
    assert reranker.rerank("anything", [], top_k=5) == []


def test_rerank_respects_top_k():
    reranker = Reranker()
    candidates = [
        {"text": f"doc {i}", "metadata": {"source": f"{i}.txt"}, "distance": i * 0.1}
        for i in range(10)
    ]
    results = reranker.rerank("doc", candidates, top_k=3)
    assert len(results) == 3
