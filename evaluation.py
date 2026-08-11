"""
Evaluation script for the Enterprise RAG Chatbot.

Fixes:
1) sources returned by pipeline are strings -> evaluation must not treat them as dicts
2) Add "soft exact match": substring + numeric match (practical for interviews)
"""
import argparse
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from rag.retrieval import RAGPipeline


@dataclass
class EvalSample:
    id: str
    question: str
    answers: List[str]
    relevant_sources: List[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d+", text or "")


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_counts: Dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    common = 0
    for t in pred_tokens:
        if gold_counts.get(t, 0) > 0:
            common += 1
            gold_counts[t] -= 1

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def soft_exact_match(pred: str, gold: str) -> float:
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)

    if pred_norm == gold_norm:
        return 1.0

    # gold inside prediction (model adds extra words)
    if gold_norm and gold_norm in pred_norm:
        return 1.0

    # numeric containment (e.g., "24" must appear)
    pred_nums = set(extract_numbers(pred))
    gold_nums = set(extract_numbers(gold))
    if gold_nums and gold_nums.issubset(pred_nums):
        return 1.0

    return 0.0


def best_exact_and_f1(pred: str, gold_answers: List[str]) -> Tuple[float, float]:
    if not gold_answers:
        return 0.0, 0.0

    best_em = 0.0
    best_f1 = 0.0
    for gold in gold_answers:
        best_em = max(best_em, soft_exact_match(pred, gold))
        best_f1 = max(best_f1, token_f1(pred, gold))
    return best_em, best_f1


def load_eval_samples(path: str) -> List[EvalSample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # supports both:
    # 1) {"samples": [...]}
    # 2) [...]
    rows = data["samples"] if isinstance(data, dict) and "samples" in data else data

    samples: List[EvalSample] = []
    for row in rows:
        samples.append(
            EvalSample(
                id=row.get("id") or row.get("question", "")[:32],
                question=row["question"],
                answers=row.get("answers", []),
                relevant_sources=row.get("relevant_sources", []),
            )
        )
    return samples


def precision_recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> Tuple[float, float]:
    if not relevant:
        return 0.0, 0.0

    retrieved_k = [r.lower() for r in retrieved[:k]]
    relevant_l = [r.lower() for r in relevant]

    tp = 0
    for r in retrieved_k:
        if any(rel in r or r in rel for rel in relevant_l):
            tp += 1

    precision = tp / len(retrieved_k) if retrieved_k else 0.0
    recall = tp / len(set(relevant_l)) if relevant_l else 0.0
    return precision, recall


def evaluate(samples: List[EvalSample], k: int = 5, use_groq: bool = False) -> None:
    pipeline = RAGPipeline()

    precisions: List[float] = []
    recalls: List[float] = []
    ems: List[float] = []
    f1s: List[float] = []

    print(f"Evaluating {len(samples)} samples with K={k}, use_groq={use_groq}")
    print("-" * 80)

    for s in samples:
        print(f"\n[Sample {s.id}]")
        print(f"Q: {s.question}")

        result = pipeline.process_question(question=s.question, image_base64=None, use_groq=use_groq)
        answer = (result.get("answer") or "").strip()
        sources = result.get("sources") or []
        err = result.get("error")

        if err:
            print(f"  !! RAG error: {err}")

        print(f"  Model answer: {answer}")

        # ✅ sources are strings already
        retrieved_sources = []
        for item in sources[:k]:
            if isinstance(item, dict):
                meta = item.get("metadata", {}) or {}
                fname = meta.get("source") or meta.get("source_file") or meta.get("doc_name") or "Unknown"
                retrieved_sources.append(fname)
            else:
        # if already string
                retrieved_sources.append(str(item))

        print(f"  Retrieved sources (top {len(retrieved_sources)}): {retrieved_sources}")
        print(f"  Relevant sources: {s.relevant_sources}")

        p, r = precision_recall_at_k(retrieved_sources, s.relevant_sources, k)
        precisions.append(p)
        recalls.append(r)

        em, f1 = best_exact_and_f1(answer, s.answers)
        ems.append(em)
        f1s.append(f1)

        print(f"  Retrieval: Precision@K = {p:.3f}, Recall@K = {r:.3f}")
        print(f"  Answer: EM = {em:.3f}, F1 = {f1:.3f}")

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print("\n" + "=" * 80)
    print("OVERALL METRICS:")
    print(f"  Mean Precision@K: {avg(precisions):.3f}")
    print(f"  Mean Recall@K:    {avg(recalls):.3f}")
    print(f"  Mean Exact Match: {avg(ems):.3f}")
    print(f"  Mean F1:          {avg(f1s):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Enterprise RAG Chatbot.")
    parser.add_argument("--data-path", type=str, default="evaluation_data.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--use-groq", action="store_true")
    args = parser.parse_args()

    samples = load_eval_samples(args.data_path)
    evaluate(samples, k=args.k, use_groq=args.use_groq)


if __name__ == "__main__":
    main()
