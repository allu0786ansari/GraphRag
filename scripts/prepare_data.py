#!/usr/bin/env python3
"""
scripts/prepare_data.py — Transform raw MultiHopRAG dataset into pipeline format.

Reads:
  data/raw/corpus.json          — 609 news articles (title, body, source, ...)
  data/raw/MultiHopRAG.json     — 2,556 multi-hop Q&A pairs

Writes:
  data/raw/articles/            — One .json file per article, chunker-ready
      mashable_001.json         — {"text": "...", "metadata": {...}}
      ...

  data/evaluation/questions.json — All unique questions from MultiHopRAG
      [{"id": "q001", "question": "...", "answer": "...", "type": "...", ...}]

  data/evaluation/evidence_map.json — Maps question → supporting article URLs
      (used to measure retrieval accuracy: did the system retrieve the right articles?)

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --corpus data/raw/corpus.json
    python scripts/prepare_data.py --max-articles 50   # dev mode
    python scripts/prepare_data.py --dry-run           # preview without writing

After running:
    python scripts/run_indexing.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_CORPUS    = _PROJECT_ROOT / "data" / "raw" / "corpus.json"
_DEFAULT_MULTIHOP  = _PROJECT_ROOT / "data" / "raw" / "MultiHopRAG.json"
_DEFAULT_OUT_DIR   = _PROJECT_ROOT / "data" / "raw" / "articles"
_DEFAULT_EVAL_DIR  = _PROJECT_ROOT / "data" / "evaluation"


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MultiHopRAG dataset for the GraphRAG indexing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full preparation (recommended first run)
  python scripts/prepare_data.py

  # Dev mode: only first 50 articles (fast, cheap to index)
  python scripts/prepare_data.py --max-articles 50

  # Preview what would be written, without writing anything
  python scripts/prepare_data.py --dry-run

  # Custom paths
  python scripts/prepare_data.py \\
      --corpus path/to/corpus.json \\
      --multihop path/to/MultiHopRAG.json
        """,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_DEFAULT_CORPUS,
        help=f"Path to corpus.json (default: {_DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--multihop",
        type=Path,
        default=_DEFAULT_MULTIHOP,
        help=f"Path to MultiHopRAG.json (default: {_DEFAULT_MULTIHOP})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=f"Output directory for per-article files (default: {_DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=_DEFAULT_EVAL_DIR,
        help=f"Output directory for evaluation files (default: {_DEFAULT_EVAL_DIR})",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        metavar="N",
        help="Only prepare first N articles (dev / cost-control mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing any files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing article files (default: skip existing)",
    )
    return parser.parse_args()


# ── Slugify helper ────────────────────────────────────────────────────────────

def _slugify(text: str, max_len: int = 50) -> str:
    """Convert a string to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    text = text.strip("_")
    return text[:max_len]


def _make_article_filename(article: dict, index: int) -> str:
    """
    Generate a deterministic filename for an article.

    Format: {source_slug}_{index:04d}.json
    e.g. "mashable_0001.json", "techcrunch_0042.json"

    Using source + index ensures:
      - Filenames are unique
      - Source is human-readable in the filename
      - Index is stable (same corpus → same IDs → pipeline is resumable)
    """
    source = _slugify(article.get("source") or "unknown", max_len=30)
    return f"{source}_{index:04d}.json"


# ── Article preparation ────────────────────────────────────────────────────────

def prepare_article(article: dict, filename: str) -> dict:
    """
    Transform a corpus.json article into the chunker's expected format.

    Input (corpus.json entry):
        {
            "title": "...",
            "author": "...",
            "source": "Mashable",
            "published_at": "2023-11-27T08:45:59+00:00",
            "category": "entertainment",
            "url": "https://...",
            "body": "Full article text..."
        }

    Output (chunker-ready):
        {
            "text": "Title: ...\n\n{body text}",
            "metadata": {
                "title": "...",
                "source": "...",
                "author": "...",
                "published_at": "...",
                "category": "...",
                "url": "...",
                "filename": "mashable_0001.json"
            }
        }

    Why prepend the title to text?
      The chunker splits on token boundaries. If the title is only in metadata,
      the extraction LLM never sees it when processing a mid-document chunk.
      Prepending ensures the title is in the first chunk and provides context.
    """
    body = (article.get("body") or "").strip()
    title = (article.get("title") or "").strip()

    # Build the text: title first, then body
    # The chunker will split this into ≤600-token windows
    if title:
        text = f"{title}\n\n{body}"
    else:
        text = body

    metadata = {
        "title":        title,
        "source":       article.get("source") or "",
        "author":       article.get("author") or "",
        "published_at": article.get("published_at") or "",
        "category":     article.get("category") or "",
        "url":          article.get("url") or "",
        "filename":     filename,
    }

    return {"text": text, "metadata": metadata}


# ── Question preparation ───────────────────────────────────────────────────────

def prepare_questions(
    multihop_data: list[dict],
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Transform MultiHopRAG entries into evaluation questions.

    Returns:
        questions:    List of {id, question, answer, type, evidence_urls}
        evidence_map: {question_id → [evidence entries]} for retrieval evaluation
    """
    questions = []
    evidence_map: dict[str, list[dict]] = {}

    for i, entry in enumerate(multihop_data, 1):
        q_id = f"q{i:04d}"
        question = (entry.get("query") or "").strip()
        answer   = (entry.get("answer") or "").strip()
        q_type   = (entry.get("question_type") or "unknown").strip()
        evidence = entry.get("evidence_list") or []

        if not question:
            continue

        # Collect URLs of supporting articles for retrieval accuracy measurement
        evidence_urls = [e.get("url", "") for e in evidence if e.get("url")]

        questions.append({
            "id":           q_id,
            "question":     question,
            "answer":       answer,
            "type":         q_type,
            "evidence_urls": evidence_urls,
            "evidence_count": len(evidence),
        })

        # Full evidence entries (title, source, fact snippet) for detailed analysis
        evidence_map[q_id] = [
            {
                "title":        e.get("title", ""),
                "author":       e.get("author", ""),
                "source":       e.get("source", ""),
                "url":          e.get("url", ""),
                "category":     e.get("category", ""),
                "published_at": e.get("published_at", ""),
                "fact":         e.get("fact", ""),
            }
            for e in evidence
        ]

    return questions, evidence_map


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Validate input files ──────────────────────────────────────────────────
    missing = [p for p in [args.corpus, args.multihop] if not p.exists()]
    if missing:
        for p in missing:
            print(f"\n  ERROR: File not found: {p}")
        print("\n  Make sure corpus.json and MultiHopRAG.json are in data/raw/\n")
        sys.exit(1)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print(f"\n  Loading corpus from:   {args.corpus}")
    with open(args.corpus, encoding="utf-8") as f:
        corpus: list[dict] = json.load(f)

    print(f"  Loading questions from: {args.multihop}")
    with open(args.multihop, encoding="utf-8") as f:
        multihop_raw = json.load(f)

    # MultiHopRAG.json can be a list directly or wrapped in a key
    if isinstance(multihop_raw, dict):
        multihop: list[dict] = (
            multihop_raw.get("data")
            or multihop_raw.get("queries")
            or multihop_raw.get("questions")
            or list(multihop_raw.values())[0]
        )
    else:
        multihop = multihop_raw

    print(f"\n  ── Dataset summary ──────────────────────────────────")
    print(f"  Articles in corpus:    {len(corpus)}")
    print(f"  Q&A pairs (MultiHop):  {len(multihop)}")

    # ── Apply limits ──────────────────────────────────────────────────────────
    articles_to_process = corpus
    if args.max_articles:
        articles_to_process = corpus[: args.max_articles]
        print(f"  ⚠  --max-articles={args.max_articles}: dev mode, not full corpus")

    # ── Analyse corpus ────────────────────────────────────────────────────────
    categories: dict[str, int] = defaultdict(int)
    sources: dict[str, int] = defaultdict(int)
    empty_body = 0

    for article in articles_to_process:
        categories[article.get("category") or "unknown"] += 1
        sources[article.get("source") or "unknown"] += 1
        if not (article.get("body") or "").strip():
            empty_body += 1

    print(f"\n  Categories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {cat:<25} {count:>4}")

    print(f"\n  Top sources:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1])[:10]:
        print(f"    {src:<30} {count:>4}")

    if empty_body:
        print(f"\n  ⚠  {empty_body} articles have empty body — will be skipped")

    # ── Analyse questions ─────────────────────────────────────────────────────
    qtypes: dict[str, int] = defaultdict(int)
    for q in multihop:
        qtypes[q.get("question_type") or "unknown"] += 1

    print(f"\n  Question types:")
    for qtype, count in sorted(qtypes.items(), key=lambda x: -x[1]):
        print(f"    {qtype:<30} {count:>4}")

    # ── Dry run: preview only ─────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n  ── Dry run — would write ─────────────────────────────")
        print(f"  {len(articles_to_process)} article files → {args.out_dir}/")
        print(f"  questions.json          → {args.eval_dir}/questions.json")
        print(f"  evidence_map.json       → {args.eval_dir}/evidence_map.json")
        print(f"\n  Run without --dry-run to write files.\n")
        return

    # ── Create output directories ─────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.eval_dir.mkdir(parents=True, exist_ok=True)

    # ── Write article files ───────────────────────────────────────────────────
    print(f"\n  ── Writing articles → {args.out_dir}/")
    written = skipped = empty = 0

    for i, article in enumerate(articles_to_process, 1):
        body = (article.get("body") or "").strip()
        if not body:
            empty += 1
            continue

        filename = _make_article_filename(article, i)
        out_path = args.out_dir / filename

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        prepared = prepare_article(article, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(prepared, f, ensure_ascii=False, indent=2)

        written += 1
        if i % 50 == 0 or i == len(articles_to_process):
            print(f"\r  Written {written} | Skipped {skipped} | Empty {empty} "
                  f"({i}/{len(articles_to_process)})", end="", flush=True)

    print(f"\n\n  ✓ Articles: {written} written, {skipped} skipped (already exist), "
          f"{empty} empty (skipped)")

    # ── Verify a sample article ───────────────────────────────────────────────
    sample_files = sorted(args.out_dir.glob("*.json"))[:1]
    if sample_files:
        with open(sample_files[0], encoding="utf-8") as f:
            sample = json.load(f)
        print(f"\n  Sample article ({sample_files[0].name}):")
        print(f"    title:      {sample['metadata']['title'][:60]}")
        print(f"    source:     {sample['metadata']['source']}")
        print(f"    category:   {sample['metadata']['category']}")
        print(f"    text chars: {len(sample['text'])}")
        print(f"    text start: {sample['text'][:80].replace(chr(10), ' ')}...")

    # ── Write evaluation files ────────────────────────────────────────────────
    print(f"\n  ── Writing evaluation files → {args.eval_dir}/")

    questions, evidence_map = prepare_questions(multihop)

    questions_path = args.eval_dir / "questions.json"
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"  ✓ questions.json:    {len(questions)} questions")

    evidence_path = args.eval_dir / "evidence_map.json"
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence_map, f, ensure_ascii=False, indent=2)
    print(f"  ✓ evidence_map.json: {len(evidence_map)} entries")

    # ── Final summary ─────────────────────────────────────────────────────────
    total_articles = len(list(args.out_dir.glob("*.json")))
    avg_words = sum(
        len((a.get("body") or "").split())
        for a in articles_to_process
        if (a.get("body") or "").strip()
    ) // max(1, written + skipped)

    print(f"\n  ── Summary ──────────────────────────────────────────")
    print(f"  Article files on disk: {total_articles}")
    print(f"  Avg words per article: ~{avg_words}")
    print(f"  Est. chunks at 600t:   ~{total_articles * max(1, avg_words // 450)}")
    print(f"  Questions prepared:    {len(questions)}")
    print(f"  Evidence map entries:  {len(evidence_map)}")

    print(f"""
  ── Next steps ───────────────────────────────────────
  1. Update your .env:
       RAW_DATA_DIR=../../data/raw/articles

  2. Quick test (first 50 articles, 0 gleaning rounds):
       python scripts/run_indexing.py \\
           --data-dir data/raw/articles \\
           --max-chunks 100 \\
           --gleaning-rounds 0

  3. Full index (all {total_articles} articles, paper parameters):
       python scripts/run_indexing.py \\
           --data-dir data/raw/articles

  4. Run evaluation:
       python scripts/run_evaluation.py \\
           --questions-file data/evaluation/questions.json \\
           --max-questions 20
""")


if __name__ == "__main__":
    main()