from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import LiteratureAnalysisAgent
from .config import default_db_path, default_parsed_dir


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    agent = LiteratureAnalysisAgent(db_path=args.db, parsed_dir=args.parsed_dir)

    if args.command == "init":
        paths = agent.init()
        print("Initialized literature analysis agent:")
        for name, path in paths.items():
            print(f"- {name}: {path}")
        return

    if args.command == "convert-pdfs":
        result = agent.convert_pdfs(round_id=args.round_id, limit=args.limit, force=args.force)
        print("PDF conversion finished:")
        for name, value in result.items():
            print(f"- {name}: {value}")
        return

    if args.command == "index-corpus":
        result = agent.index_corpus(force=not args.incremental)
        _print_result("Corpus indexing finished:", result)
        return

    if args.command == "search":
        rows = agent.search(args.query, limit=args.limit)
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for row in rows:
                print(f"- [{row['score']}] {row['title']} :: {row['section_path']} :: {row['text'][:220]}")
        return

    if args.command == "build-paper-cards":
        _print_result("Paper Cards build finished:", agent.build_paper_cards())
        return

    if args.command == "build-graph":
        _print_result("Graph build finished:", agent.build_graph())
        return

    if args.command == "build-wiki":
        _print_result("Wiki build finished:", agent.build_wiki())
        return

    if args.command == "build-all":
        result = agent.build_all()
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_result("Knowledge pipeline finished:", result)
        return

    parser.print_help()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lit-analysis-agent", description="Literature PDF analysis and corpus conversion agent.")
    parser.add_argument("--db", type=Path, default=default_db_path(), help="SQLite database path from the literature retrieval agent.")
    parser.add_argument("--parsed-dir", type=Path, default=default_parsed_dir(), help="Markdown/JSON conversion output directory.")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Initialize output directories and conversion tables.")

    convert = sub.add_parser("convert-pdfs", help="Convert collected PDFs to Markdown and structured JSON.")
    convert.add_argument("--round-id", type=int, default=None, help="Only convert PDFs for this exploration round.")
    convert.add_argument("--limit", type=int, default=None, help="Limit number of PDFs converted.")
    convert.add_argument("--force", action="store_true", help="Re-convert PDFs even if outputs already exist.")

    index = sub.add_parser("index-corpus", help="Build the SQLite FTS5 and local embedding index from parsed JSON.")
    index.add_argument("--incremental", action="store_true", help="Keep existing chunks instead of rebuilding the index.")

    search = sub.add_parser("search", help="Search the indexed corpus with hybrid BM25/vector ranking.")
    search.add_argument("query", help="Search query.")
    search.add_argument("--limit", type=int, default=10, help="Maximum number of results.")
    search.add_argument("--json", action="store_true", help="Print JSON results for frontend integration.")

    sub.add_parser("build-paper-cards", help="Generate structured Paper Cards from parsed papers.")
    sub.add_parser("build-graph", help="Generate lightweight knowledge graph tables and data/graph/graph.json.")
    sub.add_parser("build-wiki", help="Generate evidence-linked Wiki pages for topic nodes.")
    build_all = sub.add_parser("build-all", help="Run index, Paper Cards, Graph, and Wiki in sequence.")
    build_all.add_argument("--json", action="store_true", help="Print JSON summary.")
    return parser


def _print_result(title: str, result: dict[str, object]) -> None:
    print(title)
    for name, value in result.items():
        print(f"- {name}: {value}")


if __name__ == "__main__":
    main()
