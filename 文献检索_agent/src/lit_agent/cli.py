from __future__ import annotations

import argparse
from pathlib import Path

from .agents import OrchestratorAgent
from .config import default_db_path, default_parsed_dir, default_pdf_dir, default_report_dir
from .env import load_local_env


def main(argv: list[str] | None = None) -> None:
    load_local_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    orchestrator = OrchestratorAgent(
        db_path=args.db,
        config_path=args.config,
        pdf_dir=args.pdf_dir,
        report_dir=args.report_dir,
        parsed_dir=args.parsed_dir,
    )

    if args.command == "init":
        paths = orchestrator.init()
        print("Initialized literature agent:")
        for name, path in paths.items():
            print(f"- {name}: {path}")
        return

    if args.command == "search":
        sources = args.sources.split(",") if args.sources else None
        result = orchestrator.search(
            from_year=args.from_year,
            to_year=args.to_year,
            sources=sources,
            query_limit=args.query_limit,
            max_results_per_query=args.max_results_per_query,
            mode=args.mode,
            llm_review=not args.no_llm_review,
        )
        print("Search pipeline finished:")
        for name, value in result.items():
            print(f"- {name}: {value}")
        return

    if args.command == "plan-queries":
        sources = args.sources.split(",") if args.sources else None
        plan = orchestrator.plan_queries(
            from_year=args.from_year,
            to_year=args.to_year,
            sources=sources,
            query_limit=args.query_limit,
            max_results_per_query=args.max_results_per_query,
            mode=args.mode,
        )
        print(f"Domain: {plan.domain}")
        print(f"Years: {plan.from_year}-{plan.to_year}")
        print("Sources: " + ", ".join(plan.sources))
        print(f"Queries: {len(plan.queries)}")
        for query in plan.queries[: args.limit]:
            print(f"- {query}")
        if len(plan.queries) > args.limit:
            print(f"... {len(plan.queries) - args.limit} more")
        return

    if args.command == "discover":
        sources = args.sources.split(",") if args.sources else None
        counts = orchestrator.discover(
            from_year=args.from_year,
            to_year=args.to_year,
            sources=sources,
            query_limit=args.query_limit,
            max_results_per_query=args.max_results_per_query,
            mode=args.mode,
        )
        print("Discovery finished:")
        for name, value in counts.items():
            print(f"- {name}: {value}")
        return

    if args.command == "normalize":
        count = orchestrator.normalize()
        print(f"Normalized candidates: {count}")
        return

    if args.command == "dedup":
        count = orchestrator.dedup()
        print(f"Canonical papers: {count}")
        return

    if args.command == "judge-relevance":
        sources = args.sources.split(",") if args.sources else None
        count = orchestrator.judge_relevance(from_year=args.from_year, to_year=args.to_year, sources=sources)
        print(f"Relevance-scored papers: {count}")
        return

    if args.command == "resolve-oa":
        count = orchestrator.resolve_oa()
        print(f"Resolved access records: {count}")
        return

    if args.command == "review-relevance":
        result = orchestrator.review_relevance()
        print("LLM relevance review finished:")
        for name, value in result.items():
            print(f"- {name}: {value}")
        return

    if args.command == "goal":
        if args.goal_command == "create":
            goal_id = orchestrator.create_goal(
                title=args.title,
                description=args.description,
                domain=args.domain,
                target_count=args.target_count,
            )
            print(f"Scientific goal created: {goal_id}")
            return
        if args.goal_command == "list":
            for goal in orchestrator.list_goals():
                print(f"{goal['id']}: {goal['title']} [{goal['status']}] target={goal['default_target_count']}")
            return

    if args.command == "round":
        if args.round_command == "plan":
            sources = args.sources.split(",") if args.sources else None
            result = orchestrator.plan_round(
                goal_id=args.goal_id,
                target_count=args.target_count,
                max_results_per_query=args.max_results_per_query,
                query_limit=args.query_limit,
                sources=sources,
            )
            print("Round planned:")
            for name, value in result.items():
                print(f"- {name}: {value}")
            print("Next: python3 run.py round approve --round-id " + str(result["round_id"]))
            return
        if args.round_command == "approve":
            orchestrator.approve_round(args.round_id)
            print(f"Round approved: {args.round_id}")
            return
        if args.round_command == "acquire":
            result = orchestrator.acquire_round(args.round_id)
            print("Round acquisition finished:")
            for name, value in result.items():
                print(f"- {name}: {value}")
            return
        if args.round_command == "intake-manual":
            count = orchestrator.intake_manual_round(args.round_id)
            print(f"Manual PDFs linked: {count}")
            return
        if args.round_command == "analyze":
            count = orchestrator.analyze_round(args.round_id)
            print(f"Round analyses created: {count}")
            return
        if args.round_command == "propose-next":
            result = orchestrator.propose_next_round(args.round_id)
            print("Next queries:")
            for query in result["next_queries"]:
                print(f"- {query}")
            return
        if args.round_command == "report":
            result = orchestrator.round_report(args.round_id)
            print(f"Goal: {result['goal']['title']}")
            print(f"Round: {result['round']['id']} status={result['round']['status']}")
            print(f"Candidates: {len(result['candidates'])}")
            print(f"Manual download tasks: {len(result['manual_tasks'])}")
            print(f"Analyses: {len(result['analyses'])}")
            if result["synthesis"]:
                print("Synthesis: " + result["synthesis"]["summary"])
            return

    if args.command == "download":
        count = orchestrator.download(limit=args.limit)
        print(f"Downloaded PDF files: {count}")
        return

    if args.command == "convert-pdfs":
        result = orchestrator.convert_pdfs(round_id=args.round_id, limit=args.limit, force=args.force)
        print("PDF conversion finished:")
        for name, value in result.items():
            print(f"- {name}: {value}")
        return

    if args.command == "audit":
        count = orchestrator.audit()
        print(f"Audit findings created: {count}")
        return

    if args.command == "report":
        outputs = orchestrator.report()
        print("Reports written:")
        for name, path in outputs.items():
            print(f"- {name}: {path}")
        return

    if args.command == "dashboard":
        path = orchestrator.dashboard()
        print(f"Dashboard written: {path}")
        return

    if args.command == "export":
        path = orchestrator.export(args.format)
        print(f"Export written: {path}")
        return

    if args.command == "web":
        from .web_v3 import serve

        serve(host=args.host, port=args.port)
        return

    parser.print_help()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lit-agent", description="General-purpose scientific literature discovery agent.")
    parser.add_argument("--db", type=Path, default=default_db_path(), help="SQLite database path.")
    parser.add_argument("--config", type=Path, default=None, help="Domain config path.")
    parser.add_argument("--pdf-dir", type=Path, default=default_pdf_dir(), help="PDF output directory.")
    parser.add_argument("--report-dir", type=Path, default=default_report_dir(), help="Report output directory.")
    parser.add_argument("--parsed-dir", type=Path, default=default_parsed_dir(), help="Markdown/JSON conversion output directory.")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Initialize database and output directories.")

    search = sub.add_parser("search", help="Run discovery, normalization, deduplication, relevance, and OA resolution.")
    search.add_argument("--domain", default="general_research", help="Optional label for the current research scope.")
    search.add_argument("--from-year", type=int, default=None)
    search.add_argument("--to-year", type=int, default=None)
    search.add_argument("--sources", default=None, help="Comma-separated source names.")
    search.add_argument("--mode", choices=["smoke", "pilot", "full"], default="full", help="Reproducible run scale preset.")
    search.add_argument("--query-limit", type=int, default=None, help="Limit number of generated queries for smoke runs.")
    search.add_argument("--max-results-per-query", type=int, default=None, help="Override per-source results per query.")
    search.add_argument("--no-llm-review", action="store_true", help="Skip boundary-case LLM relevance review.")

    plan = sub.add_parser("plan-queries", help="Print the generated domain query plan.")
    plan.add_argument("--from-year", type=int, default=None)
    plan.add_argument("--to-year", type=int, default=None)
    plan.add_argument("--sources", default=None, help="Comma-separated source names.")
    plan.add_argument("--mode", choices=["smoke", "pilot", "full"], default="full", help="Reproducible run scale preset.")
    plan.add_argument("--limit", type=int, default=30, help="Number of queries to print.")
    plan.add_argument("--query-limit", type=int, default=None, help="Limit number of generated queries for smoke runs.")
    plan.add_argument("--max-results-per-query", type=int, default=None, help="Override per-source results per query.")

    discover = sub.add_parser("discover", help="Run only SourceDiscoveryAgent.")
    discover.add_argument("--from-year", type=int, default=None)
    discover.add_argument("--to-year", type=int, default=None)
    discover.add_argument("--sources", default=None, help="Comma-separated source names.")
    discover.add_argument("--mode", choices=["smoke", "pilot", "full"], default="full", help="Reproducible run scale preset.")
    discover.add_argument("--query-limit", type=int, default=None, help="Limit number of generated queries for smoke runs.")
    discover.add_argument("--max-results-per-query", type=int, default=None, help="Override per-source results per query.")

    sub.add_parser("normalize", help="Run only MetadataNormalizeAgent.")
    sub.add_parser("dedup", help="Run only DeduplicationAgent.")

    judge = sub.add_parser("judge-relevance", help="Run only RelevanceJudgeAgent.")
    judge.add_argument("--from-year", type=int, default=None)
    judge.add_argument("--to-year", type=int, default=None)
    judge.add_argument("--sources", default=None, help="Comma-separated source names.")

    sub.add_parser("resolve-oa", help="Resolve OA status and DOI access hints for existing papers.")
    sub.add_parser("review-relevance", help="Run LLM relevance review for boundary cases.")

    goal = sub.add_parser("goal", help="Create or list scientific goals.")
    goal_sub = goal.add_subparsers(dest="goal_command", required=True)
    goal_create = goal_sub.add_parser("create", help="Create a scientific goal.")
    goal_create.add_argument("--title", required=True)
    goal_create.add_argument("--description", default=None)
    goal_create.add_argument("--domain", default="general_research")
    goal_create.add_argument("--target-count", type=int, default=20)
    goal_sub.add_parser("list", help="List scientific goals.")

    round_parser = sub.add_parser("round", help="Plan and run iterative exploration rounds.")
    round_sub = round_parser.add_subparsers(dest="round_command", required=True)
    round_plan = round_sub.add_parser("plan", help="Plan one exploration round.")
    round_plan.add_argument("--goal-id", type=int, required=True)
    round_plan.add_argument("--target-count", type=int, default=None)
    round_plan.add_argument("--query-limit", type=int, default=4)
    round_plan.add_argument("--max-results-per-query", type=int, default=8)
    round_plan.add_argument("--sources", default=None, help="Comma-separated source names.")
    round_approve = round_sub.add_parser("approve", help="Approve a planned round.")
    round_approve.add_argument("--round-id", type=int, required=True)
    round_acquire = round_sub.add_parser("acquire", help="Download approved round PDFs and create manual tasks.")
    round_acquire.add_argument("--round-id", type=int, required=True)
    round_intake = round_sub.add_parser("intake-manual", help="Link manually downloaded PDFs for a round.")
    round_intake.add_argument("--round-id", type=int, required=True)
    round_analyze = round_sub.add_parser("analyze", help="Analyze round PDFs and metadata.")
    round_analyze.add_argument("--round-id", type=int, required=True)
    round_next = round_sub.add_parser("propose-next", help="Propose next-round queries.")
    round_next.add_argument("--round-id", type=int, required=True)
    round_report = round_sub.add_parser("report", help="Print round status summary.")
    round_report.add_argument("--round-id", type=int, required=True)

    download = sub.add_parser("download", help="Download open or explicitly public PDF files.")
    download.add_argument("--limit", type=int, default=None)

    convert = sub.add_parser("convert-pdfs", help="Convert collected PDFs to Markdown and structured JSON.")
    convert.add_argument("--round-id", type=int, default=None, help="Only convert PDFs for this exploration round.")
    convert.add_argument("--limit", type=int, default=None, help="Limit number of PDFs converted.")
    convert.add_argument("--force", action="store_true", help="Re-convert PDFs even if outputs already exist.")

    sub.add_parser("audit", help="Create quality audit findings.")
    sub.add_parser("report", help="Write operational CSV and Markdown reports.")
    sub.add_parser("dashboard", help="Write a self-contained HTML dashboard.")

    export = sub.add_parser("export", help="Export literature database.")
    export.add_argument("--format", choices=["csv", "jsonl", "bibtex"], default="csv")

    web = sub.add_parser("web", help="Start the local v3 web UI.")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)

    return parser


if __name__ == "__main__":
    main()
