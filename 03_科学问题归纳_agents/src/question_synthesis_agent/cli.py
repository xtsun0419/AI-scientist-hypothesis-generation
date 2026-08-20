from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import QuestionSynthesisAgent
from .config import default_db_path
from .db import QuestionSynthesisDB
from .env import load_local_env


def main(argv: list[str] | None = None) -> None:
    load_local_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    db = QuestionSynthesisDB(args.db)
    db.init_schema()
    try:
        agent = QuestionSynthesisAgent(db)
        if args.command == "init":
            state = agent.state()
            print(f"Initialized question synthesis agent: {args.db}")
            print(f"Messages: {len(state['messages'])}")
            print(f"Model: {state['model_name']}")
            return
        if args.command == "state":
            state = agent.state()
            if args.json:
                print(json.dumps(state, ensure_ascii=False, indent=2))
            else:
                print(f"Session: {state['session']['title'] if state['session'] else 'none'}")
                print(f"Model: {state['model_name']}")
                for message in state["messages"]:
                    print(f"\n[{message['speaker']}]\n{message['content']}")
            return
        if args.command == "chat":
            state = agent.chat(args.message)
            print(state["messages"][-1]["content"])
            return
        if args.command == "reset":
            agent.reset()
            print("Question synthesis conversation reset.")
            return
        if args.command == "confirm":
            state = agent.confirm()
            latest = state["confirmed_questions"][0] if state["confirmed_questions"] else None
            if latest is None:
                print("No confirmed question was saved.")
                return
            if args.json:
                print(json.dumps(latest, ensure_ascii=False, indent=2))
            else:
                print(f"确认问题已保存 (#{latest['id']}):")
                print(f"- 问题陈述：{latest['problem_statement']}")
                print(f"- 关键变量：{', '.join(latest['variables']) or '待补充'}")
                print(f"- 机制假设：{latest['mechanism_hypothesis']}")
                print(f"- 验证判据：{'; '.join(latest['validation_criteria']) or '待补充'}")
                print(f"- 证据：{', '.join(latest['evidence_ids']) or '无可追溯证据'}")
                print(f"- 模式：{latest['mode']}")
            return
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="question-synthesis-agent")
    parser.add_argument("--db", type=Path, default=default_db_path())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    state = sub.add_parser("state")
    state.add_argument("--json", action="store_true")
    chat = sub.add_parser("chat")
    chat.add_argument("message")
    sub.add_parser("reset")
    confirm = sub.add_parser("confirm")
    confirm.add_argument("--json", action="store_true")
    return parser


if __name__ == "__main__":
    main()
