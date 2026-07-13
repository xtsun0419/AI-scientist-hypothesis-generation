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
    return parser


if __name__ == "__main__":
    main()
