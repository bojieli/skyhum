#!/usr/bin/env python3
"""Generate docs/cli.md from the argparse definitions. Run after changing the CLI; tests check it is current.

usage: python scripts/gen_cli_docs.py [--check]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["COLUMNS"] = "100"

from skyhum.cli import build_parser  # noqa: E402

OUT = os.path.join(ROOT, "docs", "cli.md")


def subparsers(p):
    for a in p._actions:
        if isinstance(a, argparse._SubParsersAction):
            return a.choices
    return {}


def render():
    ap = build_parser()
    parts = ["# CLI reference", "",
             "Generated from the argparse definitions by `scripts/gen_cli_docs.py`. Do not edit by hand.", "",
             "```text", ap.format_help().rstrip(), "```", ""]
    for name, sp in subparsers(ap).items():
        parts += [f"## skyhum {name}", "", "```text", sp.format_help().rstrip(), "```", ""]
        for sub, ssp in subparsers(sp).items():
            parts += [f"### skyhum {name} {sub}", "", "```text", ssp.format_help().rstrip(), "```", ""]
    return "\n".join(parts)


def main():
    text = render()
    if "--check" in sys.argv:
        cur = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if cur != text:
            print("docs/cli.md is out of date: run python scripts/gen_cli_docs.py", file=sys.stderr)
            return 1
        return 0
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
