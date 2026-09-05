#!/usr/bin/env python3
"""
list_models.py
===============

Diagnostic script — lists every Gemini model that is ACTUALLY available to
YOUR API key/project right now, and whether it supports `generateContent`
(i.e. is usable as `GEMINI_MODEL` in main.py / app.py).

Why this exists: Google adds and retires Gemini model names frequently, and
availability differs by account, project, and signup date. Any static model
name — whether suggested by me, by another AI, or by a blog post — can be
stale by the time you read it. This script asks Google directly, using your
own key, so you get ground truth instead of a guess.

Usage
-----
    export GEMINI_API_KEY="your-gemini-api-key"
    python list_models.py

(Loads a .env file automatically if python-dotenv is installed, same as
main.py.)
"""

from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
except ImportError:
    print(
        "ERROR: the 'google-genai' package is required for this script.\n"
        "It's already a dependency of langchain-google-genai, so if main.py\n"
        "works, this should too. Otherwise: pip install google-genai",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(
            "ERROR: GEMINI_API_KEY is not set. Set it and try again:\n"
            '    export GEMINI_API_KEY="your-gemini-api-key"',
            file=sys.stderr,
        )
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print("Querying Google for models available to your key/project...\n")

    usable = []
    other = []
    try:
        for model in client.models.list():
            actions = model.supported_actions or []
            entry = (model.name, model.display_name, actions)
            if "generateContent" in actions:
                usable.append(entry)
            else:
                other.append(entry)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not list models — {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("Models you can use as GEMINI_MODEL (support generateContent):")
    print("=" * 70)
    if usable:
        for name, display_name, _actions in usable:
            # `name` is typically "models/gemini-x.y-flash" — strip the
            # "models/" prefix since that's what GEMINI_MODEL expects.
            short_name = name.split("/", 1)[-1] if name else "(unknown)"
            print(f"  {short_name:<30} ({display_name})")
    else:
        print("  (none found — double-check your API key and project)")

    if other:
        print()
        print("=" * 70)
        print("Other models on your account (do NOT support generateContent,")
        print("not usable as a chat model here):")
        print("=" * 70)
        for name, display_name, _actions in other:
            short_name = name.split("/", 1)[-1] if name else "(unknown)"
            print(f"  {short_name:<30} ({display_name})")

    print()
    print("Pick a model from the FIRST list above and set it, e.g.:")
    print('    export GEMINI_MODEL="<model-name-from-list>"')
    print("Then re-run main.py or app.py.")


if __name__ == "__main__":
    main()
