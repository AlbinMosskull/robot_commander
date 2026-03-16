---
name: lint
description: Runs pylint and offers to fix the issues
---

# Lint

Run pylint on the `python/` folder and report the results.

Steps:
1. Run `uv run pylint python/` using the Bash tool
2. Show the full output, including any warnings and errors
3. Summarize the issues found grouped by category (convention, warning, error, refactor)
4. If there are fixable issues, offer to fix them