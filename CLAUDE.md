# Project Configuration

This is a Python project that uses a virtual environment located in `.venv`.

## Setup
- Activate the virtual environment with `source .venv/bin/activate`
- Install dependencies with `pip install -r requirements.txt` (if requirements file exists)

## Git Configuration
- All git commits in this project must use the author "Michal Migurski <mike@teczno.com>"
- Always add individual files to git by name, never use general commands like "git add ." or "git add -A"
- This prevents unwanted files from being committed to the repository
- Always git add all touched files after making changes, including generated SVG files (us-states.svg, us-states2.svg, us-states3.svg, us-states4.svg)

## Development
- Run tests and linting before commits using `ruff` for linting and formatting
- Follow Python best practices and PEP 8 style guidelines
- **IMPORTANT**: Always check the Makefile for correct output filenames - each render script has its own target file (e.g., render-graph4.py outputs to us-states4.svg, not us-states.svg)
- Use `make` to build SVG files to ensure correct filenames are used
- Preview SVG output by opening in Safari with `open -a Safari <filename>.svg`

## Coding Style
- Use full module imports: `import module` instead of `from module import member`
- Do not use import aliases (e.g., avoid `import xml.etree.ElementTree as ET`)
- Use built-in types for type hints: `list`, `tuple`, `dict` instead of `typing.List`, `typing.Tuple`, `typing.Dict`
- Use `from __future__ import annotations` to enable modern type hint syntax (e.g., `str | None` instead of `Optional[str]`)