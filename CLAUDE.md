# Project Configuration

This is a Python project that uses a virtual environment located in `.venv`.

## Setup
- Activate the virtual environment with `source .venv/bin/activate`
- Install dependencies with `pip install -r requirements.txt` (if requirements file exists)

## Git Configuration
- All git commits in this project must use the author "Michal Migurski <mike@teczno.com>"
- Always add individual files to git by name, never use general commands like "git add ." or "git add -A"
- This prevents unwanted files from being committed to the repository
- Always git add all touched files after making changes, including generated SVG files (us-states.svg, us-states2.svg)

## Development
- Run tests and linting before commits using `ruff` for linting and formatting
- Follow Python best practices and PEP 8 style guidelines
- Preview SVG output by opening in Safari with `open -a Safari us-states.svg`