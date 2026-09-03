# Project instructions

## Stack

- Python
- NumPy
- Pandas
- PyTest
- uv

## Commands

- Install dependencies: `uv sync`
- Add a dependency: `uv add <package>`
- Run tests: `uv run pytest`

## Engineering rules 

- Follow the architecture already used in the repository. 
- Prefer small, focused changes. 
- Do not rewrite unrelated code. 
- Do not add production dependencies without asking first. 
- Add or update tests when behavior changes. 
- Fix TypeScript errors caused by your changes. 
- Run relevant tests before declaring a task complete. 
- Do not commit or push unless explicitly instructed. 
- Never put credentials or secrets into source code. 

## Before finishing a task

Report: 
1. What you changed. 
2. Which files changed. 
3. Tests you ran. 
4. Any known limitations or follow-up work.