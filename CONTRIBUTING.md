# Contributing

Thanks for contributing to `tiny_LLM`.

## Development Setup

```powershell
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Repo Conventions

- Keep root-level changes minimal and intentional.
- Put operational scripts under `scripts/`.
- Keep experimental and machine-local artifacts out of git.
- Prefer small, reviewable pull requests.

## Run Checks Before Opening a PR

```powershell
.\scripts\check.ps1
```

Equivalent manual command:

```powershell
python -m unittest discover -s tiny-llm/tests -p "test_*.py"
```

## Pull Request Checklist

- [ ] Tests pass locally.
- [ ] Documentation is updated if behavior or commands changed.
- [ ] No large model files, checkpoints, logs, or private data were added.
- [ ] Changes are scoped and explained in the PR description.
