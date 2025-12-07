# Contributing to Docker Model Runner

First off, thank you for considering contributing to Docker Model Runner! It's people like you that make Docker Model Runner such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title** for the issue to identify the problem.
- **Describe the exact steps which reproduce the problem** in as many details as possible.
- **Provide specific examples to demonstrate the steps**.
- **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
- **Explain which behavior you expected to see instead and why.**
- **Include logs and error messages** if applicable.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title** for the issue to identify the suggestion.
- **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
- **Provide specific examples to demonstrate the steps**.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
- **Explain why this enhancement would be useful** to most Docker Model Runner users.

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the PR title
- Follow the Python style guide (PEP 8)
- Include thoughtfully-worded, well-structured tests
- Document new code
- End all files with a newline

## Development Setup

1. Fork the repo and clone your fork:

```bash
git clone https://github.com/your-username/docker-model-runner.git
cd docker-model-runner
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the development server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 7860
```

## Style Guide

### Python

- Follow PEP 8
- Use type hints where possible
- Write docstrings for all public functions
- Keep functions small and focused

### Commits

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

### Documentation

- Use Markdown for documentation
- Keep language simple and clear
- Include code examples where helpful

## Testing

Run tests with:

```bash
pytest tests/
```

Ensure all tests pass before submitting a PR.

## Questions?

Feel free to open an issue with your question or reach out to the maintainers.

Thank you for contributing!
