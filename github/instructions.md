# GitHub Instructions

## Quick Setup Guide

### For Contributors

1. **Fork the Repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/docker-model-runner.git
   cd docker-model-runner
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Run Locally**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 7860
   ```

4. **Test Your Changes**
   ```bash
   # Test Anthropic API
   curl -X POST http://localhost:7860/v1/messages \
     -H "Content-Type: application/json" \
     -d '{"model":"test","max_tokens":50,"messages":[{"role":"user","content":"Hello"}]}'
   ```

---

## Branch Naming Convention

| Type | Format | Example |
|------|--------|---------|
| Feature | `feature/description` | `feature/add-vision-support` |
| Bug Fix | `fix/description` | `fix/streaming-timeout` |
| Docs | `docs/description` | `docs/update-readme` |
| Refactor | `refactor/description` | `refactor/api-handlers` |

---

## Commit Message Format

```
<type>: <short description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Example:**
```
feat: add vision/image support for multimodal models

- Added ImageBlock handling in content parser
- Updated generate_text to handle image inputs
- Added base64 image decoding

Closes #12
```

---

## Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make Changes & Commit**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

3. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature
   ```

4. **Open Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template

---

## Issue Templates

### Bug Report
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
1. Send request to '...'
2. With payload '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.11]
- Docker: [e.g., 24.0.5]
```

### Feature Request
```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Alternatives considered**
Any alternative solutions you've considered.

**Additional context**
Any other context or screenshots.
```

---

## Code Style

- **Python**: Follow PEP 8
- **Type Hints**: Use type hints for all functions
- **Docstrings**: Google style docstrings
- **Line Length**: Max 100 characters

```python
def example_function(param1: str, param2: int) -> dict:
    """Short description.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is empty.
    """
    pass
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=. tests/

# Run specific test
pytest tests/test_api.py::test_messages_endpoint
```

---

## Docker Development

```bash
# Build image
docker build -t model-runner:dev .

# Run container
docker run -p 7860:7860 model-runner:dev

# Build and run with compose
docker-compose up --build
```

---

## Contact

- **Author**: Likhon Sheikh
- **Telegram**: [@likhonsheikh](https://t.me/likhonsheikh)
- **GitHub**: [@likhonsheikhdev](https://github.com/likhonsheikhdev)

---

*Thank you for contributing to Docker Model Runner!*
