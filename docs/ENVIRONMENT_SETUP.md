# Python Environment Setup

This document explains the Python environment setup following best practices for production-ready projects.

## Approach

This project uses **venv (virtual environment)** exclusively for Python environment management. **Note: Python 3.14+ requires installing beta numba (0.63.0b1+)**.

- ✅ **venv** - Creates isolated environments for clean dependency management
- ❌ **pyenv** - Not required (assumes system Python 3.8+)

## Why venv Only?

1. **Simplicity**: No need for Python version management if Python 3.8+ is available
2. **Standard**: venv is part of Python's standard library since Python 3.3
3. **Production Ready**: Follows Python packaging best practices
4. **Clean**: Keeps the project root tidy with essential files only

## Setup Options

### Option 1: Automated Setup (Recommended)

**Bash script:**
```bash
./scripts/setup.sh
```

**Python script (cross-platform):**
```bash
python scripts/setup_env.py
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

Following best practices for clean organization:

```
bilibili-live-transcriber/
├── README.md              # Main documentation
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── config.py              # Configuration settings
├── main.py                # Main entry point
├── *.py                   # Application modules
├── scripts/               # Automation scripts
│   ├── setup.sh          # Bash setup script
│   ├── setup_env.py      # Python setup helper
│   ├── example.py        # Usage examples
│   └── README.md         # Scripts documentation
├── src/                   # Reserved for future source organization
├── tests/                 # Test files (when added)
├── venv/                  # Virtual environment (gitignored)
├── videos/                # Downloaded videos (gitignored)
├── audio/                 # Extracted audio files (gitignored)
└── output/                # Transcription results (gitignored)
```

## Dependencies

### Production
- `openai-whisper` - Speech transcription
- `pyannote.audio` - Speaker diarization
- `yt-dlp` - Video downloading
- `torch` - ML framework
- `click` - CLI interface

### Development (optional)
- `pytest` - Testing framework
- `black` - Code formatter
- `flake8` - Linter
- `mypy` - Type checking

## Virtual Environment Best Practices

1. **Always activate before working:**
   ```bash
   source venv/bin/activate
   ```

2. **Keep dependencies up to date:**
   ```bash
   pip list --outdated
   pip install --upgrade <package>
   ```

3. **Never commit venv directory:**
   - Already in `.gitignore`

4. **Export clean requirements:**
   ```bash
   pip freeze > requirements-freeze.txt
   ```

## Troubleshooting

### Issue: "Command not found: python3"
- Install Python 3.8-3.13 from https://python.org
- Note: Python 3.14+ may have compatibility issues with numba

### Issue: "RuntimeError: Cannot install on Python version 3.14"
- You're using Python 3.14 which requires special setup
- Install beta numba: `pip install numba>=0.63.0b1 llvmlite>=0.46.0b1`
- Then reinstall openai-whisper: `pip install --force-reinstall openai-whisper`
- Alternatively, use Python 3.8-3.13 for stable support

### Issue: "venv module not found"
- Ensure you're using Python 3.3+ (venv was added in 3.3)

### Issue: "Permission denied" on setup script
- Make executable: `chmod +x scripts/setup.sh`

### Issue: Import errors after setup
- Verify activation: `which python` should point to venv
- Reinstall: `pip install -r requirements.txt`

## Additional Setup

After initial setup, you need to configure HuggingFace for speaker diarization:

```bash
# Login to HuggingFace
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN="your_token_here"
```

Visit https://huggingface.co/pyannote/speaker-diarization-3.1 to accept the model license.

