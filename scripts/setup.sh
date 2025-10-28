#!/bin/bash
# Setup script for Bilibili Live Transcriber
# This script sets up a clean Python virtual environment and installs dependencies

set -e  # Exit on error, but allow errors in conditional blocks

echo "Setting up Bilibili Live Transcriber..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed" >&2
    echo "Please install Python 3.8 or higher from https://python.org" >&2
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
min_version="3.8"
max_version="3.13"

# Extract major and minor
python_major=$(echo $python_version | cut -d'.' -f1)
python_minor=$(echo $python_version | cut -d'.' -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    echo "❌ Error: Python $python_version is not supported" >&2
    echo "Please use Python 3.8 - 3.13" >&2
    exit 1
fi

PYTHON_314=false
if [ "$python_major" -eq 3 ] && [ "$python_minor" -ge 14 ]; then
    PYTHON_314=true
    echo "✓ Python $python_version detected - using beta numba for compatibility"
else
    echo "✓ Python $python_version detected"
fi
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
    else
        echo "Using existing virtual environment"
        source venv/bin/activate
        pip install --upgrade pip -q
        pip install -r requirements.txt
        echo ""
        echo "✓ Setup complete!"
        exit 0
    fi
fi

python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
echo "Installing dependencies..."
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found" >&2
    exit 1
fi

if [ "$PYTHON_314" = true ]; then
    echo ""
    echo "Installing Python 3.14 compatible dependencies..."
    
    # Install beta numba and llvmlite with pre-releases enabled
    echo "Installing numba (beta for Python 3.14)..."
    pip install --pre numba llvmlite
    
    # Now install everything else
    echo "Installing remaining dependencies..."
    pip install -r requirements.txt
else
    pip install -r requirements.txt
fi

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import whisper; import yt_dlp; import pyannote.audio" 2>/dev/null && echo "✓ Core dependencies verified" || echo "⚠️  Some imports failed (may need HuggingFace setup)"

echo ""
echo "✓ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Setup HuggingFace authentication (required for speaker diarization):"
echo "     huggingface-cli login"
echo ""
echo "  3. Setup HuggingFace token in .env file (see env.example)"
echo "  4. Start using the transcriber:"
echo "     python -m bilibili_transcriber.main <bilibili_url>"
echo ""

