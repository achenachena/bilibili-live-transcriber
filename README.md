# Bilibili Live Transcriber

A Python tool to transcribe Bilibili live recordings with speaker diarization (speaker identification). Perfect for virtual VTuber recordings!

## Features

- Download recordings from Bilibili URLs
- Extract audio from video files
- Identify different speakers in the recording
- Transcribe speech to text using OpenAI Whisper
- Output formatted transcripts with timestamps and speaker labels in Markdown format

## Requirements

- Python 3.8 or higher (Python 3.14 requires numba>=0.63.0b1)
- ffmpeg (for audio processing)

### Installing ffmpeg

**macOS:**

```bash
brew install ffmpeg
```

**Linux:**

```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/achenachena/bilibili-live-transcriber.git
cd bilibili-live-transcriber

# Run setup script
./scripts/setup.sh
```

### Manual Setup

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd bilibili-live-transcriber
   ```

2. Create and activate a virtual environment:

   ```bash
   # Create virtual environment
   python3 -m venv venv

   # Activate it
   source venv/bin/activate  # macOS/Linux
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install torchcodec for Python 3.14+ (if using Python 3.14):

   ```bash
   pip install torchcodec --index-url https://download.pytorch.org/whl/nightly/
   ```

5. Set up HuggingFace authentication:
   - Visit [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the license
   - Visit [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept the license
   - Generate a token at [HuggingFace settings](https://huggingface.co/settings/tokens) (READ token type)
   - Create a `.env` file:

   ```bash
   cp env.example .env
   # Edit .env and add your token
   ```

## Usage

**Note:** Make sure to activate the virtual environment first:

```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate      # Windows
```

### Basic usage with Bilibili URL

```bash
# The .env file is automatically loaded
python -m bilibili_transcriber.main <bilibili_url>

# Or use the CLI command
bilibili-transcribe <bilibili_url>
```

### Process a local audio/video file

```bash
python -m bilibili_transcriber.main --file path/to/video.mp4
```

### Batch processing multiple videos

Create a text file with URLs (one per line):

```bash
# Create a batch file
cat > urls.txt << EOF
https://www.bilibili.com/video/BV1234567890
https://www.bilibili.com/video/BV0987654321
https://www.bilibili.com/video/BV1122334455
EOF

# Process all URLs in batch
python -m bilibili_transcriber.main --batch urls.txt

# Batch process with custom model
python -m bilibili_transcriber.main --batch urls.txt --model large
```

### Batch processing local files

```bash
# Create a batch file with local file paths
cat > files.txt << EOF
/path/to/video1.mp4
/path/to/video2.mkv
/path/to/audio1.wav
EOF

# Process all files in batch
python -m bilibili_transcriber.main --batch files.txt --file
```

### Advanced options

```bash
python -m bilibili_transcriber.main <bilibili_url> \
  --output-dir ./results \
  --model large \
  --whisper-language zh
```

### Options

- `--file`: Process a local file instead of downloading from URL
- `--batch`: Process multiple URLs/files from a text file (one per line)
- `--output-dir`: Directory to save output (default: ./output)
- `--model`: Whisper model size (tiny, base, small, medium, large)
- `--whisper-language`: Language code for Whisper (e.g., zh, ja, en)

## Output Format

The output will be a Markdown file with formatted timestamps and speaker labels:

```markdown
# Transcription

**[00:00:05] SPEAKER_00:** 大家好，欢迎来到直播间

**[00:00:12] SPEAKER_01:** 谢谢大家来观看

**[00:00:18] SPEAKER_00:** 今天我们要聊什么呢？
```

## Performance Notes

- **First run**: Models will be downloaded (Whisper: ~150MB, pyannote: ~500MB)
- **Processing speed**: ~10-30x real-time on CPU, much faster on GPU
- **Accuracy**: Speaker diarization works best with clear audio and distinct voices
- **Recommended**: Use at least `base` model for Chinese VTubers
- **Auto-cleanup**: Downloaded videos and temporary audio files are automatically deleted after processing
- **Batch processing**: Process multiple videos efficiently with progress tracking and error recovery

## Configuration

Edit `config.py` to customize settings:

- Change Whisper model size
- Adjust audio processing parameters
- Set output directories
- Modify diarization settings

## Troubleshooting

### Error: "HuggingFace authentication required"

- Make sure you've accepted the model licenses at:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- Create a READ token at [HuggingFace settings](https://huggingface.co/settings/tokens)
- Set the token: `export HF_TOKEN="your_token_here"`

### Error: "torchcodec" not found (Python 3.14)

- Install nightly torchcodec: `pip install torchcodec --index-url <https://download.pytorch.org/whl/nightly/>`
- This is required for torchaudio compatibility with Python 3.14

### Error: "Module not found" or "Attribute not found"

- Make sure all dependencies are installed: `pip install -r requirements.txt`
- For Python 3.14, also install: `pip install torchcodec --index-url <https://download.pytorch.org/whl/nightly/>`

### Slow processing

- Use a smaller Whisper model (tiny, base) for faster transcription
- GPU acceleration will significantly speed up processing

### Repetitive output issue

- **Problem**: Whisper sometimes generates repetitive text like "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目" even when audio content is not repetitive
- **Solution**: This project includes automatic filtering to remove repetitive segments
- **Reference**: This is a known Whisper issue documented in [GitHub discussions #2015](https://github.com/openai/whisper/discussions/2015) and [#1962](https://github.com/openai/whisper/discussions/1962)
- **Technical details**: The project uses `condition_on_previous_text=False` and other anti-hallucination parameters to minimize this issue

### Python 3.14 Compatibility

- The project now works with Python 3.14!
- Requires installing nightly builds for torch, torchaudio, and torchcodec
- See setup script for automated installation

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

**Commercial Use:** This project and the models it uses (Whisper, pyannote.audio) are MIT-licensed and allow commercial use. See [ATTRIBUTION.md](ATTRIBUTION.md) for full copyright notices.

**Important:** When using this project commercially:

1. Include the original MIT license notice
2. Retain the ATTRIBUTION.md file
3. Acknowledge OpenAI (Whisper) and pyannote.audio
4. Comply with Bilibili's terms of service when downloading content
