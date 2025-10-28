"""Setup script for Bilibili Live Transcriber."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bilibili-live-transcriber",
    version="0.1.0",
    author="Bilibili Live Transcriber Contributors",
    description="Transcribe Bilibili live recordings with speaker diarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bilibili-live-transcriber",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Video",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    python_requires=">=3.8",
    install_requires=[
        "yt-dlp>=2023.12.30",
        "openai-whisper>=20231117",
        "pyannote.audio>=3.1.1",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "ffmpeg-python>=0.2.0",
        "tqdm>=4.66.1",
        "click>=8.1.7",
    ],
    entry_points={
        "console_scripts": [
            "bilibili-transcribe=bilibili_transcriber.main:main",
        ],
    },
)
