# Wallpaper Generator

AI-powered wallpaper generator using Ollama and Stable Diffusion.

## Features

- 🤖 Automatic prompt generation with Ollama LLM
- 🎨 Image generation with Stable Diffusion
- ⬆️ Built-in upscaling to 4K (3840×2160)
- 🔧 Fully configurable via environment variables

## Requirements

- Python 3.9+
- [Ollama](https://ollama.ai/) running locally or remotely
- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) with API enabled

## Installation

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Edit `config.py` or set environment variables:

```python
# API endpoints
OLLAMA_URL = "http://192.168.1.100:11434"
STABLE_DIFFUSION_URL = "http://192.168.1.100:7860"

# Image settings
IMAGE_WIDTH = 1920        # Base resolution (will be upscaled)
IMAGE_HEIGHT = 1080       # Base resolution (will be upscaled)
UPSCALE_FACTOR = 2        # 2x = 4K output (3840×2160)
NUM_WALLPAPERS = 10       # Number of wallpapers to generate
```

## Usage

```bash
python wallpaper_generator.py
```

Generated wallpapers will be saved to `~/Images` by default.

## Customization

Edit `llm.json` to customize the prompt generation instructions for the LLM.
