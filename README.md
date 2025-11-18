# Wallpaper Generator

AI-powered wallpaper generator using Ollama and Stable Diffusion.

## Features

- 🤖 **Two prompt generation modes:**
  - LLM mode: Uses Ollama to intelligently generate prompts
  - Random mode: Generates diverse prompts by randomly combining elements
- 🎨 Image generation with Stable Diffusion
- ⬆️ Built-in upscaling to 4K (3840×2160)
- 🔧 Fully configurable via environment variables
- 🎲 Massive variety with random generation (thousands of unique combinations)

## Requirements

- Python 3.9+
- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) with API enabled
- [Ollama](https://ollama.ai/) running locally or remotely (only required if using LLM mode)

## Installation

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Edit `config.py` or set environment variables:

```python
# Prompt generation mode
USE_LLM = False           # Set to True to use Ollama LLM, False for random generation

# API endpoints
OLLAMA_URL = "http://192.168.1.100:11434"         # Only needed if USE_LLM=True
STABLE_DIFFUSION_URL = "http://192.168.1.100:7860"

# Image settings
IMAGE_WIDTH = 1920        # Base resolution (will be upscaled)
IMAGE_HEIGHT = 1080       # Base resolution (will be upscaled)
UPSCALE_FACTOR = 2        # 2x = 4K output (3840×2160)
NUM_WALLPAPERS = 10       # Number of wallpapers to generate
```

### Prompt Generation Modes

#### Random Mode (Default, `USE_LLM=False`)
Generates prompts by randomly combining elements from extensive lists:
- Character types (1girl, 1boy, 2girls, etc.)
- Appearance (hair colors, styles, eye colors, features)
- Poses and actions (standing, sitting, holding items, etc.)
- Clothing types and details
- Backgrounds and settings (outdoors, city, forest, etc.)
- Lighting and atmosphere
- Camera angles and art styles

This creates thousands of unique combinations with massive variety!

#### LLM Mode (`USE_LLM=True`)
Uses Ollama to intelligently generate creative prompts based on examples. Requires Ollama to be running.

## Usage

```bash
python wallpaper_generator.py
```

Generated wallpapers will be saved to `~/Images` by default.

## Customization

### Random Mode
The random prompt generator can be customized by editing the lists in `wallpaper_generator.py` in the `_generate_random_prompt()` method. You can:
- Add or remove elements from any category
- Adjust probability weights for different features
- Add new categories of elements

### LLM Mode
Edit `llm.json` to customize the prompt generation instructions for the LLM.

## Environment Variables

You can configure the generator using environment variables instead of editing `config.py`:

```bash
export USE_LLM=false                            # Enable/disable LLM mode
export OLLAMA_URL=http://192.168.1.100:11434
export SD_URL=http://192.168.1.100:7860
export IMAGE_WIDTH=1152
export IMAGE_HEIGHT=896
export NUM_WALLPAPERS=5
export ENABLE_UPSCALE=true
export UPSCALE_FACTOR=3.33
export OUTPUT_DIR=~/Images
```
