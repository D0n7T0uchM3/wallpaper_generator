import os
from pathlib import Path

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.1.100:11434")
STABLE_DIFFUSION_URL = os.getenv("SD_URL", "http://192.168.1.100:7860")

OLLAMA_AUTO_SELECT = os.getenv("OLLAMA_AUTO_SELECT", "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:3.8b")

IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "1152"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "896"))
NUM_WALLPAPERS = int(os.getenv("NUM_WALLPAPERS", "5"))

SD_STEPS = int(os.getenv("SD_STEPS", "26"))
SD_CFG_SCALE = float(os.getenv("SD_CFG_SCALE", "3"))
SD_SAMPLER = os.getenv("SD_SAMPLER", "Euler a")

ENABLE_UPSCALE = os.getenv("ENABLE_UPSCALE", "true").lower() == "true"
UPSCALER_MODEL = os.getenv("UPSCALER_MODEL", "R-ESRGAN 4x+")
UPSCALE_FACTOR = float(os.getenv("UPSCALE_FACTOR", "3.33"))

OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(Path.home() / "Images"))

OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
SD_TIMEOUT = int(os.getenv("SD_TIMEOUT", "300"))

REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "2"))
