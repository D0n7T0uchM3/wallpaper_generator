import json
import base64
import time
import re
import logging
from pathlib import Path
from typing import List, Tuple
import requests
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wallpaper_generator.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class WallpaperGenerator:
    def __init__(self):
        logger.info("Initializing WallpaperGenerator")
        self.ollama_url = config.OLLAMA_URL
        self.sd_url = config.STABLE_DIFFUSION_URL
        self.output_dir = Path(config.OUTPUT_DIR).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        
        if config.OLLAMA_AUTO_SELECT:
            logger.info("Auto-selecting best Ollama model")
            self.ollama_model = self._select_best_ollama_model()
        else:
            self.ollama_model = config.OLLAMA_MODEL
        
        logger.info(f"Using Ollama model: {self.ollama_model}")
        
        self.num_prompts = config.NUM_WALLPAPERS
        self.width = config.IMAGE_WIDTH
        self.height = config.IMAGE_HEIGHT
        self.upscale = config.ENABLE_UPSCALE
        self.upscaler = config.UPSCALER_MODEL
        self.upscale_factor = config.UPSCALE_FACTOR
        self.sd_steps = config.SD_STEPS
        self.cfg_scale = config.SD_CFG_SCALE
        self.sampler_name = config.SD_SAMPLER
        self.ollama_timeout = config.OLLAMA_TIMEOUT
        self.sd_timeout = config.SD_TIMEOUT
        self.request_delay = config.REQUEST_DELAY
        
        logger.info(f"Configuration: {self.num_prompts} wallpapers, "
                   f"{self.width}x{self.height}px, upscale={self.upscale}")
    
    def _extract_model_size(self, model_name: str) -> Tuple[float, str]:
        match = re.search(r':(\d+\.?\d*)b', model_name.lower())
        if match:
            size = float(match.group(1))
            return (size, model_name)
        
        return (0, model_name)
    
    def _select_best_ollama_model(self) -> str:
        try:
            logger.debug(f"Fetching available Ollama models from {self.ollama_url}")
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            data = response.json()
            models = data.get("models", [])
            
            if not models:
                logger.warning("No models found, using default model")
                return config.OLLAMA_MODEL
            
            logger.debug(f"Found {len(models)} models")
            model_info = []
            for model in models:
                name = model.get("name", "")
                size = model.get("size", 0)
                details = model.get("details", {})
                parameter_size = details.get("parameter_size", "")
                
                param_billions, _ = self._extract_model_size(name)
                
                if param_billions == 0 and parameter_size:
                    param_match = re.search(r'(\d+\.?\d*)B', parameter_size)
                    if param_match:
                        param_billions = float(param_match.group(1))
                
                model_info.append({
                    "name": name,
                    "size": size,
                    "params": param_billions,
                    "parameter_size": parameter_size
                })
            
            model_info.sort(key=lambda x: x["params"], reverse=True)
            
            best_model = model_info[0]["name"]
            logger.info(f"Selected best model: {best_model} ({model_info[0]['params']}B parameters)")
            
            return best_model
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch Ollama models: {e}")
            logger.info(f"Falling back to default model: {config.OLLAMA_MODEL}")
            return config.OLLAMA_MODEL

    def generate_prompts(self) -> List[str]:
        logger.info("Generating prompts using Ollama")
        llm_prompt_file = Path(__file__).parent / "llm.json"
        if llm_prompt_file.exists():
            logger.debug(f"Loading LLM prompt from {llm_prompt_file}")
            with open(llm_prompt_file, 'r', encoding='utf-8') as f:
                llm_data = json.load(f)
                llm_prompt = llm_data.get("prompt", "")
        else:
            logger.warning("llm.json file not found")
            llm_prompt = ""
        
        if not llm_prompt:
            logger.info("No LLM prompt available, using fallback prompts")
            return self._fallback_prompts()
        
        payload = {
            "model": self.ollama_model,
            "prompt": llm_prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            logger.debug(f"Requesting prompts from Ollama ({self.ollama_model})")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.ollama_timeout
            )

            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("response", "")
            
            try:
                prompts = json.loads(response_text)
                
                if isinstance(prompts, dict) and "prompts" in prompts:
                    prompts = prompts["prompts"]
                elif isinstance(prompts, dict) and "items" in prompts:
                    prompts = prompts["items"]
                    
                if not isinstance(prompts, list):
                    raise ValueError("Not a list")
                    
                prompts = [str(p) for p in prompts[:self.num_prompts]]
                logger.info(f"Successfully generated {len(prompts)} prompts")
                return prompts
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse Ollama response: {e}")
                logger.info("Using fallback prompts")
                return self._fallback_prompts()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            logger.info("Using fallback prompts")
            return self._fallback_prompts()
    
    def _fallback_prompts(self) -> List[str]:
        logger.debug("Using fallback prompts")
        prompts = [
            "photorealistic, 8k, detailed",
            "digital art, vibrant colors",
            "minimalist, modern design",
            "abstract, artistic",
            "cinematic lighting, dramatic",
            "nature photography, stunning",
            "futuristic, sci-fi",
            "vintage, retro aesthetic",
            "fantasy art, magical",
            "landscape photography, epic",
            "cyberpunk, neon lights",
            "watercolor painting style",
            "3D render, hyperrealistic",
            "impressionist painting",
            "low poly art style",
            "dark moody atmosphere",
            "bright and cheerful",
            "noir style, high contrast",
            "anime style illustration",
            "geometric patterns"
        ]
        return prompts[:self.num_prompts]
    
    def generate_image(self, prompt: str) -> bytes:
        logger.info(f"Generating image for prompt: {prompt[:50]}...")
        llm_prompt_file = Path(__file__).parent / "llm.json"
        negative_prompt = ""
        if llm_prompt_file.exists():
            with open(llm_prompt_file, 'r', encoding='utf-8') as f:
                llm_data = json.load(f)
                negative_prompt = llm_data.get("negative_prompt", "")
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": self.sd_steps,
            "cfg_scale": self.cfg_scale,
            "width": self.width,
            "height": self.height,
            "sampler_name": self.sampler_name
        }
        
        try:
            logger.debug(f"Sending request to Stable Diffusion API at {self.sd_url}")
            response = requests.post(
                f"{self.sd_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=self.sd_timeout
            )

            response.raise_for_status()
            
            result = response.json()
            if "images" in result and len(result["images"]) > 0:
                image_data = base64.b64decode(result["images"][0])
                logger.info(f"Successfully generated image ({len(image_data)} bytes)")
                return image_data
            else:
                logger.error("No images in Stable Diffusion response")
                raise ValueError("No images in response")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Stable Diffusion API request failed: {e}")
            raise
    
    def upscale_image(self, image_data: bytes, scale: int = 2) -> bytes:
        logger.info(f"Upscaling image by {scale}x using {self.upscaler}")
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        payload = {
            "resize_mode": 0,
            "upscaling_resize": scale,
            "upscaler_1": self.upscaler,
            "image": image_base64
        }
        
        try:
            logger.debug("Sending upscale request to Stable Diffusion API")
            response = requests.post(
                f"{self.sd_url}/sdapi/v1/extra-single-image",
                json=payload,
                timeout=self.sd_timeout
            )
            response.raise_for_status()
            
            result = response.json()
            if "image" in result:
                upscaled_data = base64.b64decode(result["image"])
                logger.info(f"Successfully upscaled image ({len(upscaled_data)} bytes)")
                return upscaled_data
            else:
                logger.error("No image in upscale response")
                raise ValueError("No image in response")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Image upscaling failed: {e}")
            logger.warning("Returning original image without upscaling")
            return image_data
    
    def save_image(self, image_data: bytes, filename: str) -> str:
        filepath = self.output_dir / filename
        logger.debug(f"Saving image to {filepath}")
        with open(filepath, 'wb') as f:
            f.write(image_data)
        logger.info(f"Image saved: {filepath}")
        return str(filepath)
    
    def generate_wallpapers(self):
        logger.info("=" * 60)
        logger.info("Starting wallpaper generation process")
        logger.info("=" * 60)
        
        prompts = self.generate_prompts()
        
        if not prompts:
            logger.error("No prompts available, aborting")
            return

        successful = 0
        failed = 0
        
        logger.info(f"Processing {len(prompts)} prompts")
        
        for i, prompt in enumerate(prompts, 1):
            try:
                logger.info(f"[{i}/{len(prompts)}] Processing wallpaper")
                image_data = self.generate_image(prompt)
                
                if self.upscale:
                    image_data = self.upscale_image(image_data, self.upscale_factor)
                
                timestamp = int(time.time())
                filename = f"wallpaper_{timestamp}_{i}.png"
                self.save_image(image_data, filename)
                
                successful += 1
                logger.info(f"[{i}/{len(prompts)}] Completed successfully")
                
                if i < len(prompts):
                    logger.debug(f"Waiting {self.request_delay}s before next request")
                    time.sleep(self.request_delay)
                    
            except Exception as e:
                failed += 1
                logger.error(f"[{i}/{len(prompts)}] Failed to generate wallpaper: {e}")
                continue
        
        logger.info("=" * 60)
        logger.info(f"Wallpaper generation complete!")
        logger.info(f"Successful: {successful}/{len(prompts)}")
        logger.info(f"Failed: {failed}/{len(prompts)}")
        logger.info("=" * 60)

def main():
    try:
        logger.info("Wallpaper Generator starting...")
        generator = WallpaperGenerator()
        generator.generate_wallpapers()
        logger.info("Wallpaper Generator finished successfully")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

