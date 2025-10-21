import json
import base64
import time
import re
from pathlib import Path
from typing import List, Tuple
import requests
import config


class WallpaperGenerator:
    def __init__(self):
        self.ollama_url = config.OLLAMA_URL
        self.sd_url = config.STABLE_DIFFUSION_URL
        self.output_dir = Path(config.OUTPUT_DIR).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if config.OLLAMA_AUTO_SELECT:
            self.ollama_model = self._select_best_ollama_model()
        else:
            self.ollama_model = config.OLLAMA_MODEL
        
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
    
    def _extract_model_size(self, model_name: str) -> Tuple[float, str]:
        match = re.search(r':(\d+\.?\d*)b', model_name.lower())
        if match:
            size = float(match.group(1))
            return (size, model_name)
        
        return (0, model_name)
    
    def _select_best_ollama_model(self) -> str:
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            data = response.json()
            models = data.get("models", [])
            
            if not models:
                return config.OLLAMA_MODEL
            
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
            
            return best_model
            
        except requests.exceptions.RequestException as e:
            return config.OLLAMA_MODEL

    def generate_prompts(self) -> List[str]:
        llm_prompt_file = Path(__file__).parent / "llm.json"
        if llm_prompt_file.exists():
            with open(llm_prompt_file, 'r', encoding='utf-8') as f:
                llm_data = json.load(f)
                llm_prompt = llm_data.get("prompt", "")
        else:
            llm_prompt = ""
        
        if not llm_prompt:
            return self._fallback_prompts()
        
        payload = {
            "model": self.ollama_model,
            "prompt": llm_prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
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
                return prompts
                
            except (json.JSONDecodeError, ValueError) as e:
                return self._fallback_prompts()
                
        except requests.exceptions.RequestException as e:
            return self._fallback_prompts()
    
    def _fallback_prompts(self) -> List[str]:
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
        llm_prompt_file = Path(__file__).parent / "llm.json"
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
            response = requests.post(
                f"{self.sd_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=self.sd_timeout
            )

            response.raise_for_status()
            
            result = response.json()
            if "images" in result and len(result["images"]) > 0:
                image_data = base64.b64decode(result["images"][0])
                return image_data
            else:
                raise ValueError("No images in response")
                
        except requests.exceptions.RequestException as e:
            raise
    
    def upscale_image(self, image_data: bytes, scale: int = 2) -> bytes:
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        payload = {
            "resize_mode": 0,
            "upscaling_resize": scale,
            "upscaler_1": self.upscaler,
            "image": image_base64
        }
        
        try:
            response = requests.post(
                f"{self.sd_url}/sdapi/v1/extra-single-image",
                json=payload,
                timeout=self.sd_timeout
            )
            response.raise_for_status()
            
            result = response.json()
            if "image" in result:
                upscaled_data = base64.b64decode(result["image"])
                return upscaled_data
            else:
                raise ValueError("No image in response")
                
        except requests.exceptions.RequestException as e:
            return image_data
    
    def save_image(self, image_data: bytes, filename: str) -> str:
        filepath = self.output_dir / filename
        with open(filepath, 'wb') as f:
            f.write(image_data)
        return str(filepath)
    
    def generate_wallpapers(self):
        prompts = self.generate_prompts()

        successful = 0
        
        for i, prompt in enumerate(prompts, 1):
            try:
                image_data = self.generate_image(prompt)
                
                if self.upscale:
                    image_data = self.upscale_image(image_data, self.upscale_factor)
                
                successful += 1
                
                if i < len(prompts):
                    time.sleep(self.request_delay)
                    
            except Exception as e:
                continue

def main():
    generator = WallpaperGenerator()
    generator.generate_wallpapers()


if __name__ == "__main__":
    main()

