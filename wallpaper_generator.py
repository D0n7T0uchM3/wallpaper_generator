import json
import base64
import time
import re
import logging
import random
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
        
        # Get 3 random example prompts to send to LLM
        all_examples = self._get_all_example_prompts()
        example_prompts = random.sample(all_examples, min(3, len(all_examples)))
        
        # Build the full prompt with examples
        examples_text = "\n\n### 📋 EXAMPLE PROMPTS (Use these as reference for structure and style):\n\n"
        for i, example in enumerate(example_prompts, 1):
            examples_text += f"{i}. \"{example}\"\n\n"
        
        # Be explicit about the expected JSON format
        format_instruction = f"""

### 🎯 OUTPUT FORMAT
Return ONLY a valid JSON array (NOT an object with keys). The format must be exactly:
["prompt1 text here", "prompt2 text here", "prompt3 text here"]

Do NOT use this format: {{"Prompt 1": "text", "Prompt 2": "text"}}
Do NOT use this format: {{"prompts": ["text"]}}

Just return a simple JSON array of strings.

Now generate {self.num_prompts} new unique prompts following the same structure and quality as the examples above."""
        
        full_prompt = llm_prompt + examples_text + format_instruction
        
        payload = {
            "model": self.ollama_model,
            "prompt": full_prompt,
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
                logger.debug(f"Parsed LLM response type: {type(prompts)}")
                
                # Handle different response formats
                if isinstance(prompts, dict):
                    # Check for standard keys first
                    if "prompts" in prompts:
                        prompts = prompts["prompts"]
                    elif "items" in prompts:
                        prompts = prompts["items"]
                    else:
                        # Handle keys like "Prompt 1", "prompt1", "Prompt 2", etc.
                        # Extract all values from the dict that look like prompts
                        prompt_values = []
                        for key in sorted(prompts.keys()):
                            # Accept keys that contain "prompt" (case-insensitive)
                            if "prompt" in key.lower():
                                value = prompts[key]
                                if isinstance(value, str) and len(value) > 10:
                                    prompt_values.append(value)
                        
                        if prompt_values:
                            prompts = prompt_values
                            logger.info(f"Extracted {len(prompts)} prompts from dict keys")
                        else:
                            logger.warning("No valid prompt keys found in dict response")
                            raise ValueError("No valid prompts in dict")
                    
                if not isinstance(prompts, list):
                    raise ValueError("Not a list after parsing")
                    
                prompts = [str(p).strip() for p in prompts[:self.num_prompts]]
                logger.info(f"Successfully generated {len(prompts)} prompts")
                
                # Log the first prompt as a sample
                if prompts:
                    logger.debug(f"Sample prompt: {prompts[0][:100]}...")
                
                return prompts
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse Ollama response: {e}")
                logger.debug(f"Raw response text: {response_text[:500]}...")
                logger.info("Using fallback prompts")
                return self._fallback_prompts()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            logger.info("Using fallback prompts")
            return self._fallback_prompts()
    
    def _get_all_example_prompts(self) -> List[str]:
        """Get all curated example prompts - used both for LLM reference and as fallback."""
        return [
            "1girl, fox tail, white hair, long hair, animal ears, masterpiece, yellow eyes, parted lips, looking at viewer, jacket, outdoors, cloud, blue sky, coffee, holding, wind, upper body, smile, portrait, breath, snow, snowing, cloud",
            "1girl, long hair, closed eyes, barefoot, breasts, black leotard, sitting, headgear, white hair, bangs, toes, covered navel, highleg leotard, feet, mecha musume, machinery, indoors, cable, android, masterpiece, very long hair, backlighting",
            "1girl, chisa \\(wuthering waves\\), black skirt, medium breasts, tacet mark \\(wuthering waves\\), white sailor collar, long sleeves, red eyes, arm cutout, solo, mole under eye, cowboy shot, red ribbon, from side, red neckerchief, long hair, black shirt, character name, masterpiece, black hair",
            "1girl, masterpiece, sameko saba, animal ears, glowing blue eyes, crazy eyes, bloodshot eyes, blonde hair, crazy smile, fangs, fish tail, tongue out, turning head, looking at viewer, head tilt, sailor dress, blue neckerchief, sailor collar, v arms, leaning forward, twisted torso, arched back, legs apart, from above, dutch angle, foreshortening, multicolored background, backlighting, dark, rim light, lifebuoy hair ornament, abstract background",
            "1girl, jiangshi, long hair, solo, open mouth, blush, hat, black hair, braid, blue skin, drooling, ofuda, detached sleeves, fingernails, blue eyes, zombie pose, looking at viewer, colored skin, saliva, qing guanmao, chinese clothes, night, wide sleeves, dress, outstretched arms, outdoors, very long hair, pink nails, long fingernails, bare shoulders, red dress, hair between eyes, large breasts, mouth drool, red headwear, single braid, sharp fingernails, no panties, solo, talisman, braided ponytail, one eye covered, bangs, hair ribbon, nail polish, long sleeves, dark, masterpiece, moon, backlighting",
            "1girl, yumemizuki mizuki, masterpiece, black choker, arm up, purple bow, open mouth, frilled sleeves, frilled hairband, blue hair, pointy ears, solo, neck bell, detached sleeves, sleeveless pink kimono",
            "1girl, arm strap, belt, belt pouch, black bikini, blonde hair, outdoors, brown belt, cowboy shot, fighting stance, gauntlets, goggles on head, long hair, mechanical arms, navel, orange eyes, pouch, serious, simple background, solo, twintails, v-shaped eyebrows, cloudy sky, pants, ahoge, sharp teeth, grin, dirty",
            "1girl, animal ears, animal ear fluff, eyes streaked hair, multicolored hair, pink hair, bangs, outdoors, looking at viewer, bush, forest, white hair, tree, slit pupils, foliage, yellow eyes, blush, hair between eyes, leaf, plant, masterpiece, smile, bow \\(weapon\\), solo, tongue, arrow \\(projectile\\), holding",
            "1girl, oni, pointy ears, pale skin, white hair, blue hair, blue eyes, slit pupils, multicolored hair, long hair, split-color hair, looking at viewer, simple background, neck tattoo, large breasts, bangs, heterochromia, white eyes, cleavage, scar on face, upper body, two-tone hair, hair between eyes, gray background, black horns, masterpiece, scar on nose, crossed arms, nipples, breast rest, parted lips",
            "1girl, power \\(chainsaw man\\), outdoors, upper body, from side, yellow eyes, twintails, pink hair, streaked hair, messy hair, looking at viewer, cloudy sky, long hair, small red horns, yellow transparent raincoat, chewing gum, street, masterpiece, portrait",
            "1girl, reze \\(chainsaw man\\), solo, black hair, single hair bun, short hair, hair between eyes, flower, choker, green eyes, looking at viewer, shirt, black ribbon, on back, white shirt, sleeveless, petals, parted lips, white flower, collared shirt, bare shoulders, upper body, lily \\(flower\\), masterpiece, portrait",
            "1girl, makima \\(chainsaw man\\), solo, standing, red hair, long braided hair, golden eyes, bangs, medium breasts, white shirt, necktie, stare, smile, (evil:1.2), looking at viewer, dark background, chains, masterpiece",
            "1girl, komeiji koishi, animal ear hood, animal ears, bell, black choker, black jacket, blush, breasts, cat ears, cat tail, choker, cleavage, cleavage cutout, cowboy shot, fake animal ears, fake tail, fang, gloves, green eyes, grey hair, heart, hood up, hooded jacket, jacket, long sleeves, looking at viewer, medium hair, open jacket, open mouth, paw gloves, smile, solo, tail, third eye, two-tone jacket, white jacket, masterpiece",
            "1girl, solo, expressionless, white hair, short hair, hair covering eyes, closed mouth, floral print kimono, wide sleeves, thighhighs, black thighhighs, skirt, japanese clothes, long sleeves, black skirt, pleated skirt, gloves, platform footwear, black gloves, black footwear, boots, oversized animal, giant skeleton snake, animal skull, bone, flower, sitting, looking at viewer, full body, rose, low light, dramatic lighting, volumetric lighting, abstract background, two-toned background, black background, red background, dutch angle, dynamic composition, limited palette, horror \\(theme\\), masterpiece",
            "1girl, solo, red eyes, multicolored hair, horns, flower, tail, black hair, two-tone hair, white hair, leotard, looking at viewer, earrings, jewelry, large breasts, bangs, scar, long hair, highleg leotard, covered navel, gauntlets, glowing, indoors, thighs, dragon girl, blunt bangs, depth of field, portrait, masterpiece",
            "1girl, blue hair, kimono, open clothes, sarashi, hair ornament, horns, multiple eyes, horror \\(theme\\), undead, extra eyes, eyes on clothes, many eyes, katana, planted sword, simple background, grey background, portrait, masterpiece",
            "1girl, oni, colored skin, (red skin:1.3), smooth horns, black horns, straight horns, mature, limited palette, film grain, colorful, negative space, surreal, chinese style, holding sword, holding katana, katana, hands on sword, from behind, night sky, facing huge snow dragon, dragon is roaring, fog, skeletal dragon, mystical dragon, white dragon, eastern dragon, huge dragon, horror \\(theme\\), dark, portrait, masterpiece",
            "1girl, komi shouko, solo, sitting, yokozuwari, petting a cat, sleeping black cat, shoes, ankles, calves, thighs, skirts, black tights, school uniform, blazer, dynamic angle, close up, lower body, cinematic lighting, volumetric lighting, outdoors, park, stairs, soft sunlight, dappled sunlight, full body, sunlight, loafers, long sleeves, pantyhose, cat, jacket, black hair, brown footwear, pleated skirt, animal, masterpiece",
            "1girl, shima rin, sitting, holding coffee, reading a book, purple eyes, knit cap, scarf, sweater, pantyhose, winter, cold, tent, one bonfire, foreground, depth of field, blurred periphery, cinematic lighting, dynamic composition, foreshortening, masterpiece",
            "1girl, yellow eyes, long hair, looking at viewer, bangs, black background, white hair, bare shoulders, blunt bangs, upper body, lips, white bra, colored eyelashes, masterpiece, light smile, finger to mouth, portrait",
            "1girl, solo, black hair, straight hair, blunt bang, hime cut, parted lips, red eyes, sideways glance, purple dougi, sleeveless jacket, floral print, hakama, holding katana, katana, petals, crimson full moon, pond, falling petals, red petals, spider lily, blurry foreground, volumetric lighting, shadow, dark background, depth of field, blurry background, red theme, portrait, masterpiece",
            "1girl, solo, long hair, straight hair, open mouth, tongue out, v over mouth, large breasts, neon lights, alley, night, dark, clean background, vignetting, shadow, volumetric lighting, upper body, portrait, masterpiece",
            "1girl, catgirl, solo, star pupils, blue eyes, twintails, maid outfit, small breasts, floral background, rose \\(flower\\), petals, hands up, eyes focus, from side, looking at viewer, white nails, eating apple, grey hair, blurry, hair between eyes, looking to the side, holding, holding fruit, flower, holding food, long sleeves, upper body, food, fruit, depth of field, soft lighting, bloom effect, masterpiece"
        ]
    
    def _fallback_prompts(self) -> List[str]:
        """Return random fallback prompts when LLM is unavailable."""
        logger.debug("Using fallback prompts")
        all_prompts = self._get_all_example_prompts()
        num_to_select = min(self.num_prompts, len(all_prompts))
        selected_prompts = random.sample(all_prompts, num_to_select)
        logger.info(f"Randomly selected {len(selected_prompts)} fallback prompts from {len(all_prompts)} available")
        return selected_prompts
    
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
        
        # Log the full request details
        logger.info("=" * 80)
        logger.info("STABLE DIFFUSION REQUEST:")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Negative Prompt: {negative_prompt}")
        logger.info(f"Steps: {self.sd_steps}, CFG Scale: {self.cfg_scale}, Sampler: {self.sampler_name}")
        logger.info(f"Resolution: {self.width}x{self.height}")
        logger.info("=" * 80)
        
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

