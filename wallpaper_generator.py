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
            "1girl, power from chainsaw man, masterpiece, outdoors, upper body, from side, yellow eyes, twintails, pink hair, color, streaked hair, messy hair, looking at viewer, jacked, cloudy sky, long hair, small red horns, glance, rabbit ears, storm, yellow transparent raincoat, chewing gum, street, in yellow transparent hood, masterpiece, best quality, amazing quality, 4k, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, forest, trees, close up view, portrait, ",
            "colorful, masterpiece, best quality, amazing quality, detailed, newest, 1girl, reze (chainsaw man), solo, black hair, single hair bun, short hair, (hair between eyes:0.9), flower, choker, green eyes, looking at viewer, shirt, black ribbon, on back, purple hair, white shirt, sleeveless, petals, parted lips, white flower, sleeveless, collared shirt, bare shoulders, upper body, medium hair, lily \(flower\), s4kur4da22_illu, masterpiece, best quality, amazing quality, 4k, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, close up view, portrait,",
            "(by konya karasua:0.6), (by egawa akira:0.4), (by rei sanbonzakura:0.45), (by yu hydra,0.55),masterpiece, newest, absurdres, incredibly absurdres, 1girl, solo, expressionless, white hair, short hair, hair covering eyes, closed mouth, floral print kimono, wide sleeves, thighhighs, black thighhighs, skirt, japanese clothes, long sleeves, black skirt, pleated skirt, gloves, platform footwear, hair ornament, socks, black gloves, black footwear, boots, oversized animal, giant skeleton snake, animal skull, bone, spine, ribs, animal skeleton, flower, sitting, looking at viewer, full body, rose, low light, dramatic lighting, volumetric lighting, abstract background, two-toned background, black background, red background, blue flower, dark foreground, dutch angle, dynamic composition, M0nster_ph4ntasy_illu, snnr2, spot color, muted color, limited palette, screentone, hatching (texture), h4llow3en_illu, horror \(theme\), weirdcore, grunge, s1_dram, masterpiece, best quality, amazing quality, 4k, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery,",
            "masterpiece, best quality,high quality, newest, highres,8K,HDR,absurdres, M0nster_ph4ntasy_illu, 1girl, red eyes, solo, breasts, multicolored hair, horns, flower, tail, black hair, two-tone hair, white hair, leotard, looking at viewer, earrings, jewelry, large breasts, bangs, scar, long hair, highleg leotard, covered navel, highleg, gauntlets, glowing, indoors, thighs, dragon girl, blunt bangs,depth of field, close up view, portrait, masterpiece, best quality, amazing quality, 4k, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, close up view, portrait,",
            "1girl, looking at viewer, solo, anime style, woman, perfect slim hourglass figure, narrow waist, depth of field,  scenery, dark lipstick, long eyelashes, , pose,  small breasts, wide hips, narrow waist, slim hourglass figure, hat , goth dress, knees socks,  garters, medium hair, blunt bangs, gloves, elbow gloves, fishnet gloves, made dress, black dress, long hair, breasts, blush, dress, bow, twintails, purple eyes, purple hair, sweatdrop, black dress, makeup, white bow, black nails,  gothic, gothic, black lips, maid hat, white hair bow, makeup, white bow, masterpiece, best quality, very aesthetic, absurdres, neon genesis, beautiful, aesthetic, chromatic aberration, ((detailed face)),highly detailed, masterpiece, best quality, very aesthetic, expressive eyes, perfect face,((beautiful eyes)),(((perfect female body))), shiny skin, super idol face, intricate details, (perfect lighting), highres, flush lips, perfect lips, black eyeshadow, thick thighs, thighs ass, indoor, light particle, sitting, sitting on knees, looks with arrogance, looking below, jewelry, extravagant dimond necklaces, lipstick,  evil expression, evil corruption, evil atmosphere,",
            "1girl,komeiji koishi,touhou,:d,animal ear fluff,animal ear hood,animal ears,animal hands,bell,black bra,black choker,black jacket,blush,bra,breasts,cat day,cat ears,cat lingerie,cat tail,choker,cleavage,cleavage cutout,clothing cutout,colored eyelashes,cowboy shot,drawn whiskers,fake animal ears,fake tail,fang,gloves,green eyes,grey hair,hand in pocket,hand up,heart,heart of string,hood,hood up,hooded jacket,jacket,jingle bell,long sleeves,looking at viewer,medium hair,meme attire,multicolored clothes,multicolored jacket,neck bell,open clothes,open jacket,open mouth,paw gloves,shiny skin,skin fang,small breasts,smile,solo,tail,thighs,third eye,tsurime,two-tone jacket,underwear,white jacket,",
            "makima \(chainsaw man\), best quality, ultra detailed, 1girl, solo, standing, red hair, long braided hair, golden eyes, bangs, medium breasts, white shirt, necktie, stare, smile, (evil:1.2), looking at viewer, (interview:1.3), (dark background, chains:1.3)",
            "masterpiece, best quality, amazing quality, 4k, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, 1girl, solo, red halo, long white hair, hair between eyes, floating hair, red eyes, turning head, looking at viewer, white eyelashes, raised inner eyebrows, smile, parted lips, black sundress, circle skirt, red angel wings, (floating, midair:1.2), knees up, hugging own legs, arched back, from side, dutch angle, portrait, upper body, night, starry sky, red full moon, dark, omnious, BREAK, rim light, backlit, luminous particles, shimmering feathers, subtle lens flare, cinematic lighting, depth of field, bokeh, volumetric lighting, dreamy atmosphere, close up view, portrait,",
            "masterpiece, best quality, amazing quality, 4k, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, 1girl, solo, purple hair, medium bob, choppy blunt bangs, sidelocks, hair flaps, antenna hair, messy hair, raised inner eyebrows, (round eyebrows:1.2), blue eyes, round eyes, tareme, (head tilt:1.2), facing to the viewer, looking to the side, averting eyes, smile, open mouth, wavy mouth, embarassed, full-face blush, sweatdrop, medium breasts, (cleavage:1.2), round red-framed glasses, (loose collared open white shirt:1.3), sleeves past wrists, loose tilted red bowtie, collarbone, no bra, navel, (hands framing own cheeks:1.2), fingers, shrugging, dappled sunlight, back light, rim light, see-through silhouette, standing, upper body, dutch angle, window, living room, BREAK, delicate hair, volumetric lighting, close up view, portrait,",
            "masterpiece, best quality, amazing quality, 4k, very aesthetic, high resolution, ultra-detailed, absurdres, newest, scenery, 1girl, solo, orange hair, inverted bob, big hair, hair band, shoulder-length hair, puffy hair, orange eyes, round eyes, light smile, parted lips, looking at viewer, turtleneck ribbed black tank top, off-shoulder orange dress, close-up, portrait, dutch angle, sunset, dappled sunlight, lens flare, orange background, BREAK, depth of field, volumetric lighting, close up view, portrait,",
            "(4k,8k,Ultra HD), masterpiece, best quality, ultra-detailed, very aesthetic, depth of field, best lighting, detailed illustration, detailed background, cinematic, beautiful face, ambient occlusion, raytracing, soft lighting, 8K, illustrating, CG, detailed background, cute girl, BREAK Aka-Oni, oni, (oni horns), colored skin, red skin, smooth horns, black horns, straight horns, BREAK white shirt, covered nipples, see-through silhouette, sitting, on sofa, choker, cross choker, round glasses, head tilt, arm hug, solo focus, out of frame, holding another's arm, close up view, portrait,",
            "lazypos, 1girl, depth of field, best lighting, detailed illustration, soft lighting, bloom effect, detailed background, cute girl, eyelashes, sketch, catgirl, solo, star pupils, blue eyes, twintails, maid outfit, small breast, floral background, rose (flower), petals, hands up, eyes focus, super close, from side, look at viewer, white nails, eating apple, grey hair, blurry, hair between eyes, looking to the side, holding, holding fruit, flower, holding food, long sleeves, breasts, upper body, food, fruit, close up view, portrait,",
            "1girl, yellow eyes, long hair, looking at viewer, bangs, black background, white hair, bare shoulders, blunt bangs, upper body, lips, white bra, colored eyelashes, masterpiece, light smile, finger to mouth, close up view, portrait,",
            "best quality, amazing quality, 4k, very aesthetic, high resolution, ultra-detailed, absurdres, newest, 1girl, medium black hair, black sleeveless shirt, shorts under skirt, white wristband, white shoes, kneeling, front view, (mountain:0.9), dynamic angle, crying, BREAK, depth of field, volumetric lighting, portrait, eyes focus, shocked, no pupils, white eyes, lightning",
            "sitting, 1girl, shima rin, Holding coffee, reading a book, Purple eyes, knit cap, scarf, sweater, pantyhose, winter, cold, tent, One bonfire, foreground, depth of field, Blurred periphery, BREAK, masterpiece, best quality, amazing quality, very aesthetic, newest, incredibly absurdres, ultra detailed, 8k, HDR, High quality digital art, official art, advertising style, detailed background, painting (medium), cinematic lighting, ray tracing, ambient occlusion, dynamic composition, foreshortening, close up view, portrait,",
            "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, 1girl, blue hair, kimono, open clothes, sarashi, hair ornament, horns, multiple eyes, horror (theme), undead, extra eyes, eyes on clothes, many eyes, katana, planted sword, simple background, grey background, close up view, portrait,",
            "1girl, beautiful face, perfect eyes, detailed eyes, mature female, portrait, dutch angle, dynamic pose, black hair, straight hair, blunt bang, hime cut, parted lips, red eyes, sideways glance, purple dougi, sleeveless jacket, floral print, hakama, holding katana, (slashing trail), katana, (petals), crimson full moon, pond, falling petals, red petals, spider lily, blurry foreground, volumetric lighting, shadow, dark background, indicate details, (depth of field:0.6), blurry background, red theme, close up view, portrait,",
            "1girl, solo, beautiful face, perfect eyes, detailed eyes, mature female, dutch angle, upper body, long hair, straight hair, open mouth, tongue out, v over mouth, large breasts, neon lights, alley, night, dark, indicate details, clean background, vignetting, shadow, volumetric lighting, indicate details, close up view, portrait,",
            "masterpiece, best quality, ultra-detailed, very aesthetic, depth of field, best lighting, detailed illustration, detailed background, beautiful face, beautiful eyes, soft lighting, bloom effect, detailed background, cute, 1girl, eyelashes, foreshortening, Aka-Oni, oni, (oni horns), colored skin, (red skin:1.3), smooth horns, black horns, straight horns, mature, limited palette, film grain, colorful, negative space, surreal, blending, chinese style, holding sword, holding katana, unsheathing, katana, hands on sword, impressionism, style is detailed and epic, sense of scale that shows how huge the dragon is, absolutely massive scale, from behind, night sky, facing huge snow dragon, dragon is roaring, soundwaves shaking the surroundings, fog and dust fills the pic, HDR, skeletal dragon, mystical necrotic dragon, white dragon, eastern dragon, huge dragon, horror (theme), grunge, dark, close up view, portrait,",
            "anime style scene, 1girl, monster skull covered in slime with slit pupils, the pupils following the camera, White transparent slime dripping, Orange backlight glow, Simulated slowly breathing movements slightly moves her jaws, as the is exited to hunt, Warm glowing smile drops and glowing particles emerge from her skin into the air, Her head starts turning to the viewer, The camera is zooming extreme close-up into her slit pupils, her pupils start glowing, the slit tightens even more, intensifying the pressuring gaze of her, The camera keeps zooming in turning the view into a burning firestorm shaped like the eye with dark and horrific scener, close up view, portrait,",
            "(masterpiece, best quality, highest quality:1.4), (ultra-detailed, intricate details:1.3), 8k, high resolution, (sharp focus, deep focus:1.2), (physically based rendering), cinematic landscape, award-winning photograph, professional landscape photography, flawless photograph, high-fidelity texture, expansive detail, epic vista, vibrant colors, fine art landscape photography, photojournalistic quality, (A tropical jungle clearing landscape, featuring a winding dirt path and a flock of birds circling high above, captured in the sunset, the sun is on the horizon during a crisp, cool spring morning, the weather is a swirling sandstorm in the desert, lit with bright, direct sunlight to create a hopeful and optimistic outlook, bright, promising, inspiring, uplifting atmosphere), drone flyover shot, lit by dusk light with backlit clouds, enhanced with lens distortion and translucent materials textures, close up view, portrait,",
            "lazypos, 1girl, komi shouko, solo, sitting, yokozuwari, Petting a cat, Sleeping black cat, Shoes, ankles, calves, thighs, skirts, black tights, school uniform, blazer, dynamic Angle, close up, lower body, (BREAK:-1), cinematic lighting, volumetric lighting, ambient occlusion, ray tracing, outdoors, Park, stairs, soft sunlight, Dappled sunlight, full body, sunlight, loafers, long sleeves, pantyhose, cat, jacket, black hair, brown footwear, pleated skirt, animal"
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

