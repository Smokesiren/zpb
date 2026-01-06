import os
import torch
import random
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm

# --- KONFIGURACJA ---
MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"
OUTPUT_DIR = "data_raw/MyGenAI/fake"  # Zapisujemy jako surowe dane FAKE
NUM_IMAGES = 2000  # Ile zdjęć wygenerować
DEVICE = "cuda"

# --- SYSTEM PROMPTÓW  ---
# Losowanie cech, żeby model nie generował ciągle tej samej twarzy
SUBJECTS = ["man", "woman", "person", "young man", "young woman", "middle-aged man", "elderly woman"]
ETHNICITIES = ["Caucasian", "African", "Asian", "Hispanic", "Middle Eastern", "Indian", "Nordic"]
EXPRESSIONS = ["neutral expression", "slight smile", "serious look", "looking at camera"]
LIGHTING = ["natural sunlight", "cinematic lighting", "studio lighting", "soft window light", "golden hour", "dramatic shadows"]
DETAILS = ["highly detailed", "8k uhd", "photorealistic", "raw photo", "dslr", "sharp focus"]

NEGATIVE_PROMPT = (
    "deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, "
    "extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, "
    "disconnected limbs, malformed hands, blurry, out of focus, long neck, surreal, cartoon, 3d, illustration"
)

def get_random_prompt():
    subj = random.choice(SUBJECTS)
    eth = random.choice(ETHNICITIES)
    expr = random.choice(EXPRESSIONS)
    light = random.choice(LIGHTING)
    
    # Konstrukcja promptu: "A photo of a [Ethnicity] [Subject], [Expression], [Lighting], [Details]"
    prompt = f"close-up portrait of a {eth} {subj}, {expr}, {light}, {', '.join(DETAILS)}"
    return prompt

def main():    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)
    
    pipe.safety_checker = None 

    for i in tqdm(range(NUM_IMAGES), desc="Generowanie"):
        prompt = get_random_prompt()
        
        image = pipe(
            prompt, 
            negative_prompt=NEGATIVE_PROMPT, 
            num_inference_steps=25,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]
        
        filename = f"gen_face_{i:05d}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        image.save(save_path)


if __name__ == "__main__":
    main()