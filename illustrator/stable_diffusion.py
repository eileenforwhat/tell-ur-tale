from typing import List
from typing import Dict
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from utils import write_illustration
from utils import StoryPage, CustomCharacter
from customization.textual_inversion import TextualInversionTrainer
from customization.dreambooth import DreamBoothTrainer

MODEL_ID = "stabilityai/stable-diffusion-2"


class StableDiffusionIllustrator(object):
    def __init__(self, **config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = config.get("model_id", MODEL_ID)
        inference_steps = config.get("inference_steps", 5)
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.scheduler.set_timesteps(inference_steps, self.device)
        self.config = config

        self.customizer = None
        self.pipe = None
        if config["custom_type"] == "dreambooth":
            self.customizer = DreamBoothTrainer(**config["custom_args"])
        elif config["custom_type"] == "textual_inversion":
            self.customizer = TextualInversionTrainer(**config["custom_args"])
        elif config["custom_type"] == "baseline":
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, scheduler=self.scheduler, torch_dtype=torch.float16
            )
            self.pipe = self.pipe.to(self.device)
        else:
            raise ValueError(f"custom_type: {config['custom_type']} is not supported")

    def customize(self, custom_characters: List[CustomCharacter]):
        """
        Train custom (finetuned) model using custom_type={dreambooth, textual_inversion} and save to self.pipe
        """
        if self.customizer is not None:
            self.pipe = self.customizer.train(custom_characters)
        self.pipe = self.pipe.to(self.device)

    def generate(self, prompts: List[str], prompt_template=None) -> List[StoryPage]:
        generated_pages = []
        custom_args = self.config['custom_args']
        if prompt_template is None:
            prompt_template = custom_args['prefix'] + "{}" + custom_args['suffix']
        negative_prompt = custom_args.get('negative_prompt', "")
        for idx, prompt in enumerate(prompts):
            image = self.pipe(prompt_template.format(prompt), negative_prompt=negative_prompt).images[0]
            page = StoryPage(page_num=idx, text=prompt, image=image)
            generated_pages.append(page)
        return generated_pages


if __name__ == "__main__":
    """
    python -m illustrator.stable_diffusion
    """
    import yaml
    with open('config/openjourney.yml', 'r') as config_file:
        config: dict = yaml.safe_load(config_file)
        
    name = "test_story"
    prompts = [
        "a momma bear walked up to a monkey.",
        "the momma bear and monkey skied down a hill.",
        "the momma bear and monkey fell down.",
    ]
    # prompt_template = "Photo of a {}. style of laurie greasley, studio ghibli, akira toriyama, james gilleard, genshin impact, trending pixiv fanbox, acrylic palette knife, 4k, vibrant colors, devinart, trending on artstation, low details, soft fur, medieval"
    # prompt_template = "Painting of {}. Style of sargent and rhads and leyendecker and greg hildebrandt evening sky, low thunder clouds foothpath with trees at indian summer with dreamy sky in background, colours green, red, blue black and white, acuarela"
    # prompt_template = "Pen and ink, illustrated by herge. {}. line art, cartoon by mitsuhiro arita"
    # prompt_template = "An anime wallpaper of {}, landscape, studio ghibli, 4 k quality, 8 k quality, high definition, digital art, matte painting, realistic painting, artstation, anime art, fanart, illustration, pixiv, danbooru, painttool sai, procreate, aesthetic"
    # prompt_template = "Photo of a {}. with symmetric face. style of laurie greasley, studio ghibli, akira toriyama, james gilleard, genshin impact, trending pixiv fanbox, acrylic palette knife, 4k, vibrant colors, devinart, trending on artstation, low details, soft fur, medieval"
    # prompt_template = "Painting of {}.  Style of sargent and rhads and leyendecker and greg hildebrandt evening sky, low thunder clouds foothpath with trees at indian summer with dreamy sky in background, colours green, red, blue black and white, acuarela"
    prompt_template = "Pixar style {}, 4k, 8k, unreal engine, octane render photorealistic by cosmicwonder, hdr, photography by cosmicwonder, high definition, symmetrical face, volumetric lighting, dusty haze, photo, octane render, 24mm, 4k, 24mm, DSLR, high quality, 60 fps, ultra realistic"
    prompt_template = config["illustrator"]['custom_args']["prefix"] + "{}" + config["illustrator"]['custom_args']["suffix"]
    illustrator = StableDiffusionIllustrator(**config["illustrator"])  # default config
    images = illustrator.generate(prompts)
    write_illustration(images, output_dir="output/"+prompt_template.format("__")[:min(len(prompt_template), 40)])
