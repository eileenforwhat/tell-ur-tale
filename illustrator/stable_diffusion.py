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
        self.customizer = None
        self.config = config

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
        Train custom (finetuned) model using custom_type={dreambooth, textual_inversion}
            and save to self.pipe
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
    with open('config/test_illustrator.yml', 'r') as config_file:
        config: dict = yaml.safe_load(config_file)
        
    with open (config['illustrator']['prompts_filename'], 'r') as prompts_file:
        prompts = [line.strip() for line in prompts_file.readlines()]
        
    output_folder = config["illustrator"]["output_folder"]
    illustrator = StableDiffusionIllustrator(**config["illustrator"])  # default config
    images = illustrator.generate(prompts)
    write_illustration(images, output_dir=output_folder)
