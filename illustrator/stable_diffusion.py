from typing import List
from typing import Dict
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from utils import write_illustration
from utils import StoryPage, CustomCharacter

MODEL_ID = "stabilityai/stable-diffusion-2"


class StableDiffusionIllustrator(object):
    def __init__(self, **config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_id = config.get("model_id", MODEL_ID)
        inference_steps = config.get("inference_steps", 5)
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.scheduler.set_timesteps(inference_steps, self.device)

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=self.scheduler, torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to(self.device)

    def customize(self, custom_characters: List[CustomCharacter], custom_type: str = "dreambooth"):
        """
        Train custom (finetuned) model using custom_type={dreambooth, textual_inversion}
            and save to self.pipe
        """
        if custom_type == "dreambooth":
            pass
        elif custom_type == "textual_inversion":
            pass
        else:
            raise ValueError(f"custom_type: {custom_type} is not supported")

    def generate(self, prompts: List[str]) -> List[StoryPage]:
        generated_pages = []
        for idx, prompt in enumerate(prompts):
            image = self.pipe(prompt).images[0]
            page = StoryPage(page_num=idx, text=prompt, image=image)
            generated_pages.append(page)
        return generated_pages


if __name__ == "__main__":
    """
    python -m illustrator.stable_diffusion
    """
    name = "test_story"
    prompts = [
        "a momma bear walked up to a monkey.",
        "the momma bear and monkey skied down a hill.",
        "the momma bear and monkey fell down.",
    ]
    illustrator = StableDiffusionIllustrator()  # default config
    images = illustrator.generate(prompts)
    write_illustration(images)
