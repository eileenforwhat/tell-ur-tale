from typing import List
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from illustrator.utils import write_illustration

MODEL_ID = "stabilityai/stable-diffusion-2"


class StableDiffusionIllustrator(object):
    def __init__(self, model_id=None):
        # create sd pipeline
        model_id = model_id or MODEL_ID
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=self.scheduler, torch_dtype=torch.float16
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = self.pipe.to(self.device)

    def generate(self, prompts: List[str]) -> List[str]:
        generated_images = []
        for idx, prompt in enumerate(prompts):
            image = self.pipe(prompt).images[0]
            generated_images.append([idx, prompt, image])
        return generated_images


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
    illustrator = StableDiffusionIllustrator()
    images = illustrator.generate(prompts)
    write_illustration(images)
