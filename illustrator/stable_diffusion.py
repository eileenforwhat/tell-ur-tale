from illustrator_interface import Illustrator
from diffusers import StableDiffusionPipeline


class StableDiffusionIllustrator(Illustrator):
    def __init__(self):
        stable_diffusion_pipeline = StableDiffusionPipeline(...)

    def generate_image(self, prompt, customizer=None):
        pass