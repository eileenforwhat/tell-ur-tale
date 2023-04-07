from typing import Dict
from story_builder import StoryBuilder
from illustrator.stable_diffusion import StableDiffusionIllustrator
import yaml


def create_illustrator(config):
    # instantiate type of illustrator from yaml config
    print("config:", config)
    illustrator = StableDiffusionIllustrator(**config)
    return illustrator


def create_story_builder(config):
    print("config:", config)
    story_builder = StoryBuilder(**config)
    return story_builder


class TellUrTalePipeline(object):
    def __init__(self, config):
        self.config = config
        self.story_builder = create_story_builder(config["story_builder"])
        self.illustrator = create_illustrator(config["illustrator"])

    def run_tut(self, story_title: str, customization: Dict):
        story_prompts = self.story_builder.generate_story_plot(story_title, customization)
        story_images = self.illustrator.generate(story_prompts)
        return story_prompts, story_images


if __name__ == '__main__':
    with open('config/test_story.yml', 'r') as config_file:
        config: dict = yaml.safe_load(config_file)
    print(f"Running {config['project_name']}...")

    title = "Little Red Riding Hood"
    character_customization = {
        "wolf": "Simon"
    }
    pipeline = TellUrTalePipeline(config)
    pipeline.run_tut(
        story_title=title,
        customization=character_customization
    )
