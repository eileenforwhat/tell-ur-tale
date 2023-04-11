from typing import Dict
from typing import List
from story_builder import StoryBuilder
from illustrator.stable_diffusion import StableDiffusionIllustrator
from utils import write_illustration, CustomCharacter
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

    def run_tut(self,
                story_title: str,
                custom_characters: List[CustomCharacter],
                write_to_output_dir=None):
        story_prompts = self.story_builder.generate_story_plot(story_title, custom_characters)
        self.illustrator = self.apply_customization(custom_characters)
        story_images = self.illustrator.generate(story_prompts)
        if write_to_output_dir:
            write_illustration(story_images, output_dir=write_to_output_dir)
        return story_prompts, story_images

    def apply_customization(self, custom_characters: List[CustomCharacter]):
        self.illustrator.customize(custom_characters)
        return self.illustrator


if __name__ == '__main__':
    with open('config/test_story.yml', 'r') as config_file:
        config: dict = yaml.safe_load(config_file)
    print(f"Running {config['project_name']}...")

    title = "Little Red Riding Hood"
    custom_characters = [
        CustomCharacter(orig_name="wolf", custom_name="Aspen", custom_img_dir="sample_images/aspen")
    ]
    pipeline = TellUrTalePipeline(config)
    pipeline.run_tut(
        story_title=title,
        custom_characters=custom_characters,
        write_to_output_dir='output'
    )
