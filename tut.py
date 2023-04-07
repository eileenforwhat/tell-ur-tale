from story_builder import StoryBuilder
from illustrator.stable_diffusion import StableDiffusionIllustrator
import yaml


def create_illustrator(config):
    # instantiate type of illustrator from yaml config
    print("config:", config)
    illustrator = StableDiffusionIllustrator(**config)
    return illustrator

def create_story_builder(config):
    story_builder = StoryBuilder(**config)
    return story_builder


class TellUrTalePipeline(object):
    def __init__(self, config):
        self.config = config
        self.story_builder = create_story_builder(config["story_builder"])
        self.illustrator = create_illustrator(config["ilustrator"])

    def run_tut(self):
        # story_name, story_prompts = self.story_builder.interactive_build()
        story_name = "test_story"
        story_prompts = [
            "a momma bear walked up to a monkey.",
            "the momma bear and monkey skied down a hill.",
            "the momma bear and monkey fell down.",
        ]
        story_images = self.illustrator.generate(story_prompts)
        # return self.illustrator.write_illustration(story_images)
        return story_images



if __name__ == '__main__':
    with open('config.yml', 'r') as config_file:
        config: dict = yaml.safe_load(config_file)
    print(f"Running {config['project_name']}...")

    pipeline = TellUrTalePipeline(config)

    # uncomment when storybuilder becomes interactive
    # while True:
    #     pipeline.run_tut()
    
    pipeline.run_tut()
