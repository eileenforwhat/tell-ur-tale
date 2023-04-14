from typing import Dict
import os
import argparse
from typing import List
from story_builder import StoryBuilder
from illustrator import Illustrator
from utils import write_illustration, write_story_prompts, CustomCharacter
import yaml

DEFAULT_OUTPUT_DIR = "output"


class TellUrTalePipeline(object):
    def __init__(self, config: Dict):
        self.story_builder = StoryBuilder(**config["story_builder"])
        self.illustrator = Illustrator(**config["illustrator"]) if config.get("illustrator") else None

    def run_tut(
        self, story_title: str, characters: List[CustomCharacter], output_dir=DEFAULT_OUTPUT_DIR
    ):
        """ Call story_builder and illustrator, write text and illustrations to output.
        """
        # call chatgpt to generate story plot
        story_prompts = self.story_builder.generate_story_plot(story_title, characters)
        write_story_prompts(story_title, story_prompts, output_dir=output_dir)

        # call illustrator to generate images (with customization)
        if self.illustrator:
            self.illustrator.customize(characters)
            story_images = self.illustrator.generate(story_prompts)
            write_illustration(story_images, output_dir=output_dir)


if __name__ == '__main__':
    """
    # run without illustrator, story saved to output/jack_and_the_beanstalk/story.txt
    python tut.py --title "Jack and the Beanstalk" --orig_name Jack --config_path config/story_only.yml
    
    # run with baseline sd2 illustrator, story saved to output/jack_and_the_beanstalk/story.txt
    python tut.py --title "Jack and the Beanstalk" --orig_name Jack --config_path config/sd2.yml
    """
    parser = argparse.ArgumentParser()

    # required args
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--orig_name", type=str, required=True)  # choose a main character from story

    # custom args
    parser.add_argument("--orig_object", type=str, required=False, default="person")
    parser.add_argument("--custom_name", type=str, required=False, default=None)
    parser.add_argument("--custom_img_dir", type=str, required=False, default=None)
    parser.add_argument("--config_path", type=str, required=False, default="config/story_only.yml")

    args = parser.parse_args()

    characters = [CustomCharacter(orig_name=args.orig_name)]
    if args.custom_img_dir:
        characters = [
            CustomCharacter(
                orig_name=args.orig_name,
                orig_object=args.orig_object,
                custom_name=args.custom_name,
                custom_img_dir=args.custom_img_dir
            )
        ]
    print("Characters: ", characters)

    title_tag = args.title.lower().replace(" ", "_")
    output_dir = f"output/{title_tag}"
    os.makedirs(output_dir, exist_ok=True)

    with open(args.config_path) as config_file:
        config: Dict = yaml.safe_load(config_file)
    print(f"Running config from {args.config_path}...")
    pipeline = TellUrTalePipeline(config)
    pipeline.run_tut(story_title=args.title, characters=characters, output_dir=output_dir)
