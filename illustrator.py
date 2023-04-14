import yaml
import argparse
import os
from typing import List
from typing import Dict
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from utils import write_illustration
from utils import StoryPage, CustomCharacter
from customization.textual_inversion import TextualInversionTrainer
from customization.dreambooth import DreamBoothTrainer


class Illustrator(object):
    def __init__(self, **config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.scheduler = EulerDiscreteScheduler.from_pretrained(config["model_id"], subfolder="scheduler")
        self.scheduler.set_timesteps(config["inference_steps"], self.device)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            config["model_id"], scheduler=self.scheduler, torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to(self.device)

        self.trainer = None
        if config.get("custom_type") == "dreambooth":
            self.trainer = DreamBoothTrainer(self.pipe, **config["custom_args"])
        elif config.get("custom_type") == "textual-inversion":
            self.trainer = TextualInversionTrainer(self.pipe, **config["custom_args"])

        self.prompt_template = f"{config['prefix']} %s {config['suffix']}"
        self.negative_prompt = config["negative_prompt"]

    def customize(self, custom_characters: List[CustomCharacter]):
        if self.trainer is not None:
            self.trainer.train(custom_characters)

    def generate(self, prompts: List[str]) -> List[StoryPage]:
        generated_pages = []
        for idx, prompt in enumerate(prompts):
            full_prompt = self.prompt_template % prompt
            print(full_prompt)
            image = self.pipe(full_prompt, negative_prompt=self.negative_prompt).images[0]
            page = StoryPage(page_num=idx, text=prompt, image=image)
            generated_pages.append(page)
        return generated_pages


if __name__ == "__main__":
    """
    # illustration with NO customization
    python illustrator.py --prompts_path output/jack_and_the_beanstalk/story.txt
    """
    parser = argparse.ArgumentParser()

    # required args
    parser.add_argument("--prompts_path", type=str, required=True)  # text file with story

    # prompt eng
    parser.add_argument("--prefix", type=str, required=False, default=None)  # prefix for text2image model
    parser.add_argument("--suffix", type=str, required=False, default=None)  # suffix for text2image model

    # custom args
    parser.add_argument("--orig_object", type=str, required=False, default="person")
    parser.add_argument("--custom_name", type=str, required=False, default=None)
    parser.add_argument("--custom_img_dir", type=str, required=False, default=None)
    parser.add_argument("--config_path", type=str, required=False, default="config/openjourney.yml")

    args = parser.parse_args()

    with open(args.prompts_path, 'r') as prompts_file:
        rows = [line.strip() for line in prompts_file.readlines()]
    title = rows[0]
    prompts = rows[1:]
    print("Title: ", title)
    print("Plot: \n", prompts)

    title_tag = title.lower().replace(" ", "_")
    output_dir = f"output/{title_tag}"
    os.makedirs(output_dir, exist_ok=True)

    with open(args.config_path) as config_file:
        config: Dict = yaml.safe_load(config_file)
    print(f"Running config from {args.config_path}...")
    if args.prefix is not None:
        config["illustrator"]["prefix"] = args.prefix
    if args.suffix is not None:
        config["illustrator"]["suffix"] = args.suffix

    illustrator = Illustrator(**config["illustrator"])
    if args.custom_img_dir:
        characters = [
            CustomCharacter(
                orig_object=args.orig_object,
                custom_name=args.custom_name,
                custom_img_dir=args.custom_img_dir
            )
        ]
        print("Custom characters: ", characters)
        illustrator.customize(characters)
    images = illustrator.generate(prompts)
    write_illustration(images, output_dir=output_dir)
