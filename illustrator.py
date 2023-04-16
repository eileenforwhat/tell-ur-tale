import dataclasses

import yaml
import argparse
import os
from typing import List
from typing import Dict
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from utils import write_story_pages, parse_line_to_story_page
from utils import StoryPage, CustomCharacter
from customization.textual_inversion import TextualInversionTrainer
from customization.dreambooth import DreamBoothTrainer


class Illustrator(object):
    def __init__(self, device="cuda:0", **config):
        self.device = device if torch.cuda.is_available() else "cpu"

        self.scheduler = EulerDiscreteScheduler.from_pretrained(config["model_id"], subfolder="scheduler")
        self.scheduler.set_timesteps(config["inference_steps"], self.device)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            config["model_id"], scheduler=self.scheduler, torch_dtype=torch.float32
        )

        self.trainer = None
        if config.get("custom_type") == "dreambooth":
            self.trainer = DreamBoothTrainer(self.pipe, **config["custom_args"], device=self.device)
        elif config.get("custom_type") == "textual-inversion":
            self.trainer = TextualInversionTrainer(self.pipe, **config["custom_args"], device=self.device)

        self.prompt_template = f"{config['prefix']}, %s, {config['suffix']}"
        self.negative_prompt = config["negative_prompt"]

    def customize(self, custom_characters: List[CustomCharacter]):
        if self.trainer is None:
            print("No customization applied since trainer not initialized.")
            return

        self.trainer.train(custom_characters)

    def generate(self, pages: List[StoryPage], custom_characters: List[CustomCharacter]=None) -> List[StoryPage]:
        illustrated_pages = []
        self.pipe = self.pipe.to(self.device, torch_dtype=torch.float32)
        for page in pages:
            prompt = page.prompt
            if self.trainer is not None and custom_characters is not None:
                for character in custom_characters:
                    placeholder_custom_token = self.trainer.get_placeholder_token(character.custom_name)
                    prompt = prompt.replace(character.orig_name, placeholder_custom_token)
            full_prompt = self.prompt_template % prompt
            print(full_prompt)
            image = self.pipe(full_prompt, negative_prompt=self.negative_prompt).images[0]
            illustrated_pages.append(dataclasses.replace(page, image=image))
        return illustrated_pages


if __name__ == "__main__":
    """
    # use trained customization model to generate images
    python illustrator.py --orig_name Goldilocks --prompts_path output/goldilocks_and_the_three_bears/story.txt \
        --prefix "mdjrny-v4 kids story illustration" \
        --suffix "drawn by Rebecca Sugar, bright engaging children's illustration, digital painting, big eyes, beautiful shading, beautiful colors, amazon kdp, happy, interesting, 2D" \
        --model_id "runs/dreambooth-model" --device "cuda:1"
        
    # illustration with NO customization
    python illustrator.py --orig_name Jack --prompts_path output/jack_and_the_beanstalk/story.txt \
        --prefix "mdjrny-v4 kids illustration showcasing the story of 'Jack and the Beanstalk'" \
        --suffix "drawn by Rebecca Sugar, bright engaging children's illustration, digital painting, big eyes, beautiful shading, beautiful colors, amazon kdp, happy, interesting, 2D" \
        --config_path "config/sd2.yml"
        
    python illustrator.py --orig_name Goldilocks --prompts_path output/goldilocks_and_the_three_bears/story.txt \
        --prefix "mdjrny-v4 kids story illustration" \
        --suffix "drawn by Rebecca Sugar, bright engaging children's illustration, digital painting, big eyes, beautiful shading, beautiful colors, amazon kdp, happy, interesting, 2D" \

    python illustrator.py --orig_name "Little Red Riding Hood" --prompts_path output/little_red_riding_hood/story.txt \
        --prefix "mdjrny-v4 kids story illustration" \
        --suffix "drawn by Rebecca Sugar, bright engaging children's illustration, digital painting, big eyes, beautiful shading, beautiful colors, amazon kdp, happy, interesting, 2D" \
        
        --suffix "rich colors, highly detailed, sharp focus, cinematic lighting, by Atey Ghailan and Beatrix Potter"
        --prefix "Rebecca Sugar style kids book illustration showcasing the story Jack and the Beanstalk" \
    
    python illustrator.py --orig_name Goldilocks --prompts_path output/goldilocks_and_the_three_bears/story.txt \
        --custom_name Simon --custom_img_dir sample_images/simon_512 \
        --prefix "mdjrny-v4 kids story illustration" \
        --suffix "drawn by Rebecca Sugar, bright engaging children's illustration, digital painting, big eyes, beautiful shading, beautiful colors, amazon kdp, happy, interesting, 2D" \
        --config_path config/dreambooth-sd1-5.yml --device cuda:1
        
    """
    parser = argparse.ArgumentParser()

    # required args
    parser.add_argument("--orig_name", type=str, required=True)  # choose a main character from story
    parser.add_argument("--prompts_path", type=str, required=True)  # text file with story

    # prompt eng
    parser.add_argument("--prefix", type=str, required=False, default=None)  # prefix for text2image model
    parser.add_argument("--suffix", type=str, required=False, default=None)  # suffix for text2image model

    parser.add_argument("--device", type=str, required=False, default="cuda:0")  # suffix for text2image model

    # custom args
    parser.add_argument("--orig_object", type=str, required=False, default="boy")
    parser.add_argument("--custom_name", type=str, required=False, default=None)
    parser.add_argument("--custom_img_dir", type=str, required=False, default=None)
    parser.add_argument("--model_id", type=str, required=False, default=None)
    parser.add_argument("--config_path", type=str, required=False, default="config/openjourney.yml")

    args = parser.parse_args()

    # read from text file into story pages
    with open(args.prompts_path, 'r') as prompts_file:
        lines = [line.strip() for line in prompts_file.readlines()]
    title = lines[0]
    pages = [parse_line_to_story_page(line) for line in lines[1:]]
    print(pages)

    # output dir
    title_tag = title.lower().replace(" ", "_")
    output_dir = f"output/{title_tag}"
    os.makedirs(output_dir, exist_ok=True)

    with open(args.config_path) as config_file:
        config: Dict = yaml.safe_load(config_file)
    print(f"Running config from {args.config_path}...")
    # overwrite prefix and suffix from default if given
    if args.prefix is not None:
        config["illustrator"]["prefix"] = args.prefix
    if args.suffix is not None:
        config["illustrator"]["suffix"] = args.suffix
    # overwrite model_id if given
    if args.model_id is not None:
        config["illustrator"]["model_id"] = args.model_id

    illustrator = Illustrator(**config["illustrator"], device=args.device)
    characters = None
    if args.custom_img_dir:
        characters = [
            CustomCharacter(
                orig_name=args.orig_name,
                orig_object=args.orig_object,
                custom_name=args.custom_name,
                custom_img_dir=args.custom_img_dir
            )
        ]
        print("Custom characters: ", characters)
        illustrator.customize(characters)
    images = illustrator.generate(pages, custom_characters=characters)
    write_story_pages(title, images, output_dir=output_dir)
