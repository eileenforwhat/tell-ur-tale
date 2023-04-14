from typing import List
import os
from dataclasses import dataclass
from PIL import Image


@dataclass
class StoryPage:
    page_num: int  # 0-indexed
    text: str
    image: Image


@dataclass
class CustomCharacter:
    orig_name: str = None  # name of the character in the original story
    orig_object: str = None  # used as initialization token, what type of object is this? ex. "boy"
    custom_name: str = None  # custom name of the character, ex. Simon
    custom_img_dir: str = None  # dir that contains images we want to customize the character to


def write_story_prompts(story_title, story_prompts: List[str], output_dir):
    text_output_path = os.path.join(output_dir, "story.txt")
    with open(text_output_path, "w") as f:
        f.write(story_title + "\n")
        for line in story_prompts:
            f.write(line + "\n")
    print(f"Wrote story_prompts to {text_output_path}")


def write_illustration(generated_pages: List[StoryPage], output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # write illustration png files
    for page in generated_pages:
        tag = "{0:03d}".format(page.page_num)
        output_fn = f"{tag}_{page.text.strip('.')}.png"
        page.image.save(os.path.join(output_dir, output_fn))
    print(f"Wrote illustration to {output_dir}")
