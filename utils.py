from typing import List

import os
import re
from dataclasses import dataclass
from PIL import Image


@dataclass
class StoryPage:
    page_num: int  # 0-indexed
    text: str  # plot
    prompt: str  # prompt for text2image
    image: Image = None


@dataclass
class CustomCharacter:
    orig_name: str  # name of the character in the original story
    orig_object: str = None  # used as initialization token, what type of object is this? ex. "boy"
    custom_name: str = None  # custom name of the character, ex. Simon
    custom_img_dir: str = None  # dir that contains images we want to customize the character to


def parse_line_to_story_page(line: str) -> StoryPage:
    # match "1. <text> [<prompt>]"
    m = re.match(r"([0-9]+)\. (.+) \[(.+)\]", line)
    if not m:
        import ipdb;ipdb.set_trace()
    page_num, text, prompt = m.group(1, 2, 3)
    page = StoryPage(
        page_num=int(page_num),
        text=text,
        prompt=prompt
    )
    return page


def write_story_pages(title: str, generated_pages: List[StoryPage], output_dir):
    """
    Out file format:
        <title>
        <page_num>. <text> [<prompt>]
        ...
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    text_path = os.path.join(output_dir, "story.txt")
    with open(text_path, "w") as f:
        f.write(title + "\n")
        for page in generated_pages:
            text_line = f"{page.page_num}. {page.text} [{page.prompt}]"
            f.write(text_line + "\n")
            if page.image:
                tag = "{0:03d}".format(page.page_num)
                output_fn = f"{tag}_{page.prompt.strip('.')}.png"
                img_path = os.path.join(output_dir, output_fn)
                page.image.save(img_path)
                print(f"Wrote illustration to {img_path}")
    print(f"Wrote story_prompts to {text_path}")
