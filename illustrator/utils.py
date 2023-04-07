from typing import List
import os
from dataclasses import dataclass
from PIL import Image

DEFAULT_OUTPUT_DIR = "static/output"


@dataclass
class StoryPage:
    page_num: int  # 0-indexed
    text: str
    image: Image


def write_illustration(generated_pages: List[StoryPage], output_dir=DEFAULT_OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for page in generated_pages:
        tag = "{0:03d}".format(page.page_num)
        output_fn = f"{tag}_{page.text.strip('.')}.png"
        page.image.save(os.path.join(output_dir, output_fn))
    return output_dir
