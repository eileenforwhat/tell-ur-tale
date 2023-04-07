import os
from dataclasses import dataclass
from PIL import Image

DEFAULT_OUTPUT_DIR = "output"


@dataclass
class StoryPage:
    page_num: int  # 0-indexed
    text: str
    image: Image


def write_illustration(generated_pages, output_dir=DEFAULT_OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for idx, (prompt, image) in enumerate(generated_pages):
        tag = "{0:03d}".format(idx)
        output_fn = f"{tag}_{prompt.strip('.')}.png"
        image.save(os.path.join(output_dir, output_fn))
