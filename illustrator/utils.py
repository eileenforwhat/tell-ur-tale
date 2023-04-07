import os

DEFAULT_OUTPUT_DIR = "output"


def write_illustration(generated_images, output_dir=DEFAULT_OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for idx, prompt, image in generated_images:
        tag = "{0:03d}".format(idx)
        output_fn = f"{tag}_{prompt.strip('.')}.png"
        image.save(os.path.join(output_dir, output_fn))
