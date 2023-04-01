import os

DEFAULT_OUTPUT_DIR = "output"


def write_illustration(self, output_dir=DEFAULT_OUTPUT_DIR):
    for idx, image in self.idx2images.items():
        tag = "{0:03d}".format(idx)
        output_fn = f"{self.name}_{tag}"
        image.save(os.path.join(output_dir, output_fn))
