from typing import List

import argparse
import itertools
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from utils import CustomCharacter

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.16.0.dev0")

logger = get_logger(__name__)


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class DreamBoothTrainer(object):
    def __init__(self, init_pipe_from, **args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.accelerator = Accelerator(
            mixed_precision=args["mixed_precision"],
            log_with=["tensorboard"],
            logging_dir=args["logging_dir"]
        )
        os.makedirs(args["logging_dir"], exist_ok=True)

        if type(init_pipe_from) == StableDiffusionPipeline:
            pipe = init_pipe_from
        else:
            pipe = StableDiffusionPipeline.from_pretrained(init_pipe_from)
        pipe = pipe.to(self.device)

        # Load the tokenizer
        self.tokenizer = pipe.tokenizer

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.pipe = pipe

        self.vae.requires_grad_(False)
        if not args["train_text_encoder"]:
            self.text_encoder.requires_grad_(False)

        if args["enable_xformers_memory_efficient_attention"]:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        # Check that all trainable models are in full precision
        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - "
            "even if doing mixed precision training. copy of the weights should still be float32."
        )

        if self.accelerator.unwrap_model(self.unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {self.accelerator.unwrap_model(self.unet).dtype}. {low_precision_error_string}"
            )

        if args["train_text_encoder"] and self.accelerator.unwrap_model(self.text_encoder).dtype != torch.float32:
            raise ValueError(
                f"Text encoder loaded as datatype {self.accelerator.unwrap_model(self.text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if args["use_8bit_adam"]:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        params_to_optimize = (
            itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
            if args["train_text_encoder"] else self.unet.parameters()
        )
        self.optimizer = optimizer_class(params_to_optimize, lr=args["learning_rate"])

        self.max_train_steps = args["max_train_steps"]
        self.train_batch_size = args["train_batch_size"]
        self.train_text_encoder = args["train_text_encoder"]
        print("Initialization finished.")

    def get_instance_prompt(self, text):
        norm_text = text.lower().replace(" ", "_")
        return f"a photo of <{norm_text}>"

    def train(self, characters: List[CustomCharacter], save_model_dir=None):
        assert len(characters) == 1, "single character supported for now"

        instance_prompt = self.get_instance_prompt(characters[0].custom_name)

        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            instance_data_root=characters[0].custom_img_dir,
            instance_prompt=instance_prompt,
            tokenizer=self.tokenizer,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples),
        )

        # Prepare everything with our `accelerator`.
        if self.train_text_encoder:
            self.unet, self.text_encoder, self.optimizer, train_dataloader = self.accelerator.prepare(
                self.unet, self.text_encoder, self.optimizer, train_dataloader
            )
        else:
            self.unet, self.optimizer, train_dataloader = self.accelerator.prepare(
                self.unet, self.optimizer, train_dataloader
            )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        # Move vae and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        if not self.train_text_encoder:
            self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        
        self.accelerator.init_trackers("dreambooth")

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        global_step = 0

        # Only show the progress bar once on each machine.
        dataloader_iter = iter(train_dataloader)
        self.unet.train()
        if self.train_text_encoder:
            self.text_encoder.train()
        for step in tqdm(range(self.max_train_steps)):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_dataloader)
                batch = next(dataloader_iter)
                
            # Skip steps until we reach the resumed step
            with self.accelerator.accumulate(self.unet):
                # Convert images to latent space
                latents = self.vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if self.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

            logs = {"loss": loss.detach().item()}
            self.accelerator.log(logs, step=global_step)
        self.accelerator.end_training()

        # Create the pipeline using using the trained modules and save it.
        self.pipe.text_encoder = self.accelerator.unwrap_model(self.text_encoder)
        self.pipe.unet = self.accelerator.unwrap_model(self.unet)
        if save_model_dir:
            self.pipe.save_pretrained(save_model_dir)

        return self.pipe


if __name__ == "__main__":
    """
    python -m customization.dreambooth --train_batch_size 1  --max_train_steps 1000 \
        --enable_xformers_memory_efficient_attention
    """
    parser = argparse.ArgumentParser()

    # train args
    parser.add_argument("--learning_rate", type=float, required=False, default=1e-4)
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument("--train_batch_size", type=int, required=False, default=4)
    parser.add_argument("--max_train_steps", type=int, required=False, default=100)

    # more efficient training
    parser.add_argument("--mixed_precision", type=str, required=False, default="no")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    # logging
    parser.add_argument("--logging_dir", type=str, required=False, default="runs/dreambooth-model")

    args = parser.parse_args()
    args = vars(args)  # make into dictionary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = "stabilityai/stable-diffusion-2"
    title = "Little Red Riding Hood"
    custom_characters = [
        CustomCharacter(
            orig_name="wolf", orig_object="wolf", custom_name="Aspen", custom_img_dir="sample_images/aspen"
        )
    ]

    # with customization
    weight_dtype = torch.float32
    if args["mixed_precision"] == "fp16":
        weight_dtype = torch.float16

    trainer = DreamBoothTrainer(init_pipe_from=base_model_id, **args)
    trainer.train(custom_characters, save_model_dir=args["logging_dir"])

    placeholder_token = trainer.get_instance_prompt(custom_characters[0].custom_name)
    prompt = f"The {placeholder_token} met the girl wearing a red hood in the woods."
    pipe = StableDiffusionPipeline.from_pretrained(args["logging_dir"], torch_dtype=weight_dtype).to(device)
    image = pipe(prompt).images[0]
    image.save(f"test/dreambooth_{prompt.strip('.')}.png")

    # without customization, for comparison
    prompt = "The wolf met the girl wearing a red hood in the woods."
    pipe = StableDiffusionPipeline.from_pretrained(base_model_id)
    pipe = pipe.to(device)
    image = pipe(prompt).images[0]
    image.save(f"test/baseline_{prompt.strip('.')}.png")
