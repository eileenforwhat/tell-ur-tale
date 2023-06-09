from typing import List, Union

import argparse
import itertools
import hashlib
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
from diffusers.optimization import get_scheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

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
        class_data_root=None,
        class_prompt=None,
        class_num=None,
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
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None
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
        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
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
    def __init__(self, init_pipe_from: Union[StableDiffusionPipeline, str], **args):
        # args["use_lora"]

        self.device = args["device"] if torch.cuda.is_available() else "cpu"
        self.accelerator = Accelerator(
            device_placement=False,
            mixed_precision=args["mixed_precision"],
            log_with=["tensorboard"],
            logging_dir=args["custom_model_dir"]
        )
        os.makedirs(args["custom_model_dir"], exist_ok=True)

        if type(init_pipe_from) == StableDiffusionPipeline:
            pipe = init_pipe_from
        else:
            pipe = StableDiffusionPipeline.from_pretrained(init_pipe_from, torch_dtype=torch.float32)

        # Load the tokenizer
        self.tokenizer = pipe.tokenizer

        # Load scheduler and models
        self.pipe = pipe.to(self.device)
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.unet = pipe.unet

        if args["use_lora"]:
            # We only train the additional adapter LoRA layers
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.unet.requires_grad_(False)
        else:
            self.vae.requires_grad_(False)
            if not args["train_text_encoder"]:
                self.text_encoder.requires_grad_(False)
            self.unet.requires_grad_(True)

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

        if args["use_lora"]:
            # TODO (if use lora, train lora weights)
            lora_attn_procs = {}
            for name in self.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]
            
                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size,
                                                          cross_attention_dim=cross_attention_dim)
            
            self.unet.set_attn_processor(lora_attn_procs)
            self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
            params_to_optimize = self.lora_layers.parameters() # todo lora parameters
            self.accelerator.register_for_checkpointing(self.lora_layers)
        else:
            params_to_optimize = (
                itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
                if args["train_text_encoder"] else self.unet.parameters()
            )
        # Optimizer creation
        self.optimizer = optimizer_class(params_to_optimize, lr=args["learning_rate"])

        # Generate class images if prior preservation is enabled.
        if args["with_prior_preservation"]:
            class_images_dir = Path(args["class_data_dir"])
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < args["num_class_images"]:
                num_new_images = args["num_class_images"] - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(args["class_prompt"], num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=1)

                sample_dataloader = self.accelerator.prepare(sample_dataloader)
                self.pipe.to(self.device)

                for example in tqdm(sample_dataloader, desc="Generating class images"):
                    images = self.pipe(example["prompt"]).images

                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.max_train_steps = args["max_train_steps"]
        self.train_batch_size = args["train_batch_size"]
        self.train_text_encoder = args["train_text_encoder"]
        self.save_model_dir = args["custom_model_dir"]
        os.makedirs(self.save_model_dir, exist_ok=True)
        # prior preservation loss
        self.with_prior_preservation = args["with_prior_preservation"]

        self.class_data_dir = args["class_data_dir"]
        self.class_prompt = args["class_prompt"]
        self.num_class_images = args["num_class_images"]
        self.prior_loss_weight = args["prior_loss_weight"]
        self.use_lora = args["use_lora"]

        print("Initialization finished.")

    def get_placeholder_token(self, text):
        norm_text = text.lower().replace(" ", "_")
        return f"<{norm_text}>"

    def get_instance_prompt(self, character: CustomCharacter):
        token = self.get_placeholder_token(character.custom_name)
        return f"a photo of {token} {character.orig_object}"

    def train(self, characters: List[CustomCharacter], save_model_dir=None):
        assert len(characters) == 1, "single character supported for now"

        instance_prompt = self.get_instance_prompt(characters[0])

        # Dataset and DataLoaders creation:
        if self.with_prior_preservation:
            train_dataset = DreamBoothDataset(
                instance_data_root=characters[0].custom_img_dir,
                instance_prompt=instance_prompt,
                class_data_root=self.class_data_dir,
                class_prompt=self.class_prompt,
                class_num=self.num_class_images,
                tokenizer=self.tokenizer,
            )
        else:
            train_dataset = DreamBoothDataset(
                instance_data_root=characters[0].custom_img_dir,
                instance_prompt=instance_prompt,
                tokenizer=self.tokenizer,
            )
            

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, self.with_prior_preservation),
        )

        # lr_scheduler = get_scheduler(
        #     args.lr_scheduler,
        #     optimizer=self.optimizer,
        #     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        #     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        #     num_cycles=args.lr_num_cycles,
        #     power=args.lr_power,
        # )
        # Prepare everything with our `accelerator`.
        # TODO: add lora if true
        if self.use_lora:
            self.lora_layers, self.optimizer, train_dataloader = self.accelerator.prepare(
                self.lora_layers, self.optimizer, train_dataloader
            )
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
        self.vae.to(self.device, dtype=weight_dtype)
        if not self.train_text_encoder:
            self.text_encoder.to(self.device, dtype=weight_dtype)
        
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
        self.unet.train().to(self.device)
        if self.train_text_encoder:
            self.text_encoder.train()
        for step in tqdm(range(self.max_train_steps)):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_dataloader)
                batch = next(dataloader_iter)
            batch["pixel_values"] = batch["pixel_values"].to(self.device)
            batch["input_ids"] = batch["input_ids"].to(self.device)

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

                if self.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + self.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                self.accelerator.backward(loss)
                # if self.accelerator.sync_gradients:
                #     params_to_clip = lora_layers.parameters()
                #     self.accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                # lr_scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()

            logs = {"loss": loss.detach().item()}
            self.accelerator.log(logs, step=global_step)
        self.accelerator.end_training()

        # Create the pipeline using using the trained modules and save it.
        self.pipe.text_encoder = self.accelerator.unwrap_model(self.text_encoder)
        self.pipe.unet = self.accelerator.unwrap_model(self.unet)
        save_model_dir = save_model_dir or self.save_model_dir
        
        if save_model_dir:
            if self.use_lora:
                self.unet = self.unet.to(torch.float32)
                self.unet.save_attn_procs(save_model_dir)
            else:
                self.pipe.save_pretrained(save_model_dir)
            print(f"saved model to {save_model_dir}")
        return self.pipe


if __name__ == "__main__":
    """
    python -m customization.dreambooth --train_batch_size 1  --max_train_steps 2000 \
        --enable_xformers_memory_efficient_attention --class_prompt "a photo of a dog" --class_data_dir sample_images/class_dog \
        --with_prior_preservation --mixed_precision
    """
    parser = argparse.ArgumentParser()

    # train args
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--only_inference", type=bool, default=False)
    parser.add_argument("--learning_rate", type=float, required=False, default=1e-4)
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument("--train_batch_size", type=int, required=False, default=4)
    parser.add_argument("--max_train_steps", type=int, required=False, default=100)

    # more efficient training
    parser.add_argument("--mixed_precision", type=str, 
                        choices=["no", "fp16"],
                        required=False, default="no")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    # logging
    parser.add_argument("--custom_model_dir", type=str, required=False, default="runs/dreambooth-model")

    args = parser.parse_args()
    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    args = vars(args)  # make into dictionary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_id = "stabilityai/stable-diffusion-v1-5"
    title = "Little Red Riding Hood"
    custom_characters = [
        CustomCharacter(
            orig_name="wolf", orig_object="dog", custom_name="Aspen", custom_img_dir="sample_images/aspen_512"
        )
    ]

    # with customization
    weight_dtype = torch.float32
    if args["mixed_precision"] == "fp16":
        weight_dtype = torch.float16
        
    trainer = DreamBoothTrainer(init_pipe_from=base_model_id, device=device, **args)
    if not args['only_inference']:
        trainer.train(custom_characters, save_model_dir=args["custom_model_dir"])

    placeholder_token = trainer.get_placeholder_token(custom_characters[0].custom_name)

    prompt = f"f{placeholder_token} facing a girl wearing a red hood in the woods, as anime illustration wondrous, expressive, by Hayao Miyazaki, by Studio Ghibli, cel-shaded, professional illustration, magical, dynamic, colorful, uhd, highly detailed, intricate, vivid colors"
    if args["use_lora"]:
        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, revision=None, torch_dtype=weight_dtype).to(device)
        pipe.unet.load_attn_procs(args["custom_model_dir"])
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args["custom_model_dir"], torch_dtype=weight_dtype).to(device)
    
    image = pipe(prompt).images[0]
    image.save(f"test/dreambooth_{prompt.strip('.')}.png")

    # without customization, for comparison
    # prompt = "The wolf met the girl wearing a red hood in the woods, as anime illustration wondrous, expressive, by Hayao Miyazaki, by Studio Ghibli, cel-shaded, professional illustration, magical, dynamic, colorful, uhd, highly detailed, intricate, vivid colors"
    # pipe = StableDiffusionPipeline.from_pretrained(base_model_id).to(device)
    # image = pipe(prompt).images[0]
    # image.save(f"test/baseline_{prompt.strip('.')}.png")
