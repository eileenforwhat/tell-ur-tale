from typing import List, Union

import argparse
import os
import random

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from utils import CustomCharacter


def save_progress(text_encoder, placeholder_token_id, accelerator, placeholder_token, save_path):
    print("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = PIL.Image.BICUBIC
        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (h, w) = (img.shape[0], img.shape[1])
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


class TextualInversionTrainer(object):
    """
    Adapted from:
        https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    """
    def __init__(self, init_pipe_from: Union[StableDiffusionPipeline, str], **args):
        self.device = args["device"] if torch.cuda.is_available() else "cpu"
        self.accelerator = Accelerator(mixed_precision=args["mixed_precision"])

        # Handle the repository creation
        os.makedirs(args["logging_dir"], exist_ok=True)

        if type(init_pipe_from) == StableDiffusionPipeline:
            pipe = init_pipe_from
        else:
            pipe = StableDiffusionPipeline.from_pretrained(init_pipe_from)
        pipe = pipe.to(self.device)

        # Load scheduler and models
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.pipe = pipe

        if args["enable_xformers_memory_efficient_attention"]:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        self.max_train_steps = args["max_train_steps"]
        self.train_batch_size = args["train_batch_size"]
        self.learning_rate = args["learning_rate"]
        print("Initialization finished.")

    def get_placeholder_token(self, text):
        norm_text = text.lower().replace(" ", "_")
        return f"<{norm_text}>"

    def train(self, characters: List[CustomCharacter], save_model_dir=None):
        """
        :return:  StableDiffusionPipeline
        """
        assert len(characters) == 1, "single character supported for now"

        placeholder_token = self.get_placeholder_token(characters[0].custom_name)
        print(f"Add the placeholder token in tokenizer: {placeholder_token}")
        num_added_tokens = self.tokenizer.add_tokens(placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        # Convert the initializer_token, placeholder_token to ids
        initializer_token = f"{characters[0].orig_object}"
        print(f"initializer_token: {initializer_token}")
        token_ids = self.tokenizer.encode(initializer_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]
        placeholder_token_id = self.tokenizer.convert_tokens_to_ids(placeholder_token)

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

        # Freeze vae and unet
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        # Freeze all parameters except for the token embeddings in text encoder
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        self.text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            self.text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
            lr=self.learning_rate,
        )

        # Dataset and DataLoaders creation
        train_dataset = TextualInversionDataset(
            data_root=characters[0].custom_img_dir,
            tokenizer=self.tokenizer,
            placeholder_token=placeholder_token,
            learnable_property="object",
            set="train",
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=4
        )

        # Prepare everything with our `accelerator`
        self.text_encoder, optimizer, train_dataloader = self.accelerator.prepare(
            self.text_encoder, optimizer, train_dataloader
        )

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        print("Move vae and unet to device and cast to weight_dtype")
        self.unet.to(self.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.accelerator.device, dtype=weight_dtype)

        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num train steps = {self.max_train_steps}")
        print(f"  Instantaneous batch size per device = {self.train_batch_size}")

        # keep original embeddings as reference
        orig_embeds_params = self.accelerator.unwrap_model(
            self.text_encoder).get_input_embeddings().weight.data.clone()

        self.text_encoder.train()
        dataloader_iter = iter(train_dataloader)
        for step in tqdm(range(self.max_train_steps)):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_dataloader)
                batch = next(dataloader_iter)

            with self.accelerator.accumulate(self.text_encoder):
                # Convert images to latent space
                latents = self.vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * self.vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = self.text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

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

                optimizer.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.arange(len(self.tokenizer)) != placeholder_token_id
                with torch.no_grad():
                    self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

        # Create the pipeline using the trained modules and save it.
        self.pipe.text_encoder = self.accelerator.unwrap_model(self.text_encoder)
        if save_model_dir:
            self.pipe.save_pretrained(save_model_dir)
            # Save the newly trained embeddings
            save_path = os.path.join(save_model_dir, "learned_embeds.bin")
            save_progress(self.text_encoder, placeholder_token_id, self.accelerator, placeholder_token, save_path)

        return self.pipe


if __name__ == "__main__":
    """
    python -m customization.textual_inversion --enable_xformers_memory_efficient_attention --train_batch_size 4  --max_train_steps 2000
    """
    parser = argparse.ArgumentParser()

    # train args
    parser.add_argument("--learning_rate", type=float, required=False, default=1e-4)
    parser.add_argument("--train_batch_size", type=int, required=False, default=16)
    parser.add_argument("--max_train_steps", type=int, required=False, default=100)
    parser.add_argument("--device", type=str, required=False, default="cuda:0")  # suffix for text2image model

    # more efficient training
    parser.add_argument("--mixed_precision", type=str, required=False, default="fp16")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    # logging
    parser.add_argument("--logging_dir", type=str, required=False, default="runs/text-inversion-model")

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

    trainer = TextualInversionTrainer(init_pipe_from=base_model_id, **args)
    trainer.train(custom_characters, save_model_dir=args["logging_dir"])

    placeholder_token = trainer.get_placeholder_token(custom_characters[0].custom_name)
    prompt = f"The {placeholder_token} met the girl wearing a red hood in the woods."
    pipe = StableDiffusionPipeline.from_pretrained(args["logging_dir"], torch_dtype=weight_dtype).to(device)
    image = pipe(prompt).images[0]
    image.save(f"test/inversion_{prompt.strip('.')}.png")

    # without customization, for comparison
    prompt = "The wolf met the girl wearing a red hood in the woods."
    pipe = StableDiffusionPipeline.from_pretrained(base_model_id)
    pipe = pipe.to(device)
    image = pipe(prompt).images[0]
    image.save(f"test/baseline_{prompt.strip('.')}.png")
