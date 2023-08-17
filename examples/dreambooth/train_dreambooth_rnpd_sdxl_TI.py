import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Optional
import subprocess
import sys

import gc
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PretrainedConfig
import bitsandbytes as bnb

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from contextlib import nullcontext
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from lora_sdxl_TI import *

logger = get_logger(__name__)

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        use_auth_token=True
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
        

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="",
        help="The prompt to specify images in the same class as provided instance images.",
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
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
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
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )

    parser.add_argument(
        "--save_n_steps",
        type=int,
        default=1,
        help=("Save the model every n global_steps"),
    )
    
    parser.add_argument(
        "--save_starting_step",
        type=int,
        default=1,
        help=("The step from which it starts saving intermediary checkpoints"),
    )
    
    parser.add_argument(
        "--stop_text_encoder_training",
        type=int,
        default=1000000,
        help=("The step at which the text_encoder is no longer trained"),
    )


    parser.add_argument(
        "--image_captions_filename",
        action="store_true",
        help="Get captions from filename",
    )    
    
    
    parser.add_argument(
        "--dump_only_text_encoder",
        action="store_true",
        default=False,        
        help="Dump only text-encoder",
    )

    parser.add_argument(
        "--train_only_unet",
        action="store_true",
        default=False,        
        help="Train only the unet",
    )
    
    parser.add_argument(
        "--train_only_text_encoder",
        action="store_true",
        default=False,        
        help="Train only the text-encoder",
    )        
    
    parser.add_argument(
        "--Resumetr",
        type=str,
        default="False",        
        help="Resume training info",
    )    
    
    parser.add_argument(
        "--Style",
        action="store_true",
        default=False,        
        help="Further reduce overfitting",
    )        
    
    parser.add_argument(
        "--Session_dir",
        type=str,
        default="",     
        help="Current session directory",
    )    

    parser.add_argument(
        "--external_captions",
        action="store_true",
        default=False,        
        help="Use captions stored in a txt file",
    )    
    
    parser.add_argument(
        "--captions_dir",
        type=str,
        default="",
        help="The folder where captions files are stored",
    )        

    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help="Offset Noise",
    )
    
    parser.add_argument(
        "--ofstnselvl",
        type=float,
        default=0.03,        
        help="Offset Noise amount",
    )
    
    parser.add_argument(
        "--dim",
        type=int,
        default=64,        
        help="LoRa dimension",
    )    


    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args



class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """    
    
    def __init__(
        self,
        instance_data_root,
        args,
        tokenizers,        
        text_encoders,
        size=512,
        center_crop=False,
        instance_prompt_hidden_states=None,
        instance_unet_added_conditions=None,    
    ):
        self.size = size
        self.tokenizers=tokenizers
        self.text_encoders=text_encoders
        self.center_crop = center_crop
        self.instance_prompt=None,
        self.instance_prompt_hidden_states = instance_prompt_hidden_states
        self.instance_unet_added_conditions = instance_unet_added_conditions
        self.image_captions_filename = None

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        #self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if args.image_captions_filename:
            self.image_captions_filename = True
        
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

    def __getitem__(self, index, args=parse_args()):
        example = {}
        path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        if self.image_captions_filename:
            filename = Path(path).stem
            
            pt=''.join([i for i in filename if not i.isdigit()])
            pt=pt.replace("_"," ")
            pt=pt.replace("(","")
            pt=pt.replace(")","")
            pt=pt.replace("-","")
            pt=pt.replace("conceptimagedb","")  
            
            if args.external_captions:
              cptpth=os.path.join(args.captions_dir, filename+'.txt')
              if os.path.exists(cptpth):
                with open(cptpth, "r") as f:
                    instance_prompt=f.read()
              else:
                instance_prompt=pt
            else:
                instance_prompt = pt
        
        self.instance_prompt=instance_prompt
        
        example["instance_images"] = self.image_transforms(instance_image)
        
        example["instance_prompt"]=instance_prompt


        return example

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


def encode_prompt(args, text_encoders, tokenizers, prompt):
    prompt_embeds_list = []

   
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(text_encoder.device)
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(
            text_input_ids,
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds    
    

def collate_fn(examples, args):

    pixel_values = [example["instance_images"] for example in examples]
    
    input_ids = [example["instance_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "unet_added_conditions": {}
        }
        
    return batch

def compute_embeddings(args, prompt, text_encoders, tokenizers):
    original_size = (args.resolution, args.resolution)
    target_size = (args.resolution, args.resolution)
    crops_coords_top_left = (0, 0)
    
    prompt_embeds, pooled_prompt_embeds = encode_prompt(args, text_encoders, tokenizers, prompt)
    add_text_embeds = pooled_prompt_embeds
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])

    prompt_embeds = prompt_embeds.to('cuda')
    add_text_embeds = add_text_embeds.to('cuda')
    add_time_ids = add_time_ids.to('cuda', dtype=prompt_embeds.dtype)
    unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    return prompt_embeds, unet_added_cond_kwargs
    
    
class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache, cond_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache
        self.cond_cache = cond_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index], self.cond_cache[index]


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    
    i=args.save_starting_step
       
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )


    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
        use_auth_token=True,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        use_fast=False,
        use_auth_token=True
    )

    
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    text_encoder_one = text_encoder_cls_one.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to('cuda')
    text_encoder_two = text_encoder_cls_two.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2").to('cuda')
            
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", use_auth_token=True).to('cuda')
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", use_auth_token=True).to('cuda')
    
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]  

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    model_path = os.path.join(args.Session_dir, os.path.basename(args.Session_dir) + "_TI.safetensors")
    network = create_network(1, args.dim, 20000, text_encoders)

    unet.enable_xformers_memory_efficient_attention()

    network.apply_to(text_encoders, True)
    network.prepare_optimizer_params(args.learning_rate)
    trainable_params = network.parameters()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_class = bnb.optim.AdamW8bit

    params_to_optimize = trainable_params
    
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", use_auth_token=True)

    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        tokenizers=tokenizers,
        text_encoders=text_encoders,
        size=args.resolution,
        center_crop=args.center_crop,
        args=args
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args)
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    

    text_encoder_one, text_encoder_two, network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder_one, text_encoder_two, network, optimizer, train_dataloader, lr_scheduler)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    network.requires_grad_(True)
    unet.eval()
    vae.eval()
    text_encoder_two.eval()
    text_encoder_one.eval()
   
    
    latents_cache = []
    text_encoder_cache = []
    cond_cache= []
    for batch in train_dataloader:
               
        batch["input_ids"] = batch["input_ids"]
        batch["pixel_values"]=vae.encode(batch["pixel_values"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample() * vae.config.scaling_factor

        latents_cache.append(batch["pixel_values"])
        text_encoder_cache.append(batch["input_ids"])
        cond_cache.append(batch["input_ids"])

    train_dataset = LatentsDataset(latents_cache, text_encoder_cache, cond_cache)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

    #del vae
    gc.collect()
    torch.cuda.empty_cache()

    
   
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    def bar(prg):
       br='|'+'█' * prg + ' ' * (25-prg)+'|'
       return br

     
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    for epoch in range(args.num_train_epochs):
        network.train()
        with accelerator.accumulate(network):
            for step, batch in enumerate(train_dataloader):
                with torch.no_grad():
                    model_input = batch[0][0]

                if args.offset_noise:
                    noise = torch.randn_like(model_input)
                else:
                    noise = torch.randn_like(model_input)

                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)


                # Predict the noise residual                
                te_hidden_states=[]
                unet_added_conditions= {"text_embeds": [], "time_ids": []}
                for prompt in batch[0][1]:
                    with accelerator.autocast():
                        hidden_states, unet_added_cond= compute_embeddings(args, prompt, text_encoders, tokenizers)
                        te_hidden_states.append(hidden_states)
                        unet_added_conditions["text_embeds"].append(unet_added_cond["text_embeds"])
                        unet_added_conditions["time_ids"].append(unet_added_cond["time_ids"])


                hidden_states=torch.cat(te_hidden_states, dim=0)
                unet_added_conditions["text_embeds"] = torch.cat(unet_added_conditions["text_embeds"], dim=0)
                unet_added_conditions["time_ids"]=torch.cat(unet_added_conditions["time_ids"], dim=0)

                with accelerator.autocast():
                    model_pred = unet(noisy_model_input, timesteps, hidden_states, added_cond_kwargs=unet_added_conditions).sample

                # Get the target for loss depending on the prediction type
                target = noise
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                fll=round((global_step*100)/args.max_train_steps)
                fll=round(fll/4)
                pr=bar(fll)

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                progress_bar.set_description_str("Progress")
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
         network = accelerator.unwrap_model(network)
    accelerator.end_training()
    network.save_weights(model_path, torch.float32, None)
      
    accelerator.end_training()

if __name__ == "__main__":
    main()
