import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import re, sys, os, json
from peft import PeftModel
import tqdm
from unsafe_datasets import *

def load_pretrained_model(lora_path, model_base, device):
    compute_dtype = torch.float16
    
    from llava.model import LlavaLlamaForCausalLM
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    model =  LlavaLlamaForCausalLM.from_pretrained(model_base,
                                                   device_map={"": device}, 
                                                   torch_dtype=compute_dtype)
    
    model.config.torch_dtype = compute_dtype
    
    if lora_path is not None:
        model = PeftModel.from_pretrained(
                model,
                lora_path,
                is_trainable=False,
            )
        model = model.merge_and_unload()

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device).to(compute_dtype)
    vision_tower.requires_grad_(False)
    image_processor = vision_tower.image_processor
    
    mm_projector = model.get_model().mm_projector
    mm_projector.to(device=device).to(compute_dtype)
    mm_projector.requires_grad_(False)
    
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    
    model.resize_token_embeddings(len(tokenizer))
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    return tokenizer, model, image_processor, context_len

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def format_dataset(dataset_name):

    # load test prompts (same as the alignment prompts)
    test_prompts = PROMPTS["alignment"]
    
    # load images
    if dataset_name == "UnsafeConcepts_TEST":
        dataset = fetch_UnsafeConcepts_TEST_dataset()
        
    elif dataset_name in ["SMID", "NSFW"]:
        dataset = fetch_generalization_dataset(dataset_name=dataset_name, sample_num=None)

    seed = 42
    np.random.seed(seed)
    
    # randomly pair with one of the alignment prompts
    num_images = len(dataset)
    sampled_indices = np.random.choice(len(test_prompts), size=num_images, replace=True)
    prompts = [test_prompts[i] for i in sampled_indices]

    paired = [
        {"image_fname": item["image_fname"], "prompt": pr}
        for item, pr in zip(dataset, prompts)
    ]

    return paired
    
def main(args):
    
    working_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Model
    disable_torch_init()

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.lora_path, args.model_base, device="cuda"
    )
        
    def generate(image_file, prompt):
        qs = prompt
        args.image_file = image_file

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv_mode = "llava_v1"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        image_files = image_parser(args)
        images = load_images(image_files)
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        model.config.use_cache = True
        model.config.cache_shape = (
            input_ids.shape[-1] + args.max_new_tokens + model.get_vision_tower().num_patches,
        )
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs
    
    dataset_name = args.dataset_name
    print("dataset_name:", dataset_name)
    if dataset_name in ["LLaVABench", "MME"]:
        
        dataset = fetch_general_capability_dataset(dataset_name)
        
        result = []
        for idx, item in tqdm.tqdm(enumerate(dataset)):

            image_fname, prompt = item["image_fname"], item["prompt"]

            output = generate(image_file=image_fname, prompt=prompt)
            print(image_fname)
            print(output)
            
            result.append({
                "idx": str(idx),
                "prompt": prompt,
                "image_fname": image_fname,
                "output": output
            })
        
        Path(args.save_dir).mkdir(exist_ok=True, parents=True)
        json.dump(result, open(os.path.join(args.save_dir, f"{dataset_name}_result.json"), "w"), indent=2)
    
    elif dataset_name=="context":
        context_settings = ["safe", "unsafe"]
        
        for context in context_settings:
            
            dataset = fetch_contextualized_data("alignment_context", context=context)
            testset = fetch_UnsafeConcepts_TEST_dataset()
            testset_image_fnames = [os.path.join(working_dir, item["image_fname"]) for item in testset]

            result = []
            for idx, item in enumerate(dataset):
                image_fname = item["image_fname"]
                if image_fname not in testset_image_fnames:
                    continue
                prompt = item["prompt"]

                output = generate(image_file=image_fname, prompt=prompt)
                print(output)
                
                result.append({
                    "idx": str(idx),
                    "prompt": prompt,
                    "image_fname": image_fname.replace(working_dir, ""),
                    "output": output
                })

            Path(args.save_dir).mkdir(exist_ok=True, parents=True)
            json.dump(result, open(os.path.join(args.save_dir, f"context_{context}_result.json"), "w"))
    else:
        
        dataset = format_dataset(dataset_name)
        print("dataset size:", len(dataset))
        
        result = []
        for idx, item in tqdm.tqdm(enumerate(dataset)):
            
            image_fname = item["image_fname"]
            prompt = item["prompt"]
    
            output = generate(image_file=image_fname, prompt=prompt)
            print(image_fname)
            print(output)
            
            result.append({
                "idx": str(idx),
                "prompt": prompt,
                "image_fname": image_fname,
                "output": output
            })


        Path(args.save_dir).mkdir(exist_ok=True, parents=True)
        json.dump(result, open(os.path.join(args.save_dir, f"{dataset_name}_result.json"), "w"), indent=2)


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    # customized
    parser.add_argument("--save_dir", type=str, default="outputs/original")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--dataset_name", type=str, default="MME")
    # default
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    main(args)