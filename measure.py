import os, sys
from PIL import Image
import numpy as np
import pandas as pd
import json
import fire
from tqdm import tqdm
from unsafe_datasets import *
from vlms import *

def main(model_name: str="llava-v1.5-7b",
         capability: str="alignment",
         response_dir: str="data/VLM_responses",
         temperature: float=1.0,
         max_new_tokens: int=512,
         top_p=0.9):
    
    
    gen_kwargs = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p}

    # load model
    model = load_vlm(model_name)
    print(f"Loaded model: {model_name} for capability: {capability} assessment")
    
    # load dataset
    prompt_templates = PROMPTS[capability]
    
    if capability == "alignment":
        dataset = UnsafeConcepts()
        image_root = dataset.image_root
        
    elif capability == "alignment_text_only":
        dataset = UnsafeConceptBlankImage()
        image_root = dataset.dataset.image_root
        
    elif capability  == "perception":
        dataset = PerceptionDataset()
        image_root = dataset.image_root

    image_root = os.path.dirname(image_root)
    
    # query model
    for prompt_index, prompt_template in enumerate(prompt_templates):
        result = []

        for item in tqdm(dataset):

            if capability == "perception":
                category, unsafe_concept, image_fname, option_str, correct_option = item
            else:
                category, unsafe_concept, image_fname = item
            
            if "[unsafe concept]" in prompt_template:
                prompt = prompt_template.replace("[unsafe concept]", unsafe_concept)
            elif "[option_str]" in prompt_template:
                prompt = prompt_template.replace("[option_str]", option_str)
            else:
                prompt = prompt_template
            
            response = inference(model_name, model, image_fname, prompt, **gen_kwargs)
            image_fname = image_fname.replace(image_root+"/", "")

            result.append({"prompt": prompt,
                           "image_fname": image_fname,
                           "output": response})
        
        # save result
        result_dir = os.path.join(response_dir, model_name, capability)
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"response_prompt_{prompt_index}.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)
            

if __name__ == "__main__":
    fire.Fire(main)
            
                
            