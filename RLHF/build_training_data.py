import os, sys
import re
import random
import numpy as np
import pandas as pd
abspath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(abspath))
from unsafe_datasets import *

def build_finetuning_dataset_with_concepts(method, train_test_split=0.8, save_dir="data/rlhf"):
    
    os.makedirs(save_dir, exist_ok=True)
    
    def get_conversation(method, image_fname, prompt, label, rejected_label, concept, prompt_type):
        
        label_formatting = {
            "Safe": {
                PROMPTS["alignment"][0]: "Yes, it is safe for viewing on social media. The image shows [concept], which is considered harmless.",
                PROMPTS["alignment"][1]: "Yes, it is safe to generate such an image using AI. The image shows [concept], which is considered harmless.",
                PROMPTS["alignment"][2]: "Yes, the image adheres to community guidelines for online forums. The image shows [concept], which is considered harmless.",
                PROMPTS["alignment"][3]: "Yes, the image can be shown to a general audience without causing harm or distress. The image shows [concept], which is considered harmless.",
                PROMPTS["alignment"][4]: "Yes, the image is free from elements that could be considered harmful or exploitative. The image shows [concept], which is considered harmless."
                },
            "Unsafe":{
                PROMPTS["alignment"][0]: "No, it is not safe for viewing on social media. The image shows [concept], which is considered harmful.",
                PROMPTS["alignment"][1]: "No, it is not safe to generate such an image using AI. The image shows [concept], which is considered harmful.",
                PROMPTS["alignment"][2]: "No, the image does not adhere to community guidelines for online forums. The image shows [concept], which is considered harmful.",
                PROMPTS["alignment"][3]: "No, the image cannot be shown to a general audience without causing harm or distress. The image shows [concept], which is considered harmful.",
                PROMPTS["alignment"][4]: "No, the image contains elements that could be considered harmful or exploitative. The image shows [concept], which is considered harmful."
            }
                
        }
        
        concept = concept.lower()
        
        if method == "sft":
            response = label_formatting[label][prompt].replace("[concept]", concept)
            prompt = prompt.replace("[unsafe concept]", concept)
            return {
                    "image": image_fname,
                    "conversations":[
                        {"from": "user", "value": "<image>\n"+prompt},
                        {"from": "assistant", "value": response}
                    ]
                }
        elif method == "dpo":
            chosen_response = label_formatting[label][prompt].replace("[concept]", concept)
            rejected_response = label_formatting[rejected_label][prompt].replace("[concept]", concept)
            prompt = prompt.replace("[unsafe concept]", concept)
            return {
                "image": image_fname,
                "prompt": "<image>\n"+prompt,
                "chosen": chosen_response,
                "rejected": rejected_response
            }
        elif method == "ppo":
            if "[unsafe concept]" in prompt:
                prompt = prompt.replace("[unsafe concept]", concept)
                
            return {
                "image": image_fname,
                "label": label,
                "concept": concept,
                "type": prompt_type,
                "conversations": [{"from": "human", "value": "<image>\n"+prompt},
                                    {"from": "gpt", "value": ""}]
            }

    image_root = os.path.join(os.path.dirname(abspath), "data/UnsafeConcepts")
    unsafe_train, unsafe_test = UnsafeConcept_train_test_split(split_ratio=train_test_split, image_root=image_root)

    image_root = os.path.join(os.path.dirname(abspath), "data/imagenet_1k")
    safe_train, safe_test = SafeConcept_train_test_split(split_ratio=train_test_split, image_root=image_root)
    
    # balance dataset
    safe_train = safe_train[:len(unsafe_train)]
    safe_test = safe_test[:len(unsafe_test)]
    
    data = []
    
    prompts = PROMPTS["alignment"]
    print("prompt:", prompts)
    
    prompt_types = {p: "YORN" for p in prompts} # yes or no type of prompts, we can also add other types of prompts e.g., open-ended prompts 
    
    # if method == "ppo":
        
    #     open_prompts = ["Describe the image in detail.",
    #         "What details in the image catch your attention first?",
    #         "What do you see from the image?"]
        
    #     prompts.extend(open_prompts)
    
    #     prompt_types.update({p: "Open-ended" for p in open_prompts})

    # unsafe images
    image_folder = abspath.replace("RLHF", "data/")
    
    for item in unsafe_train:
        prompt = random.sample(prompts, 1)[0]
        category, unsafe_concept, image_fname = item
        prompt_type = prompt_types[prompt]

        image_fname = image_fname.replace(image_folder, "")
        data.append(get_conversation(method, image_fname, prompt, label="Unsafe", rejected_label="Safe", concept=unsafe_concept, prompt_type=prompt_type))
    
    for item in safe_train:
        image_fname = item["image_filename"]
        concept = item["concept"]
        prompt = random.sample(prompts, 1)[0]
        prompt_type = prompt_types[prompt]

        image_fname = image_fname.replace(image_folder, "")
        data.append(get_conversation(method, image_fname, prompt, "Safe", "Unsafe", concept=concept, prompt_type=prompt_type))
    
    random.shuffle(data)
    print("train dataset size:", len(data))
    
    # save train data
    json.dump(data, open(f"{save_dir}/{method}_train.json", "w"), indent=2)
    
    # if os.path.exists(f"{save_dir}/test_data.json"):
    #     return
    
    test_data = []
    for item in unsafe_test:
        category, unsafe_concept, image_fname = item
        image_fname = image_fname.replace(image_folder, "")
        test_data.append(
            {
                "image_fname": image_fname,
                "concept": unsafe_concept,
                "label": "Unsafe"
            }
        )
    for item in safe_test:
        test_data.append(
            {
                "image_fname": item["image_filename"].replace(image_folder, ""),
                "concept": item["concept"],
                "label": "Safe"
            }
        )
    
    json.dump(test_data, open(f"{save_dir}/test_data.json", "w"), indent=2)
    print("test data size:", len(test_data))


def build_finetuning_dataset_with_contexts(method, train_test_split=0.8, save_dir="data/contexts"):
    
    def get_conversation(method, image_fname, prompt, label, rejected_label, concept, prompt_type):
        
        label_formatting = {
            "Safe": "Yes. The image shows [concept], which is considered harmless.",
            "Unsafe": "No. The image shows [concept], which is considered harmful."
        }
        
        concept = concept.lower()
        
        if method == "sft":
            response = label_formatting[label].replace("[concept]", concept)
            prompt = prompt.replace("[unsafe concept]", concept)
            return {
                    "image": image_fname,
                    "conversations":[
                        {"from": "user", "value": "<image>\n"+prompt},
                        {"from": "assistant", "value": response}
                    ]
                }
        elif method == "dpo":
            chosen_response = label_formatting[label].replace("[concept]", concept)
            rejected_response = label_formatting[rejected_label].replace("[concept]", concept)
            prompt = prompt.replace("[unsafe concept]", concept)
            return {
                "image": image_fname,
                "prompt": "<image>\n"+prompt,
                "chosen": chosen_response,
                "rejected": rejected_response
            }
        elif method == "ppo":
            if "[unsafe concept]" in prompt:
                prompt = prompt.replace("[unsafe concept]", concept)
                
            return {
                "image": image_fname,
                "label": label,
                "concept": concept,
                "type": prompt_type,
                "conversations": [{"from": "human", "value": "<image>\n"+prompt},
                                    {"from": "gpt", "value": ""}]
            }

    image_root = os.path.join(os.path.dirname(abspath), "data/UnsafeConcepts")
    print("image root:", image_root)
    image_folder = os.path.abspath("data")
    
    unsafe_train, unsafe_test = UnsafeConcept_train_test_split(split_ratio=train_test_split, image_root=image_root)
    
    data = []
    
    data_dir = os.path.join(save_dir, "generated_questions.json")
    context_data = json.load(open(data_dir, "r"))
    
    concept2questions = {}
    for item in context_data:
        category = item["category"]
        unsafe_concept = item["unsafe concept"]
        output_text = item["output"]
        safe_match = re.search(r"Safe Context:\s*(.+?)\s*(?=Unsafe Context:)", output_text, re.DOTALL)
        unsafe_match = re.search(r"Unsafe Context:\s*(.+)", output_text, re.DOTALL)

        safe_question = safe_match.group(1).strip() if safe_match else None
        unsafe_question = unsafe_match.group(1).strip() if unsafe_match else None
        # save safe questions
        concept2questions[unsafe_concept] = (safe_question, unsafe_question)
        # save unsafe questions
    
    for item in unsafe_train:
        category, unsafe_concept, image_fname = item
        image_fname = image_fname.replace(image_folder, "")

        # safe contexts questions
        prompt_type = "YORN"
        safe_context_prompt = concept2questions[unsafe_concept][0]
        data.append(get_conversation(method, image_fname, safe_context_prompt, label="Safe", rejected_label="Unsafe", concept=unsafe_concept, prompt_type=prompt_type))

        # unsafe contexts questions
        unsafe_context_prompt = concept2questions[unsafe_concept][1]
        data.append(get_conversation(method, image_fname, unsafe_context_prompt, label="Unsafe", rejected_label="Safe", concept=unsafe_concept, prompt_type=prompt_type))

    random.shuffle(data)
    print("train dataset size:", len(data))
    
    # save train data
    json.dump(data, open(f"{save_dir}/{method}_train.json", "w"), indent=2)

    for context_setting in ["safe", "unsafe"]:
        test_data = []
        for item in unsafe_test:
            category, unsafe_concept, image_fname = item
            image_fname = image_fname.replace(image_folder, "")

            if context_setting == "safe":
                safe_context_prompt = concept2questions[unsafe_concept][0]
                test_data.append(
                    {
                        "image_fname": image_fname,
                        "concept": unsafe_concept,
                        "prompt": safe_context_prompt,
                        "label": "Safe"
                    }
                )
            else:
                unsafe_context_prompt = concept2questions[unsafe_concept][1]

                test_data.append(
                    {
                        "image_fname": image_fname,
                        "concept": unsafe_concept,
                        "prompt": unsafe_context_prompt,
                        "label": "Unsafe"
                    }
                )
        
        json.dump(test_data, open(f"{save_dir}/test_data_{context_setting}.json", "w"), indent=2)
        print("test data size:", len(test_data))

if __name__ == "__main__":

    methods = ["sft", "dpo", "ppo"]
    for method in methods:
        print(f"Building dataset for {method}...")
        build_finetuning_dataset_with_concepts(method, train_test_split=0.8, save_dir=abspath.replace("RLHF", "data/rlhf"))
        # build_finetuning_dataset_with_contexts(method, train_test_split=0.8, save_dir=abspath.replace("RLHF", "data/contexts"))