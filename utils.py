import numpy as np
import pandas as pd
import base64
from PIL import Image
from openai import OpenAI

rule_dict = {
    'llava_bench_conv': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'},  # noqa: E501
    'llava_bench_detail': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'},  # noqa: E501
    'llava_bench_complex': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'}  # noqa: E501
}

def judge_eval(prompt, **gen_kwargs):
    
    model_name = gen_kwargs["model_name"]
    client = OpenAI(api_key=gen_kwargs["api_key"])

    system_prompt = gen_kwargs["system_prompt"]

    messages = [{"role": "system",
                 "content": system_prompt
        
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]
                }
                ]
    
    completion = client.chat.completions.create(
        model=model_name,
        store=True,
        messages=messages,
        temperature=gen_kwargs["temperature"],
        max_tokens=gen_kwargs["max_tokens"]
    )
    output = completion.choices[0].message.content
    return output
    

def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise ValueError("Invalid score format")
    except Exception as e:
        raise ValueError("Invalid score format")


def build_prompt(line):
    cap_str = line['caption']
    question = line['question']
    ans1 = line['gpt4_ans']
    ans2 = line['prediction']
    category = 'llava_bench_' + line['category']
    rule = rule_dict[category]
    role, prompt = rule['role'], rule['prompt']

    content = (f'[Context]\n{cap_str}\n\n'
               f'[Question]\n{question}\n\n'
               f'[{role} 1]\n{ans1}\n\n[End of {role} 1]\n\n'
               f'[{role} 2]\n{ans2}\n\n[End of {role} 2]\n\n'
               f'[System]\n{prompt}\n\n')
    return content
