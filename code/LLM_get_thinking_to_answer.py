import os
import json
import torch
import platform
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
from fractions import Fraction


def parse_string(s):
    try:
        # Try to convert directly to a float
        return float(s)
    except ValueError:
        # If there is a percentage, remove it and divide by 100
        if '%' in s:
            s = s.replace('%', '')  # Remove the percentage sign
            # Check for mixed numbers in percentages
            if ' ' in s:
                whole_number, fraction_part = s.split(' ')
                whole_number = float(whole_number)
                fraction_value = float(Fraction(fraction_part))
                return (whole_number + fraction_value) / 100
            else:
                try:
                    return float(s) / 100
                except ValueError:
                    pass                
        # If there is a fraction, try to evaluate it
        elif '/' in s:
            try:
                # Handle mixed numbers (e.g., "7 3/4")
                if ' ' in s:
                    whole_number, fraction_part = s.split(' ')
                    whole_number = float(whole_number)
                    fraction_value = float(Fraction(fraction_part))
                    return whole_number + fraction_value
                else:
                    return float(Fraction(s))
            except ValueError:
                pass  # If it fails, do nothing and the original string will be returned
        # If there is 'frac', try to evaluate it assuming it's LaTeX code for a fraction
        elif 'frac' in s:
            try:
                # Extract numerator and denominator using regex
                numerator, denominator = re.findall(r'frac\{(.+?)\}\{(.+?)\}', s)[0]
                return float(numerator) / float(denominator)
            except (ValueError, IndexError):
                pass  # If it fails, do nothing and the original string will be returned
    # If all conversions fail, return None to indicate no conversion could be made
    return s.replace('$','')

def init_model():
    model_path="/work/share/public/weights/Qwen-14B-0925/Qwen-14B-Chat" # "/work/share/public/weights/Qwen-72B-Chat" #"/work/share/public/weights/Qwen-14B-0925/Qwen-14B-Chat"
    print("init model ...")
    print(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer

def main():
    model, tokenizer = init_model()
    output = []
    with open('/aaai2024comp/math_dataset/MATH/train.json','r',encoding='utf8') as file:
        data = json.load(file)
        for line in tqdm(data):
            thinking, question = line['thinking'], line['instruction']
            pre_prompt = '问题：'+question+'\n解题过程：'+thinking+'\n根据上述问题和解题过程，问题的最终答案是多少？请以浮点数的形式直接返回答案。'
            text = question
            messages = pre_prompt + text #+ end_prompt
            res, _ = model.chat(tokenizer, messages, history=[]) 
            #res = parse_string(res)
            temp = {'instruction':question ,'input':'','output':res, 'thinking':thinking}
            output.append(temp)

    with open('/aaai2024comp/train_dataset/math_train_summ.json', 'w', encoding='utf8') as file:
        json.dump(output, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
