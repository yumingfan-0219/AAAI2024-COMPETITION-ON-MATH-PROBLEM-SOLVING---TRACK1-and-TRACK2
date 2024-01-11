import os
import re
import json
import torch
import platform
import openpyxl
from openpyxl import Workbook
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
from fractions import Fraction


def init_model():
    model_path = "/work/cache/model/aaai24-math/14B-TAL-SAQ5K-math-ape210k-math23k-GSM8K/1224-13-5nodes"
    # "/work/cache/model/aaai24-math/14B-TAL-SAQ5K/1224-00-5nodes"
    # "/work/cache/model/aaai24-math/14B-TAL-SAQ5K-cot-math-cot/1224-16-5nodes"
    # "/work/share/public/weights/Qwen-14B-0925/Qwen-14B-Chat"
    # /checkpoint-2100" 
    # /work/share/public/weights/Qwen-72B-Chat /work/share/public/weights/Qwen-14B-0925/Qwen-14B-Chat
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
                return float(s) / 100
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


def main():
    model, tokenizer = init_model()
    result_dict = {}
    count = 0
    with open('/work/cache/paper_zhuanli_game/aaai2024comp/TAL-SAQ7K-CN.jsonl','r',encoding='utf8') as file:
        for line in tqdm(file):
            count+=1
            q_id, question = json.loads(line)['queId'], json.loads(line)['problem']
            #pre_prompt = '你是一位奥林匹克数学竞赛冠军，请回答给定的数学问题，你的回答应该包括问题分析和最终答案，最终答案应当是一个浮点数。问题如下\n'
            pre_prompt = '你是一位奥林匹克数学竞赛冠军，对于给定的数学问题，请直接返回答案，不需要输出思考过程，除了最终的答案以外，不允许输出任何信息。问题如下\n'
            #end_prompt = '\n请直接返回答案，不需要输出思考过程，除了最终的答案以外，不允许输出任何信息。'
            messages = pre_prompt + question #+ end_prompt
            res, _ = model.chat(tokenizer, messages, history=[]) 
            #res, _ = model.generate(messages) 
            #res = parse_string(res)
            result_dict[q_id] = res
            if count%200==0:
                print(res)

    with open('/work/cache/paper_zhuanli_game/aaai2024comp/result/TAL_SAQ7K_CN_prediction_ans_only.json', 'w', encoding='utf-8') as file:
        json_str = json.dumps(result_dict, indent=4)
        file.write(json_str + '\n')


if __name__ == "__main__":
    main()
