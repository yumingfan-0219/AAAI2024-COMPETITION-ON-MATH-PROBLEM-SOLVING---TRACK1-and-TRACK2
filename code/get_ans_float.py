import os
import json
import torch
import platform
import time
import re
from tqdm import tqdm
from fractions import Fraction
def parse_string(s):
    s = s.replace('$','').replace('\\', '')
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
            s = s.replace('(', '').replace(')', '')
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



with open('/work/cache/paper_zhuanli_game/aaai2024comp/result/TAL_SAQ7K_CN_prediction_14b_all.json','r',encoding='utf-8') as f:
    my_dict = json.load(f)
    for key in my_dict:
        my_dict[key] = parse_string(my_dict[key])

with open('/work/cache/paper_zhuanli_game/aaai2024comp/result/TAL_SAQ7K_CN_prediction.json', 'w',encoding='utf-8') as file:
    json_str = json.dumps(my_dict, indent=4)
    file.write(json_str + '\n')        