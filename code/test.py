import json
from fractions import Fraction
import re

'''
with open('/work/cache/paper_zhuanli_game/aaai2024comp/math_dataset/MATH/test_summ.json','r',encoding='utf-8') as file:
    output = []
    data = json.load(file)
    for i in data:
        i['output'] = str(i['output'])
        output.append(i)

with open('/work/cache/paper_zhuanli_game/aaai2024comp/train_dataset/math_test_summ.json', 'w', encoding='utf8') as file:
    json.dump(output, file, ensure_ascii=False, indent=4)'''


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

def extract_first_number(s):
    # 使用正则表达式匹配第一个整数或浮点数
    number_pattern = r'-?\d+(\.\d+)?'
    match = re.search(number_pattern, s)
    
    # 如果找到数字，转换并返回它；否则返回None
    if match:
        return float(match.group())
    else:
        return 'not find'

import re
from fractions import Fraction

def convert_numerical_expressions(s):
    # 定义转换函数
    def replace_with_float(match):
        text = match.group(0)
        if '%' in text:
            return str(float(text.replace('%', '')) / 100)
        elif '/' in text:
            try:
                if ' ' in text:
                    whole_number, fraction_part = text.split(' ')
                    return str(float(whole_number) + float(Fraction(fraction_part)))
                else:
                    return str(float(Fraction(text)))
            except ValueError:
                return text
        elif 'frac' in text:
            try:
                numerator, denominator = re.findall(r'frac\{(.+?)\}\{(.+?)\}', text)[0]
                return str(float(numerator) / float(denominator))
            except (ValueError, IndexError):
                return text
        else:
            return text

    # 使用正则表达式匹配数值表达式，并进行转换
    pattern = r'-?\d+\s+\d+/\d+|-?\d+/\d+|\d+%|frac\{.+?\}\{.+?\}'
    return re.sub(pattern, replace_with_float, s)

count = 0
with open('/work/cache/paper_zhuanli_game/aaai2024comp/result/TAL_SAQ7K_CN_prediction_skymath_summ.json', 'r') as file:
    f_cn = open('/work/cache/paper_zhuanli_game/aaai2024comp/result/TAL_SAQ7K_CN_prediction.json', 'r')
    output_f_cn = json.load(f_cn)
    output = json.load(file)
    
    for key in output:
        output[key] = extract_first_number(convert_numerical_expressions(output[key]))
        if output[key]=='not find':
            count += 1
            output[key] = output_f_cn[key]
            
print(count)

with open('/work/cache/paper_zhuanli_game/aaai2024comp/result_submit/TAL_SAQ7K_CN_prediction.json', 'w', encoding='utf-8') as file: 
    json.dump(output, file, ensure_ascii=False, indent=4)
