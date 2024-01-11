import json


with open('/work/cache/paper_zhuanli_game/aaai2024comp/train_dataset/math_train_summ.json','r',encoding='utf-8') as file:
    math_train = json.load(file)
with open('/work/cache/paper_zhuanli_game/aaai2024comp/train_dataset/math_test_summ.json','r',encoding='utf-8') as file_test:
    math_test = json.load(file_test)

math = math_train + math_test
output = []

for data in math:
    out_dict = {}
    out_dict['instruction'] = '你是一位奥林匹克数学竞赛冠军，请回答给定的数学问题。问题如下\n'
    out_dict['input'] = data['instruction'] + '\n问题分析：'+ data['thinking']
    out_dict['output'] = '\n最终答案为'+data['output']
    output.append(out_dict)


with open('/work/cache/paper_zhuanli_game/aaai2024comp/train_dataset/math_cot_input.json', 'w', encoding='utf8') as file:
    json.dump(output, file, ensure_ascii=False, indent=4)