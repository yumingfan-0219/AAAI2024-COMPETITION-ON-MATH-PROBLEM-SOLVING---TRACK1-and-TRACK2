import json
#已经放进来的数据集：ape210k GSM8K Math23K 
#需要构建答案才能放进来的数据集：school_math_0.25M MATH 

def read_json_objects(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_objects = []
        current_object = ""
        bracket_count = 0
        for line in file:
            bracket_count += line.count('{')
            bracket_count -= line.count('}')
            current_object += line
            if bracket_count == 0 and current_object.strip():
                json_objects.append(json.loads(current_object))
                current_object = ""
    return json_objects

pre_prompt = '你是一位奥林匹克数学竞赛冠军，对于给定的数学问题，请直接返回答案，不需要输出思考过程，除了最终的答案以外，不允许输出任何信息。问题如下'


######################################################################################################################################################
output = []
with open('/work/cache/paper_zhuanli_game/aaai2024comp/math_dataset/ape210k-master/data/train.ape.json','r',encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        temp = {'instruction':pre_prompt + data['original_text'],'input':'','output': data['ans']}
        output.append(temp)

with open('/work/cache/paper_zhuanli_game/aaai2024comp/math_dataset/ape210k-master/data/test.ape.json','r',encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        temp = {'instruction':pre_prompt + data['original_text'],'input':'','output': data['ans']}
        output.append(temp)

with open('/work/cache/paper_zhuanli_game/aaai2024comp/math_dataset/ape210k-master/data/valid.ape.json','r',encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        temp = {'instruction':pre_prompt + data['original_text'],'input':'','output': data['ans']}
        output.append(temp)   

with open('/work/cache/paper_zhuanli_game/aaai2024comp/train_dataset/ape210k.json', 'w', encoding='utf-8') as file:
    json.dump(output, file, ensure_ascii=False, indent=4)   


######################################################################################################################################################
output = []


with open('/work/cache/paper_zhuanli_game/aaai2024comp/math_dataset/GSM8K_zh.json','r',encoding='utf-8') as gsmfile:
    datas = json.load(gsmfile)
    for data in datas:
        temp = {'instruction':pre_prompt + data['question_zh'],'input':'','output': data['answer_only']}
        output.append(temp)

with open('/work/cache/paper_zhuanli_game/aaai2024comp/train_dataset/GSM8K.json', 'w', encoding='utf-8') as file:
    json.dump(output, file, ensure_ascii=False, indent=4)        

######################################################################################################################################################         
output = []

json_objects = read_json_objects('/work/cache/paper_zhuanli_game/aaai2024comp/math_dataset/Math23K/math23k_train.json')
for data in json_objects:
    temp = {'instruction':pre_prompt + data['original_text'],'input':'','output': data['ans']}
    output.append(temp)

json_objects = read_json_objects('/work/cache/paper_zhuanli_game/aaai2024comp/math_dataset/Math23K/math23k_test.json')
for data in json_objects:
    temp = {'instruction':pre_prompt + data['original_text'],'input':'','output': data['ans']}
    output.append(temp)     

with open('/work/cache/paper_zhuanli_game/aaai2024comp/train_dataset/math_23k.json', 'w', encoding='utf-8') as file:
    json.dump(output, file, ensure_ascii=False, indent=4)        


exit()

output = []

with open('/work/cache/paper_zhuanli_game/aaai2024comp/math_dataset/MATH/train.json','r',encoding='utf-8') as MATHfile:
    MATHdata = json.load(MATHfile)
    for data in MATHdata:
        temp = {'instruction':pre_prompt + data['problem'],'input':'','output': data['answer_only'], 'thinking': solution['answer_only']}
        output.append(temp)        

with open('/work/cache/paper_zhuanli_game/aaai2024comp/train_dataset/math.json', 'w', encoding='utf-8') as file:
    json.dump(output, file, ensure_ascii=False, indent=4)       