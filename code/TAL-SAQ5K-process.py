import json

output = []

with open('TAL-SCQ5K-CN_train.jsonl','r',encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        problem = data["problem"]
        options = []
        for option_group in data["answer_option_list"]:
            for option in option_group:
                options.append(f"{option['aoVal']}. {option['content'].strip()}")
        problem_and_option = f"{problem}\n" #+ "\n".join(options)
        ans_analysis = data["answer_analysis"][0]
        correct_answer_value = data["answer_value"]
        # 初始化变量以存储最终的答案
        final_answer = None
        # 遍历选项列表，查找与正确答案值匹配的选项
        for option_group in data["answer_option_list"]:
            for option in option_group:
                if option["aoVal"] == correct_answer_value:
                    final_answer = option["content"].strip().replace('$','')
                    break
        out_dict = {}
        out_dict['instruction'] = '你是一位奥林匹克数学竞赛冠军，请回答给定的数学问题，最终答案应当是一个浮点数。问题如下\n'
        #out_dict['instruction'] = '你是一位奥林匹克数学竞赛冠军，对于给定的数学问题，请直接返回答案，不需要输出思考过程，除了最终的答案以外，不允许输出任何信息。问题如下\n'
        out_dict['input'] = problem_and_option #+ '\n思考过程如下：\n' + ans_analysis
        out_dict['output'] = final_answer
        output.append(out_dict)
'''
with open('TAL-SCQ5K-CN_test.jsonl','r',encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        problem = data["problem"]
        options = []
        for option_group in data["answer_option_list"]:
            for option in option_group:
                options.append(f"{option['aoVal']}. {option['content'].strip()}")
        problem_and_option = f"{problem}\n" #+ "\n".join(options)
        ans_analysis = data["answer_analysis"][0]
        correct_answer_value = data["answer_value"]
        # 初始化变量以存储最终的答案
        final_answer = None
        # 遍历选项列表，查找与正确答案值匹配的选项
        for option_group in data["answer_option_list"]:
            for option in option_group:
                if option["aoVal"] == correct_answer_value:
                    final_answer = option["content"].strip().replace('$','')
                    break
        #print(problem_and_option,ans_analysis,correct_answer_value,final_answer)
        out_dict = {}
        out_dict['instruction'] = '你是一位奥林匹克数学竞赛冠军，请回答给定的数学问题。问题如下\n'
        #out_dict['instruction'] = '你是一位奥林匹克数学竞赛冠军，请回答给定的数学问题，你的回答应该包括问题分析和最终答案，最终答案应当是一个浮点数。问题如下\n'
        #out_dict['instruction'] = '你是一位奥林匹克数学竞赛冠军，对于给定的数学问题，请直接返回答案，不需要输出思考过程，除了最终的答案以外，不允许输出任何信息。问题如下\n'
        out_dict['input'] = problem_and_option + '\n思考过程如下：\n' + ans_analysis
        out_dict['output'] = final_answer
        output.append(out_dict)'''


with open('/work/cache/paper_zhuanli_game/aaai2024comp/math_dataset/TAL-SAQ5K/TAL-SAQ5K-ans.json', 'w', encoding='utf8') as file:
    json.dump(output, file, ensure_ascii=False, indent=4)