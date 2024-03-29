import os
import torch
import platform
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


def init_model():
    # model_path="/work/cache/model/QWen-after-pretrain/0907-09-5nodes-QWen-after-pretrain-0906-full-3epoch"
    # model_path="/work/share/public/weights/Qwen-7B-0925-modelscope/chat/qwen/Qwen-7B-Chat"
    model_path="/work/share/cmj/gungzhou/model/0926-QWen-canton-chat-full-3e-5-3epoch-925-v2"
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


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(Fore.YELLOW + Style.BRIGHT + "欢迎使用QWen大模型，输入进行对话，clear 清空历史，CTRL+C 中断生成，stream 开关流式生成，exit 结束。")
    return []


def main(stream=True):
    model, tokenizer = init_model()
    # init_model()

    messages = clear_screen()
    while True:
        prompt = input(Fore.GREEN + Style.BRIGHT + "\n用户：" + Style.NORMAL)
        if prompt.strip() == "exit":
            break
        if prompt.strip() == "clear":
            messages = clear_screen()
            continue
        print(Fore.CYAN + Style.BRIGHT + "\nQWen：" + Style.NORMAL, end='')
        if prompt.strip() == "stream":
            stream = not stream
            print(Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"), end='')
            continue
        messages=prompt
        if stream:
            position = 0
            try:
                for response in model.chat_stream(tokenizer=tokenizer, query=messages, history=None):                    
                    print(response[position:], end='', flush=True)
                    position = len(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            except KeyboardInterrupt:
                pass
            print()
        else:
            response,history = model.chat(tokenizer, messages, history=None)
            print(response)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        # messages = []
        # messages.append({"role": "assistant", "content": response})
        # print(messages)

    print(Style.RESET_ALL)


if __name__ == "__main__":
    main()