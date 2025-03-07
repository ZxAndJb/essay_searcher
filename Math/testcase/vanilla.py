import torch
import transformers
import accelerate 
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
import time
from vllm import LLM, SamplingParams
import os
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def parseanswer(text):
    pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(pattern, text)

    return matches[-1] if len(matches) > 0 else None

def compute_accuracy(predictions,labels):
    total = 0
    correct = 0
    for prediction, label in zip(predictions,labels):
        if prediction==label:
            correct +=1
        total+=1
    return correct/total

def main():
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        damp_percent=0.01,
        desc_act=False,
    )     


    model_path = "deepseek-ai/deepseek-math-7b-instruct"
    # model_path = "/home/s2475654/Math/model/deepseek-math-7b-instruct-gptq"

    # BnBConfig = BitsAndBytesConfig(
    #     load_in_4bit = True,
    #     bnb_4bit_compute_dtype = torch.float16,
    #     bnb_4bit_quant_type = "nf4",
    #     bnb_4bit_use_double_quant = True
    # )

    sampling_params = SamplingParams(temperature=0.89645, top_p=0.9, max_tokens=2048)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # model = AutoModelForCausalLM.from_pretrained(model_path,device_map = "auto")
    data = pd.read_csv("train.csv")
    questions, labels = data["problem"], data["answer"]

    llm = LLM(model=model_path, gpu_memory_utilization=0.8)

    inputs = []
    answers = []
    cot_prompt = "Please reason step by step, and put your final answer within \\boxed{}. "
    pot_prompt = "Please reason step by step first and write a python pragram to solve the problem. After that, put your final answer within \\boxed{}. "


    for q in questions:
        message = [
            {
                "role": "user", 
                "content":  q + "\n" + cot_prompt
            }
        ]
        inputs.append(tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True))

    start_time = time.time()
    outputs = llm.generate(inputs, sampling_params)
    end_time = time.time()
    inferencetime = end_time - start_time

    with open('answers.txt', 'w') as f:
        for idx,output in enumerate(outputs):
            answers.append(parseanswer(output.outputs[0].text))
            f.write(output.outputs[0].text + '\n')

    print("predictions: ", answers)
    print("labels: ", labels)
    print(f"Elapsed time: {inferencetime:.2f} seconds")
    accuracy = compute_accuracy(answers, labels)
    print("accuracy: ", accuracy)

if __name__ == '__main__':
     main()