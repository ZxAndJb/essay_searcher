from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_path = "../model/deepseek-math-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

quantization_config = GPTQConfig(
     bits=4,
     tokenizer=tokenizer,
     group_size=128,
     dataset="wikitext2",
     desc_act=False,
)

quantized_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=quantization_config)

quantized_model_dir = "../model/deepseek-math-7b-instruct-gptq"
quantized_model.save_pretrained(quantized_model_dir)
tokenizer.save_pretrained(quantized_model_dir)
print("Finish quantization")