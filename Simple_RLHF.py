import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

if __name__ == "__main__":
    ppo_config = PPOConfig(model_name='gpt2', learning_rate=1e-4,batch_size=4,mini_batch_size=2)
    dataset = load_dataset("HuggingFaceH4/cherry_picked_prompts", split="train")
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token


    def tokenize(sample):
        sample['input_ids'] = tokenizer.encode(sample['prompt'], add_special_tokens=False, return_tensors='pt').squeeze()
        return sample

    def collate_fn(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    dataset = dataset.map(tokenize, batched=False, remove_columns=['completion', 'meta'])
    dataset.set_format(type="torch")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
    rl_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
    ppo_trainer = PPOTrainer(ppo_config,rl_model, ref_model,tokenizer,dataset = dataset,data_collator = collate_fn)
    generation_kwargs = {
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "temperature": 0.8,
        "max_new_tokens": 20
    }

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        questions = batch['input_ids']

        responses = []
        for q in questions:
            r = ppo_trainer.generate(q, **generation_kwargs)
            responses.append(r.squeeze())

        rewards = [torch.ones(1)] * 4

        stats = ppo_trainer.step(questions, responses, rewards)