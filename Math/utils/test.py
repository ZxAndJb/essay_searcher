from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial

def concat_message(messages, tokenizer):
    # the role sequence should be system, user, assistant
    message_text = ''
    for m in messages:
        if m['role'] == "user":
            message_text += "<|user|>\n" + m["content"].strip() + "\n"
        elif m['role'] == "assistant":
            message_text += "<|assistant|>\n" + m["content"].strip() + tokenizer.eos_token +"\n"
        elif m["role"] == "system":
            message_text += "<|system|>\n" + m["content"].strip() + "\n"
        else:
            raise ValueError(f"Invalid role {m['role']}")
    return message_text


def encode_message(example, tokenizer, max_seq_len, mask_prompt = True, use_dafault_template = False):
    if "messages" not in example.keys():
        raise ValueError("messages field must exist!")
    messages = example['messages']
    if use_dafault_template:
        text = tokenizer.apply_chat_template(messages, max_length = max_seq_len, truncation = True ,return_tensors="pt")
    else:
        text = concat_message(messages, tokenizer).strip()
        text = tokenizer(text, return_tensors = "pt",max_length = max_seq_len, truncation = True)

    input_ids = text.input_ids
    labels = input_ids.clone()
    

    # mask the loss of user and system part
    # mask the answer in the message, because the answer should be given by the python interpreter

    if mask_prompt:
        filter_message = [x for x in messages if x['role']!='assistant']
        mask_message = concat_message(filter_message,tokenizer) + "<|assistant|>\n"
        mask_len = tokenizer(mask_message,return_tensors = "pt",max_length = max_seq_len, truncation = True).input_ids.shape[1]
        labels[:, :mask_len] = -100
        for message in messages:
            # mask the part inside the '''output\n '''
            if message['role'] == 'assistant':
                solution = message['content'].strip()
                output_start = solution.find("```output\n")
                output_end = solution.rfind("```\n")
                if output_start == -1 or output_end == -1:
                    raise ValueError('The form of the message is incorrect')
                output_start_id = tokenizer(mask_message + solution[:output_start] ,return_tensors = "pt",max_length = max_seq_len, truncation = True).input_ids.shape[1]
                output_end_id = tokenizer(mask_message + solution[:output_end] ,return_tensors = "pt",max_length = max_seq_len, truncation = True).input_ids.shape[1]
                labels[:, output_start_id:output_end_id] = -100
            
    return {
        'input_ids':input_ids.flatten(),
        'attention_mask': text.attention_mask.flatten(),
        'labels': labels.flatten(),
    }
               
modelpath = "../model/deepseek-math-7b-instruct"
datasetpath = "../dataset/MuMath_Code"
tokenizer = AutoTokenizer.from_pretrained(modelpath)

train_dataset = load_dataset(datasetpath)
encode_function = partial(encode_message, tokenizer = tokenizer, max_seq_len= 2048)
lm_dataset = train_dataset.map(encode_function, batched=False,
                               remove_columns=[name for name in train_dataset.column_names if name not in ["input_ids", "labels", "attention_mask"]])

print(lm_dataset[0])


