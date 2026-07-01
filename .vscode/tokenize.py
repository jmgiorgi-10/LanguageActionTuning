# Tokenize the dataset

from transformers import AutoTokenizer

model_checkpoint = "xlm-roberta-base" # Excellent base for multilingual (ES/EN) tasks
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

from datasets import load_dataset

# Load the local jsonl file directly into a dataset object
dataset = load_dataset("json", data_files="synthetic_dataset.jsonl")

# Extract unique intents dynamically from your dataset
unique_intents = sorted(dataset.unique("intent"))

# Create lookup dictionaries
intent2id = {intent: idx for idx, intent in enumerate(unique_intents)}
id2intent = {idx: intent for idx, intent in enumerate(unique_intents)}

def preprocess_function(examples):
    prompts = examples["user_query"]
    # Map the string intents to integers using our dictionary
    intent_labels = [intent2id[intent] for intent in examples["intent"]]

    formatted_texts = []
    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt}
        ]
        # Format with the model's official template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_texts.append(text)

    # Tokenize the user prompts
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=512, 
        padding=False,
    )

    # Assign the integer IDs to labels
    tokenized["labels"] = intent_labels
    
    return tokenized

tokenized_dataset = dataset.map(preprocess_function, batched=True)