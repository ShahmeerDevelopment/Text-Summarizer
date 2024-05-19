from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset("multi_news")
print(f"Features: {dataset['train'].column_names}")

dataset

sample = dataset["train"][1]
print(f"""Document (excerpt of 2000 characters, total length: {len(sample["document"])}):""")
print(sample["document"][:2000])
print(f'\nSummary (length: {len(sample["summary"])}):')
print(sample["summary"])

from transformers import BartForConditionalGeneration, AutoTokenizer

model_ckpt = "sshleifer/distilbart-cnn-6-6"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BartForConditionalGeneration.from_pretrained(model_ckpt)

d_len = [len(tokenizer.encode(s)) for s in dataset["validation"]["document"]]
s_len = [len(tokenizer.encode(s)) for s in dataset["validation"]["summary"]]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
axes[0].hist(d_len, bins=20, color="C0", edgecolor="C0")
axes[0].set_title("Document Token Length")
axes[0].set_xlabel("Length")

axes[0].set_ylabel("Count")
axes[1].hist(s_len, bins=20, color="C0", edgecolor="C0")
axes[1].set_title("Summary Token Length")
axes[1].set_xlabel("Length")
plt.tight_layout()
plt.show()

def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch["document"], max_length=1024, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch["summary"], max_length=256, truncation=True)
        
    return {"input_ids": input_encodings["input_ids"], 
           "attention_mask": input_encodings["attention_mask"], 
           "labels": target_encodings["input_ids"]}

dataset_tf = dataset.map(convert_examples_to_features, batched=True)

columns = ["input_ids", "labels", "attention_mask"]
dataset_tf.set_format(type="torch", columns=columns)

from transformers import DataCollatorForSeq2Seq
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir='bart-multi-news', num_train_epochs=1, warmup_steps=500, 
                                  per_device_train_batch_size=1, per_device_eval_batch_size=1, 
                                  weight_decay=0.01, logging_steps=10, push_to_hub=False, 
                                  evaluation_strategy='steps', eval_steps=500, save_steps=1e6, 
                                  gradient_accumulation_steps=16)

trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer, 
                  data_collator=seq2seq_data_collator, 
                  train_dataset=dataset_tf["train"], 
                  eval_dataset=dataset_tf["validation"])

trainer.train()

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_text = dataset["test"][1]["document"]
reference = dataset["test"][1]["summary"]

input_ids = tokenizer(sample_text, max_length=1024, truncation=True, 
                   padding='max_length', return_tensors='pt').to(device)
summaries = model.generate(input_ids=input_ids['input_ids'], 
                           attention_mask=input_ids['attention_mask'], 
                           max_length=256)

decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                      clean_up_tokenization_spaces=True) 
                    for s in summaries]

print("Document:")
print(sample_text)
print("\nReference Summary:")
print(reference)
print("\nModel Summary:")
print(decoded_summaries[0])

from transformers import BartForConditionalGeneration, AutoTokenizer

model_ckpt = "sshleifer/distilbart-cnn-6-6"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BartForConditionalGeneration.from_pretrained(model_ckpt)

save_directory = "/kaggle/working/"  
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)



