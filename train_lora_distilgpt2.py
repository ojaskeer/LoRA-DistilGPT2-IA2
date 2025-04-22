from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Optional: for safety (unused warning suppression)
import torch

# 1. Load AG News dataset
dataset = load_dataset("ag_news")

# 2. Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # Prevent padding issues

# 3. Preprocessing
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize, batched=True)

# 4. Load base model
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# 5. LoRA configuration
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)

# 6. Apply LoRA
model = get_peft_model(base_model, lora_config)

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none",
    save_total_limit=2
)

# 8. Data collator for Causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Not masked language modeling — causal instead
)

# 9. Custom Trainer to handle LoRA + GPT2 properly
class GPT2LoRATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        inputs = {k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask', 'labels']}
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# 10. Initialize trainer
trainer = GPT2LoRATrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"].select(range(1000)),
    eval_dataset=tokenized["test"].select(range(200)),
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 11. Start training
trainer.train()

# 11. Start training
trainer.train()


