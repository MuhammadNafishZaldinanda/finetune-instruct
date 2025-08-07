# 1. Import Dependencies
from unsloth import FastLanguageModel # Menggunakan framework Unsloth
from transformers import TrainingArguments
from datasets import Dataset
import torch
import pandas as pd
import re

# 2. Load Pretrained Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B",
    max_seq_length = 2048,   # Context length 
    load_in_4bit = True,     # Pemanfaatan QLoRA yaitu menggunakan basemodel kuantisasi precision 4-bit
    load_in_8bit = False,    
    full_finetuning = False,
    dtype = None
)

# 3. Preprocessing Dataset
data_train = pd.read_csv("data_train.csv")
data_eval = pd.read_csv("data_eval.csv")

formatted_train = []
formatted_eval = []

# Preprocessing data_train
for _, row in data_train.iterrows():
    # Mengubah dataset ke format chat berbasis "role" dan "content" 
    chat_list = [
        { "role": "user", "content": row["prompt"] },
        { "role": "assistant", "content": row["output"] }
    ]
    try:
        # Konversi ke format chat template dari model Qwen3
        formatted = tokenizer.apply_chat_template(chat_list, tokenize=False)
        formatted_train.append(formatted)
    except Exception as e:
        print("Error formatting row (train):", row)
        print("Exception:", e)

# Preprocessing data_eval
for _, row in data_eval.iterrows():
    # Mengubah dataset ke format chat berbasis "role" dan "content" 
    chat_list = [
        { "role": "user", "content": row["prompt"] },
        { "role": "assistant", "content": row["output"] }
    ]
    try:
        # Konversi ke format chat template dari model Qwen3
        formatted = tokenizer.apply_chat_template(chat_list, tokenize=False)
        formatted_eval.append(formatted)
    except Exception as e:
        print("Error formatting row (eval):", row)
        print("Exception:", e)

# Konversi list ke Dataset HF
train_dataset = Dataset.from_dict({"text": formatted_train})
eval_dataset = Dataset.from_dict({"text": formatted_eval})

# 4. LoRA configuration
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,           # LoRA rank, semakin besar rank meningkatkan kapasitas dan memori
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", # Layer attention dan MLP yang jadi target modules
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Pilih LoRA alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)

# 5. Traininig Argument Config
from transformers import TrainingArguments
args = TrainingArguments(
    per_device_train_batch_size = 2,          # Ukuran batch per-device saat training
    per_device_eval_batch_size = 2,           # Ukuran batch per-device saat evaluasi
    gradient_accumulation_steps = 4,          # Akumulasi gradien untuk mensimulasikan batch besar
    max_seq_length = 1024,                    # Sesuaikan dengan hasil preprocessing
    warmup_steps = 500,                       # Langkah awal untuk warmup learning rate
    learning_rate = 2e-4,                     # Learning rate awal. Kurangi jika training panjang
    num_train_epochs = 2,                     # Jumlah epoch training penuh
    eval_strategy = "epoch",                  # Evaluasi dilakukan di akhir setiap epoch
    logging_steps = 1,                        # Logging setiap 1 step (tergantung total langkah)
    optim = "adamw_8bit",                     # Optimizer dengan dukungan quantized (8-bit AdamW)
    save_strategy = "epoch",                  # Simpan checkpoint di setiap akhir epoch
    save_total_limit = 2,                     # Maksimal checkpoint yang disimpan
    weight_decay = 0.01,                      # Regularisasi untuk mencegah overfitting
    lr_scheduler_type = "linear",             # Learning rate schedule: turun linear dari awal
    seed = 3407,                              # Seed
    report_to = "wandb",                      # Kirim log ke Weights & Biases (WandB) untuk monitoring 
    output_dir = "checkpoints"                # Direktori untuk menyimpan checkpoint model
    
)

from transformers import EarlyStoppingCallback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=2,    # Berhenti jika 2 evaluasi berturut-turut tidak membaik
    early_stopping_threshold=0.0  # Ambang minimum perbaikan (0 artinya harus benar-benar turun)
)


# 6. Train
from trl import SFTTrainer, SFTConfig         # Memanfaatkan library trl (Supervised Finetuning Trainer)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    dataset_text_field = "text",
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, 
    args = args,
    callbacks=[early_stopping] 
)
trainer.train()

# 7. Save Model
model.save_pretrained("checkpoints/lora_model")
tokenizer.save_pretrained("checkpoints/lora_model")
print("TASK DONE!!!")