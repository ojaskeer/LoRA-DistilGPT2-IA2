# 🚀 LoRA-Based Fine-Tuning of DistilGPT2 on AG News (CPU-Only)

This project demonstrates lightweight fine-tuning of a pretrained DistilGPT2 model using **Low-Rank Adaptation (LoRA)**, implemented on a **CPU-only environment** (Dell laptop) with Hugging Face Transformers and PEFT libraries.

> 📁 Part of IA2 Research Submission — Department of Computer Engineering, KJSCE

---

## 📌 Overview

- **Model**: `distilgpt2` (from Hugging Face)
- **Technique**: LoRA — Low-Rank Adaptation
- **Dataset**: AG News (text classification)
- **Tools**: `transformers`, `peft`, `datasets`, `Trainer API`
- **Device**: Intel-based CPU (no GPU needed)
- **Results**:  
  - Train Loss: `4.65`  
  - Eval Loss: `4.39`  
  - Runtime: ~6 minutes (1 epoch, 1000 samples)

---

## 📦 Setup

> ⚠️ Tested on Python 3.11 — No GPU required

### 1. Clone this repo or unzip the project:
```bash
git clone https://github.com/YOUR_USERNAME/LoRA-DistilGPT2-IA2.git
cd LoRA-DistilGPT2-IA2
```

### 2. Install dependencies:
```bash
pip install transformers datasets peft accelerate
```

> Optional (safer on CPU):  
```bash
pip install torch==2.0.0
```

---

## 🧠 Training the Model

```bash
python train_lora_distilgpt2.py
```

- Model will be fine-tuned using LoRA on ~1000 training samples
- Training artifacts saved in `./results/` and `./logs/`

---

## 📂 Project Structure

```
📦 LoRA-DistilGPT2-IA2
├── train_lora_distilgpt2.py        # Main training script
├── README.md                       # This file
├── results/                        # Saved model weights + training logs
└── logs/                           # Optional: TensorBoard logs (if any)
```

---

## 📊 Sample Output

```
Final Train Loss: 4.65
Eval Loss: 4.39
Training Time: ~6m 17s
```

---

## 📝 Citation (APA)
Hu, E., Shen, Y., Wallis, P., et al. (2021). **LoRA: Low-Rank Adaptation of Large Language Models**. arXiv:2106.09685.

---

## 📧 Author

**Ojas Keer**  
Dept. of Computer Engineering, KJSCE  
ojas.keer@somaiya.edu

---

## 📌 Note

This was implemented without a GPU. LoRA enabled fine-tuning by updating <1M parameters in total, proving effective on standard hardware. Refer to the paper in `/doc` for more details.
# LoRA-DistilGPT2-IA2
