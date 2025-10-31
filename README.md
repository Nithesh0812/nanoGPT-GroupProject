nanoGPT Group Project — Shakespeare Character-Level Language Model

This repository contains our **nanoGPT fine-tuning experiments** conducted as part of the group assignment.  
The project explores different hyperparameter configurations of a GPT-like model trained on the **Shakespeare character-level dataset**.

---

 Group Members
| Member | Configuration | Key Hyperparameters | Output Directory |
|:--------|:---------------|:--------------------|:----------------|
| **Nithesh Mudsu** | Small model | `batch_size=8`, `n_layer=4`, `n_head=4`, `n_embd=128`, `dropout=0.1`, `max_iters=1000` | `out/member1/bs64_L4_H4_E128_B8_D0.1_I1000` |
| **Khaja Fasi Ahmed** | Medium model | `batch_size=8`, `n_layer=6`, `n_head=8`, `n_embd=256`, `dropout=0.2`, `max_iters=1000` | `out/member2/bs64_L6_H8_E256_B8_D0.2_I1000` |
| **Priyabrata Behera** | Large model | `batch_size=16`, `n_layer=4`, `n_head=8`, `n_embd=256`, `dropout=0.1`, `max_iters=1000` | `out/member3/bs128_L4_H8_E256_B16_D0.1_I1000` |

---

Execution on Google Colab

1. Open [`nano_gpt-Assignment.ipynb`](./nano_gpt-Assignment.ipynb) in **Google Colab**.  
2. Set runtime to **GPU (CUDA)**:  
   *Runtime → Change Runtime Type → Hardware Accelerator → GPU*  
3. Run all cells sequentially to:
   - Prepare the Shakespeare dataset
   - Train each member’s configuration
   - Generate text samples and checkpoints

4. Training logs, checkpoints, and generated samples will appear in the `out/` and `logs/` directories.

---

Repository Contents
| File / Folder | Description |
|:---------------|:-------------|
| `train.py` | Main training script (single or distributed GPU) |
| `model.py` | GPT model architecture and configuration class |
| `sample.py` | Text generation / sampling script |
| `configurator.py` | Command-line argument handler |
| `data/shakespeare_char/` | Tokenized training and validation data |
| `out/` | Model outputs and checkpoints for each experiment |
| `logs/` | Training logs for plotting and loss tracking |
| `nano_gpt-Assignment.ipynb` | Full Colab notebook (runnable end-to-end) |
| `Report_Group.pdf` | Final written report and analysis |

---

Results Summary
- All three runs successfully trained for ~1000 iterations on CUDA.  
- Each configuration shows different convergence behavior and text coherence.  
- Example generated text excerpts (post-training) are included in the Colab output cells and in the `out/` folders.
- 
Notes
- All models were trained using mixed-precision (AMP) and checkpointed at each evaluation interval.
- Checkpoints are automatically saved under their respective member directories as `ckpt.pt`.

---

Reproducibility
To reproduce any experiment locally or on Colab:
```bash
python train.py \
  --device=cuda \
  --compile=False \
  --dataset=shakespeare_char \
  --out_dir=out/memberX/... \
  --block_size=64 \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=128 \
  --batch_size=8 \
  --dropout=0.1 \
  --max_iters=1000 \
  --eval_interval=250 \
  --eval_iters=200
