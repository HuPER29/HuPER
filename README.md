# HuPER: A Human-Inspired Framework for Phonetic Perception

[![arXiv](https://img.shields.io/badge/arXiv-2602.01634-b31b1b.svg)](https://arxiv.org/abs/2602.01634)
[![🤗 Model](https://img.shields.io/badge/🤗%20Model-huper_recognizer-yellow.svg)](https://huggingface.co/huper29/huper_recognizer)
[![🤗 Model](https://img.shields.io/badge/🤗%20Model-huper_corrector-yellow.svg)](https://huggingface.co/huper29/huper_corrector)
[![🤗 Dataset](https://img.shields.io/badge/🤗%20Dataset-huper--clean100--proxyphones-yellow.svg)](https://huggingface.co/datasets/huper29/huper-clean100-proxyphones)

HuPER is a human-inspired framework that models phonetic perception as adaptive inference integrating acoustic–phonetic evidence and linguistic knowledge.

- Paper (arXiv): https://arxiv.org/abs/2602.01634  
- Code: this repo  
- Pretrained models & data: Hugging Face (links below)

---

## Released artifacts (Hugging Face)

- **HuPER Recognizer (phone recognition model)**  
  https://huggingface.co/huper29/huper_recognizer

- **HuPER Corrector (phoneme→phone correction model)**  
  https://huggingface.co/huper29/huper_corrector

- **Training dataset (Clean100 proxyphones)**  
  https://huggingface.co/datasets/huper29/huper-clean100-proxyphones  
---

## Quickstart

### 1) Clone this repo
```bash
git clone https://github.com/HuPER29/HuPER.git
cd HuPER
```

### 2) Usage examples

See runnable examples under:

-   `notebooks/`
    

### 3) HuPER Corrector: minimal inference snippet

The Corrector takes (1) a canonical phoneme sequence (e.g., ARPAbet, space-separated) and (2) discrete audio tokens, and predicts edit operations to better match realized phones.

```python
import os, sys
from huggingface_hub import snapshot_download

repo_dir = snapshot_download("huper29/huper_corrector")
sys.path.append(repo_dir)

from edit_seq_speech.inference import PhonemeCorrectionInference

ckpt_path  = os.path.join(repo_dir, "model.safetensors")
vocab_path = os.path.join(repo_dir, "edit_seq_speech/config/vocab.json")

infer = PhonemeCorrectionInference(
    checkpoint_path=ckpt_path,
    vocab_path=vocab_path,
)

wav_path = "your.wav"
text = "AY R OW T AH L EH T ER"   # space-separated ARPAbet
final_phns, log = infer.predict(wav_path, text)
print(final_phns)
```

---

## Repository layout

-   `wavlm_ft/` — training / fine-tuning code for the phone recognizer
    
-   `correction/` — training code for the phoneme→phone Corrector
    
-   `notebooks/` — end-to-end usage examples and demos
    

---

## Citation

If you use HuPER, please cite:

```bibtex
@article{guo2026huper,
  title   = {HuPER: A Human-Inspired Framework for Phonetic Perception},
  author  = {Guo, Chenxu and Lian, Jiachen and Liu, Yisi and Huang, Baihe and Narayanan, Shriyaa and Cho, Cheol Jun and Anumanchipalli, Gopala},
  journal = {arXiv preprint arXiv:2602.01634},
  year    = {2026}
}