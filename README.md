# HuPER: A Human-Inspired Framework for Phonetic Perception

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