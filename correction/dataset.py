import json
import torch
from torch.utils.data import Dataset

class PhonemeCorrectionDataset(Dataset):
    def __init__(self, jsonl_path, vocab_path, max_len=256):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Load vocab
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            
        self.vocab = vocab_data
        self.insert_to_id = vocab_data['insert_to_id']
        self.op_to_id = vocab_data['op_to_id']
        
        # Create reverse mapping for text phonemes (use insert_to_id, skip <NONE> and <PAD>)
        self.text_vocab = {k: v for k, v in self.insert_to_id.items() if k not in ['<NONE>', '<PAD>']}
        self.pad_idx = self.insert_to_id.get("<PAD>", 1)
        self.max_len = max_len

    def text_to_ids(self, phn_list):
        """Convert phoneme list to IDs using text_vocab."""
        return [self.text_vocab.get(p, self.text_vocab.get("AA", 2)) for p in phn_list]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        op_ids = item['op_ids']
        ins_ids = item['ins_ids']
        audio_tokens = item['audio_tokens']
        text_phonemes = item['text_phonemes']
        
        input_ids = self.text_to_ids(text_phonemes)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "audio_tokens": torch.tensor(audio_tokens, dtype=torch.long),
            "label_op": torch.tensor(op_ids, dtype=torch.long),
            "label_ins": torch.tensor(ins_ids, dtype=torch.long),
            "length_text": item['src_len'],
            "length_audio": len(audio_tokens),
            "pad_idx": self.pad_idx
        }

def collate_fn(batch):
    pad_idx = batch[0]['pad_idx']
    
    # Determine max lengths in this batch
    max_text = max(x['length_text'] for x in batch)
    max_audio = max(x['length_audio'] for x in batch)
    
    # Prepare batch tensors
    b_size = len(batch)
    
    # Inputs
    input_ids = torch.full((b_size, max_text), pad_idx, dtype=torch.long)
    audio_tokens = torch.zeros(b_size, max_audio, dtype=torch.long)
    
    # Labels - pad with 0 (KEEP for ops, NONE for insertions)
    label_op = torch.zeros(b_size, max_text, dtype=torch.long)
    label_ins = torch.zeros(b_size, max_text, dtype=torch.long)

    # Masks
    text_mask = torch.zeros(b_size, max_text, dtype=torch.bool)
    audio_mask = torch.zeros(b_size, max_audio, dtype=torch.bool)

    for i, x in enumerate(batch):
        txt_len = x['length_text']
        aud_len = x['length_audio']
        
        input_ids[i, :txt_len] = x['input_ids']
        audio_tokens[i, :aud_len] = x['audio_tokens']
        
        label_op[i, :txt_len] = x['label_op']
        label_ins[i, :txt_len] = x['label_ins']
        
        text_mask[i, :txt_len] = 1
        audio_mask[i, :aud_len] = 1
        
    return {
        "input_ids": input_ids,
        "audio_tokens": audio_tokens,
        "labels": {
            "op": label_op,
            "ins": label_ins
        },
        "masks": {
            "text": text_mask,
            "audio": audio_mask
        }
    }