import json
import re

INPUT_FILE = "DATASET_JSONL_FILE.jsonl"
VOCAB_FILE = "vocab.json"

def clean_stress(phn_list):
    """Removes digits (stress markers) from phonemes."""
    # Example: "AE1" -> "AE", "R" -> "R"
    return [re.sub(r'\d+', '', p) for p in phn_list]

def build_vocab():
    unique_symbols = set()
    IGNORED_TOKENS = {"SIL", "'"}
    
    print("Scanning dataset to build vocabulary...")
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Clean and collect from Source (g2p)
            src = clean_stress(data['phonemes'])
            unique_symbols.update(src)
            
            # Clean and collect from Target (Real/WavLM)
            # We clean target too, so the model learns 'AE' matches 'AE', not 'AE1' matches 'AE'
            tgt = clean_stress(data['real_phonemes'])
            tgt = [p for p in tgt if p not in IGNORED_TOKENS]
            unique_symbols.update(tgt)

    # Sort for reproducibility
    sorted_symbols = sorted(list(unique_symbols))
    
    # Create Mapping
    # 0 is usually PAD. 
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1, 
        "<BOS>": 2, # Beginning of Sequence (optional, but good for generation)
        "<EOS>": 3  # End of Sequence
    }
    
    start_idx = len(vocab)
    for i, sym in enumerate(sorted_symbols):
        vocab[sym] = start_idx + i
        
    print(f"Found {len(sorted_symbols)} unique phonemes.")
    print(f"Total Vocab Size: {len(vocab)}")
    print(f"Sample: {list(vocab.items())[:10]}...")
    
    with open(VOCAB_FILE, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved to {VOCAB_FILE}")

if __name__ == "__main__":
    build_vocab()