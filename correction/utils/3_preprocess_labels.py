import json
import os
import Levenshtein
from tqdm import tqdm
from g2p_en import G2p

# CONFIG
INPUT_FILE = "DATASET_JSONL_FILE.jsonl" # Your existing file with text/audio/real_phn
OUTPUT_FILE = "PROCESSED_OUTPUT_FILE.jsonl"
VOCAB_FILE = "vocab.json"

def clean_phn(phn_list):
    """Remove stress digits and ignored tokens."""
    IGNORED = {"SIL", "'", "SPN"} # Add SPN if you want to ignore noise
    return [p.rstrip('012') for p in phn_list if p.rstrip('012') not in IGNORED]

def build_maps(phoneme_set):
    """Creates the explicit maps you requested."""
    sorted_phns = sorted(list(phoneme_set))
    
    # 1. Operation Map
    op_to_id = {"KEEP": 0, "DEL": 1}
    # Add SUB:PHONEME
    for i, p in enumerate(sorted_phns):
        op_to_id[f"SUB:{p}"] = i + 2
        
    # 2. Insertion Map
    insert_to_id = {"<NONE>": 0}
    for i, p in enumerate(sorted_phns):
        insert_to_id[p] = i + 1
        
    return op_to_id, insert_to_id

def main():
    data_entries = []
    all_phonemes = set()
    
    print("Pass 1: Collecting Phonemes...")
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Collect unique phonemes from both text and audio ground truth
            src = clean_phn(item['phonemes'])
            tgt = clean_phn(item['real_phonemes'])
            all_phonemes.update(src)
            all_phonemes.update(tgt)

    # Build Dictionaries
    op_map, ins_map = build_maps(all_phonemes)
    
    print(f"Stats: {len(op_map)} Operations, {len(ins_map)} Insertion types.")

    print("Pass 2: Generating Labels...")
    with open(OUTPUT_FILE, 'w') as f_out:
        with open(INPUT_FILE, 'r') as f_in:
            for line in tqdm(f_in):
                item = json.loads(line)
                
                src = clean_phn(item['phonemes'])
                tgt = clean_phn(item['real_phonemes'])
                
                # Default Labels
                L = len(src)
                # op_ids: default to KEEP (0)
                op_ids = [op_map["KEEP"]] * L
                # ins_ids: default to <NONE> (0)
                ins_ids = [ins_map["<NONE>"]] * L
                
                # Levenshtein
                ops = Levenshtein.editops(src, tgt)
                
                for action, s_i, t_i in ops:
                    if s_i >= L: continue

                    if action == 'replace':
                        target_phn = tgt[t_i]
                        # Create label like "SUB:AA"
                        label_str = f"SUB:{target_phn}"
                        op_ids[s_i] = op_map.get(label_str, op_map["KEEP"]) # Fallback safety
                        
                    elif action == 'delete':
                        op_ids[s_i] = op_map["DEL"]
                        
                    elif action == 'insert':
                        # Attach insertion to previous token
                        target_phn = tgt[t_i]
                        attach_idx = max(0, s_i - 1)
                        ins_ids[attach_idx] = ins_map.get(target_phn, 0)

                # Save Processed Item
                new_item = {
                    "audio_tokens": item['audio_tokens'], # Pass through
                    "text_ids": item.get('text_ids'), # Optional if you have it
                    "src_len": L,
                    "op_ids": op_ids,   # List of Ints
                    "ins_ids": ins_ids, # List of Ints
                    "text_phonemes": src # Helpful for debug
                }
                f_out.write(json.dumps(new_item) + "\n")

    # Save Config
    config_out = {
        "op_to_id": op_map,
        "insert_to_id": ins_map,
        "stats": {
            "num_ops": len(op_map),
            "num_inserts": len(ins_map)
        }
    }
    with open(VOCAB_FILE, 'w') as f:
        json.dump(config_out, f, indent=2)
        
    print(f"Saved processed data to {OUTPUT_FILE}")
    print(f"Saved vocab config to {VOCAB_FILE}")

if __name__ == "__main__":
    main()