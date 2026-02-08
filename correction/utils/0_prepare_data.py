import json
import os
import soundfile as sf
from datasets import load_dataset
from g2p_en import G2p
from tqdm import tqdm

def prepare_librispeech():
    # 1. Setup paths
    output_dir = os.path.abspath("YOUR_WAV_FILES")
    os.makedirs(output_dir, exist_ok=True)
    
    jsonl_file = "YOUR_JSONL_FILE.jsonl"
    
    # 2. Initialize G2P
    g2p = G2p()

    print("Loading LibriSpeech clean.100 (this may take a moment)...")
    ds = load_dataset("librispeech_asr", "clean", split="train.100")

    print(f"Processing {len(ds)} samples...")
    print(f"Audio will be saved to: {output_dir}")
    
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in tqdm(ds):
            # OLD ERROR LINE: file_id = item['file'] 
            
            # CORRECT LINE: Use 'id' (clean string) or basename of the path
            file_id = item['id'] 
            
            text = item['text']
            
            audio_array = item['audio']['array']
            sample_rate = item['audio']['sampling_rate']
            
            # Construct new absolute path
            new_filename = f"{file_id}.wav"
            new_path = os.path.join(output_dir, new_filename)
            
            # Save the wav file to disk
            sf.write(new_path, audio_array, sample_rate)
            
            # --- G2P ---
            phonemes = g2p(text)
            phonemes = [p for p in phonemes if p.strip() != ""]

            # --- ENTRY ---
            entry = {
                "id": file_id,
                "audio_path": new_path,
                "text": text,
                "phonemes": phonemes
            }
            
            f.write(json.dumps(entry) + "\n")

    print(f"Done! Data saved to '{jsonl_file}' and audio in '{output_dir}'")

if __name__ == "__main__":
    prepare_librispeech()