import sys
import os
import json
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor, HubertModel

# --- CONFIGURATION ---
INPUT_JSONL = "INPUT_JSONL_FILE.jsonl" 
OUTPUT_JSONL = "OUTPUT_JSONL_FILE.jsonl" 

# Paths from your screenshot
PROJECT_ROOT = 'PROJECT_ROOT_DIRECTORY'
CHECKPOINT_PATH = "ROCOGNITION_MODEL_CHECKPOINT.pth" # Path to your phoneme inference model checkpoint
VOCAB_PATH = os.path.join(PROJECT_ROOT, 'config/vocab.json')

# Audio Tokenizer Model (Adjust if you use a specific quantizer)
TOKENIZER_MODEL_ID = "facebook/hubert-large-ls960-ft" # Placeholder for xcodec-hubert
# If you have a local path for xcodec, use that instead.

# --- 1. SETUP USER'S PHONEME INFERENCE ---
sys.path.append(PROJECT_ROOT)
try:
    from utils.inference import PhonemeInference
    print("✅ Successfully imported PhonemeInference")
except ImportError:
    print(f"❌ Could not import PhonemeInference. Make sure {PROJECT_ROOT} is correct.")
    sys.exit(1)

def load_phoneme_model():
    print("Loading Phoneme Inference Model...")
    inferencer = PhonemeInference(
        checkpoint_path=CHECKPOINT_PATH,
        vocab_path=VOCAB_PATH
    )
    return inferencer

# --- 2. SETUP AUDIO TOKENIZER (XCODEC) ---
def load_audio_tokenizer():
    print(f"Loading Audio Tokenizer: {TOKENIZER_MODEL_ID}...")
    # NOTE: Standard HuBERT returns continuous features. 
    # If your xcodec model includes a K-Means clusterizer or VQ-VAE, load it here.
    processor = Wav2Vec2Processor.from_pretrained(TOKENIZER_MODEL_ID)
    model = HubertModel.from_pretrained(TOKENIZER_MODEL_ID).eval().cuda()
    return processor, model

def get_audio_tokens(audio_path, processor, model):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Prepare input
    inputs = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)
    input_values = inputs.input_values.cuda()

    with torch.no_grad():
        outputs = model(input_values)
    
    # --- IMPORTANT: CONVERT TO DISCRETE TOKENS ---
    # Standard HuBERT output is (Batch, Time, 1024). 
    # You need the discrete code IDs (integers).
    # If xcodec is a VQ model, `outputs.codebook_indices` might exist.
    # If it is standard HuBERT, you usually project to clusters.
    
    # FOR NOW: I am returning argmax just to give you integer shapes. 
    # YOU MUST REPLACE THIS with your specific K-Means or Quantizer call 
    # associated with 'hf-audio/xcodec-hubert-librispeech'.
    features = outputs.last_hidden_state # (1, T, 1024)
    tokens = torch.argmax(features, dim=-1).squeeze().cpu().tolist() # Placeholder logic    
    return tokens

# --- 3. MAIN LOOP ---
def main():
    # Load models
    phn_model = load_phoneme_model()
    tokenizer_proc, tokenizer_model = load_audio_tokenizer()
    
    # Read input file
    with open(INPUT_JSONL, 'r') as f:
        lines = f.readlines()

    # Open output file
    with open(OUTPUT_JSONL, 'w') as f_out:
        for line in tqdm(lines, desc="Processing Audio"):
            data = json.loads(line)
            audio_path = data['audio_path']
            
            try:
                # A. Get Ground Truth Phonemes (User's Model)
                # Your class method returns (phonemes, logits)
                real_phonemes, _ = phn_model.transcribe(audio_path)
                
                # B. Get Audio Tokens (25Hz)
                audio_tokens = get_audio_tokens(audio_path, tokenizer_proc, tokenizer_model)
                
                # C. Save
                data['real_phonemes'] = real_phonemes # List of strings ['L', 'AE', ...]
                data['audio_tokens'] = audio_tokens   # List of ints [34, 102, ...]
                
                f_out.write(json.dumps(data) + "\n")
                f_out.flush() # Ensure we save progress
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue

    print(f"Done! Saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()