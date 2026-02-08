import torch
import torch.nn as nn
import torchaudio
import json
import re
import os
from g2p_en import G2p
import pytorch_lightning as pl

from .model import PhonemeCorrector
from transformers import Wav2Vec2Processor, HubertModel

class PhonemeCorrectionInference:
    def __init__(self, checkpoint_path, vocab_path, audio_model_name="facebook/hubert-large-ls960-ft", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Vocab / Config
        print(f"Loading config from {vocab_path}...")
        with open(vocab_path, 'r') as f:
            self.config = json.load(f)
            
        self.op_map = self.config['op_to_id']
        self.ins_map = self.config['insert_to_id']
        
        # Create Reverse Maps (ID -> String)
        self.id2op = {v: k for k, v in self.op_map.items()}
        self.id2ins = {v: k for k, v in self.ins_map.items()}
        
        # 2. Load G2P
        self.g2p = G2p()
        
        # 3. Load Model
        print(f"Loading model from {checkpoint_path}...")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            hparams = checkpoint.get('hyper_parameters', {})
            
            vocab_size = max(self.ins_map.values()) + 1
            audio_vocab_size = hparams.get('audio_vocab_size', 2048)
            
            self.model = PhonemeCorrector.load_from_checkpoint(
                checkpoint_path,
                map_location=self.device,
                vocab_size=vocab_size,
                audio_vocab_size=audio_vocab_size
            )
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        self.model.to(self.device)
        self.model.eval()

        # 4. Load Audio Tokenizer
        print(f"Loading Audio Tokenizer: {audio_model_name}")
        self.audio_processor = Wav2Vec2Processor.from_pretrained(audio_model_name)
        self.audio_model = HubertModel.from_pretrained(audio_model_name).eval().to(self.device)

    def _clean_phn(self, phn_list):
        """Standard cleaning to match training."""
        IGNORED = {"SIL", "'", "SPN", " "} 
        return [p.rstrip('012') for p in phn_list if p.rstrip('012') not in IGNORED]

    def _get_audio_tokens(self, wav_path):
        """
        Runs the audio tokenizer. 
        IMPORTANT: This must match your training data generation logic.
        """
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            
        inputs = self.audio_processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.audio_model(input_values)
            
        # Placeholder Quantization (Argmax) - Replace if using K-Means
        features = outputs.last_hidden_state
        tokens = torch.argmax(features, dim=-1).squeeze()
        
        # Downsample to 25Hz (Assuming model is 50Hz)
        tokens = tokens[::2]
        return tokens.unsqueeze(0) # (1, T)

    def predict(self, wav_path, text):
        # A. Prepare Inputs
        # 1. Text -> Phonemes -> IDs
        # raw_phns = self.g2p(text)
        raw_phns = text.split()  # Assuming input text is already phonemized for inference
        src_phns = self._clean_phn(raw_phns)
        
        # Create text vocab from insert_to_id (same as dataset)
        text_vocab = {k: v for k, v in self.ins_map.items() if k not in ['<NONE>', '<PAD>']}
        text_ids = [text_vocab.get(p, text_vocab.get("AA", 2)) for p in src_phns]
        text_tensor = torch.tensor([text_ids], dtype=torch.long).to(self.device)
        
        # 2. Audio -> Tokens
        audio_tensor = self._get_audio_tokens(wav_path)
        
        # B. Run Model
        with torch.no_grad():
            # Create masks
            txt_mask = torch.ones_like(text_tensor)
            aud_mask = torch.ones_like(audio_tensor)
            
            logits_op, logits_ins = self.model(
                text_tensor, audio_tensor, txt_mask, aud_mask
            )
            
            # C. Decode
            pred_ops = torch.argmax(logits_op, dim=-1).squeeze().tolist()
            pred_ins = torch.argmax(logits_ins, dim=-1).squeeze().tolist()
            
        # Ensure lists
        if not isinstance(pred_ops, list): pred_ops = [pred_ops]
        if not isinstance(pred_ins, list): pred_ins = [pred_ins]

        # D. Reconstruct Sequence
        final_phonemes = []
        log = []
        
        for i, (orig, op_id, ins_id) in enumerate(zip(src_phns, pred_ops, pred_ins)):
            
            # 1. Apply Operation
            op_str = self.id2op.get(op_id, "KEEP")
            curr_log = {"src": orig, "op": op_str, "ins": "NONE"}
            
            if op_str == "KEEP":
                final_phonemes.append(orig)
            elif op_str == "DEL":
                pass # Do not append
            elif op_str.startswith("SUB:"):
                # Extract phoneme: "SUB:AA" -> "AA"
                new_phn = op_str.split(":")[1]
                final_phonemes.append(new_phn)
            
            # 2. Apply Insertion
            ins_str = self.id2ins.get(ins_id, "<NONE>")
            if ins_str != "<NONE>":
                final_phonemes.append(ins_str)
                curr_log["ins"] = ins_str
                
            log.append(curr_log)
            
        return final_phonemes, log

if __name__ == "__main__":
    ckpt_path = "/data/chenxu/checkpoints/edit_seq_speech/phoneme-corrector/last.ckpt"
    vocab_path = "edit_seq_speech/config/vocab.json"
    wav_file = "test.wav"
    text_input = "Last Sunday"
    
    if os.path.exists(ckpt_path) and os.path.exists(wav_file):
        infer = PhonemeCorrectionInference(ckpt_path, vocab_path)
        result, details = infer.predict(wav_file, text_input)
        
        print(f"Input Text: {text_input}")
        print(f"Result Phn: {result}")
        print("-" * 20)
        for step in details:
            print(f"{step['src']} -> {step['op']} + Insert({step['ins']})")
    else:
        print("Please set valid paths for checkpoint and wav file.")