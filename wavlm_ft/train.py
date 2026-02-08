import json
import torch
import torchaudio
import os
import yaml
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Union, Any
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2Processor, 
    WavLMForCTC, 
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor,
    WavLMConfig
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# Setup logging (will be reconfigured in main with file handler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. DATASET ---
class DistilledPhonemeDataset(Dataset):
    def __init__(self, jsonl_path, processor):
        self.data = []
        # Load JSONL
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Load Audio
        # Ensure path exists
        if not os.path.exists(item['audio_path']):
            # Return dummy if file missing (prevents crash, but ideally fix data)
            logger.warning(f"Audio file not found: {item['audio_path']}")
            return None 

        waveform, sr = torchaudio.load(item['audio_path'])
        
        # Resample if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # 2. Process Audio (Feature Extraction)
        # input_values shape: (Length,)
        input_values = self.processor(
            waveform.squeeze().numpy(), 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_values[0]
        
        # 3. Process Labels
        # We join phonemes with a space to treat them as "words" for the tokenizer
        # The tokenizer splits by space and maps to IDs
        # Example: ['L', 'AE', 'S', 'T'] -> "L AE S T"
        label_str = " ".join(item['label_phonemes'])
        
        # Use text argument instead of deprecated as_target_processor
        labels = self.processor(text=label_str, return_tensors="pt").input_ids[0]
            
        return {
            "input_values": input_values,
            "labels": torch.tensor(labels)
        }

# --- 2. COLLATOR ---
@dataclass
class DataCollatorCTC:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Filter out Nones (missing files)
        features = [f for f in features if f is not None]
        if len(features) == 0:
            raise ValueError("All items in batch were None!")

        # Separate inputs and labels
        input_values = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad Inputs (Audio)
        batch = self.processor.pad(
            input_values,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Pad Labels
        # Use tokenizer directly for padding labels (instead of deprecated as_target_processor)
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )
            
        # Replace padding with -100 to ignore in CrossEntropy
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        return batch

# --- 3. LIGHTNING MODULE ---
class WavLMFinetuner(pl.LightningModule):
    def __init__(
        self,
        model_name,
        vocab_size,
        processor,
        lr=3e-5,
        weight_decay=0.005,
        warmup_ratio=0.1,
        ctc_loss_reduction="mean",
        num_training_batches=None,
        init_from_pretrained=True
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['processor']) # Don't save processor in hparams
        self.processor = processor
        self.num_training_batches = num_training_batches  # Store for scheduler calculation
        
        # Load Model
        # ctc_loss_reduction="mean" is standard
        if init_from_pretrained:
            self.model = WavLMForCTC.from_pretrained(
                model_name, 
                ctc_loss_reduction=ctc_loss_reduction, 
                pad_token_id=processor.tokenizer.pad_token_id,
                vocab_size=vocab_size
            )
        else:
            config = WavLMConfig.from_pretrained(model_name)
            config.ctc_loss_reduction = ctc_loss_reduction
            config.pad_token_id = processor.tokenizer.pad_token_id
            config.vocab_size = vocab_size
            self.model = WavLMForCTC(config)
        
        # Freeze all parameters except the linear head (lm_head)
        # This ensures only the CTC classification head is trained
        for name, param in self.model.named_parameters():
            if 'lm_head' not in name:
                param.requires_grad = False
        
        # Count trainable parameters for logging
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Track if transformer layers have been unfrozen
        self.transformer_unfrozen = False

    def forward(self, input_values, labels=None):
        return self.model(input_values=input_values, labels=labels)
    
    def on_train_batch_start(self, batch, batch_idx):
        """Unfreeze transformer layers after warmup period"""
        if self.transformer_unfrozen:
            return  # Already unfrozen
        
        # Calculate warmup steps - use same logic as configure_optimizers
        warmup_ratio = float(self.hparams.get('warmup_ratio', 0.1))
        
        # Get total training steps - use same calculation as scheduler
        max_epochs = self.trainer.max_epochs if self.trainer else 10
        accumulate_grad_batches = self.trainer.accumulate_grad_batches if self.trainer else 1
        
        # Calculate steps per epoch considering gradient accumulation
        if self.num_training_batches is not None:
            num_batches = self.num_training_batches
        elif self.trainer and hasattr(self.trainer, 'num_training_batches'):
            num_batches = self.trainer.num_training_batches
        else:
            # Can't calculate, skip for now
            if self.global_step == 0:
                logger.warning("Cannot calculate total steps, will retry later")
            return
        
        # Calculate steps per epoch (with gradient accumulation)
        if isinstance(accumulate_grad_batches, int) and accumulate_grad_batches > 1:
            import math
            steps_per_epoch = max(1, math.ceil(num_batches / accumulate_grad_batches))
        else:
            steps_per_epoch = num_batches
        
        total_steps = steps_per_epoch * max_epochs
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        current_step = self.global_step
        
        # Debug info (only log once at the beginning)
        if self.global_step == 0:
            logger.info("="*70)
            logger.info("Training Configuration:")
            logger.info(f"  Total batches per epoch: {num_batches}")
            logger.info(f"  Gradient accumulation: {accumulate_grad_batches}")
            logger.info(f"  Steps per epoch: {steps_per_epoch}")
            logger.info(f"  Max epochs: {max_epochs}")
            logger.info(f"  Total training steps: {total_steps}")
            logger.info(f"  Warmup ratio: {warmup_ratio}")
            logger.info(f"  Warmup steps: {warmup_steps}")
            logger.info(f"  Transformer will unfreeze at step: {warmup_steps}")
            logger.info("="*70)
        
        # Unfreeze transformer layers after warmup
        # Note: We need to use optimizer step, not global_step, because scheduler uses optimizer step
        # Calculate optimizer step considering gradient accumulation
        accumulate_grad_batches = self.trainer.accumulate_grad_batches if self.trainer else 1
        if isinstance(accumulate_grad_batches, int) and accumulate_grad_batches > 1:
            optimizer_step = current_step // accumulate_grad_batches
        else:
            optimizer_step = current_step
        
        if optimizer_step >= warmup_steps and not self.transformer_unfrozen:
            logger.info("="*70)
            logger.info("Warmup complete! Unfreezing transformer layers")
            logger.info(f"  Global step: {current_step}")
            logger.info(f"  Optimizer step: {optimizer_step}")
            logger.info(f"  Warmup steps: {warmup_steps}")
            logger.info(f"  Current epoch: {self.current_epoch if hasattr(self, 'current_epoch') else 'N/A'}")
            logger.info("="*70)
            
            # Unfreeze transformer layers but keep feature encoder frozen
            # Feature encoder and feature projection stay frozen, everything else becomes trainable
            for name, param in self.model.named_parameters():
                if 'feature_projection' in name or 'feature_extractor' in name or 'feature_encoder' in name:
                    param.requires_grad = False  # Keep feature encoder frozen
                else:
                    # Unfreeze everything else (transformer encoder layers + lm_head)
                    param.requires_grad = True
            
            self.transformer_unfrozen = True
            
            # Count trainable parameters after unfreezing
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Trainable parameters after unfreezing: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
            
            # Update optimizer to include newly unfrozen parameters
            # Get current optimizer
            optimizer = self.optimizers()
            if optimizer is not None:
                # Get all newly unfrozen parameters (transformer layers)
                new_params = []
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        # Check if this parameter is not already in optimizer
                        already_in_optimizer = any(param is p for group in optimizer.param_groups for p in group['params'])
                        if not already_in_optimizer:
                            new_params.append(param)
                
                if new_params:
                    # Add new parameters to optimizer with same settings as existing group
                    current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else float(self.hparams.lr)
                    current_weight_decay = optimizer.param_groups[0].get('weight_decay', float(self.hparams.get('weight_decay', 0.005))) if optimizer.param_groups else float(self.hparams.get('weight_decay', 0.005))
                    
                    optimizer.add_param_group({
                        'params': new_params,
                        'lr': current_lr,
                        'weight_decay': current_weight_decay
                    })
                    logger.info(f"Optimizer updated: Added {len(new_params)} newly unfrozen parameters")

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_values=batch['input_values'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_values=batch['input_values'],
            labels=batch['labels']
        )
        val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        # Optional: Quick decode check (first item in batch)
        # Uncomment to see predictions during training logs
        # pred_ids = torch.argmax(outputs.logits[0], dim=-1)
        # pred_str = self.processor.decode(pred_ids)
        # print(f"\nPred: {pred_str}")
        
        return val_loss

    def configure_optimizers(self):
        # Ensure all values are floats (in case they were saved as strings)
        # Convert to float to handle both string and numeric inputs
        lr = float(self.hparams.lr)
        weight_decay = float(self.hparams.get('weight_decay', 0.005))
        warmup_ratio = float(self.hparams.get('warmup_ratio', 0.1))
        
        # Only optimize parameters that require gradients (i.e., the linear head)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        
        # Calculate total training steps for warmup
        # With gradient accumulation, optimizer steps per epoch = batches / accumulate_grad_batches
        max_epochs = self.trainer.max_epochs if self.trainer else 10
        
        # Get number of batches - prefer stored value, then trainer, then datamodule
        num_batches = None
        if self.num_training_batches is not None:
            num_batches = self.num_training_batches
        elif self.trainer and hasattr(self.trainer, 'num_training_batches'):
            num_batches = self.trainer.num_training_batches
        elif self.trainer and self.trainer.datamodule:
            try:
                train_loader = self.trainer.datamodule.train_dataloader()
                num_batches = len(train_loader)
            except:
                pass
        
        if num_batches and num_batches > 0:
            accumulate_grad_batches = self.trainer.accumulate_grad_batches if self.trainer else 1
            if isinstance(accumulate_grad_batches, int) and accumulate_grad_batches > 1:
                import math
                steps_per_epoch = max(1, math.ceil(num_batches / accumulate_grad_batches))
            else:
                steps_per_epoch = num_batches
            
            total_steps = steps_per_epoch * max_epochs
            warmup_steps = max(1, int(total_steps * warmup_ratio))
        else:
            # Fallback: use conservative estimates
            total_steps = 1000 * max_epochs  # Assume ~1000 steps per epoch
            warmup_steps = max(1, int(total_steps * warmup_ratio))
        
        # Create warmup + constant LR scheduler using LambdaLR
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup: increase from 0 to lr over warmup_steps
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Constant LR after warmup
                return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

# --- 4. MAIN ---
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train WavLM for Phoneme Recognition")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract config values
    DATA_FILE = config['data']['data_file']
    VOCAB_FILE = config['data']['vocab_file']
    TRAIN_VAL_SPLIT = config['data']['train_val_split']
    
    MODEL_NAME = config['model']['model_name']
    FREEZE_FEATURE_EXTRACTOR = config['model']['freeze_feature_extractor']
    CTC_LOSS_REDUCTION = config['model']['ctc_loss_reduction']
    
    BATCH_SIZE = config['training']['batch_size']
    ACCUMULATE_GRAD = config['training']['accumulate_grad']
    LEARNING_RATE = float(config['training']['learning_rate'])  # Ensure float conversion
    NUM_EPOCHS = config['training']['num_epochs']
    WEIGHT_DECAY = float(config['training']['weight_decay'])  # Ensure float conversion
    GRADIENT_CLIP_VAL = float(config['training']['gradient_clip_val'])  # Ensure float conversion
    
    WARMUP_RATIO = float(config['optimizer'].get('warmup_ratio', 0.1))  # Warmup ratio (default 0.1)
    
    NUM_WORKERS = config['dataloader']['num_workers']
    PIN_MEMORY = config['dataloader']['pin_memory']
    SHUFFLE_TRAIN = config['dataloader']['shuffle_train']
    
    FEATURE_SIZE = config['feature_extractor']['feature_size']
    SAMPLING_RATE = config['feature_extractor']['sampling_rate']
    PADDING_VALUE = config['feature_extractor']['padding_value']
    DO_NORMALIZE = config['feature_extractor']['do_normalize']
    RETURN_ATTENTION_MASK = config['feature_extractor']['return_attention_mask']
    
    UNK_TOKEN = config['tokenizer']['unk_token']
    PAD_TOKEN = config['tokenizer']['pad_token']
    WORD_DELIMITER_TOKEN = config['tokenizer']['word_delimiter_token']
    DO_LOWER_CASE = config['tokenizer']['do_lower_case']
    
    OUTPUT_DIR = config['output']['output_dir']
    CHECKPOINT_FILENAME = config['output']['checkpoint_filename']
    SAVE_TOP_K = config['output']['save_top_k']
    MONITOR_METRIC = config['output']['monitor_metric']
    MONITOR_MODE = config['output']['monitor_mode']
    
    ACCELERATOR = config['trainer']['accelerator']
    DEVICES = config['trainer']['devices']
    PRECISION = config['trainer']['precision']
    
    WANDB_PROJECT = config['wandb']['project']
    WANDB_NAME = config['wandb']['name']
    WANDB_TAGS = config['wandb']['tags']
    WANDB_LOG_MODEL = config['wandb']['log_model']
    WANDB_OFFLINE = config['wandb']['offline']
    
    # Setup logging to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "processor.log")
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add file handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging to file: {log_file}")
    
    # A. Setup Processor
    logger.info(f"Loading vocab from {VOCAB_FILE}...")
    
    # 1. Tokenizer
    # word_delimiter_token="|" allows us to use space as a separator between phonemes
    # We treat phonemes like "words" in a sentence: "L AE S T"
    tokenizer = Wav2Vec2CTCTokenizer(
        VOCAB_FILE, 
        unk_token=UNK_TOKEN, 
        pad_token=PAD_TOKEN, 
        word_delimiter_token=WORD_DELIMITER_TOKEN,
        do_lower_case=DO_LOWER_CASE
    )
    
    # 2. Feature Extractor (Standard 16k)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=FEATURE_SIZE, 
        sampling_rate=SAMPLING_RATE, 
        padding_value=PADDING_VALUE, 
        do_normalize=DO_NORMALIZE, 
        return_attention_mask=RETURN_ATTENTION_MASK
    )
    
    # 3. Processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # Save processor to output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processor_path = os.path.join(OUTPUT_DIR, "processor")
    processor.save_pretrained(processor_path)
    logger.info(f"Processor saved to {processor_path}")
    
    # B. Dataset
    logger.info(f"Loading data from {DATA_FILE}...")
    full_ds = DistilledPhonemeDataset(DATA_FILE, processor)
    
    # Split Train/Val
    train_size = int(TRAIN_VAL_SPLIT * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
    
    logger.info(f"Train Size: {len(train_ds)}, Val Size: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        collate_fn=DataCollatorCTC(processor),
        num_workers=NUM_WORKERS,
        shuffle=SHUFFLE_TRAIN,
        pin_memory=PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        collate_fn=DataCollatorCTC(processor),
        num_workers=NUM_WORKERS,
        shuffle=False
    )
    
    # C. Model
    # Calculate number of training batches for scheduler
    num_training_batches = len(train_loader)
    
    model = WavLMFinetuner(
        model_name=MODEL_NAME, 
        vocab_size=len(tokenizer), 
        processor=processor,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        ctc_loss_reduction=CTC_LOSS_REDUCTION,
        num_training_batches=num_training_batches
    )
    
    # Note: All parameters except lm_head are frozen in the model's __init__
    # The FREEZE_FEATURE_EXTRACTOR config is now ignored since we freeze everything except the head
    
    # D. Setup Wandb Logger
    # Prepare config dictionary for wandb
    wandb_config = {
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "accumulate_grad": ACCUMULATE_GRAD,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_val": GRADIENT_CLIP_VAL,
        "train_val_split": TRAIN_VAL_SPLIT,
        "freeze_feature_extractor": FREEZE_FEATURE_EXTRACTOR,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }
    
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        name=WANDB_NAME,
        tags=WANDB_TAGS if WANDB_TAGS else None,
        log_model=WANDB_LOG_MODEL,
        offline=WANDB_OFFLINE,
        config=wandb_config  # Pass config directly to WandbLogger
    )
    
    # Note: Model hyperparameters (lr, weight_decay, etc.) are also automatically logged via save_hyperparameters()
    
    # E. Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename=CHECKPOINT_FILENAME,
        save_top_k=SAVE_TOP_K,
        monitor=MONITOR_METRIC,
        mode=MONITOR_MODE
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Determine accelerator
    if ACCELERATOR == "gpu" and not torch.cuda.is_available():
        logger.warning("GPU requested but not available. Falling back to CPU.")
        accelerator = "cpu"
        strategy = "auto"
    else:
        accelerator = ACCELERATOR if torch.cuda.is_available() else "cpu"
        # Use DDP strategy with find_unused_parameters=True when using multiple GPUs
        # This is needed because the frozen feature encoder parameters don't participate in loss computation
        if isinstance(DEVICES, list) and len(DEVICES) > 1:
            strategy = DDPStrategy(find_unused_parameters=True)
        elif isinstance(DEVICES, int) and DEVICES > 1:
            strategy = DDPStrategy(find_unused_parameters=True)
        else:
            strategy = "auto"
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=DEVICES,
        strategy=strategy,
        max_epochs=NUM_EPOCHS,
        precision=PRECISION,
        accumulate_grad_batches=ACCUMULATE_GRAD,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=GRADIENT_CLIP_VAL,
        logger=wandb_logger
    )
    
    # F. Run
    logger.info("Starting Training...")
    trainer.fit(model, train_loader, val_loader)
