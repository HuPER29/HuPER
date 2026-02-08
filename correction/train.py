import yaml
import torch
import os
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from dataset import PhonemeCorrectionDataset, collate_fn
from model import PhonemeCorrector

def train(config):
    # 1. Data
    train_ds = PhonemeCorrectionDataset(
        jsonl_path=config['data']['data_path'],
        vocab_path=config['data']['vocab_path']
    )
    
    # Calculate vocab_size before splitting (Subset doesn't have vocab attribute)
    # vocab_size = max ID in insert_to_id + 1
    vocab_size = max(train_ds.insert_to_id.values()) + 1
    
    # Validation data
    val_data_path = config['data'].get('val_data_path')
    if val_data_path:
        val_ds = PhonemeCorrectionDataset(
            jsonl_path=val_data_path,
            vocab_path=config['data']['vocab_path']
        )
    else:
        # Use a subset of training data for validation if no separate validation set
        val_split = config['data'].get('val_split', 0.1)  # 10% for validation by default
        total_size = len(train_ds)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        train_ds, val_ds = random_split(train_ds, [train_size, val_size])
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['training']['batch_size'], 
        collate_fn=collate_fn, 
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    # 2. Model
    model = PhonemeCorrector(
        vocab_size=vocab_size,
        audio_vocab_size=config['data']['audio_vocab_size'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 0.01),
        scheduler_config=config.get('scheduler', {}),
        optimizer_config=config.get('optimizer', {})
    )
    
    # 3. Setup output directory
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = config['output'].get('experiment_name', 'phoneme-corrector')
    
    # 4. Setup Wandb logger
    loggers = []
    if config.get('wandb', {}).get('enabled', False):
        wandb_config = config['wandb']
        wandb_logger = WandbLogger(
            project=wandb_config.get('project', 'phoneme-correction'),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('name'),
            tags=wandb_config.get('tags', []),
            log_model=wandb_config.get('log_model', False),
            save_code=wandb_config.get('save_code', True),
            save_dir=str(output_dir)
        )
        loggers.append(wandb_logger)
    
    # 5. Setup checkpoint callback
    checkpoint_dir = output_dir / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='{epoch}-{step}-{train_loss:.4f}',
        monitor=config['checkpoint'].get('monitor', 'train_loss'),
        mode=config['checkpoint'].get('mode', 'min'),
        save_top_k=config['checkpoint'].get('save_top_k', 3),
        save_last=config['checkpoint'].get('save_last', True),
        every_n_epochs=config['checkpoint'].get('every_n_epochs', 1),
        verbose=True
    )
    
    # 6. Setup learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # 7. Trainer configuration
    trainer_config = config['trainer'].copy()
    trainer_config['logger'] = loggers if loggers else True  # Use default logger if wandb disabled
    trainer_config['callbacks'] = [checkpoint_callback, lr_monitor]
    
    # Add training-specific configs
    if 'gradient_clip_val' in config['training']:
        trainer_config['gradient_clip_val'] = config['training']['gradient_clip_val']
    if 'accumulate_grad_batches' in config['training']:
        trainer_config['accumulate_grad_batches'] = config['training']['accumulate_grad_batches']
    if 'val_check_interval' in config['training']:
        trainer_config['val_check_interval'] = config['training']['val_check_interval']
    if 'precision' in config['training']:
        trainer_config['precision'] = config['training']['precision']
    
    trainer = Trainer(**trainer_config)
    
    # 8. Start Training
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    # Load configuration from YAML
    with open("edit_seq_speech/config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)
