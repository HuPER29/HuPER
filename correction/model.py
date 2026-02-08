import torch
import torch.nn as nn
import math
import pytorch_lightning as pl

class PhonemeCorrector(pl.LightningModule):
    def __init__(self, vocab_size, audio_vocab_size, d_model=256, nhead=4, num_layers=4, dropout=0.1, lr=1e-4, 
                 weight_decay=0.01, scheduler_config=None, optimizer_config=None):
        super().__init__()
        self.save_hyperparameters()
        self.scheduler_config = scheduler_config or {}
        self.optimizer_config = optimizer_config or {}
        
        # 1. Embeddings
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.audio_embedding = nn.Embedding(audio_vocab_size, d_model)
        
        # Positional Encoding (Standard Sinusoidal)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 2. The Core Transformer (Text querying Audio)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 3. Prediction Heads - 2-head architecture
        # Head 1: Operation (KEEP, DEL, SUB:AA, SUB:AE, ...)
        # num_ops = vocab_size + 2 (KEEP=0, DEL=1, SUB:phonemes=2+)
        # This matches the precomputed op_ids format
        num_ops = vocab_size + 2
        self.head_op = nn.Linear(d_model, num_ops)
        
        # Head 2: Insertion (NONE=0, AA, AE, ...)
        # num_inserts = vocab_size (NONE=0, then phonemes)
        num_inserts = vocab_size
        self.head_ins = nn.Linear(d_model, num_inserts)

    def forward(self, text_ids, audio_ids, text_mask=None, audio_mask=None):
        """
        text_ids: (Batch, Text_Len)
        audio_ids: (Batch, Audio_Len)
        masks: (Batch, Len) - 1 for valid, 0 for pad.
        """
        text_emb = self.pos_encoder(self.text_embedding(text_ids))
        audio_emb = self.pos_encoder(self.audio_embedding(audio_ids))
        
        txt_pad_mask = (text_mask == 0) if text_mask is not None else None
        aud_pad_mask = (audio_mask == 0) if audio_mask is not None else None

        encoded_features = self.transformer(
            tgt=text_emb, 
            memory=audio_emb,
            tgt_key_padding_mask=txt_pad_mask,
            memory_key_padding_mask=aud_pad_mask
        )
        
        logits_op = self.head_op(encoded_features)
        logits_ins = self.head_ins(encoded_features)
        
        return logits_op, logits_ins

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        audio_tokens = batch['audio_tokens']
        lbl_op = batch['labels']['op']
        lbl_ins = batch['labels']['ins']
        txt_mask = batch['masks']['text']
        audio_mask = batch['masks']['audio']
        
        logits_op, logits_ins = self(input_ids, audio_tokens, txt_mask, audio_mask)
        
        # Active loss mask (only compute loss on valid text tokens)
        active_loss = txt_mask.view(-1) == 1
        
        # OP LOSS (includes KEEP, DEL, and all SUB:phoneme operations)
        num_ops = self.hparams.vocab_size + 2
        loss_op = nn.functional.cross_entropy(
            logits_op.view(-1, num_ops)[active_loss], 
            lbl_op.view(-1)[active_loss]
        )
        
        # INS LOSS
        loss_ins = nn.functional.cross_entropy(
            logits_ins.view(-1, self.hparams.vocab_size)[active_loss],
            lbl_ins.view(-1)[active_loss]
        )
        
        loss = loss_op + loss_ins
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_loss_op', loss_op)
        self.log('train_loss_ins', loss_ins)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        audio_tokens = batch['audio_tokens']
        lbl_op = batch['labels']['op']
        lbl_ins = batch['labels']['ins']
        txt_mask = batch['masks']['text']
        audio_mask = batch['masks']['audio']
        
        logits_op, logits_ins = self(input_ids, audio_tokens, txt_mask, audio_mask)
        
        # Compute losses
        active_loss = txt_mask.view(-1) == 1
        num_ops = self.hparams.vocab_size + 2
        
        loss_op = nn.functional.cross_entropy(
            logits_op.view(-1, num_ops)[active_loss], 
            lbl_op.view(-1)[active_loss]
        )
        
        loss_ins = nn.functional.cross_entropy(
            logits_ins.view(-1, self.hparams.vocab_size)[active_loss],
            lbl_ins.view(-1)[active_loss]
        )
        
        loss = loss_op + loss_ins
        
        # Compute accuracy
        pred_op = torch.argmax(logits_op, dim=-1)
        pred_ins = torch.argmax(logits_ins, dim=-1)
        
        # OP accuracy
        op_correct = (pred_op == lbl_op) & txt_mask
        op_acc = op_correct.sum().float() / txt_mask.sum().float()
        
        # INS accuracy
        ins_correct = (pred_ins == lbl_ins) & txt_mask
        ins_acc = ins_correct.sum().float() / txt_mask.sum().float()
        
        # Overall accuracy: correct OP prediction
        overall_acc = op_acc
        
        # Per-operation accuracy (KEEP=0, DEL=1, SUB>=2)
        keep_mask = (lbl_op == 0) & txt_mask
        del_mask = (lbl_op == 1) & txt_mask
        sub_op_mask = (lbl_op >= 2) & txt_mask
        
        keep_acc = torch.tensor(0.0, device=loss.device)
        del_acc = torch.tensor(0.0, device=loss.device)
        sub_op_acc = torch.tensor(0.0, device=loss.device)
        
        if keep_mask.sum() > 0:
            keep_correct = (pred_op == lbl_op) & keep_mask
            keep_acc = keep_correct.sum().float() / keep_mask.sum().float()
        
        if del_mask.sum() > 0:
            del_correct = (pred_op == lbl_op) & del_mask
            del_acc = del_correct.sum().float() / del_mask.sum().float()
        
        if sub_op_mask.sum() > 0:
            sub_op_correct = (pred_op == lbl_op) & sub_op_mask
            sub_op_acc = sub_op_correct.sum().float() / sub_op_mask.sum().float()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_loss_op', loss_op, sync_dist=True)
        self.log('val_loss_ins', loss_ins, sync_dist=True)
        self.log('val_acc', overall_acc, prog_bar=True, sync_dist=True)
        self.log('val_acc_op', op_acc, sync_dist=True)
        self.log('val_acc_ins', ins_acc, sync_dist=True)
        self.log('val_acc_keep', keep_acc, sync_dist=True)
        self.log('val_acc_del', del_acc, sync_dist=True)
        self.log('val_acc_sub_op', sub_op_acc, sync_dist=True)
        
        return {
            'val_loss': loss,
            'val_acc': overall_acc,
            'val_acc_op': op_acc,
            'val_acc_ins': ins_acc
        }

    def configure_optimizers(self):
        # Get optimizer configuration
        optimizer_name = self.optimizer_config.get("name", "adamw").lower()
        lr = self.hparams.lr
        weight_decay = getattr(self.hparams, 'weight_decay', 0.01)
        
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.optimizer_config.get("betas", [0.9, 0.999]),
                eps=self.optimizer_config.get("eps", 1.0e-8)
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.optimizer_config.get("betas", [0.9, 0.999]),
                eps=self.optimizer_config.get("eps", 1.0e-8)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Configure scheduler
        scheduler_type = self.scheduler_config.get("type", "cosine").lower()
        
        # Calculate total training steps
        max_epochs = getattr(self.trainer, 'max_epochs', 50)
        if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches'):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # Fallback: estimate steps per epoch
            estimated_steps_per_epoch = 1000  # Conservative estimate
            total_steps = max_epochs * estimated_steps_per_epoch
        
        warmup_ratio = self.scheduler_config.get("warmup_ratio", 0.1)
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        
        if scheduler_type == "cosine":
            # Use transformers' cosine scheduler with warmup
            try:
                from transformers import get_cosine_schedule_with_warmup
                eta_min = self.scheduler_config.get("eta_min", 1.0e-6)
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                    num_cycles=0.5,  # Default cosine cycles
                    last_epoch=-1
                )
            except ImportError:
                # Fallback to PyTorch implementation
                from torch.optim.lr_scheduler import LambdaLR
                import math
                eta_min = self.scheduler_config.get("eta_min", 1.0e-6)
                def lr_lambda(step):
                    if step < warmup_steps:
                        return step / warmup_steps
                    else:
                        # Cosine annealing after warmup
                        progress = (step - warmup_steps) / (total_steps - warmup_steps)
                        cosine_value = 0.5 * (1 + math.cos(math.pi * progress))
                        return eta_min / lr + (1 - eta_min / lr) * cosine_value
                scheduler = LambdaLR(optimizer, lr_lambda)
            
        elif scheduler_type == "linear":
            # Use transformers' linear scheduler with warmup
            try:
                from transformers import get_linear_schedule_with_warmup
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
            except ImportError:
                # Fallback to PyTorch implementation
                from torch.optim.lr_scheduler import LambdaLR
                def lr_lambda(step):
                    if step < warmup_steps:
                        return step / warmup_steps
                    else:
                        progress = (step - warmup_steps) / (total_steps - warmup_steps)
                        return max(0.0, 1.0 - progress)
                scheduler = LambdaLR(optimizer, lr_lambda)
            
        elif scheduler_type == "polynomial":
            # Use transformers' polynomial scheduler with warmup
            try:
                from transformers import get_polynomial_decay_schedule_with_warmup
                power = self.scheduler_config.get("power", 1.0)
                scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                    power=power
                )
            except ImportError:
                # Fallback: use linear scheduler
                from torch.optim.lr_scheduler import LambdaLR
                def lr_lambda(step):
                    if step < warmup_steps:
                        return step / warmup_steps
                    else:
                        progress = (step - warmup_steps) / (total_steps - warmup_steps)
                        return max(0.0, (1.0 - progress) ** power)
                scheduler = LambdaLR(optimizer, lr_lambda)
            
        elif scheduler_type == "reduce_on_plateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_config.get("factor", 0.5),
                patience=self.scheduler_config.get("patience", 3),
                min_lr=self.scheduler_config.get("min_lr", 1.0e-6),
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        else:
            # No scheduler
            return optimizer
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

# Helper for Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq, Dim)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)