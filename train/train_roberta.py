"""
Optimized RoBERTa Training for Google Colab Free Tier
Features:
- Sequential parameter training (train in segments)
- Minimal memory usage
- Mixed precision training
- Gradient accumulation
- Memory efficient data loading
Adapted from BERT SPST methodology
"""

import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW  # Import from torch.optim, not transformers
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc

# Import configuration
try:
    from config import (
        BERT_CONFIG, TRAIN_CONFIG, DATA_FILES, DEVICE_CONFIG,
        SEQUENTIAL_CONFIG, MEMORY_CONFIG, DATA_LOADER_CONFIG, LOGGING_CONFIG
    )
    # Adapt config for RoBERTa
    ROBERTA_CONFIG = BERT_CONFIG.copy()
    ROBERTA_CONFIG['model_name'] = 'distilroberta-base'  # Smaller, faster RoBERTa
    ROBERTA_CONFIG['save_dir'] = BERT_CONFIG['save_dir'].parent / 'roberta'
    ROBERTA_CONFIG['log_dir'] = BERT_CONFIG['log_dir'].parent / 'roberta'
    
except ImportError:
    print("Error: config.py not found.")
    print("Make sure config.py is in the same directory.")
    sys.exit(1)

# Setup logging
ROBERTA_CONFIG['log_dir'].mkdir(parents=True, exist_ok=True)
log_file = ROBERTA_CONFIG['log_dir'] / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FakeNewsDataset(Dataset):
    """Memory-efficient dataset class."""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_and_split_data():
    """Load unified dataset and split into train/val/test."""
    logger.info("Loading unified dataset...")
    
    # Load the unified dataset
    df = pd.read_csv(DATA_FILES['train'], sep='\t')
    
    logger.info(f"Total samples loaded: {len(df)}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Map labels to binary (0: fake, 1: true)
    label_mapping = {
        'fake': 0,
        'false': 0,
        'true': 1,
        'real': 1,
    }
    
    df['label'] = df['label'].str.lower().map(label_mapping)
    
    # Remove rows with missing labels or text
    df = df.dropna(subset=['label', 'text'])
    
    logger.info(f"After cleaning: {len(df)} samples")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Split data: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, 
        test_size=(TRAIN_CONFIG['val_split'] + TRAIN_CONFIG['test_split']),
        random_state=TRAIN_CONFIG['random_seed'],
        stratify=df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TRAIN_CONFIG['test_split'] / (TRAIN_CONFIG['val_split'] + TRAIN_CONFIG['test_split']),
        random_state=TRAIN_CONFIG['random_seed'],
        stratify=temp_df['label']
    )
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df

def create_data_loaders(train_df, val_df, test_df, tokenizer, config):
    """Create memory-efficient data loaders."""
    
    train_dataset = FakeNewsDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_length=config['max_length']
    )
    
    val_dataset = FakeNewsDataset(
        texts=val_df['text'].values,
        labels=val_df['label'].values,
        tokenizer=tokenizer,
        max_length=config['max_length']
    )
    
    test_dataset = FakeNewsDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_length=config['max_length']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=DATA_LOADER_CONFIG['num_workers'],
        pin_memory=DATA_LOADER_CONFIG['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=DATA_LOADER_CONFIG['num_workers'],
        pin_memory=DATA_LOADER_CONFIG['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=DATA_LOADER_CONFIG['num_workers'],
        pin_memory=DATA_LOADER_CONFIG['pin_memory']
    )
    
    return train_loader, val_loader, test_loader

def freeze_layers(model, layers_to_train=None, freeze_bert=False):
    """
    Freeze specific layers for sequential training.
    
    Args:
        model: RoBERTa model
        layers_to_train: List of layer names to train (None = train all)
        freeze_bert: If True, freeze all RoBERTa encoder layers
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # If training all layers
    if layers_to_train is None:
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Training ALL layers")
        return
    
    # Unfreeze specific layers
    for name, param in model.named_parameters():
        for layer_name in layers_to_train:
            if layer_name in name:
                param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    logger.info(f"Training layers: {layers_to_train}")

def calculate_metrics(predictions, labels, probabilities=None):
    """Calculate evaluation metrics."""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Calculate AUC if probabilities provided
    if probabilities is not None:
        try:
            # Get probability of positive class
            pos_probs = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
            auc = roc_auc_score(labels, pos_probs)
            metrics['auc'] = auc
        except:
            pass
    
    return metrics

def train_epoch(model, data_loader, optimizer, scheduler, device, scaler, config, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(data_loader, desc=f'Training Epoch {epoch}')
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass with mixed precision
        if MEMORY_CONFIG['use_mixed_precision']:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / config['gradient_accumulation_steps']
            
            # Backward pass
            scaler.scale(loss).backward()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / config['gradient_accumulation_steps']
            loss.backward()
        
        total_loss += loss.item()
        
        # Gradient accumulation
        if (step + 1) % config['gradient_accumulation_steps'] == 0:
            if MEMORY_CONFIG['use_mixed_precision']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MEMORY_CONFIG['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MEMORY_CONFIG['max_grad_norm'])
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        # Track predictions
        with torch.no_grad():
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        
        # Clear memory periodically
        if step % MEMORY_CONFIG['empty_cache_frequency'] == 0:
            clear_memory()
        
        progress_bar.set_postfix({'loss': f"{loss.item() * config['gradient_accumulation_steps']:.4f}"})
    
    avg_loss = total_loss / len(data_loader)
    metrics = calculate_metrics(predictions, true_labels)
    metrics['loss'] = avg_loss
    
    clear_memory()
    return metrics

def evaluate(model, data_loader, device, desc='Evaluating'):
    """Evaluate the model."""
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if MEMORY_CONFIG['use_mixed_precision']:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            total_loss += outputs.loss.item()
            
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    probabilities = np.array(probabilities)
    metrics = calculate_metrics(predictions, true_labels, probabilities)
    metrics['loss'] = avg_loss
    
    clear_memory()
    return metrics

def save_checkpoint(model, tokenizer, optimizer, epoch, segment_name, metrics, config):
    """Save model checkpoint."""
    checkpoint_dir = config['save_dir'] / f"{segment_name}_epoch_{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save training state
    torch.save({
        'epoch': epoch,
        'segment': segment_name,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, checkpoint_dir / 'training_state.pt')
    
    logger.info(f"Checkpoint saved to {checkpoint_dir}")

def train_segment(model, train_loader, val_loader, tokenizer, device, segment_config, segment_idx):
    """Train a specific segment of the model."""
    segment_name = segment_config['name']
    logger.info(f"\n{'='*80}")
    logger.info(f"Training Segment {segment_idx + 1}: {segment_name}")
    logger.info(f"{'='*80}")
    
    # Freeze/unfreeze layers
    freeze_layers(
        model, 
        layers_to_train=segment_config['layers_to_train'],
        freeze_bert=segment_config['freeze_bert']
    )
    
    # Setup optimizer for this segment
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=segment_config['learning_rate'],
        weight_decay=ROBERTA_CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * segment_config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=ROBERTA_CONFIG['warmup_steps'] // len(SEQUENTIAL_CONFIG['segments']),
        num_training_steps=total_steps
    )
    
    scaler = GradScaler() if MEMORY_CONFIG['use_mixed_precision'] else None
    
    best_val_f1 = 0
    
    for epoch in range(segment_config['epochs']):
        logger.info(f"\nSegment: {segment_name} | Epoch {epoch + 1}/{segment_config['epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, ROBERTA_CONFIG, epoch + 1
        )
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, desc=f'Validation {segment_name}')
        logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model for this segment
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_checkpoint(model, tokenizer, optimizer, epoch, segment_name, val_metrics, ROBERTA_CONFIG)
            logger.info(f"✓ New best for {segment_name}! F1: {best_val_f1:.4f}")
    
    return best_val_f1

def main():
    """Main training function."""
    logger.info("="*80)
    logger.info("RoBERTa Training for Fake News Detection - Colab Optimized")
    logger.info("="*80)
    logger.info(f"Model: {ROBERTA_CONFIG['model_name']}")
    logger.info(f"Batch size: {ROBERTA_CONFIG['batch_size']}")
    logger.info(f"Effective batch size: {ROBERTA_CONFIG['batch_size'] * ROBERTA_CONFIG['gradient_accumulation_steps']}")
    logger.info(f"Max length: {ROBERTA_CONFIG['max_length']}")
    logger.info(f"Sequential training: {SEQUENTIAL_CONFIG['enabled']}")
    logger.info(f"Mixed precision: {MEMORY_CONFIG['use_mixed_precision']}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and DEVICE_CONFIG['use_cuda'] else 'cpu')
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    train_df, val_df, test_df = load_and_split_data()
    
    # Initialize tokenizer and model
    logger.info(f"\nLoading model: {ROBERTA_CONFIG['model_name']}")
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_CONFIG['model_name'])
    model = RobertaForSequenceClassification.from_pretrained(
        ROBERTA_CONFIG['model_name'],
        num_labels=ROBERTA_CONFIG['num_labels']
    )
    
    # Enable gradient checkpointing for memory efficiency
    if MEMORY_CONFIG['use_gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled")
    
    model.to(device)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, tokenizer, ROBERTA_CONFIG
    )
    
    clear_memory()
    
    # Sequential training
    if SEQUENTIAL_CONFIG['enabled']:
        logger.info("\n" + "="*80)
        logger.info("SEQUENTIAL TRAINING MODE (SPST)")
        logger.info(f"Training {len(SEQUENTIAL_CONFIG['segments'])} segments")
        logger.info("="*80)
        
        for idx, segment_config in enumerate(SEQUENTIAL_CONFIG['segments']):
            best_f1 = train_segment(
                model, train_loader, val_loader, tokenizer, device, segment_config, idx
            )
            logger.info(f"Segment {idx + 1} complete. Best F1: {best_f1:.4f}\n")
    
    else:
        # Normal training (not recommended for Colab free tier)
        logger.info("\nNormal training mode (WARNING: May run out of memory)")
        # ... implement normal training if needed
    
    # Final evaluation
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*80)
    test_metrics = evaluate(model, test_loader, device, desc='Testing')
    logger.info(f"Test Results:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {test_metrics['f1']:.4f}")
    if 'auc' in test_metrics:
        logger.info(f"  AUC:       {test_metrics['auc']:.4f}")
    
    # Save final model
    final_dir = ROBERTA_CONFIG['save_dir'] / 'final_model'
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"\nFinal model saved to {final_dir}")
    
    # Save results
    results_file = ROBERTA_CONFIG['save_dir'] / 'results.txt'
    with open(results_file, 'w') as f:
        f.write("RoBERTa Fake News Detection - Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {ROBERTA_CONFIG['model_name']}\n")
        f.write(f"Test Metrics:\n")
        for key, value in test_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"  Batch size: {ROBERTA_CONFIG['batch_size']}\n")
        f.write(f"  Max length: {ROBERTA_CONFIG['max_length']}\n")
        f.write(f"  Sequential training: {SEQUENTIAL_CONFIG['enabled']}\n")
    
    logger.info(f"\nTraining complete! Results saved to {results_file}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
