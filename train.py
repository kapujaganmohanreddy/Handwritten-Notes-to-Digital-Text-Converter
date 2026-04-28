import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import Levenshtein
import argparse
import random

from dataset import build_line_dataset, build_charset, CVLLineDataset, collate_fn
from model import CRNN

def ctc_greedy_decode(log_probs, charset):
    # log_probs: (T, B, C)
    preds = log_probs.argmax(2) # (T, B)
    preds = preds.transpose(0, 1) # (B, T)
    
    decoded_strings = []
    for i in range(preds.size(0)):
        pred = preds[i]
        
        # Collapse repeated
        collapsed = []
        for j in range(pred.size(0)):
            if pred[j] != 0: # 0 is blank
                if j == 0 or pred[j] != pred[j-1]:
                    collapsed.append(pred[j].item())
                    
        # Map to chars
        chars = [charset[c-1] for c in collapsed if c-1 < len(charset)]
        decoded_strings.append(''.join(chars))
        
    return decoded_strings

def calculate_metrics(preds, targets):
    cer_list = []
    wer_list = []
    
    for pred, target in zip(preds, targets):
        if len(target) == 0:
            continue
            
        # CER
        cer = Levenshtein.distance(pred, target) / len(target)
        cer_list.append(cer)
        
        # WER
        pred_words = pred.split()
        target_words = target.split()
        if len(target_words) > 0:
            wer = Levenshtein.distance(pred_words, target_words) / len(target_words)
        else:
            wer = 1.0 if len(pred_words) > 0 else 0.0
        wer_list.append(wer)
        
    return sum(cer_list)/len(cer_list) if cer_list else 0, sum(wer_list)/len(wer_list) if wer_list else 0

def evaluate(model, val_loader, charset, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels, label_lengths, input_lengths in val_loader:
            images = images.to(device)
            outputs = model(images) # (B, T, C)
            log_probs = outputs.permute(1, 0, 2).log_softmax(2) # (T, B, C)
            
            preds = ctc_greedy_decode(log_probs, charset)
            all_preds.extend(preds)
            
            # Decode targets
            start = 0
            for length in label_lengths:
                target_seq = labels[start:start+length]
                start += length
                target_str = ''.join([charset[c-1] for c in target_seq])
                all_targets.append(target_str)
                
    cer, wer = calculate_metrics(all_preds, all_targets)
    return cer, wer, all_preds, all_targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_dir', type=str, default='cvl-database-1-1')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dir = os.path.join(args.data_dir, 'trainset')
    if not os.path.exists(train_dir):
        print(f"Dataset directory not found: {train_dir}")
        return
        
    all_data = build_line_dataset(train_dir)
    print(f"Total English line samples found: {len(all_data)}")
    
    if len(all_data) == 0:
        print("No training data found. Please check dataset path.")
        return
        
    # Train/Val split by writer to avoid data leakage
    writer_dict = {}
    for path, label in all_data:
        writer = path.split(os.sep)[-2]
        if writer not in writer_dict:
            writer_dict[writer] = []
        writer_dict[writer].append((path, label))
        
    writers = sorted(list(writer_dict.keys()))  # sorted for reproducibility
    random.seed(42)  # fixed seed so resume gets the same split
    shuffled_writers = writers[:]
    random.shuffle(shuffled_writers)
    split_idx = int(len(shuffled_writers) * 0.9)
    train_writers = shuffled_writers[:split_idx]
    val_writers = shuffled_writers[split_idx:]
    
    train_data = []
    for w in train_writers:
        train_data.extend(writer_dict[w])
        
    val_data = []
    for w in val_writers:
        val_data.extend(writer_dict[w])
        
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    charset = build_charset([label for _, label in train_data])
    print(f"Charset length: {len(charset)}")
    
    train_dataset = CVLLineDataset(train_data, charset, is_train=True)
    val_dataset = CVLLineDataset(val_data, charset, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # ---- Build test set ----
    test_dir = os.path.join(args.data_dir, 'testset')
    test_loader = None
    if os.path.exists(test_dir):
        test_data = build_line_dataset(test_dir)
        if test_data:
            test_dataset = CVLLineDataset(test_data, charset, is_train=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
            print(f"Test samples: {len(test_data)}")
    else:
        print(f"Test directory not found: {test_dir}, skipping test evaluation")
    
    # ---- Print dataset summary ----
    augmentations = [
        "Rotation (±3°)", "Affine shear (±3°)", "Resolution degradation (30-70%)",
        "Perspective warp", "Gaussian blur (0.3-2.0)", "Brightness (0.6-1.4)",
        "Contrast (0.5-1.5)", "Uneven lighting gradient", "Gaussian noise",
        "JPEG compression (q15-50)", "Erosion/Dilation"
    ]
    print(f"\nDataset Summary:")
    print(f"  Train: {len(train_data)} original × {train_dataset.num_augments} augments = {len(train_dataset)} samples per epoch")
    print(f"  Val:   {len(val_data)} samples (no augmentation)")
    print(f"  Test:  {len(test_data) if test_loader else 0} samples (no augmentation)")
    print(f"  Augmentations ({len(augmentations)} types): {', '.join(augmentations)}\n")
    
    num_classes = len(charset) + 1 # +1 for blank
    model = CRNN(num_classes).to(device)
    
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
    scaler = GradScaler('cuda')
    
    os.makedirs('checkpoints', exist_ok=True)
    best_cer = float('inf')
    start_epoch = 0
    early_stop_patience = 15
    epochs_no_improve = 0
    
    # ---- Resume from last checkpoint ----
    last_ckpt_path = 'checkpoints/last_checkpoint.pth'
    if args.resume and os.path.exists(last_ckpt_path):
        print(f"Resuming from {last_ckpt_path}...")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_cer = ckpt['best_cer']
        epochs_no_improve = ckpt['epochs_no_improve']
        print(f"Resumed at epoch {start_epoch}, best CER so far: {best_cer:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Starting training: epochs {start_epoch+1} to {args.epochs}")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, labels, label_lengths, input_lengths) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(images) # (B, T, C)
                log_probs = outputs.permute(1, 0, 2).log_softmax(2) # (T, B, C)
                loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
                
        avg_loss = total_loss / num_batches
        
        # Validate
        val_cer, val_wer, val_preds, val_targets = evaluate(model, val_loader, charset, device)
        
        # Test
        test_cer, test_wer = 0.0, 0.0
        if test_loader:
            test_cer, test_wer, _, _ = evaluate(model, test_loader, charset, device)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_cer)
        
        # ---- Print epoch summary ----
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.1f}s | Loss: {avg_loss:.4f} | Val CER: {val_cer:.4f} | Val WER: {val_wer:.4f} | Test CER: {test_cer:.4f} | Test WER: {test_wer:.4f} | LR: {current_lr:.6f}")
        
        # Show one sample prediction
        if val_preds and val_targets:
            idx = random.randint(0, len(val_preds) - 1)
            print(f"  Sample GT:   {val_targets[idx]}")
            print(f"  Sample Pred: {val_preds[idx]}")
        print(f"{'─'*70}")
        
        # ---- Save checkpoint every epoch (for resume) ----
        ckpt_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'charset': charset,
            'val_cer': val_cer,
            'test_cer': test_cer,
            'best_cer': best_cer if val_cer >= best_cer else val_cer,
            'epochs_no_improve': epochs_no_improve,
        }
        torch.save(ckpt_data, last_ckpt_path)
        
        # Also save numbered checkpoint
        torch.save(ckpt_data, f'checkpoints/epoch_{epoch+1}.pth')
        
        # ---- Track best model ----
        if val_cer < best_cer:
            best_cer = val_cer
            epochs_no_improve = 0
            torch.save(ckpt_data, 'checkpoints/best_model.pth')
            print(f"  ★ New best model saved (CER: {best_cer:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered")
                break
        
        # Update best_cer in the last checkpoint
        ckpt_data['best_cer'] = best_cer
        ckpt_data['epochs_no_improve'] = epochs_no_improve
        torch.save(ckpt_data, last_ckpt_path)

if __name__ == '__main__':
    main()
