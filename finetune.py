"""
Fine-tune the HTR model on degraded CVL data + phone/screenshot labeled data.
Loads the best model checkpoint and continues training with a low learning rate.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
import numpy as np
import random
import Levenshtein
import argparse

from model import CRNN


def ctc_greedy_decode(log_probs, charset):
    preds = log_probs.argmax(2).transpose(0, 1)
    decoded_strings = []
    for i in range(preds.size(0)):
        pred = preds[i]
        collapsed = []
        for j in range(pred.size(0)):
            if pred[j] != 0:
                if j == 0 or pred[j] != pred[j-1]:
                    collapsed.append(pred[j].item())
        chars = [charset[c-1] for c in collapsed if c-1 < len(charset)]
        decoded_strings.append(''.join(chars))
    return decoded_strings


def calculate_metrics(preds, targets):
    cer_list, wer_list = [], []
    total_correct_chars, total_chars = 0, 0
    total_correct_words, total_words = 0, 0
    exact_matches = 0
    
    for pred, target in zip(preds, targets):
        if len(target) == 0:
            continue
        
        # CER
        char_dist = Levenshtein.distance(pred, target)
        cer = char_dist / len(target)
        cer_list.append(cer)
        
        # Character accuracy (correct chars / total chars)
        correct_chars = max(0, len(target) - char_dist)
        total_correct_chars += correct_chars
        total_chars += len(target)
        
        # WER
        pred_words, target_words = pred.split(), target.split()
        word_dist = Levenshtein.distance(pred_words, target_words)
        wer = word_dist / max(len(target_words), 1)
        wer_list.append(wer)
        
        # Word accuracy (exact word matches)
        for pw, tw in zip(pred_words, target_words):
            if pw == tw:
                total_correct_words += 1
        total_words += len(target_words)
        
        # Line accuracy (exact full line match)
        if pred.strip() == target.strip():
            exact_matches += 1
    
    n = len(cer_list) if cer_list else 1
    avg_cer = sum(cer_list) / n if cer_list else 0
    avg_wer = sum(wer_list) / n if wer_list else 0
    char_acc = total_correct_chars / total_chars * 100 if total_chars else 0
    word_acc = total_correct_words / total_words * 100 if total_words else 0
    line_acc = exact_matches / len(preds) * 100 if preds else 0
    
    return avg_cer, avg_wer, char_acc, word_acc, line_acc


class FineTuneDataset(Dataset):
    """Dataset for fine-tuning with labeled line crops."""
    def __init__(self, data, charset, is_train=True, num_augments=3):
        self.charset = charset
        self.char_to_idx = {c: i+1 for i, c in enumerate(charset)}
        self.is_train = is_train
        
        if is_train and num_augments > 1:
            self.data = data * num_augments
        else:
            self.data = data
    
    def augment(self, img):
        """Light augmentation for fine-tuning (don't want to change domain too much)."""
        if not self.is_train:
            return img
        
        from PIL import ImageFilter, ImageEnhance
        import io
        
        # Slight rotation
        if random.random() < 0.3:
            angle = random.uniform(-2, 2)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor='white')
        
        # Brightness/contrast
        if random.random() < 0.4:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.4:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
        
        # Slight blur
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))
        
        # Noise
        if random.random() < 0.2:
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(2, 8), arr.shape)
            img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
        
        return img
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        img = Image.open(img_path).convert('L')
        img = self.augment(img)
        
        w, h = img.size
        new_h = 64
        new_w = int(w * (new_h / h))
        if new_w > 1024:
            new_w = 1024
        
        img = img.resize((new_w, new_h), Image.BILINEAR)
        img_arr = 255 - np.array(img)  # Invert to match training
        
        padded = np.zeros((new_h, 1024), dtype=np.float32)
        padded[:, :new_w] = img_arr / 255.0
        
        img_tensor = torch.FloatTensor(padded).unsqueeze(0)
        encoded = [self.char_to_idx.get(c, 0) for c in label if c in self.char_to_idx]
        
        return (img_tensor,
                torch.LongTensor(encoded),
                torch.LongTensor([len(encoded)]),
                torch.LongTensor([1024 // 4 - 1]))


def collate_fn(batch):
    images, labels, label_lengths, input_lengths = zip(*batch)
    return (torch.stack(images),
            torch.cat(labels),
            torch.cat(label_lengths),
            torch.cat(input_lengths))


def load_labels_file(labels_path, images_dir):
    """Load labels.txt → list of (image_path, label)"""
    data = []
    skipped = 0
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue
            fname, label = line.split('|', 1)
            label = label.strip()
            if label == 'LABEL_HERE' or not label:
                skipped += 1
                continue
            img_path = os.path.join(images_dir, fname.strip())
            if os.path.exists(img_path):
                data.append((img_path, label))
    if skipped > 0:
        print(f"  Skipped {skipped} unlabeled entries (LABEL_HERE)")
    return data


def main():
    parser = argparse.ArgumentParser(description='Fine-tune HTR model')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (low for fine-tuning)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--finetune_dir', type=str, default='finetune_data')
    parser.add_argument('--num_augments', type=int, default=3)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load base model
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    charset = checkpoint['charset']
    print(f"Loaded base model from {args.checkpoint}")
    print(f"Charset: {len(charset)} characters")
    
    # Collect fine-tuning data from all sources
    all_data = []
    
    # Option A: Auto-degraded CVL data
    auto_dir = os.path.join(args.finetune_dir, 'auto_degraded')
    auto_labels = os.path.join(auto_dir, 'labels.txt')
    if os.path.exists(auto_labels):
        auto_data = load_labels_file(auto_labels, auto_dir)
        print(f"Auto-degraded CVL data: {len(auto_data)} samples")
        all_data.extend(auto_data)
    
    # Option B: Phone/screenshot data
    phone_dir = os.path.join(args.finetune_dir, 'phone_lines')
    phone_labels = os.path.join(phone_dir, 'labels.txt')
    if os.path.exists(phone_labels):
        phone_data = load_labels_file(phone_labels, phone_dir)
        print(f"Phone/screenshot data: {len(phone_data)} labeled samples")
        if phone_data:
            # Oversample phone data to balance with auto data (phone data is more valuable)
            oversample = max(1, len(all_data) // (len(phone_data) * 5)) if all_data else 1
            phone_data_oversampled = phone_data * oversample
            print(f"  Oversampled {oversample}x -> {len(phone_data_oversampled)} samples")
            all_data.extend(phone_data_oversampled)
    
    if not all_data:
        print("No fine-tuning data found!")
        print("Run: python prepare_finetune.py --auto     (for degraded CVL data)")
        print("Run: python prepare_finetune.py --phone    (for phone photo data)")
        return
    
    # Split into train/val (90/10)
    random.seed(42)
    random.shuffle(all_data)
    split = int(len(all_data) * 0.9)
    train_data = all_data[:split]
    val_data = all_data[split:]
    
    print(f"\nFine-tuning dataset:")
    print(f"  Train: {len(train_data)} samples x {args.num_augments} augments = {len(train_data) * args.num_augments}")
    print(f"  Val:   {len(val_data)} samples")
    
    train_dataset = FineTuneDataset(train_data, charset, is_train=True, num_augments=args.num_augments)
    val_dataset = FineTuneDataset(val_data, charset, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    
    # Build model
    num_classes = len(charset) + 1
    model = CRNN(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully")
    
    # Fine-tuning with low learning rate
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    os.makedirs('checkpoints', exist_ok=True)
    best_cer = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning: {args.epochs} epochs, LR={args.lr}")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        num_batches = 0
        
        for images, labels, label_lengths, input_lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = label_lengths.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(images)
                log_probs = outputs.permute(1, 0, 2).log_softmax(2)
                loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        
        # Validate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, labels, label_lengths, input_lengths in val_loader:
                images = images.to(device)
                outputs = model(images)
                log_probs = outputs.permute(1, 0, 2).log_softmax(2)
                preds = ctc_greedy_decode(log_probs, charset)
                all_preds.extend(preds)
                start = 0
                for length in label_lengths:
                    target_seq = labels[start:start+length]
                    start += length
                    all_targets.append(''.join([charset[c-1] for c in target_seq]))
        
        cer, wer, char_acc, word_acc, line_acc = calculate_metrics(all_preds, all_targets)
        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{args.epochs} | {elapsed:.1f}s | Loss: {avg_loss:.4f}")
        print(f"  CER: {cer:.4f} | WER: {wer:.4f} | Char Acc: {char_acc:.1f}% | Word Acc: {word_acc:.1f}% | Line Acc: {line_acc:.1f}%")
        
        # Show sample
        if all_preds and all_targets:
            idx = random.randint(0, len(all_preds)-1)
            print(f"  GT:   {all_targets[idx][:80]}")
            print(f"  Pred: {all_preds[idx][:80]}")
        
        # Save best
        if cer < best_cer:
            best_cer = cer
            save_data = {
                'model_state_dict': model.state_dict(),
                'charset': charset,
                'val_cer': cer,
                'epoch': epoch,
                'fine_tuned': True,
            }
            torch.save(save_data, 'checkpoints/best_finetuned.pth')
            print(f"  * New best finetuned model! CER={best_cer:.4f}")
        
        print(f"{'_'*60}")
    
    print(f"\nFine-tuning complete!")
    print(f"Best CER: {best_cer:.4f}")
    print(f"Model saved to: checkpoints/best_finetuned.pth")
    print(f"\nTo use: python inference.py --image your_photo.png --checkpoint checkpoints/best_finetuned.pth")


if __name__ == '__main__':
    main()
