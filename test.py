import os
import torch
from torch.utils.data import DataLoader
import argparse
import Levenshtein

from dataset import build_line_dataset, CVLLineDataset, collate_fn
from model import CRNN
from train import evaluate, ctc_greedy_decode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='cvl-database-1-1')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    charset = checkpoint['charset']
    
    test_dir = os.path.join(args.data_dir, 'testset')
    if not os.path.exists(test_dir):
        print(f"Test dataset directory not found: {test_dir}")
        return
        
    test_data = build_line_dataset(test_dir)
    print(f"Total test samples: {len(test_data)}")
    
    if len(test_data) == 0:
        return
        
    test_dataset = CVLLineDataset(test_data, charset, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    num_classes = len(charset) + 1
    model = CRNN(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels, label_lengths, input_lengths in test_loader:
            images = images.to(device)
            outputs = model(images)
            log_probs = outputs.permute(1, 0, 2).log_softmax(2)
            
            preds = ctc_greedy_decode(log_probs, charset)
            all_preds.extend(preds)
            
            start = 0
            for length in label_lengths:
                target_seq = labels[start:start+length]
                start += length
                target_str = ''.join([charset[c-1] for c in target_seq])
                all_targets.append(target_str)
                
    cer_list = []
    wer_list = []
    
    with open('test_results.txt', 'w', encoding='utf-8') as f:
        for pred, target in zip(all_preds, all_targets):
            if len(target) == 0:
                continue
            cer = Levenshtein.distance(pred, target) / len(target)
            
            pred_words = pred.split()
            target_words = target.split()
            if len(target_words) > 0:
                wer = Levenshtein.distance(pred_words, target_words) / len(target_words)
            else:
                wer = 1.0 if len(pred_words) > 0 else 0.0
                
            cer_list.append(cer)
            wer_list.append(wer)
            
            f.write(f"Target: {target}\n")
            f.write(f"Pred:   {pred}\n")
            f.write(f"CER: {cer:.4f} | WER: {wer:.4f}\n")
            f.write("-" * 50 + "\n")
            
        avg_cer = sum(cer_list)/len(cer_list) if cer_list else 0
        avg_wer = sum(wer_list)/len(wer_list) if wer_list else 0
        
        f.write(f"\nFinal Test CER: {avg_cer:.4f}\n")
        f.write(f"Final Test WER: {avg_wer:.4f}\n")
        
    print(f"Test CER: {avg_cer:.4f} | Test WER: {avg_wer:.4f}")
    print("Detailed results saved to test_results.txt")

if __name__ == '__main__':
    main()
