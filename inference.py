import os
import cv2
import numpy as np
import torch
import json
import argparse
from scipy.ndimage import gaussian_filter1d
from PIL import Image

from model import CRNN
from train import ctc_greedy_decode

def crop_handwritten_region(gray):
    """
    CVL pages have printed/digital text at the top and handwritten text below.
    Scan from the top, find the first content block (printed header), then find
    the gap after it. Crop from after that gap.
    """
    from scipy.ndimage import gaussian_filter1d as gf1d
    
    h_proj = np.sum(255 - gray, axis=1).astype(float)
    h_smooth = gf1d(h_proj, sigma=10)
    
    max_val = np.max(h_smooth)
    if max_val == 0:
        return gray
    
    # Only search in the top 40% of the page for the header
    search_limit = int(gray.shape[0] * 0.40)
    content_threshold = max_val * 0.02  # anything above this is "content"
    
    # Step 1: Find where the first content block starts (top of header)
    first_content = 0
    for i in range(search_limit):
        if h_smooth[i] > content_threshold:
            first_content = i
            break
    
    # Step 2: From the first content, find the end of the header block.
    # The header ends when we hit a gap of 40+ consecutive empty rows.
    min_gap = 40
    gap_count = 0
    header_end = None
    
    for i in range(first_content, search_limit):
        if h_smooth[i] < content_threshold:
            gap_count += 1
            if gap_count >= min_gap:
                header_end = i
                break
        else:
            gap_count = 0
    
    if header_end is None:
        # No clear header found in top 40% — return full image
        print("  No header detected, using full page")
        return gray
    
    # Step 3: Skip past the gap to where handwriting starts
    crop_y = header_end
    for i in range(header_end, gray.shape[0]):
        if h_smooth[i] > content_threshold:
            crop_y = max(0, i - 10)  # small padding above first handwritten line
            break
    
    print(f"  Header detected: cropping from row {crop_y} (page height: {gray.shape[0]})")
    return gray[crop_y:, :]

def preprocess_page(image_path):
    img = cv2.imread(image_path)
    if img is None:
        # Fallback: use Pillow for .tif files that OpenCV can't read
        try:
            pil_img = Image.open(image_path)
            img = np.array(pil_img)
            if len(img.shape) == 2:
                gray = img
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        except Exception:
            raise ValueError(f"Could not read image: {image_path}")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Crop out the printed header
    gray = crop_handwritten_region(gray)
    
    # Upscale low-resolution images (screenshots, phone photos)
    # CVL training pages are ~2500x3500, model expects similar detail level
    h, w = gray.shape
    min_height = 1000
    max_scale = 3.0  # Don't upscale more than 3x (over-upscaling blurs details)
    if h < min_height:
        scale = min(min_height / h, max_scale)
        new_w, new_h = int(w * scale), int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"  Low-res input upscaled: {w}x{h} -> {new_w}x{new_h} ({scale:.1f}x)")
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 15)
                                    
    # Denoise
    denoised = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2,2)))
    return gray, denoised

def segment_lines(binary_image):
    # Sum pixel values per row -> horizontal projection
    h_proj = np.sum(binary_image, axis=1)
    
    # Smooth the projection to handle noise
    h_proj_smooth = gaussian_filter1d(h_proj, sigma=5)
    
    # Find line regions (threshold 5% of max to ignore noise)
    threshold = np.max(h_proj_smooth) * 0.05
    in_line = h_proj_smooth > threshold
    
    lines = []
    start = None
    for i, val in enumerate(in_line):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start > 30:  # Minimum line height (filters noise blobs)
                lines.append((start, i))
            start = None
    if start is not None and len(in_line) - start > 30:
        lines.append((start, len(in_line)))
    
    # Merge lines that are too close (likely split by noise)
    merged = []
    for y1, y2 in lines:
        if merged and y1 - merged[-1][1] < 20:
            merged[-1] = (merged[-1][0], y2)  # Merge with previous
        else:
            merged.append((y1, y2))
    lines = merged
        
    # Add padding
    pad = 10
    h, w = binary_image.shape
    line_positions = []
    
    for y1, y2 in lines:
        y1_pad = max(0, y1 - pad)
        y2_pad = min(h, y2 + pad)
        line_positions.append((y1_pad, y2_pad))
        
    return line_positions

def detect_paragraphs(line_positions):
    """Group lines into paragraphs based on vertical gap size"""
    if len(line_positions) <= 1:
        return [[i for i in range(len(line_positions))]]
        
    gaps = [line_positions[i+1][0] - line_positions[i][1] 
            for i in range(len(line_positions)-1)]
    median_gap = np.median(gaps)
    
    paragraphs = []
    current_para = [0]
    for i, gap in enumerate(gaps):
        if gap > median_gap * 1.5:
            paragraphs.append(current_para)
            current_para = [i+1]
        else:
            current_para.append(i+1)
    paragraphs.append(current_para)
    return paragraphs

def prepare_line_image(line_crop):
    """Prepare a line crop for the model - matches training preprocessing exactly."""
    img = Image.fromarray(line_crop)
    
    w, h = img.size
    new_h = 64
    new_w = int(w * (new_h / h))
    if new_w > 1024:
        new_w = 1024
        
    img = img.resize((new_w, new_h), Image.BILINEAR)
    img_arr = 255 - np.array(img)  # Invert: text=white, bg=black (matches training)
    
    padded_img = np.zeros((new_h, 1024), dtype=np.float32)
    padded_img[:, :new_w] = img_arr / 255.0
    
    img_tensor = torch.FloatTensor(padded_img).unsqueeze(0).unsqueeze(0) # (1, 1, 64, 1024)
    return img_tensor

def recognize_page(image_path, model, charset, device):
    gray, binary = preprocess_page(image_path)
    line_positions = segment_lines(binary)
    paragraphs = detect_paragraphs(line_positions)
    
    w = gray.shape[1]
    
    line_texts = []
    lines_metadata = []
    
    model.eval()
    with torch.no_grad():
        for i, (y1, y2) in enumerate(line_positions):
            line_crop = gray[y1:y2, :]
            
            processed = prepare_line_image(line_crop).to(device)
            output = model(processed)
            log_probs = output.permute(1, 0, 2).log_softmax(2)
            
            text = ctc_greedy_decode(log_probs, charset)[0]
            line_texts.append(text)
            
            lines_metadata.append({
                "line_id": i + 1,
                "text": text,
                "bbox": [0, int(y1), int(w), int(y2)]
            })
            
    output_paragraphs = []
    json_paragraphs = []
    
    for i, para_indices in enumerate(paragraphs):
        para_text = ' '.join(line_texts[j] for j in para_indices)
        output_paragraphs.append(para_text)
        
        para_lines = [lines_metadata[j] for j in para_indices]
        json_paragraphs.append({
            "paragraph_id": i + 1,
            "lines": para_lines,
            "full_text": para_text
        })
        
    plain_text = '\n\n'.join(output_paragraphs)
    json_output = {"paragraphs": json_paragraphs}
    
    return plain_text, json_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Path to input full page image")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    charset = checkpoint['charset']
    
    num_classes = len(charset) + 1
    model = CRNN(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Processing image: {args.image}")
    plain_text, json_output = recognize_page(args.image, model, charset, device)
    
    out_txt_path = 'output.txt'
    out_json_path = 'output.json'
    
    with open(out_txt_path, 'w', encoding='utf-8') as f:
        f.write(plain_text)
        
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2)
        
    print(f"\n--- Output Text ---\n{plain_text}\n-------------------\n")
    print(f"Saved results to {out_txt_path} and {out_json_path}")

if __name__ == '__main__':
    main()
