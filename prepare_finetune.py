"""
Prepare fine-tuning data for HTR model.
Two modes:
  1. --auto: Generate degraded CVL line crops (simulates screenshots/phone photos)
  2. --phone: Extract lines from phone/screenshot photos for manual labeling
"""
import os
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
import argparse
import random

# Reuse inference segmentation
def segment_from_gray(gray):
    """Segment lines from a grayscale image, tuned for phone photos.
    Uses valley detection instead of absolute threshold (handles noisy backgrounds).
    """
    # Use Otsu to get a good binary for phone photos
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    denoised = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    
    # Horizontal projection
    h_proj = np.sum(denoised, axis=1).astype(np.float64)
    h_proj_smooth = gaussian_filter1d(h_proj, sigma=5)
    
    # Find valleys (local minima) in the projection = gaps between lines
    from scipy.signal import find_peaks
    
    # Invert projection to find valleys as peaks
    inverted = np.max(h_proj_smooth) - h_proj_smooth
    
    # Find peaks in inverted signal (= valleys in original)
    # Prominence ensures we only find significant valleys
    peak_prominence = np.max(h_proj_smooth) * 0.15
    min_distance = 20  # Minimum distance between line boundaries
    
    peaks, properties = find_peaks(inverted, prominence=peak_prominence, distance=min_distance)
    
    if len(peaks) == 0:
        # Fallback: try lower prominence
        peak_prominence = np.max(h_proj_smooth) * 0.05
        peaks, properties = find_peaks(inverted, prominence=peak_prominence, distance=min_distance)
    
    if len(peaks) == 0:
        # No valleys found — return whole image as one line
        return [(0, gray.shape[0])]
    
    # Build line regions from valleys
    # Lines are the regions BETWEEN consecutive valleys
    boundaries = [0] + list(peaks) + [gray.shape[0]]
    
    lines = []
    for i in range(len(boundaries) - 1):
        y1, y2 = boundaries[i], boundaries[i + 1]
        region_height = y2 - y1
        
        # Skip very small regions (noise at edges)
        if region_height < 15:
            continue
        
        # Check this region actually has content
        region_proj = h_proj_smooth[y1:y2]
        if np.max(region_proj) < np.max(h_proj_smooth) * 0.05:
            continue
        
        lines.append((y1, y2))
    
    # Add padding
    pad = 5
    h = gray.shape[0]
    result = []
    for y1, y2 in lines:
        result.append((max(0, y1 - pad), min(h, y2 + pad)))
    
    return result if result else [(0, gray.shape[0])]


def prepare_auto(data_dir, output_dir, num_scales=5):
    """
    Option A: Generate degraded CVL line crops that simulate screenshot/phone resolution.
    Uses CVL testset pages -> downscale -> extract lines -> match to known labels.
    """
    from dataset import build_line_dataset
    
    print("=" * 60)
    print("Option A: Generating degraded CVL line crops")
    print("=" * 60)
    
    # Load CVL line labels from testset
    testset_dir = os.path.join(data_dir, 'testset')
    line_data = build_line_dataset(testset_dir)
    print(f"Found {len(line_data)} labeled lines in testset")
    
    # Also use trainset for more data
    trainset_dir = os.path.join(data_dir, 'trainset')
    train_line_data = build_line_dataset(trainset_dir)
    print(f"Found {len(train_line_data)} labeled lines in trainset")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # For each line image, create degraded versions at different resolutions
    scales = [0.15, 0.20, 0.25, 0.30, 0.35]  # Aggressive downscale to simulate screenshots
    
    all_data = line_data + random.sample(train_line_data, min(500, len(train_line_data)))
    random.shuffle(all_data)
    
    labels_file = os.path.join(output_dir, 'labels.txt')
    count = 0
    
    with open(labels_file, 'w', encoding='utf-8') as f:
        for img_path, label in all_data:
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    pil_img = Image.open(img_path).convert('L')
                    img = np.array(pil_img)
            except:
                continue
                
            h, w = img.shape
            
            for scale in scales:
                # Downscale then upscale (simulates screenshot resolution loss)
                small_h = max(10, int(h * scale))
                small_w = max(10, int(w * scale))
                small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
                restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
                
                # Add some JPEG compression artifacts
                if random.random() < 0.5:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(20, 60)]
                    _, enc = cv2.imencode('.jpg', restored, encode_param)
                    restored = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
                
                # Add slight brightness/contrast variation
                if random.random() < 0.3:
                    alpha = random.uniform(0.7, 1.3)  # contrast
                    beta = random.uniform(-20, 20)     # brightness
                    restored = np.clip(alpha * restored + beta, 0, 255).astype(np.uint8)
                
                # Save
                fname = f"degraded_{count:05d}.png"
                cv2.imwrite(os.path.join(output_dir, fname), restored)
                f.write(f"{fname}|{label}\n")
                count += 1
                
            if count % 500 == 0:
                print(f"  Generated {count} degraded samples...")
    
    print(f"\nDone! Generated {count} degraded line crops in {output_dir}/")
    print(f"Labels saved to {labels_file}")
    return count


def prepare_phone(photo_dir, output_dir):
    """
    Option B: Extract lines from phone/screenshot photos for manual labeling.
    """
    print("=" * 60)
    print("Option B: Extracting lines from phone/screenshot photos")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    photos = []
    for f in sorted(os.listdir(photo_dir)):
        if any(f.lower().endswith(e) for e in exts):
            photos.append(os.path.join(photo_dir, f))
    
    if not photos:
        print(f"No images found in {photo_dir}")
        return 0
    
    print(f"Found {len(photos)} photos")
    
    labels_file = os.path.join(output_dir, 'labels.txt')
    count = 0
    
    with open(labels_file, 'w', encoding='utf-8') as f:
        for photo_path in photos:
            print(f"\nProcessing: {os.path.basename(photo_path)}")
            
            # Load image
            img = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                try:
                    pil_img = Image.open(photo_path).convert('L')
                    img = np.array(pil_img)
                except:
                    print(f"  Skipped (can't read)")
                    continue
            
            # Upscale if too small
            h, w = img.shape
            if h < 500:
                scale = 500 / h
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
                print(f"  Upscaled: {w}x{h} -> {img.shape[1]}x{img.shape[0]}")
            
            # Segment lines
            line_positions = segment_from_gray(img)
            print(f"  Found {len(line_positions)} lines")
            
            for j, (y1, y2) in enumerate(line_positions):
                line_crop = img[y1:y2, :]
                
                # Skip very thin or very small crops (likely noise)
                if line_crop.shape[0] < 15 or line_crop.shape[1] < 50:
                    continue
                
                fname = f"phone_line_{count:04d}.png"
                cv2.imwrite(os.path.join(output_dir, fname), line_crop)
                f.write(f"{fname}|LABEL_HERE\n")  # Placeholder for manual labeling
                count += 1
    
    print(f"\nDone! Extracted {count} line crops to {output_dir}/")
    print(f"\nNEXT STEPS:")
    print(f"  1. Open the line images in {output_dir}/ to see each line")
    print(f"  2. Edit {labels_file}")
    print(f"  3. Replace 'LABEL_HERE' with the actual text for each line")
    print(f"  4. Example: phone_line_0001.png|While I gazed this fissure rapidly widened")
    print(f"  5. Then run: python finetune.py")
    return count


def main():
    parser = argparse.ArgumentParser(description='Prepare fine-tuning data')
    parser.add_argument('--auto', action='store_true', help='Option A: Generate degraded CVL crops')
    parser.add_argument('--phone', action='store_true', help='Option B: Extract lines from phone photos')
    parser.add_argument('--data_dir', default='cvl-database-1-1', help='CVL dataset directory')
    parser.add_argument('--photo_dir', default='finetune_data/raw_photos', help='Directory with phone photos')
    parser.add_argument('--output_dir', default='finetune_data', help='Output directory')
    args = parser.parse_args()
    
    if not args.auto and not args.phone:
        print("Specify --auto and/or --phone")
        print("  --auto   Generate degraded CVL line crops (no manual work)")
        print("  --phone  Extract lines from photos for manual labeling")
        return
    
    auto_dir = os.path.join(args.output_dir, 'auto_degraded')
    phone_dir = os.path.join(args.output_dir, 'phone_lines')
    
    total = 0
    
    if args.auto:
        total += prepare_auto(args.data_dir, auto_dir)
    
    if args.phone:
        os.makedirs(args.photo_dir, exist_ok=True)
        if len(os.listdir(args.photo_dir)) == 0:
            print(f"\nPlease put your phone/screenshot photos in: {args.photo_dir}/")
            print("Then run this script again with --phone")
        else:
            total += prepare_phone(args.photo_dir, phone_dir)
    
    print(f"\nTotal fine-tuning samples: {total}")


if __name__ == '__main__':
    main()
