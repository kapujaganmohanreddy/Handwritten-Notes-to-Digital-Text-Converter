import os
import glob
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

def is_english(filename):
    """Return True if the file is from an English page (not page 6)"""
    # Pattern: {writerID}-{pageID}-...
    parts = filename.split('-')
    if len(parts) >= 2:
        page_id = parts[1]
        return page_id != '6'  # Page 6 = German
    return False

def build_line_dataset(split_dir):
    """
    Scans words directory to reconstruct line labels,
    and pairs them with corresponding line images.
    Returns: list of (line_image_path, line_text_label)
    """
    words_dir = os.path.join(split_dir, 'words')
    lines_dir = os.path.join(split_dir, 'lines')
    
    line_dict = {}  # (writer, page, line) -> list of (wordIdx, label)
    
    # Check if directory exists
    if not os.path.exists(words_dir):
        return []
        
    for writer_id in os.listdir(words_dir):
        writer_path = os.path.join(words_dir, writer_id)
        if not os.path.isdir(writer_path):
            continue
            
        for word_file in os.listdir(writer_path):
            if not word_file.endswith('.tif') and not word_file.endswith('.png'):
                continue
                
            if not is_english(word_file):
                continue
                
            # Parse filename: {writer}-{page}-{line}-{wordIdx}-{label}.tif
            name_without_ext = os.path.splitext(word_file)[0]
            parts = name_without_ext.split('-')
            
            if len(parts) < 5:
                continue
                
            writer = parts[0]
            page = parts[1]
            line = parts[2]
            word_idx = int(parts[3])
            label = "-".join(parts[4:])
            
            key = (writer, page, line)
            if key not in line_dict:
                line_dict[key] = []
            line_dict[key].append((word_idx, label))
            
    dataset = []
    for (writer, page, line), words in line_dict.items():
        # Sort words by wordIdx
        words.sort(key=lambda x: x[0])
        line_label = " ".join([w[1] for w in words])
        
        # Try both .tif and .png
        line_img_name_tif = f"{writer}-{page}-{line}.tif"
        line_img_name_png = f"{writer}-{page}-{line}.png"
        
        line_img_path = os.path.join(lines_dir, writer, line_img_name_tif)
        if not os.path.exists(line_img_path):
            line_img_path = os.path.join(lines_dir, writer, line_img_name_png)
            
        if os.path.exists(line_img_path):
            dataset.append((line_img_path, line_label))
            
    return dataset

def build_charset(labels):
    charset = set()
    for label in labels:
        charset.update(list(label))
    return sorted(list(charset))

class CVLLineDataset(Dataset):
    def __init__(self, data, charset, is_train=True, num_augments=5):
        self.charset = charset
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(charset)} # 0 is CTC blank
        self.is_train = is_train
        
        # Multiply training data: 1 original + (num_augments-1) augmented copies
        if is_train and num_augments > 1:
            self.data = data * num_augments
            self.num_augments = num_augments
        else:
            self.data = data
            self.num_augments = 1
        
    def __len__(self):
        return len(self.data)
        
    def augment(self, img):
        if not self.is_train:
            return img
            
        # Random rotation (slightly more aggressive)
        if random.random() < 0.5:
            angle = random.uniform(-3, 3)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor='white')
            
        # Random affine (shear)
        if random.random() < 0.5:
            shear = random.uniform(-3, 3)
            img = img.transform(img.size, Image.AFFINE, (1, shear * np.pi / 180, 0, 0, 1, 0), fillcolor='white')
        
        # ---- Phone/Screenshot simulation augmentations ----
        
        # Resolution degradation: downscale then upscale (simulates low-res phone capture)
        if random.random() < 0.4:
            w, h = img.size
            scale = random.uniform(0.3, 0.7)  # shrink to 30-70% then back up
            small_w, small_h = max(10, int(w * scale)), max(10, int(h * scale))
            img = img.resize((small_w, small_h), Image.BILINEAR)
            img = img.resize((w, h), Image.BILINEAR)
        
        # Perspective warp (simulates angled phone photos)
        if random.random() < 0.3:
            w, h = img.size
            margin = int(min(w, h) * 0.05)
            coeffs = [
                random.randint(-margin, margin), random.randint(-margin, margin),
                random.randint(-margin, margin), random.randint(-margin, margin),
                random.randint(-margin, margin), random.randint(-margin, margin),
                random.randint(-margin, margin), random.randint(-margin, margin),
            ]
            img = img.transform(img.size, Image.PERSPECTIVE, coeffs, fillcolor='white')
        
        # Gaussian blur (stronger range for phone defocus)
        if random.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 2.0)))
        
        # Brightness/contrast (wider range for phone lighting variation)
        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.6, 1.4))
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.5, 1.5))
        
        # Uneven lighting gradient (simulates phone flash / shadow)
        if random.random() < 0.3:
            w, h = img.size
            img_arr = np.array(img, dtype=np.float32)
            # Create a horizontal or vertical gradient
            if random.random() < 0.5:
                gradient = np.linspace(random.uniform(0.7, 0.9), random.uniform(1.0, 1.3), w)
                gradient = np.tile(gradient, (h, 1))
            else:
                gradient = np.linspace(random.uniform(0.7, 0.9), random.uniform(1.0, 1.3), h)
                gradient = np.tile(gradient.reshape(-1, 1), (1, w))
            img_arr = np.clip(img_arr * gradient, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr)
        
        # Gaussian noise (simulates phone sensor noise)
        if random.random() < 0.3:
            img_arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(3, 15), img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr)
        
        # JPEG compression artifacts
        if random.random() < 0.3:
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=random.randint(15, 50))
            buffer.seek(0)
            img = Image.open(buffer).convert('L')
        
        # Erosion/dilation (pen thickness variation)
        if random.random() < 0.2:
            img_arr = np.array(img)
            kernel = np.ones((2, 2), np.uint8)
            if random.random() < 0.5:
                img_arr = cv2.erode(img_arr, kernel, iterations=1)
            else:
                img_arr = cv2.dilate(img_arr, kernel, iterations=1)
            img = Image.fromarray(img_arr)
            
        return img
        
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        img = Image.open(img_path).convert('L')
        img = self.augment(img)
        
        # Resize to fixed height 64, pad width to 1024
        w, h = img.size
        new_h = 64
        new_w = int(w * (new_h / h))
        
        # Cap width at 1024
        if new_w > 1024:
            new_w = 1024
            
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Convert to numpy array, invert so text is white (higher value), background is black (0)
        img_arr = 255 - np.array(img)
        
        padded_img = np.zeros((new_h, 1024), dtype=np.float32)
        padded_img[:, :new_w] = img_arr / 255.0
        
        img_tensor = torch.FloatTensor(padded_img).unsqueeze(0) # (1, 64, 1024)
        
        # Encode label
        encoded_label = [self.char_to_idx.get(c, 0) for c in label if c in self.char_to_idx]
        
        encoded_label = torch.LongTensor(encoded_label)
        label_length = torch.LongTensor([len(encoded_label)])
        input_length = torch.LongTensor([1024 // 4 - 1]) # CRNN width reduction
        
        return img_tensor, encoded_label, label_length, input_length

def collate_fn(batch):
    images, labels, label_lengths, input_lengths = zip(*batch)
    images = torch.stack(images)
    labels = torch.cat(labels)
    label_lengths = torch.cat(label_lengths)
    input_lengths = torch.cat(input_lengths)
    return images, labels, label_lengths, input_lengths
