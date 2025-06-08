import os
import random
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold

base_path = "path_to/RDD2022"
output_path = "path_/RDD_Kfold"

govs = "China_MotorBike"
damageTypes = {1: "D00", 2: "D10", 3: "D20", 4: "D40", 5: "Repair"} 

k_folds = 5
random_seed = 42

os.makedirs(output_path, exist_ok=True)

with open(os.path.join(output_path, 'classes.txt'), 'w') as f:
    for class_id, class_name in damageTypes.items():
        f.write(f"{class_name}\n")

def convert_annotation(xml_path, img_width, img_height):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_annotations = []
    
    for obj in root.findall('.//object'):
        name = obj.find('name').text
        if name not in damageTypes.values():
            continue  # Skip classes not in our specified damage types
        
        # Find the class_id for this damage type
        class_id = None
        for id, damage_type in damageTypes.items():
            if damage_type == name:
                class_id = id - 1  # Subtract 1 for 0-indexed YOLO format
                break
        
        if class_id is None:
            continue
            
        bndbox = obj.find('bndbox')
        # Convert to float first and then to int to handle decimal values
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        
        # Convert to YOLO format (normalized centroids)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # Ensure values are within [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_annotations


all_data = []
for gov in tqdm(govs, desc="Processing governments"):
    xml_path = os.path.join(base_path, gov, 'train', 'annotations', 'xmls')
    img_path = os.path.join(base_path, gov, 'train', 'images')
    
    if not os.path.exists(xml_path) or not os.path.exists(img_path):
        print(f"Skipping {gov}: paths don't exist")
        continue
    
    xml_files = [f for f in os.listdir(xml_path) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        img_file = xml_file.replace('.xml', '.jpg')
        img_file_path = os.path.join(img_path, img_file)
        
    # Get image dimensions from XML
        tree = ET.parse(os.path.join(xml_path, xml_file))
        root = tree.getroot()
        size = root.find('size')
        width = int(float(size.find('width').text))
        height = int(float(size.find('height').text))
        
        # Check if this XML contains any of our target damage types
        has_target_damage = False
        for obj in root.findall('.//object'):
            name = obj.find('name').text
            if name in damageTypes.values():
                has_target_damage = True
                break
        
        # Only add to our dataset if it has at least one target damage type
        if has_target_damage:
            all_data.append({
                'xml_path': os.path.join(xml_path, xml_file),
                'img_path': img_file_path,
                'width': width,
                'height': height,
                'gov': gov,
                'filename': os.path.splitext(xml_file)[0]
            })

print(f"Total data with target damage types: {len(all_data)}")



from math import ceil

num_samples = len(all_data)
indices = list(range(num_samples))
random.seed(random_seed)
random.shuffle(indices)

# Calculate split sizes for 80:10:10
train_size = int(0.8 * num_samples)
val_size = int(0.1 * num_samples)
test_size = num_samples - train_size - val_size

fold_size = ceil(num_samples / k_folds)
fold_indices = [indices[i*fold_size:(i+1)*fold_size] for i in range(k_folds)]

for fold in range(k_folds):
    # Use current fold as test set
    test_idx = fold_indices[fold]
    
    # Remaining indices for train+val
    remaining_idx = [i for j, idxs in enumerate(fold_indices) if j != fold for i in idxs]
    
    # Split remaining into train and val (80:10 ratio from the remaining 90%)
    val_size_fold = len(remaining_idx) // 9  # 10% of total ≈ 1/9 of remaining
    val_idx = remaining_idx[:val_size_fold]
    train_idx = remaining_idx[val_size_fold:]

    fold_path = os.path.join(output_path, f"fold_{fold + 1}")
    os.makedirs(fold_path, exist_ok=True)
    
    train_path = os.path.join(fold_path, "train")
    val_path = os.path.join(fold_path, "val")
    test_path = os.path.join(fold_path, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Process all three splits
    for idx, split_type in zip([train_idx, val_idx, test_idx], ["train", "val", "test"]):
        if split_type == "train":
            split_path = train_path
        elif split_type == "val":
            split_path = val_path
        else:
            split_path = test_path
        
        for i in idx:
            data = all_data[i]
            xml_path = data['xml_path']
            img_path = data['img_path']
            width = data['width']
            height = data['height']
            
            # Copy image file
            shutil.copy(img_path, os.path.join(split_path, os.path.basename(img_path)))
            
            # Convert and save YOLO annotations
            yolo_annotations = convert_annotation(xml_path, width, height)
            annotation_file = os.path.join(split_path, f"{data['filename']}.txt")
            with open(annotation_file, 'w') as f:
                f.write("\n".join(yolo_annotations))
    
    # Print split statistics for this fold
    print(f"Fold {fold + 1}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    print(f"  Ratios: Train={len(train_idx)/num_samples:.1%}, Val={len(val_idx)/num_samples:.1%}, Test={len(test_idx)/num_samples:.1%}")