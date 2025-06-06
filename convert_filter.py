import os
import random
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# Path configuration
base_path = "path_to/RDD2022"
output_path = "path_to/RDD_YOLOv11_format"  # Output path for YOLOv11 format

govs = ["China_Drone", "China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States"]
damageTypes = {1: "D00", 2: "D01", 3: "D10", 4: "D11", 5: "D20", 6: "D40", 7: "D43", 8: "D44"}

# Create necessary directories
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(output_path, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(output_path, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(output_path, "labels", "val"), exist_ok=True)

# Save class names to file
with open(os.path.join(output_path, 'classes.txt'), 'w') as f:
    for class_id, class_name in damageTypes.items():
        f.write(f"{class_name}\n")

# Function to convert XML to YOLO format
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

# Collect all data
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
        
        # Check if image exists
        if not os.path.exists(img_file_path):
            # Try PNG format
            img_file = xml_file.replace('.xml', '.png')
            img_file_path = os.path.join(img_path, img_file)
            if not os.path.exists(img_file_path):
                print(f"Image not found for {xml_file}")
                continue
        
        # Get image dimensions from XML
        try:
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
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")

# Split data with random seed 42
random.seed(42)
np.random.seed(42)
train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)

print(f"Total data with target damage types: {len(all_data)}")
print(f"Training data: {len(train_data)}")
print(f"Validation data: {len(val_data)}")

# Process training data
for item in tqdm(train_data, desc="Processing training data"):
    # Convert annotations
    yolo_annotations = convert_annotation(item['xml_path'], item['width'], item['height'])
    
    # Skip if no annotations
    if not yolo_annotations:
        continue
    
    # Create YOLO label file
    label_filename = f"{item['filename']}.txt"
    with open(os.path.join(output_path, "labels", "train", label_filename), 'w') as f:
        f.write('\n'.join(yolo_annotations))
    
    # Copy image
    img_ext = os.path.splitext(item['img_path'])[1]
    dst_img = os.path.join(output_path, "images", "train", f"{item['filename']}{img_ext}")
    shutil.copy2(item['img_path'], dst_img)

# Process validation data
for item in tqdm(val_data, desc="Processing validation data"):
    # Convert annotations
    yolo_annotations = convert_annotation(item['xml_path'], item['width'], item['height'])
    
    # Skip if no annotations
    if not yolo_annotations:
        continue
    
    # Create YOLO label file
    label_filename = f"{item['filename']}.txt"
    with open(os.path.join(output_path, "labels", "val", label_filename), 'w') as f:
        f.write('\n'.join(yolo_annotations))
    
    # Copy image
    img_ext = os.path.splitext(item['img_path'])[1] 
    dst_img = os.path.join(output_path, "images", "val", f"{item['filename']}{img_ext}")
    shutil.copy2(item['img_path'], dst_img)

# Create data.yaml file for YOLOv11
damage_names = [damageTypes[i] for i in sorted(damageTypes.keys())]
yaml_content = f"""path: {output_path}
train: images/train
val: images/val

nc: {len(damageTypes)}
names: {damage_names}
"""

with open(os.path.join(output_path, 'data.yaml'), 'w') as f:
    f.write(yaml_content)

print("Conversion complete!")