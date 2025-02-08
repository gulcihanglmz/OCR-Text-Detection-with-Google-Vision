import os
import shutil
from sklearn.model_selection import train_test_split

# Dataset1 klasör yolun
dataset1_path = r"D:\Atik Nakit\billboard\dataset1"

# Yeni dataset yapısı
dataset_path = r"D:\Atik Nakit\billboard\dataset"
train_images_path = os.path.join(dataset_path, 'images/train')
test_images_path = os.path.join(dataset_path, 'images/test')
train_labels_path = os.path.join(dataset_path, 'labels/train')
test_labels_path = os.path.join(dataset_path, 'labels/test')

# Yeni dizinleri oluştur
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)
os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

# Fotoğraf ve ilgili txt dosyalarını al
files = [f for f in os.listdir(dataset1_path) if f.endswith(('.jpg', '.png'))]
image_files = []
label_files = []

for file in files:
    label_file = file.rsplit('.', 1)[0] + '.txt'
    if os.path.exists(os.path.join(dataset1_path, label_file)):
        image_files.append(file)
        label_files.append(label_file)

# Eğer image_files boşsa, bir mesaj yazdır
if not image_files:
    print("No image and label pairs found in the specified directory.")
else:
    print(f"Found {len(image_files)} image-label pairs.")

# Train-Test Split işlemi
train_images, test_images, train_labels, test_labels = train_test_split(image_files, label_files, test_size=0.2, random_state=42)

# Fotoğrafları ve etiketleri taşı
for img, lbl in zip(train_images, train_labels):
    shutil.copy(os.path.join(dataset1_path, img), os.path.join(train_images_path, img))
    shutil.copy(os.path.join(dataset1_path, lbl), os.path.join(train_labels_path, lbl))

for img, lbl in zip(test_images, test_labels):
    shutil.copy(os.path.join(dataset1_path, img), os.path.join(test_images_path, img))
    shutil.copy(os.path.join(dataset1_path, lbl), os.path.join(test_labels_path, lbl))

# detect.yaml dosyasını oluştur
yaml_content = f"""
train: {os.path.join(dataset_path, 'images/train')}
val: {os.path.join(dataset_path, 'images/test')}

nc: 1
names: "billboard"
"""

with open(os.path.join(dataset_path, 'detect.yaml'), 'w') as f:
    f.write(yaml_content)

print("Dataset split and YAML creation complete.")
