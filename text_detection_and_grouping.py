import os
import cv2
import re
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import vision

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"path_to_your_google_cloud_credentials.json"

def setup_dataset_structure(dataset1_path, dataset_path):
    """Creates a structured dataset with train/test split."""
    train_images_path = os.path.join(dataset_path, 'images/train')
    test_images_path = os.path.join(dataset_path, 'images/test')
    train_labels_path = os.path.join(dataset_path, 'labels/train')
    test_labels_path = os.path.join(dataset_path, 'labels/test')
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(test_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(test_labels_path, exist_ok=True)

    image_files, label_files = [], []
    for file in os.listdir(dataset1_path):
        if file.endswith(('.jpg', '.png')):
            label_file = file.rsplit('.', 1)[0] + '.txt'
            if os.path.exists(os.path.join(dataset1_path, label_file)):
                image_files.append(file)
                label_files.append(label_file)

    if not image_files:
        print("No image and label pairs found.")
        return

    train_images, test_images, train_labels, test_labels = train_test_split(
        image_files, label_files, test_size=0.2, random_state=42
    )
    for img, lbl in zip(train_images, train_labels):
        shutil.copy(os.path.join(dataset1_path, img), os.path.join(train_images_path, img))
        shutil.copy(os.path.join(dataset1_path, lbl), os.path.join(train_labels_path, lbl))
    for img, lbl in zip(test_images, test_labels):
        shutil.copy(os.path.join(dataset1_path, img), os.path.join(test_images_path, img))
        shutil.copy(os.path.join(dataset1_path, lbl), os.path.join(test_labels_path, lbl))
    
    yaml_content = f"""
    train: {train_images_path}
    val: {test_images_path}
    nc: 1
    names: "billboard"
    """
    with open(os.path.join(dataset_path, 'detect.yaml'), 'w') as f:
        f.write(yaml_content)
    print("Dataset setup complete.")

def detect_text_in_image(image_path):
    """Uses Google Vision API to detect text in an image."""
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    vision_image = vision.Image(content=content)
    response = client.text_detection(image=vision_image, image_context={"language_hints": ["tr", "en"]})
    if response.error and response.error.message:
        raise Exception(f"Error: {response.error.message}")
    texts = response.text_annotations
    return [
        {'text': text.description, 'bounds': [(v.x, v.y) for v in text.bounding_poly.vertices]}
        for text in texts[1:] if re.match(r'^[a-zA-Z]+$', text.description) and len(text.description) > 2
    ]

def group_texts(extracted_texts, threshold=10):
    """Groups words that are close together into meaningful segments."""
    extracted_texts.sort(key=lambda x: x['bounds'][0][0])
    groups, current_group = [], [extracted_texts[0]] if extracted_texts else []
    for i in range(1, len(extracted_texts)):
        prev_x_end = extracted_texts[i - 1]['bounds'][2][0]
        current_x_start = extracted_texts[i]['bounds'][0][0]
        if current_x_start - prev_x_end <= threshold:
            current_group.append(extracted_texts[i])
        else:
            groups.append(current_group)
            current_group = [extracted_texts[i]]
    if current_group:
        groups.append(current_group)
    return groups

def process_images(image_folder, output_folder, excel_output):
    """Processes images: detects text, annotates, and saves results in Excel."""
    os.makedirs(output_folder, exist_ok=True)
    data = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_folder, filename)
            output_path = os.path.join(output_folder, f"output_{filename}")
            extracted_texts = detect_text_in_image(image_path)
            groups = group_texts(extracted_texts)
            image = cv2.imread(image_path)
            for text in extracted_texts:
                bounds = np.array(text['bounds'], dtype=np.int32)
                cv2.polylines(image, [bounds], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(image, text['text'], (bounds[0][0], bounds[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            for group in groups:
                x_min, y_min = min(word['bounds'][0][0] for word in group), min(word['bounds'][0][1] for word in group)
                x_max, y_max = max(word['bounds'][2][0] for word in group), max(word['bounds'][2][1] for word in group)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.imwrite(output_path, image)
            grouped_texts = "\n".join([" ".join([word['text'] for word in group]) for group in groups])
            data.append({
                "Detected Text": grouped_texts,
                "Hyperlink": f'=HYPERLINK("{os.path.abspath(output_path)}", "{os.path.basename(output_path)}")'
            })
    pd.DataFrame(data).to_excel(excel_output, index=False)
    print(f"✅ Excel saved: {excel_output}")

if __name__ == "__main__":
    dataset1_path = r"D:\\Atik Nakit\\billboard\\dataset1"
    dataset_path = r"D:\\Atik Nakit\\billboard\\dataset"
    setup_dataset_structure(dataset1_path, dataset_path)
    image_folder = r"your_input_images_folder"
    output_folder = r"your_output_folder"
    excel_output = r"your_output_excel_file.xlsx"
    process_images(image_folder, output_folder, excel_output)
