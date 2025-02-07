import os
import cv2
import re
import pandas as pd
from google.cloud import vision
import numpy as np

# Set up Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"path_to_your_google_cloud_credentials.json"


def detect_text_in_image(image_path):
    client = vision.ImageAnnotatorClient()
    image = cv2.imread(image_path)

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    vision_image = vision.Image(content=content)
    response = client.text_detection(image=vision_image, image_context={"language_hints": ["tr", "en"]})
    texts = response.text_annotations

    if response.error and response.error.message:
        raise Exception(f"Error occurred: {response.error.message}\nDetails: {response.error.details}")

    extracted_texts = []
    if texts:
        for text in texts[1:]:
            word = text.description
            if re.match(r'^[a-zA-Z]+$', word) and len(word) > 2:
                bounds = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                extracted_texts.append({'text': word, 'bounds': bounds})

    return extracted_texts, image


def group_nearby_words(extracted_texts, threshold=10):
    extracted_texts.sort(key=lambda x: x['bounds'][0][0])  # Sort by X coordinate
    groups = []
    # The X coordinate of the left corner of each word is used to ensure left-to-right sorting

    """
    Nearby words are grouped together by checking the distance between the start and end points of each word.
    The distance between the start of one word and the end of the previous word is checked for grouping.
    """
    current_group = [extracted_texts[0]] if extracted_texts else []
    for i in range(1, len(extracted_texts)):
        prev_word = extracted_texts[i - 1]
        current_word = extracted_texts[i]

        prev_x_end = prev_word['bounds'][2][0]
        current_x_start = current_word['bounds'][0][0]

        if current_x_start - prev_x_end <= threshold:
            current_group.append(current_word)
        else:
            groups.append(current_group)
            current_group = [current_word]

    if current_group:
        groups.append(current_group)

    return groups


def process_images(image_folder, output_folder, excel_output):
    os.makedirs(output_folder, exist_ok=True)
    data = []

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_folder, filename)
            output_path = os.path.join(output_folder, f"output_{filename}")

            extracted_texts, image = detect_text_in_image(image_path)
            groups = group_nearby_words(extracted_texts)  # Grouping words here

            for text in extracted_texts:
                bounds = np.array(text['bounds'], dtype=np.int32)
                cv2.polylines(image, [bounds], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(image, text['text'],(bounds[0][0], bounds[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            for group in groups:
                x_min = min(word['bounds'][0][0] for word in group)
                y_min = min(word['bounds'][0][1] for word in group)
                x_max = max(word['bounds'][2][0] for word in group)
                y_max = max(word['bounds'][2][1] for word in group)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            cv2.imwrite(output_path, image)

            # Format grouped texts into a single string
            grouped_texts = "\n".join([" ".join([word['text'] for word in group]) for group in groups])

            # Add hyperlink format for easy access
            data.append({
                "Detected Text": grouped_texts,
                "Hyperlink": f'=HYPERLINK("{os.path.abspath(output_path)}", "{os.path.basename(output_path)}")'
            })

    # Create DataFrame and save to Excel
    df = pd.DataFrame(data)
    df.to_excel(excel_output, index=False)
    print(f"✅ Excel saved: {excel_output}")

if __name__ == "__main__":
    image_folder = r"your_input_images_folder"
    output_folder = r"your_output_folder"
    excel_output = r"your_output_excel_file.xlsx"
    process_images(image_folder, output_folder, excel_output)
