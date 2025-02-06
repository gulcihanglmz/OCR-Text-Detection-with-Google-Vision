import re
import numpy as np
import pandas as pd
from google.cloud import vision
import os
import cv2

# Set up Google Cloud credentials (this should be the first line!)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"path_to_your_google_cloud_credentials.json"

def detect_text_in_image(image_path, output_path, excel_writer):
    client = vision.ImageAnnotatorClient()

    # Read the image file
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Prepare the image in the format required by Vision API
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    vision_image = vision.Image(content=content)

    # Perform OCR (with support for Turkish and English)
    response = client.text_detection(image=vision_image, image_context={"language_hints": ["tr", "en"]})
    texts = response.text_annotations

    if response.error and response.error.message:
        raise Exception(
            f"Error occurred: {response.error.message}\nDetails: {response.error.details}"
        )

    extracted_texts = []
    ocr_data = []  # Collect OCR data here

    if texts:
        full_text = texts[0].description
        print(f"📌 Detected Full Text:\n{full_text}\n" + "-" * 50)

        for text in texts[1:]:
            word = text.description

            # Only include words consisting of letters and with length greater than 2
            if re.match(r'^[a-zA-Z]+$', word) and len(word) > 2:
                extracted_texts.append({
                    'text': word,
                    'bounds': [(vertex.x, vertex.y)
                               for vertex in text.bounding_poly.vertices]
                })

                # Draw bounding box and write the text on the image
                points = np.array([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices], np.int32)
                points = points.reshape((-1, 1, 2))

                # Draw the box
                cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

                # Write the text above the box
                x_min = min(points[:, 0, 0])
                y_min = min(points[:, 0, 1])
                cv2.putText(image, word, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Prepare data to save to Excel
                image_filename = os.path.basename(image_path)
                output_image_filename = os.path.basename(output_path)

                # Here we add the full output path for the annotated image
                annotated_image_path = os.path.abspath(output_path)  # This returns the full path

                # Append OCR data to the list
                ocr_data.append([image_filename, word, str([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]), annotated_image_path])

    # Save the result as an image
    cv2.imwrite(output_path, image)

    # Save data to Excel
    if ocr_data:
        new_df = pd.DataFrame(ocr_data, columns=["Image Name", "Detected Text", "Bounding Box", "Annotated Image Path"])
        excel_writer = pd.concat([excel_writer, new_df], ignore_index=True)  # Append to existing DataFrame

    return extracted_texts, excel_writer

# Example usage
if __name__ == "__main__":
    image_folder = r"your_input_images_folder"
    output_folder = r"your_output_folder"
    excel_output = r"your_output_excel_file.xlsx"

    # Create Excel file
    df_columns = ["Image Name", "Detected Text", "Bounding Box", "Annotated Image Path"]
    excel_writer = pd.DataFrame(columns=df_columns)

    try:
        # Process images in the folder
        for filename in os.listdir(image_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Check image files
                image_path = os.path.join(image_folder, filename)
                output_path = os.path.join(output_folder, f"output_{filename}")

                results, excel_writer = detect_text_in_image(image_path, output_path, excel_writer)

                if not results:
                    print(f"❌ No text found in {filename}.")
                else:
                    print(f"✅ {filename} - Detected texts:")
                    for result in results:
                        print(f"✅ Detected word: {result['text']}")
                        print(f"📍 Location (Bounding Box): {result['bounds']}\n")

        # Save the Excel file
        excel_writer['Annotated Image Path'] = excel_writer['Annotated Image Path'].apply(
            lambda x: f'=HYPERLINK("{x}", "{x}")'
        )

        excel_writer.to_excel(excel_output, index=False)
        print(f"✅ Excel file successfully saved: {excel_output}")

    except Exception as e:
        print(f"⚠️ Error: {e}")
