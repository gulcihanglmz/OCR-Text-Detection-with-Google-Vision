# OCR Text Detection with Google Vision

This project utilizes the Google Cloud Vision API to detect and extract text from images. The text detected is then annotated back onto the images and the results are saved in an Excel file with details like the bounding box coordinates and image paths.

## Features

- Detects text from images using Google Cloud Vision API.
- Annotates images with the detected text and bounding boxes.
- Saves the detected text, bounding box coordinates, and annotated image paths in an Excel file.
- Supports text detection in both Turkish and English languages.

## Requirements

- Python 3.6 or higher
- Google Cloud Vision API credentials
- `opencv-python`, `google-cloud-vision`, `pandas`, `numpy`, `openpyxl`

## 1. Install the necessary libraries:

  Set up your Google Cloud Vision credentials:
  Download the JSON key file from Google Cloud Console.
  Set the path to your credentials file:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path_to_your_google_cloud_credentials.json"
 ```

## Usage
Place your input images in the folder your_input_images_folder.

The output images and the generated Excel file will be saved in the folder your_output_folder.

Run the script:
```bash
python ocr_text_detection.py
```
| Detected Text  | Detected Text  | Detected Text  | Detected Text |
|---|---|---|---|
| ![Annotated Image 1](https://github.com/user-attachments/assets/3a122ec7-63e7-408a-896b-fc9d34cd37c4) | ![Annotated Image 2](https://github.com/user-attachments/assets/8112b06d-7753-47cd-8dad-ed2cd844d38c) | ![Annotated Image 3](https://github.com/user-attachments/assets/109cac88-ac62-4ef3-8b7b-3e31e4451125) | ![Annotated Image 4](https://github.com/user-attachments/assets/654d9c56-548f-4d97-85db-f1c1bef7fdf7) |
| Annotated Image | Annotated Image | Annotated Image | Grouping Texts |




