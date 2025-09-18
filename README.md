# Face Detection and Recognition

This repository contains a facial detection and recognition system using OpenCV and DeepFace. The system detects faces from a webcam stream and recognizes them based on precomputed embeddings.

## Features

- Real-time face detection using RetinaFace.
- Face recognition using DeepFace embeddings.
- GPU acceleration support (CUDA).
- Easy-to-update face database.

## Getting Started

### 1. Clone the repository

```sh
git clone <repo-url>
cd Detection\ and\ recognition
```

### 2. Install dependencies

It is recommended to use a virtual environment:

```sh
python -m venv myenv
myenv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3. Add your own pictures

**You must add your own face images to the `images/` directory.**  
Supported formats: `.jpg`, `.jpeg`, `.png`.  
Each image filename (without extension) will be used as the person's name in the database.

### 4. Generate face embeddings

Run the following script to process the images and generate embeddings:

```sh
python conv_img_to_emedding.py
```

This will create or update the `face_embeddings.csv` file.

### 5. Run the main application

```sh
python main.py
```

A webcam window will open. Detected faces will be recognized if they match the database.

### 6. Test CUDA (optional)

To check if CUDA is available for acceleration:

```sh
python test_cuda.py
```

## Notes

- If you add or remove images in the `images/` folder, rerun `conv_img_to_emedding.py` to update the embeddings.
- Recognition accuracy depends on image quality and lighting.
- The threshold for recognition can be adjusted in `main.py`.

## File Structure

- `images/` — Place your face images here.
- `conv_img_to_emedding.py` — Script to convert images to embeddings.
- `face_embeddings.csv` — Generated face embeddings database.
- `main.py` — Main application for real-time recognition.
- `test_cuda.py` — Script to test CUDA availability.
- `requirements.txt` — Python dependencies.


