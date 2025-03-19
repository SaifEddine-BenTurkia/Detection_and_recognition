import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepface import DeepFace

# Configuration
IMAGE_DIR = r"images"  # Raw string for Windows paths
OUTPUT_CSV = "face_embeddings.csv"


def process_images():
    data = {"name": [], "embedding": []}
    valid_images = 0

    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
 
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, img_file)
        print(f"Processing {img_file} ({idx+1}/{len(image_files)})...")

        try:
            # Detect faces with retinaface (more reliable than yolov8)
            faces = DeepFace.extract_faces(img_path, detector_backend="retinaface" , align = True)
            
            if not faces:
                print(f"  No faces detected in {img_file}")
                continue

            # Process first detected face
            face = faces[0]
            face_img = face["face"]
            
            # Generate embedding
            embedding = DeepFace.represent(
                face_img, 
                model_name="ArcFace",
                enforce_detection=False
            )[0]["embedding"]
            
            name = os.path.splitext(img_file)[0]

            # Store data
            data["name"].append(name)
            data["embedding"].append(" ".join(map(str, embedding)))
            valid_images += 1


        except Exception as e:
            print(f"  Error processing {img_file}: {str(e)}")

    # Save embeddings
    if valid_images > 0:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccessfully processed {valid_images}/{len(image_files)} images")
        print(f"Embeddings saved to {OUTPUT_CSV}")
    else:
        print("No valid images processed")

if __name__ == "__main__":
    process_images()