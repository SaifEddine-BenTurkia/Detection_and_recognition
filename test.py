import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import pandas as pd

# Use raw string for Windows paths
path = r"images\saif 3.jpg"

# Initialize data structure
data = {"name": [], "embedding": []}

# Read image
image = cv2.imread(path)
if image is None:
    raise FileNotFoundError(f"Could not read image at {path}")

# Convert to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

try:
    # Detect faces
    face_objs = DeepFace.extract_faces(img_path=path, detector_backend="yolov8")
    
    if not face_objs:
        raise ValueError("No faces detected")
    
    # Process first detected face
    face = face_objs[0]
    face_img = face["face"]
    
    # Generate embedding
    embedding = DeepFace.represent(
        face_img, 
        model_name="Facenet",
        enforce_detection=False
    )[0]["embedding"]
    
    # Add to data
    data["name"].append('saif')
    data["embedding"].append(" ".join(map(str, embedding)))
    
    # Draw bounding box
    facial_area = face["facial_area"]
    x = int(facial_area["x"])
    y = int(facial_area["y"])
    w = int(facial_area["w"])
    h = int(facial_area["h"])
    
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

except Exception as e:
    print(f"Error: {str(e)}")
    exit()

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("face_embeddings.csv", index=False)
print("Embeddings saved successfully")

# Display results
plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.axis("off")
plt.title("YOLOv8 Face Detection")
plt.show() 