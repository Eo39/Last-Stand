import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# YOLO Modell laden
model = YOLO("yolov8n.pt")

st.title("T-Shirt Farb-Erkennung 👕")

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "png", "jpeg"])

def detect_color(image_crop):
    # Bild verkleinern für schnellere Berechnung
    img = cv2.resize(image_crop, (100, 100))
    
    # Durchschnittsfarbe berechnen
    avg_color = img.mean(axis=0).mean(axis=0)  # BGR
    
    b, g, r = avg_color

    # Einfache Logik
    if r > b and r > g:
        return "Rot", r / (r+g+b)
    elif b > r and b > g:
        return "Blau", b / (r+g+b)
    else:
        return "Schwarz/Dunkel", (r+g+b)/3 / 255

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    results = model(img_array)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Nur T-Shirts
            if label in ["t-shirt", "shirt"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                crop = img_array[y1:y2, x1:x2]

                color, confidence = detect_color(crop)

                st.image(crop, caption=f"{color} ({confidence*100:.2f}%)")

                st.write(f"Erkannt als: **{color}** mit {confidence*100:.2f}%")
