import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

st.title("Fundbüro Kleidungserkennung 👕")

# Modell laden
model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

def detect_color(image_crop):

    img = cv2.resize(image_crop, (100, 100))

    avg_color = img.mean(axis=0).mean(axis=0)

    b, g, r = avg_color

    if r > g and r > b:
        return "Rot"

    elif g > r and g > b:
        return "Grün"

    elif b > r and b > g:
        return "Blau"

    else:
        return "Schwarz/Dunkel"

if uploaded_file:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        img_array = np.array(image, dtype=np.uint8)

        # RGB -> BGR
        img_array = cv2.cvtColor(
            img_array,
            cv2.COLOR_RGB2BGR
        )

        results = model.predict(
            source=img_array,
            imgsz=640,
            conf=0.25
        )

        found = False

        for r in results:

            for box in r.boxes:

                cls_id = int(box.cls[0])

                label = model.names[cls_id]

                if label == "person":

                    found = True

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    # Oberkörper
                    torso = img_array[
                        y1:int((y1+y2)/2),
                        x1:x2
                    ]

                    color = detect_color(torso)

                    st.image(
                        torso,
                        channels="BGR",
                        caption=f"Farbe: {color}"
                    )

                    st.success(
                        f"Erkannt: {color}"
                    )

        if not found:
            st.warning("Keine Person erkannt.")

    except Exception as e:

        st.error(f"Fehler: {e}")
