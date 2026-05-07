import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Fundbüro",
    page_icon="💀",
    layout="centered"
)

# ----------------------------
# PIXEL STYLE CSS
# ----------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

html, body, [class*="css"]  {
    background-color: #111111;
    color: #f5f5f5;
    font-family: 'Press Start 2P', cursive;
}

/* MAIN CONTAINER */
.main {
    background: linear-gradient(
        180deg,
        #161616 0%,
        #0f0f0f 100%
    );
}

/* TITLE */
.pixel-title {
    text-align: center;
    font-size: 38px;
    color: #ffffff;
    text-shadow:
        4px 4px 0px #ff004c,
        8px 8px 0px #00d9ff;
    margin-top: 20px;
    margin-bottom: 40px;
}

/* SUBTEXT */
.pixel-sub {
    text-align: center;
    color: #bbbbbb;
    font-size: 12px;
    margin-bottom: 30px;
}

/* BUTTON */
.stButton>button {
    background-color: #ff004c;
    color: white;
    border: 4px solid white;
    border-radius: 0px;
    padding: 12px 20px;
    font-family: 'Press Start 2P', cursive;
    font-size: 12px;
    box-shadow: 6px 6px 0px #000000;
}

.stButton>button:hover {
    background-color: #00d9ff;
    color: black;
}

/* FILE UPLOADER */
section[data-testid="stFileUploader"] {
    border: 3px dashed #ff004c;
    padding: 20px;
    background-color: #1b1b1b;
}

/* SUCCESS BOX */
.stSuccess {
    background-color: #1e4620;
    border: 3px solid #00ff66;
}

/* WARNING BOX */
.stWarning {
    background-color: #4d3a00;
    border: 3px solid #ffcc00;
}

/* IMAGE */
img {
    border: 4px solid white;
    image-rendering: pixelated;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------

st.markdown("""
<div class="pixel-title">
💀 FUNDBÜRO 💀
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pixel-sub">
Verlorene Kleidung erkennen wie ein Dungeon Loot Scanner
</div>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------

model = YOLO("yolov8n.pt")

# ----------------------------
# UPLOADER
# ----------------------------

uploaded_file = st.file_uploader(
    "BILD HOCHLADEN",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# COLOR DETECTION
# ----------------------------

def detect_color(image_crop):

    img = cv2.resize(image_crop, (100, 100))

    avg_color = img.mean(axis=0).mean(axis=0)

    b, g, r = avg_color

    if r > g and r > b:
        return "ROT 🔴"

    elif g > r and g > b:
        return "GRÜN 🟢"

    elif b > r and b > g:
        return "BLAU 🔵"

    else:
        return "SCHWARZ ⚫"

# ----------------------------
# IMAGE PROCESSING
# ----------------------------

if uploaded_file:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        img_array = np.array(image, dtype=np.uint8)

        img_array = cv2.cvtColor(
            img_array,
            cv2.COLOR_RGB2BGR
        )

        with st.spinner("Dungeon Scan läuft..."):

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

                    torso = img_array[
                        y1:int((y1+y2)/2),
                        x1:x2
                    ]

                    color = detect_color(torso)

                    st.image(
                        torso,
                        channels="BGR",
                        caption=f"ERKANNTE FARBE: {color}"
                    )

                    st.success(
                        f"ITEM GEFUNDEN → {color}"
                    )

        if not found:
            st.warning("KEIN CHARAKTER ERKANNT")

    except Exception as e:

        st.error(f"FEHLER: {e}")
