import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="Fundbüro",
    page_icon="🐋",
    layout="centered"
)

# --------------------------------
# PIXEL OCEAN STYLE
# --------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

html, body, [class*="css"]  {
    font-family: 'Press Start 2P', cursive;
    overflow-x: hidden;
}

/* BACKGROUND */
.stApp {
    background: linear-gradient(
        180deg,
        #001f3f 0%,
        #003f7f 40%,
        #0074D9 100%
    );
    color: white;
}

/* WAVES */
.wave {
    position: fixed;
    width: 200%;
    height: 200px;
    background: rgba(255,255,255,0.05);
    top: 0;
    left: 0;
    border-radius: 45%;
    animation: wave 12s linear infinite;
    z-index: 0;
}

.wave2 {
    top: 100px;
    animation-duration: 18s;
    opacity: 0.4;
}

@keyframes wave {
    from {
        transform: translateX(0);
    }
    to {
        transform: translateX(-50%);
    }
}

/* WHALES */
.whale {
    position: fixed;
    font-size: 70px;
    animation: swim 25s linear infinite;
    z-index: 1;
    opacity: 0.9;
}

.whale2 {
    top: 70%;
    animation-duration: 32s;
    font-size: 55px;
}

.whale3 {
    top: 40%;
    animation-duration: 20s;
    font-size: 80px;
}

@keyframes swim {
    from {
        transform: translateX(-20vw);
    }
    to {
        transform: translateX(120vw);
    }
}

/* CONTENT */
.main-container {
    position: relative;
    z-index: 5;
}

/* TITLE */
.pixel-title {
    text-align: center;
    font-size: 42px;
    color: #ffffff;
    margin-top: 20px;
    margin-bottom: 20px;

    text-shadow:
        4px 4px 0px #001f3f,
        8px 8px 0px #00c3ff;
}

/* SUBTITLE */
.pixel-sub {
    text-align: center;
    color: #d6f4ff;
    font-size: 12px;
    margin-bottom: 35px;
}

/* FILE UPLOADER */
section[data-testid="stFileUploader"] {
    border: 4px dashed #7FDBFF;
    background-color: rgba(0,0,0,0.25);
    padding: 25px;
    border-radius: 0px;
}

/* BUTTON */
.stButton>button {
    background-color: #39CCCC;
    color: black;
    border: 4px solid white;
    border-radius: 0px;
    font-family: 'Press Start 2P', cursive;
    box-shadow: 6px 6px 0px #001f3f;
}

.stButton>button:hover {
    background-color: #7FDBFF;
    color: black;
}

/* SUCCESS */
.stSuccess {
    background-color: rgba(0, 255, 200, 0.2);
    border: 3px solid #39CCCC;
}

/* WARNING */
.stWarning {
    background-color: rgba(255,255,0,0.2);
    border: 3px solid yellow;
}

/* IMAGES */
img {
    border: 4px solid white;
    image-rendering: pixelated;
}

</style>

<div class="wave"></div>
<div class="wave wave2"></div>

<div class="whale" style="top:20%;">🐋</div>
<div class="whale whale2">🐳</div>
<div class="whale whale3">🐋</div>

""", unsafe_allow_html=True)

# --------------------------------
# HEADER
# --------------------------------

st.markdown("""
<div class="main-container">

<div class="pixel-title">
🐋 FUNDBÜRO 🐋
</div>

<div class="pixel-sub">
Verlorene Kleidung aus den Tiefen des Ozeans finden
</div>

</div>
""", unsafe_allow_html=True)

# --------------------------------
# MODEL
# --------------------------------

model = YOLO("yolov8n.pt")

# --------------------------------
# UPLOAD
# --------------------------------

uploaded_file = st.file_uploader(
    "BILD HOCHLADEN",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------
# COLOR DETECTION
# --------------------------------

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

# --------------------------------
# PROCESS IMAGE
# --------------------------------

if uploaded_file:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        img_array = np.array(image, dtype=np.uint8)

        img_array = cv2.cvtColor(
            img_array,
            cv2.COLOR_RGB2BGR
        )

        with st.spinner("🐋 OZEANSCAN LÄUFT..."):

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
                        caption=f"GEFUNDENE FARBE: {color}"
                    )

                    st.success(
                        f"🐳 ITEM GEFUNDEN → {color}"
                    )

        if not found:
            st.warning("KEIN CHARAKTER IM OZEAN GEFUNDEN")

    except Exception as e:

        st.error(f"FEHLER: {e}")
