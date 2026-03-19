import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import streamlit as st
from tf_keras.models import model_from_json
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

# ------------------ LOAD MODELS ------------------

def load_legacy_h5_model(model_path: Path):
    with h5py.File(model_path, "r") as h5_file:
        config = h5_file.attrs["model_config"]
        if isinstance(config, bytes):
            config = config.decode("utf-8")

    model_config = json.loads(config)

    for layer in model_config.get("config", {}).get("layers", []):
        layer_config = layer.get("config", {})
        dtype = layer_config.get("dtype")

        if isinstance(dtype, dict) and dtype.get("class_name") == "DTypePolicy":
            layer_config["dtype"] = dtype.get("config", {}).get("name", "float32")

        if layer.get("class_name") == "InputLayer":
            if "batch_shape" in layer_config:
                layer_config["batch_input_shape"] = layer_config.pop("batch_shape")
            layer_config.pop("optional", None)

        if layer.get("class_name") == "Dense":
            layer_config.pop("quantization_config", None)

    model = model_from_json(json.dumps(model_config))
    model.load_weights(model_path)
    return model

@st.cache_resource(show_spinner=False)
def load_models():
    yolo_model = YOLO("yolov8n.pt")
    drowsy_model = load_legacy_h5_model(BASE_DIR / "drowsiness_model.h5")

    age_net = cv2.dnn.readNetFromCaffe(
        str(BASE_DIR / "age_model" / "age_deploy.prototxt"),
        str(BASE_DIR / "age_model" / "age_net.caffemodel")
    )

    return yolo_model, drowsy_model, age_net

yolo_model, drowsy_model, age_net = load_models()

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# ------------------ SESSION STATE ------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "image" not in st.session_state:
    st.session_state.image = None

# ------------------ FUNCTIONS ------------------

def predict_age(face_img):
    try:
        face_img = cv2.resize(face_img, (227, 227))
        blob = cv2.dnn.blobFromImage(
            face_img,
            1.0,
            (227, 227),
            (78.426, 87.768, 114.896),
            swapRB=False
        )
        age_net.setInput(blob)
        preds = age_net.forward()
        return AGE_LIST[preds[0].argmax()]
    except:
        return "Unknown"


def process_image(image):

    frame = image.copy()
    h, w, _ = frame.shape

    results = yolo_model(frame)

    persons = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2 - x1) < 60 or (y2 - y1) < 60:
                    continue

                persons.append((x1, y1, x2, y2))

    sleeping_count = 0
    sleeping_ages = []

    for (x1, y1, x2, y2) in persons:

        y_mid = y1 + (y2 - y1) // 2
        person_img = frame[y1:y_mid, x1:x2]

        if person_img.size == 0:
            continue

        img = cv2.resize(person_img, (64, 64))
        img = img / 255.0
        img = img.reshape(1, 64, 64, 3)

        pred = drowsy_model.predict(img, verbose=0)[0]
        sleep_prob = pred[1]

        if sleep_prob > 0.7:
            sleeping_count += 1

            age = predict_age(person_img)
            sleeping_ages.append(age)

            color = (0, 0, 255)
            text = f"Sleeping | Age: {age}"
        else:
            color = (0, 255, 0)
            text = "Awake"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, len(persons), sleeping_count, sleeping_ages


# ------------------ UI ------------------

st.title("Drowsiness Detection System")
st.write("Detect sleeping people and their age from an image")

# ------------------ UPLOAD SECTION ------------------

st.header("Upload Image")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection", type="primary"):

        with st.spinner("Processing..."):

            try:
                output, total, sleeping, ages = process_image(image)

                st.session_state.result = (output, total, sleeping, ages)

            except Exception as e:
                st.error(f"Error: {e}")

# ------------------ RESULT DISPLAY ------------------

if st.session_state.result is not None:

    output, total, sleeping, ages = st.session_state.result

    st.subheader("Result")

    st.image(output, caption="Processed Image", use_column_width=True)

    st.write(f"Total People: {total}")
    st.write(f"Sleeping People: {sleeping}")

    # 🔥 ALERT (POPUP STYLE)
    if sleeping > 0:
        st.warning(f"⚠️ ALERT: {sleeping} people are sleeping!")
        st.write("Ages:", ages)
    else:
        st.success("All people are awake")

# ------------------ FOOTER ------------------

st.markdown("*Drowsiness Detection | YOLO + CNN + Age Model*")
