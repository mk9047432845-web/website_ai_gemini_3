from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, io, requests, tempfile
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import torch
from torchvision import models, transforms

# =====================
# CONFIG
# =====================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_LABELS = ["benign", "malignant", "normal"]
ALLOWED_EXT = {"jpg", "jpeg", "png"}
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# HUGGING FACE FILES
# =====================
HF_BASE = "https://huggingface.co/mani880740255/skin_care_tflite/resolve/main/"

HF_MODELS = {
    "tflite": HF_BASE + "skin_model_quantized.tflite",
    "mobilenetv2": "skin_cancer_mobilenetv2%20(1).h5",
    "b3": HF_BASE + "efficientnet_b3_skin_cancer.pth"
}

# =====================
# CHATBOT DATA
# =====================
# =====================
# UPDATED CHATBOT DATA
# =====================
CHAT_RESPONSES = {
    "what is skin care?": "Skin care is the practice of maintaining healthy, clean, and protected skin through proper hygiene and protection.",
    "why is skin care important?": "Proper skin care helps prevent infections, premature aging, and various skin diseases.",
    "what is a benign lesion?": "A benign skin lesion is a non-cancerous growth that does not spread to other parts of the body.",
    "what is a malignant lesion?": "A malignant skin lesion is a cancerous growth that can spread and damage surrounding tissues.",
    "difference: benign vs malignant": "Benign lesions are non-cancerous and generally harmless, while malignant lesions are cancerous and dangerous.",
    "signs of skin cancer": "Common signs include irregular shapes, color changes, bleeding, and rapid growth of a mole or spot.",
    "can benign turn malignant?": "While most benign lesions stay that way, some can become malignant if not monitored or treated properly.",
    "what causes skin cancer?": "Skin cancer is mainly caused by prolonged exposure to ultraviolet (UV) radiation from the sun or tanning beds.",
    "how to prevent skin cancer?": "Prevention involves using sunscreen (SPF 30+), wearing protective clothing, and avoiding excessive sun exposure.",
    "why is early detection key?": "Early detection significantly increases treatment success rates and reduces the risk of the cancer spreading to other organs."
}

# =====================
# HELPERS
# =====================
def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def download_file(url):
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"Model download failed: {url}")
    return io.BytesIO(r.content)

# =====================
# MODEL PREDICTIONS
# =====================
def predict_tflite(img_path):
    model_bytes = download_file(HF_MODELS["tflite"])
    interpreter = tf.lite.Interpreter(model_content=model_bytes.read())
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(preds))
    return CLASS_LABELS[idx], float(preds[idx]), preds.tolist()

def predict_keras(img_path):
    model_bytes = download_file(HF_MODELS["mobilenetv2"])
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(model_bytes.read())
        tmp_path = tmp.name
    try:
        model = tf.keras.models.load_model(tmp_path)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224,224))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)
        preds = model.predict(img)[0]
        idx = int(np.argmax(preds))
        return CLASS_LABELS[idx], float(preds[idx]), preds.tolist()
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def predict_b3(img_path):
    model_bytes = download_file(HF_MODELS["b3"])
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Linear(1536, 3)
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp.write(model_bytes.read())
        tmp_path = tmp.name
    try:
        state_dict = torch.load(tmp_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device); model.eval()
        transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor()])
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img)
            probs = torch.softmax(out, dim=1)[0]
        idx = int(torch.argmax(probs))
        return CLASS_LABELS[idx], float(probs[idx]), probs.cpu().tolist()
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# =====================
# ROUTES
# =====================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "model" not in request.form:
        return jsonify({"error": "image + model required"}), 400
    model_choice = request.form["model"]
    file = request.files["image"]
    if model_choice not in HF_MODELS or not allowed_file(file.filename):
        return jsonify({"error": "invalid model/file"}), 400
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    try:
        if model_choice == "tflite": pred, conf, probs = predict_tflite(path)
        elif model_choice == "mobilenetv2": pred, conf, probs = predict_keras(path)
        else: pred, conf, probs = predict_b3(path)
        os.remove(path)
        return jsonify({
            "model_used": model_choice, "prediction": pred, "confidence": conf,
            "probabilities": {CLASS_LABELS[i]: probs[i] for i in range(3)}
        })
    except Exception as e:
        if os.path.exists(path): os.remove(path)
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "").lower().strip()
    response = CHAT_RESPONSES.get(user_msg, "I only understand specific skin health questions. Try the buttons below!")
    return jsonify({"reply": response, "suggestions": list(CHAT_RESPONSES.keys())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)