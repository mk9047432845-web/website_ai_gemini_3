from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, requests, gc, io, tempfile
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import torch
from torchvision import models, transforms

# =====================
# CONFIG & PATHS
# =====================
app = Flask(__name__)
CORS(app)

MODEL_DIR = "models_cache"
UPLOAD_FOLDER = "uploads"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_LABELS = ["benign", "malignant", "normal"]
ALLOWED_EXT = {"jpg", "jpeg", "png"}

# URLs for models
HF_BASE = "https://huggingface.co/mani880740255/skin_care_tflite/resolve/main/"
URLS = {
    "tflite": HF_BASE + "skin_model_quantized.tflite",
    "mobilenetv2": HF_BASE + "skin_cancer_mobilenetv2%20(1).h5",
    "b3": HF_BASE + "efficientnet_b3_skin_cancer.pth"
}

# Local Paths
PATHS = {
    "tflite": os.path.join(MODEL_DIR, "skin_model.tflite"),
    "mobilenetv2": os.path.join(MODEL_DIR, "mobilenetv2.h5"),
    "b3": os.path.join(MODEL_DIR, "efficientnet_b3.pth")
}

# =====================
# FULL CHATBOT DATA
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
# HELPERS & CACHE MANAGER
# =====================
def ensure_model_exists(model_key):
    """Checks if model exists locally; if not, downloads it once."""
    path = PATHS[model_key]
    if not os.path.exists(path):
        print(f"Downloading {model_key} model... please wait.")
        r = requests.get(URLS[model_key], stream=True)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            raise Exception(f"Failed to download model from {URLS[model_key]}")
    return path

def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# =====================
# PREDICTION ENGINES
# =====================

def predict_tflite(img_path):
    model_path = ensure_model_exists("tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    img = cv2.imread(img_path)
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]
    
    return int(np.argmax(preds)), preds.tolist()

def predict_keras(img_path):
    model_path = ensure_model_exists("mobilenetv2")
    model = tf.keras.models.load_model(model_path)
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_arr = np.expand_dims(np.array(img)/255.0, axis=0)
    
    preds = model.predict(img_arr)[0]
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return int(np.argmax(preds)), preds.tolist()

def predict_b3(img_path):
    model_path = ensure_model_exists("b3")
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Linear(1536, 3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])
    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
    
    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)[0].tolist()
    
    del model
    gc.collect()
    return int(np.argmax(probs)), probs

# =====================
# ROUTES
# =====================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "model" not in request.form:
        return jsonify({"error": "image + model selection required"}), 400
    
    model_choice = request.form["model"]
    file = request.files["image"]
    
    if not (file and allowed_file(file.filename)):
        return jsonify({"error": "Invalid file type"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    
    try:
        if model_choice == "tflite":
            idx, probs = predict_tflite(path)
        elif model_choice == "mobilenetv2":
            idx, probs = predict_keras(path)
        elif model_choice == "b3":
            idx, probs = predict_b3(path)
        else:
            return jsonify({"error": "Invalid model choice"}), 400

        return jsonify({
            "model_used": model_choice,
            "prediction": CLASS_LABELS[idx],
            "confidence": float(probs[idx]),
            "probabilities": {CLASS_LABELS[i]: probs[i] for i in range(3)}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path): os.remove(path)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "").lower().strip()
    response = CHAT_RESPONSES.get(user_msg, "I only understand specific skin health questions. Try the buttons below!")
    return jsonify({"reply": response, "suggestions": list(CHAT_RESPONSES.keys())})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
