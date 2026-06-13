import onnxruntime as ort
import numpy as np
from PIL import Image
import json

# ─── Class Names (must match training order) ───────────────────────────────────
CLASS_NAMES = sorted([
    "Corn_Common_Rust", "Corn_Gray_Leaf_Spot", "Corn_Healthy",
    "Corn_Northern_Leaf_Blight", "Pepper_Bacterial_Spot", "Pepper_Healthy",
    "Potato_Early_Blight", "Potato_Healthy", "Potato_Late_Blight",
    "Rice_Brown_Spot", "Rice_Healthy", "Rice_Hispa", "Rice_Leaf_Blast",
    "Rice_Neck_Blast", "Sugarcane_Bacterial_Blight", "Sugarcane_Healthy",
    "Sugarcane_Red_Rot", "Tomato_Bacterial_Spot", "Tomato_Early_Blight",
    "Tomato_Healthy", "Tomato_Late_Blight", "Tomato_Leaf_Mold",
    "Tomato_Mosaic_Virus", "Tomato_Septoria_Leaf_Spot", "Tomato_Spider_Mites",
    "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Wheat_Aphid",
    "Wheat_Black_Rust", "Wheat_Blast", "Wheat_Brown_Rust",
    "Wheat_Common_Root_Rot", "Wheat_Fusarium_Head_Blight", "Wheat_Healthy",
    "Wheat_Mildew", "Wheat_Mite", "Wheat_Septoria", "Wheat_Tan_Spot",
    "Wheat_Yellow_Rust"
])

# ─── Severity Rules ─────────────────────────────────────────────────────────────
# Based on confidence + disease type
# Healthy classes always → None
# Confidence thresholds: >= 0.90 → severe, >= 0.70 → moderate, < 0.70 → mild

HEALTHY_CLASSES = {
    "Corn_Healthy", "Pepper_Healthy", "Potato_Healthy",
    "Rice_Healthy", "Sugarcane_Healthy", "Tomato_Healthy", "Wheat_Healthy"
}

# Diseases known to spread rapidly → bump severity one level up
HIGH_SPREAD_DISEASES = {
    "Rice_Leaf_Blast", "Rice_Neck_Blast", "Tomato_Late_Blight",
    "Potato_Late_Blight", "Wheat_Blast", "Wheat_Yellow_Rust",
    "Tomato_Yellow_Leaf_Curl_Virus"
}

def get_severity(disease: str, confidence: float) -> str:
    if disease in HEALTHY_CLASSES:
        return "None"

    # Base severity from confidence
    if confidence >= 0.90:
        severity = "Severe"
    elif confidence >= 0.70:
        severity = "Moderate"
    else:
        severity = "Mild"

    # Bump up one level for fast-spreading diseases
    if disease in HIGH_SPREAD_DISEASES:
        if severity == "Mild":
            severity = "Moderate"
        elif severity == "Moderate":
            severity = "Severe"

    return severity


# ─── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)

    arr = np.array(img).astype(np.float32) / 255.0

    # ImageNet normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std

    # HWC → CHW → NCHW
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)

    return arr


# ─── Softmax ────────────────────────────────────────────────────────────────────
def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


# ─── Main Inference ─────────────────────────────────────────────────────────────
class PlantDiseaseClassifier:

    def __init__(self, model_path: str = "plant_disease.onnx"):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_path: str) -> dict:
        # Preprocess
        input_tensor = preprocess(image_path)

        # Run ONNX inference
        logits = self.session.run(None, {self.input_name: input_tensor})[0][0]

        # Probabilities
        probs = softmax(logits)

        # Top prediction
        top_idx        = int(np.argmax(probs))
        top_confidence = float(probs[top_idx])
        disease        = CLASS_NAMES[top_idx]
        severity       = get_severity(disease, top_confidence)

        # Top-3 predictions (useful for debugging / low-confidence cases)
        top3_idx  = np.argsort(probs)[::-1][:3]
        top3      = [
            {
                "disease":    CLASS_NAMES[i],
                "confidence": round(float(probs[i]) * 100, 2)
            }
            for i in top3_idx
        ]

        return {
            "disease":        disease,
            "confidence_pct": round(top_confidence * 100, 2),
            "severity":       severity,
            "is_healthy":     disease in HEALTHY_CLASSES,
            "top3":           top3
        }


# ─── CLI usage ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "plant_disease.onnx"

    classifier = PlantDiseaseClassifier(model_path)
    result     = classifier.predict(image_path)

    print(json.dumps(result, indent=2))
