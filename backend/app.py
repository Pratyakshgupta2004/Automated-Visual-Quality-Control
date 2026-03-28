from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np
import base64
import io
import cv2
from flask_cors import CORS

# ─── CONFIG ───────────────────────────────
MODEL_PATH = r"C:\avqc_project\runs\pcb_defect_v1\weights\best.pt"
CONFIDENCE = 0.25
IMG_SIZE = 640
# ─────────────────────────────────────────

app = Flask(__name__)
CORS(app)

print("🧠 Loading YOLO model...")
model = YOLO(MODEL_PATH)

# 🔥 Warmup (first request fast karega)
model(np.zeros((IMG_SIZE, IMG_SIZE, 3)), verbose=False)
print("✅ Model ready!")

# 🎨 Colors for defects
COLORS = {
    "missing_hole": (255, 0, 0),
    "mouse_bite": (255, 165, 0),
    "open_circuit": (255, 255, 0),
    "short": (0, 0, 255),
    "spur": (255, 0, 255),
    "spurious_copper": (0, 255, 0),
}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Check file
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # ✅ Read image
        img = Image.open(file.stream).convert("RGB")

        # ⚡ Resize for speed
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_np = np.array(img)

        # 📦 Original image → base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        original_b64 = base64.b64encode(buffer.getvalue()).decode()

        # 🤖 YOLO inference
        results = model(img_np, conf=CONFIDENCE, verbose=False)

        result_img = img_np.copy()
        detections = []

        # 🔍 Process detections
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                color = COLORS.get(cls_name, (255, 255, 255))

                # 🟦 Draw bounding box
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

                label = f"{cls_name} {conf:.0%}"
                cv2.putText(result_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                detections.append({
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })

        # 📦 Result image → base64
        result_pil = Image.fromarray(result_img)
        buffer2 = io.BytesIO()
        result_pil.save(buffer2, format="JPEG")
        result_b64 = base64.b64encode(buffer2.getvalue()).decode()

        # 📊 Response
        return jsonify({
            "status": "OK" if len(detections) == 0 else "DEFECT",
            "total_defects": len(detections),
            "detections": detections,
            "original": original_b64,
            "result": result_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"message": "AVQC Backend Running 🚀"})


if __name__ == "__main__":
    print("\n🌐 Server running: http://localhost:5000")
    app.run(debug=True)