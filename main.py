# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import PlantDiseaseClassifier
import tempfile, shutil, os
from fastapi.responses import JSONResponse

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

clf = PlantDiseaseClassifier("plant_disease.onnx")  # loaded once at startup

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#         shutil.copyfileobj(file.file, tmp)
#         tmp_path = tmp.name
#     try:
#         result = clf.predict(tmp_path)
#         return result
#     finally:
#         os.unlink(tmp_path)

# @app.post("/predict")
# def predict():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files["image"]

#     if file.filename == "":
#         return jsonify({"error": "Empty filename"}), 400

#     # ── Detect extension from uploaded file ───────────────────────────────────
#     ext = os.path.splitext(file.filename or "")[-1].lower()
#     allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

#     if ext not in allowed:
#         return jsonify({"error": f"Unsupported file type: {ext}"}), 415

#     # ── Save to temp file with correct extension ──────────────────────────────
#     with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
#         file.save(tmp.name)
#         tmp_path = tmp.name

#     try:
#         result = clf.predict(tmp_path)
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         os.unlink(tmp_path)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    ext = os.path.splitext(image.filename or "")[-1].lower()

    if ext not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {ext}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = clf.predict(tmp_path)
        return JSONResponse(content=result)
        # print(result)
        # return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/health")
def health():
    return {
        "health": "ok"
    }