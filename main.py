# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import PlantDiseaseClassifier
import tempfile, shutil, os
from fastapi.responses import JSONResponse
import requests

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

clf = PlantDiseaseClassifier("plant_disease.onnx")  # loaded once at startup

def get_soil_health(lat: str, lon: str):
    try:
        url = f"https://kaegro.com/farms/api/soil?lat={lat}&lon={lon}"

        response = requests.get(
            url,
            headers={
                "Accept": "application/json",
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        return {
            "soilType": data["soil_type"]["texture_class"],
            "faoClassification": data["soil_type"]["fao_classification"],
            "sandPct": data["physical"]["sand_pct"],
            "siltPct": data["physical"]["silt_pct"],
            "clayPct": data["physical"]["clay_pct"],
            "bulkDensity": data["physical"]["bulk_density_g_cm3"],
            "ph": data["chemical"]["ph_h2o"],
            "organicMatter": data["chemical"]["organic_matter_pct"],
            "nitrogen": data["chemical"]["nitrogen_g_kg"],
            "cec": data["chemical"]["cec_cmol_kg"],
            "capacityFieldVol": data["water"]["capacity_field_vol_pct"],
            "capacityWiltVol": data["water"]["capacity_wilt_vol_pct"],
        }

    except Exception as e:
        print(f"get_soil_health failed (non-fatal): {e}")
        return None


@app.get("/soil")
def soil(lat: float, lon: float):

    if not lat or not lon:
        return jsonify({"error": "lat and lon are required"}), 400

    result = get_soil_health(lat, lon)

    if result is None:
        return JSONResponse({"error": "Failed to fetch soil data"}), 500

    return JSONResponse(result)


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