# To run, don't forget to change into directory and then run uvicorn API:app --reload

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
#import cv2
#import PIL
from PIL import Image
import numpy as np
import tensorflow as tf
import io
#import base64
#import json
import os
from metrics_and_loss import CombinedLoss, M_IoUMetric


# Load your trained model
#try:
#    model = tf.keras.models.load_model('inception_aug.keras')
#    print("Model loaded successfully!")
#except Exception as e:
#    print(f"Error loading model: {e}")
#    model = None

app = FastAPI()

# --- Chargement du modèle avec custom_objects ---
base = os.path.dirname(__file__)
MODEL_PATH = os.path.join(base, "inception_aug.keras")
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "CombinedLoss": CombinedLoss,
            "M_IoUMetric": M_IoUMetric
        }
    )
    print(f"Modèle chargé depuis {MODEL_PATH}")
except Exception as e:
    print(f"Échec du chargement du modèle : {e}")
    model = None



# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- helpers ---
def preprocess_image(image_np: np.ndarray):
    """Preprocesses the input image (NumPy array) for the model."""
    #resized_image = cv2.resize(image_np, (512, 512)) # Modified to 512x512
    #resized_image = image_np.resize((512, 512))
    normalized_image = image_np / 255.0
    expanded_image = np.expand_dims(normalized_image, axis=0) # Add batch dimension
    return expanded_image

def postprocess_mask_indices(prediction: np.ndarray):
    """Returns the raw mask indices as a list."""
    mask = np.squeeze(prediction)
    mask_index = np.argmax(mask, axis=-1).astype(np.uint8)
    return mask_index.tolist() # Convert to list for JSON serialization

# --- routes ---
@app.post("/predict/")
async def predict(image_file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(content={'error': 'Model not loaded'}, status_code=500)

    if not image_file.filename:
        return JSONResponse(content={'error': 'No image selected'}, status_code=400)

    try:
        contents = await image_file.read()
        #nparr = np.frombuffer(contents, np.uint8)
        #image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        buffer = io.BytesIO(contents)
        image_np = Image.open(buffer).convert('RGB')
        image_np = image_np.resize((512, 512))

        if image_np is None:
            return JSONResponse(content={'error': 'Failed to decode image'}, status_code=400)
        image_np = np.array(image_np)
        processed_image = preprocess_image(image_np)
        prediction = model.predict(processed_image)
        mask_indices = postprocess_mask_indices(prediction)

        return JSONResponse(content={'mask_indices': mask_indices}, status_code=200) # Return raw indices

    except Exception as e:
        return JSONResponse(content={'error': f'Prediction error: {e}'}, status_code=500)


