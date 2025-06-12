# To run, don't forget to change into directory and then run uicorn API:app --reload

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tensorflow as tf
import io
import base64
import uvicorn
import json
from metrics_and_loss import CombinedLoss, IoUMetric


# Load your trained model
try:
    model = tf.keras.models.load_model('../models_in_progress/second_model.keras')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = FastAPI()

def preprocess_image(image_np: np.ndarray):
    """Preprocesses the input image (NumPy array) for the model."""
    resized_image = cv2.resize(image_np, (512, 512)) # Modified to 512x512
    normalized_image = resized_image / 255.0
    expanded_image = np.expand_dims(normalized_image, axis=0) # Add batch dimension
    return expanded_image

def postprocess_mask_indices(prediction: np.ndarray):
    """Returns the raw mask indices as a list."""
    mask = np.squeeze(prediction)
    mask_index = np.argmax(mask, axis=-1).astype(np.uint8)
    return mask_index.tolist() # Convert to list for JSON serialization

@app.post("/predict/")
async def predict(image_file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(content={'error': 'Model not loaded'}, status_code=500)

    if not image_file.filename:
        return JSONResponse(content={'error': 'No image selected'}, status_code=400)

    try:
        contents = await image_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_np is None:
            return JSONResponse(content={'error': 'Failed to decode image'}, status_code=400)

        processed_image = preprocess_image(image_np)
        prediction = model.predict(processed_image)
        mask_indices = postprocess_mask_indices(prediction)

        return JSONResponse(content={'mask_indices': mask_indices}, status_code=200) # Return raw indices

    except Exception as e:
        return JSONResponse(content={'error': f'Prediction error: {e}'}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
