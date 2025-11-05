"""
FastAPI deployment for Skin Disease Classifier with recommendation engine
"""
import os
import io
import numpy as np
from typing import Dict, List
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import config
from src.model import load_model
from src.data_pipeline import prepare_image_from_bytes


# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
MODEL = None


@app.on_event("startup")
async def load_model_on_startup():
    """Load model when API starts"""
    global MODEL

    model_path = config.BEST_MODEL_PATH

    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("Trying final model path...")
        model_path = config.FINAL_MODEL_PATH

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        MODEL = load_model(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Error: No trained model found!")
        print(f"Please train a model first using: python src/train.py")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Skin Disease Classifier API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #34495e;
                    margin-top: 30px;
                }
                .endpoint {
                    background-color: #ecf0f1;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    border-left: 4px solid #3498db;
                }
                .method {
                    color: #27ae60;
                    font-weight: bold;
                }
                code {
                    background-color: #2c3e50;
                    color: #ecf0f1;
                    padding: 2px 6px;
                    border-radius: 3px;
                }
                .classes {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-top: 10px;
                }
                .class-badge {
                    background-color: #3498db;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏥 AI Skin Disease Classifier API</h1>
                <p>Deep learning-powered skin disease classification with medical recommendations</p>

                <h2>Supported Conditions</h2>
                <div class="classes">
                    <div class="class-badge">Acne</div>
                    <div class="class-badge">Carcinoma</div>
                    <div class="class-badge">Eczema</div>
                    <div class="class-badge">Keratosis</div>
                    <div class="class-badge">Milia</div>
                    <div class="class-badge">Rosacea</div>
                </div>

                <h2>API Endpoints</h2>

                <div class="endpoint">
                    <span class="method">GET</span> <code>/</code>
                    <p>This page - API documentation</p>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> <code>/health</code>
                    <p>Health check endpoint</p>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> <code>/classes</code>
                    <p>Get list of supported disease classes</p>
                </div>

                <div class="endpoint">
                    <span class="method">POST</span> <code>/predict</code>
                    <p>Upload an image for disease prediction</p>
                    <p><strong>Request:</strong> Form-data with <code>file</code> parameter (image file)</p>
                    <p><strong>Response:</strong> JSON with prediction, confidence, probabilities, and medical recommendations</p>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> <code>/recommendation/{disease}</code>
                    <p>Get medical recommendations for a specific disease</p>
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> <code>/docs</code>
                    <p>Interactive API documentation (Swagger UI)</p>
                </div>

                <h2>Quick Test</h2>
                <p>Use the following command to test the API:</p>
                <code style="display: block; padding: 10px; margin: 10px 0;">
                    curl -X POST "http://localhost:8000/predict" -F "file=@your_image.jpg"
                </code>

                <h2>⚠️ Medical Disclaimer</h2>
                <p style="color: #e74c3c; font-weight: bold;">
                    This is an AI-powered prediction tool and should NOT replace professional medical diagnosis.
                    Always consult a qualified dermatologist or healthcare provider for proper diagnosis and treatment.
                </p>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = MODEL is not None

    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "api_version": config.API_VERSION,
        "model_name": config.MODEL_NAME,
        "num_classes": config.NUM_CLASSES
    }


@app.get("/classes")
async def get_classes():
    """Get supported disease classes"""
    return {
        "classes": config.CLASSES,
        "num_classes": len(config.CLASSES),
        "descriptions": {
            cls: config.RECOMMENDATIONS[cls]['description']
            for cls in config.CLASSES
        }
    }


@app.get("/recommendation/{disease}")
async def get_recommendation(disease: str):
    """
    Get medical recommendations for a specific disease

    Args:
        disease: Disease name (e.g., 'Acne', 'Carcinoma')

    Returns:
        dict: Medical recommendations
    """
    if disease not in config.RECOMMENDATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Disease '{disease}' not found. Available: {config.CLASSES}"
        )

    return {
        "disease": disease,
        "recommendation": config.RECOMMENDATIONS[disease]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict skin disease from uploaded image

    Args:
        file: Uploaded image file

    Returns:
        dict: Prediction results with recommendations
    """
    # Check if model is loaded
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    try:
        # Read image
        image_bytes = await file.read()

        # Validate image can be opened
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )

        # Prepare image for prediction
        img_array = prepare_image_from_bytes(image_bytes)

        # Make prediction
        predictions = MODEL.predict(img_array, verbose=0)

        # Get results
        pred_class_idx = int(np.argmax(predictions[0]))
        pred_class_name = config.CLASSES[pred_class_idx]
        confidence = float(predictions[0][pred_class_idx])

        # Get all probabilities
        all_probabilities = {
            config.CLASSES[i]: float(predictions[0][i])
            for i in range(len(config.CLASSES))
        }

        # Sort probabilities
        sorted_probs = dict(sorted(
            all_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        # Get recommendation
        recommendation = config.RECOMMENDATIONS[pred_class_name]

        # Check confidence threshold
        low_confidence = confidence < config.CONFIDENCE_THRESHOLD

        # Prepare response
        response = {
            "success": True,
            "prediction": {
                "disease": pred_class_name,
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%",
                "low_confidence_warning": low_confidence
            },
            "all_probabilities": sorted_probs,
            "recommendation": recommendation,
            "medical_disclaimer": (
                "This is an AI-powered prediction and should NOT replace professional "
                "medical diagnosis. Please consult a qualified dermatologist or "
                "healthcare provider for proper diagnosis and treatment."
            )
        }

        # Add warning if low confidence
        if low_confidence:
            response["warning"] = (
                f"Confidence is below threshold ({config.CONFIDENCE_THRESHOLD:.0%}). "
                "Consider getting multiple opinions or professional consultation."
            )

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Predict multiple images at once

    Args:
        files: List of uploaded image files

    Returns:
        dict: Batch prediction results
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )

    results = []
    errors = []

    for idx, file in enumerate(files):
        try:
            # Read and prepare image
            image_bytes = await file.read()
            img_array = prepare_image_from_bytes(image_bytes)

            # Predict
            predictions = MODEL.predict(img_array, verbose=0)
            pred_class_idx = int(np.argmax(predictions[0]))
            pred_class_name = config.CLASSES[pred_class_idx]
            confidence = float(predictions[0][pred_class_idx])

            results.append({
                "filename": file.filename,
                "prediction": pred_class_name,
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%"
            })

        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })

    return {
        "success": True,
        "total_files": len(files),
        "successful_predictions": len(results),
        "failed_predictions": len(errors),
        "results": results,
        "errors": errors if errors else None
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*70)
    print("SKIN DISEASE CLASSIFIER API")
    print("="*70)
    print(f"Starting server...")
    print(f"API Documentation: http://localhost:8000")
    print(f"Swagger UI: http://localhost:8000/docs")
    print(f"ReDoc: http://localhost:8000/redoc")
    print("="*70 + "\n")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
