from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import logging
from datetime import datetime
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pneumonia Detection API",
    description="AI-powered chest X-ray pneumonia detection using ResNet50",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your mobile app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
try:
    model = load_model('outputs/models/best_model.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Response models
class PredictionResponse(BaseModel):
    success: bool
    timestamp: str
    diagnosis: str
    confidence: float
    risk_level: str
    probability_scores: dict
    recommendations: list
    model_info: dict
    disclaimer: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    timestamp: str

@app.get("/", response_model=dict)
def root():
    """API root endpoint"""
    return {
        "message": "Pneumonia Detection API",
        "version": "1.0.0",
        "author": "Somveer Kaidwal",
        "institution": "RV Institute of Technology, Bijnor",
        "endpoints": {
            "POST /predict": "Upload chest X-ray for pneumonia detection",
            "GET /health": "Check API health status",
            "GET /docs": "Interactive API documentation",
            "GET /model-info": "Get model performance metrics"
        }
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.get("/model-info")
def model_info():
    """Get detailed model performance information"""
    return {
        "model_architecture": "ResNet50 (Transfer Learning)",
        "input_size": "224x224x3",
        "training_dataset": "Chest X-ray Pneumonia (Kaggle)",
        "performance_metrics": {
            "recall_sensitivity": "98.97%",
            "precision": "76%",
            "accuracy": "80%",
            "false_negative_rate": "1.03%",
            "specificity": "47.44%"
        },
        "training_details": {
            "epochs": 50,
            "optimizer": "Adam",
            "loss_function": "Binary Cross-Entropy",
            "best_epoch": 40,
            "validation_loss": 0.1358
        },
        "limitations": [
            "Trained on pediatric patients (ages 1-5)",
            "Binary classification only (NORMAL vs PNEUMONIA)",
            "Single institution dataset",
            "Requires clinical validation"
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_pneumonia(file: UploadFile = File(...)):
    """
    Predict pneumonia from chest X-ray image
    
    Args:
        file: Uploaded image file (JPEG/PNG)
    
    Returns:
        PredictionResponse with diagnosis and confidence
    """
    
    # Validate model loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an image (JPEG/PNG)"
        )
    
    try:
        # Read and preprocess image
        contents = await file.read()
        
        # Check file size (limit to 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        img = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize and normalize
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        logger.info(f"Processing prediction for file: {file.filename}")
        prediction_prob = float(model.predict(img_array, verbose=0)[0][0])
        
        # Interpret results
        if prediction_prob > 0.5:
            diagnosis = "PNEUMONIA"
            confidence = prediction_prob * 100
            risk_level = "HIGH" if confidence > 90 else "MODERATE"
            recommendations = [
                "Immediate consultation with a radiologist recommended",
                "Further diagnostic tests may be required (blood work, sputum culture)",
                "Monitor symptoms: fever, cough, difficulty breathing",
                "Consider antibiotic treatment if bacterial pneumonia suspected",
                "Follow-up chest X-ray after treatment"
            ]
        else:
            diagnosis = "NORMAL"
            confidence = (1 - prediction_prob) * 100
            risk_level = "LOW"
            recommendations = [
                "No signs of pneumonia detected",
                "Continue regular health monitoring",
                "Consult doctor if symptoms develop",
                "Maintain good respiratory hygiene"
            ]
        
        response = PredictionResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            diagnosis=diagnosis,
            confidence=round(confidence, 2),
            risk_level=risk_level,
            probability_scores={
                "NORMAL": round((1 - prediction_prob) * 100, 2),
                "PNEUMONIA": round(prediction_prob * 100, 2)
            },
            recommendations=recommendations,
            model_info={
                "model_name": "ResNet50-PneumoniaDetector",
                "recall": "98.97%",
                "false_negative_rate": "1.03%",
                "note": "High recall ensures minimal missed cases"
            },
            disclaimer="This is an AI screening tool. Final diagnosis must be made by qualified medical professionals. Not a replacement for professional medical advice."
        )
        
        logger.info(f"Prediction: {diagnosis} ({confidence:.2f}%)")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Batch prediction for multiple X-ray images
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    for file in files:
        try:
            result = await predict_pneumonia(file)
            results.append({
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"batch_results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
