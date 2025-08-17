"""
AI-Powered Loan Eligibility & Risk Scoring System
FastAPI Backend with REST endpoints for loan default prediction
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import logging
from contextlib import asynccontextmanager

from models.schemas import LoanRequest, PredictionResponse, ModelInsightsResponse
from utils.feature_engineering import engineer_features
from utils.data_validation import validate_loan_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and insights
model = None
model_insights = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and insights on startup"""
    global model, model_insights
    
    try:
        # Load the trained model
        model_path = Path("models/artifacts/loan_default_pipeline.joblib")
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found at {model_path}")
            
        # Load model insights
        insights_path = Path("models/artifacts/model_insights.json")
        if insights_path.exists():
            with open(insights_path, 'r') as f:
                model_insights = json.load(f)
            logger.info("Model insights loaded successfully")
        else:
            logger.error(f"Model insights file not found at {insights_path}")
            
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Loan Risk Scoring System",
    description="REST API for loan default risk prediction and model insights",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    try:
        with open("static/index.html", "r", encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Welcome to Loan Risk Scoring API</h1><p>Static files not found.</p>")

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_loan_risk(loan_data: LoanRequest) -> PredictionResponse:
    """
    Predict loan default risk based on borrower features
    
    Args:
        loan_data: LoanRequest object containing borrower information
        
    Returns:
        PredictionResponse with risk score and prediction
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame([loan_data.dict()])
        
        # Validate data
        validation_errors = validate_loan_data(df)
        if validation_errors:
            raise HTTPException(status_code=400, detail=f"Data validation errors: {validation_errors}")
        
        # Apply feature engineering
        df_engineered = engineer_features(df)
        
        # Make prediction
        prediction = model.predict(df_engineered)[0]
        risk_score = model.predict_proba(df_engineered)[0][1]  # Probability of default
        
        # Determine risk category
        if risk_score < 0.3:
            risk_category = "Low Risk"
        elif risk_score < 0.7:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"
        
        logger.info(f"Prediction made: risk_score={risk_score:.4f}, prediction={prediction}")
        
        return PredictionResponse(
            prediction=int(prediction),
            risk_score=float(risk_score),
            risk_category=risk_category,
            recommendation="Approved" if prediction == 0 else "Rejected"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/insights", response_model=ModelInsightsResponse)
async def get_model_insights() -> ModelInsightsResponse:
    """
    Get model performance metrics and feature importance
    
    Returns:
        ModelInsightsResponse with performance metrics and feature importance
    """
    try:
        if model_insights is None:
            raise HTTPException(status_code=500, detail="Model insights not loaded")
        
        return ModelInsightsResponse(**model_insights)
        
    except Exception as e:
        logger.error(f"Insights error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")

@app.get("/api/charts/{chart_name}")
async def get_chart(chart_name: str):
    """
    Serve model performance charts
    
    Args:
        chart_name: Name of the chart (confusion_matrix.png or roc_curve.png)
        
    Returns:
        Chart image file
    """
    try:
        chart_path = Path(f"models/artifacts/charts/{chart_name}")
        if not chart_path.exists():
            raise HTTPException(status_code=404, detail="Chart not found")
        
        return FileResponse(chart_path)
        
    except Exception as e:
        logger.error(f"Chart error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chart: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "insights_loaded": model_insights is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)