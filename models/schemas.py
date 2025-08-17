"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional

class LoanRequest(BaseModel):
    """Schema for loan prediction request"""
    
    Age: int = Field(..., ge=18, le=100, description="Borrower age (18-100)")
    Income: float = Field(..., gt=0, description="Annual income in dollars")
    LoanAmount: float = Field(..., gt=0, description="Requested loan amount")
    CreditScore: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    MonthsEmployed: int = Field(..., ge=0, description="Months of employment")
    NumCreditLines: int = Field(..., ge=0, description="Number of credit lines")
    InterestRate: float = Field(..., ge=0, le=30, description="Interest rate percentage")
    LoanTerm: int = Field(..., gt=0, description="Loan term in months")
    DTIRatio: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    Education: str = Field(..., description="Education level")
    EmploymentType: str = Field(..., description="Employment type")
    MaritalStatus: str = Field(..., description="Marital status")
    HasMortgage: str = Field(..., description="Has mortgage (Yes/No)")
    HasDependents: str = Field(..., description="Has dependents (Yes/No)")
    LoanPurpose: str = Field(..., description="Purpose of loan")
    HasCoSigner: str = Field(..., description="Has co-signer (Yes/No)")
    
    @validator('Education')
    def validate_education(cls, v):
        valid_education = ['High School', 'Bachelor\'s', 'Master\'s', 'PhD']
        if v not in valid_education:
            raise ValueError(f'Education must be one of: {valid_education}')
        return v
    
    @validator('EmploymentType')
    def validate_employment_type(cls, v):
        valid_employment = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed']
        if v not in valid_employment:
            raise ValueError(f'EmploymentType must be one of: {valid_employment}')
        return v
    
    @validator('MaritalStatus')
    def validate_marital_status(cls, v):
        valid_marital = ['Single', 'Married', 'Divorced']
        if v not in valid_marital:
            raise ValueError(f'MaritalStatus must be one of: {valid_marital}')
        return v
    
    @validator('HasMortgage', 'HasDependents', 'HasCoSigner')
    def validate_yes_no(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Must be Yes or No')
        return v
    
    @validator('LoanPurpose')
    def validate_loan_purpose(cls, v):
        valid_purposes = ['Home', 'Auto', 'Education', 'Business', 'Other']
        if v not in valid_purposes:
            raise ValueError(f'LoanPurpose must be one of: {valid_purposes}')
        return v

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    
    prediction: int = Field(..., description="Prediction (0=No Default, 1=Default)")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score (0-1)")
    risk_category: str = Field(..., description="Risk category (Low/Medium/High)")
    recommendation: str = Field(..., description="Loan recommendation")

class FeatureImportance(BaseModel):
    """Schema for feature importance"""
    
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Feature importance score")

class PerformanceMetrics(BaseModel):
    """Schema for model performance metrics"""
    
    accuracy: float = Field(..., description="Model accuracy")
    roc_auc: float = Field(..., description="ROC AUC score")
    classification_report: Dict[str, Any] = Field(..., description="Classification report")

class ModelInsightsResponse(BaseModel):
    """Schema for model insights response"""
    
    performance_metrics: PerformanceMetrics
    feature_importance: List[FeatureImportance]
    model_parameters: Dict[str, Any]
