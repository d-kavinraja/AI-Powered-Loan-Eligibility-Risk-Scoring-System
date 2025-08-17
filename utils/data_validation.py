"""
Data validation utilities for loan risk prediction
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

def validate_loan_data(df: pd.DataFrame) -> List[str]:
    """
    Validate loan data for common issues
    
    Args:
        df: DataFrame containing loan data
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Check for missing values
    missing_cols = df.isnull().sum()
    if missing_cols.any():
        errors.append(f"Missing values found in columns: {missing_cols[missing_cols > 0].index.tolist()}")
    
    # Check for negative values in financial fields
    financial_fields = ['Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines']
    for field in financial_fields:
        if field in df.columns and (df[field] < 0).any():
            errors.append(f"Negative values found in {field}")
    
    # Check credit score range
    if 'CreditScore' in df.columns:
        invalid_scores = df[(df['CreditScore'] < 300) | (df['CreditScore'] > 850)]
        if not invalid_scores.empty:
            errors.append("Credit scores must be between 300 and 850")
    
    # Check DTI ratio range
    if 'DTIRatio' in df.columns:
        invalid_dti = df[(df['DTIRatio'] < 0) | (df['DTIRatio'] > 1)]
        if not invalid_dti.empty:
            errors.append("DTI ratio must be between 0 and 1")
    
    # Check age range
    if 'Age' in df.columns:
        invalid_age = df[(df['Age'] < 18) | (df['Age'] > 100)]
        if not invalid_age.empty:
            errors.append("Age must be between 18 and 100")
    
    return errors

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for loan data
    
    Args:
        df: DataFrame containing loan data
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'numerical_summary': df.describe().to_dict(),
        'categorical_summary': {}
    }
    
    # Add categorical summaries
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_summary'][col] = df[col].value_counts().to_dict()
    
    return summary
