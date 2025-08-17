"""
Feature engineering utilities for loan risk prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to loan data
    
    Args:
        df: Input DataFrame with loan features
        
    Returns:
        DataFrame with engineered features
    """
    df_eng = df.copy()
    
    # Handle zero income to avoid division by zero
    df_eng['Income'] = df_eng['Income'].replace(0, 1e-6)
    
    # Create derived features
    df_eng['LoanIncomeRatio'] = df_eng['LoanAmount'] / df_eng['Income']
    df_eng['CreditScoreIncome'] = df_eng['CreditScore'] * df_eng['Income']
    df_eng['MonthsEmployedCreditScore'] = df_eng['MonthsEmployed'] * df_eng['CreditScore']
    df_eng['InterestRateLoanTerm'] = df_eng['InterestRate'] * df_eng['LoanTerm']
    
    return df_eng

def get_feature_descriptions() -> Dict[str, str]:
    """
    Get descriptions of all features including engineered ones
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    return {
        'Age': 'Borrower age in years',
        'Income': 'Annual income in dollars',
        'LoanAmount': 'Requested loan amount in dollars',
        'CreditScore': 'Credit score (300-850)',
        'MonthsEmployed': 'Number of months employed',
        'NumCreditLines': 'Number of active credit lines',
        'InterestRate': 'Interest rate percentage',
        'LoanTerm': 'Loan term in months',
        'DTIRatio': 'Debt-to-income ratio',
        'Education': 'Education level',
        'EmploymentType': 'Type of employment',
        'MaritalStatus': 'Marital status',
        'HasMortgage': 'Whether borrower has a mortgage',
        'HasDependents': 'Whether borrower has dependents',
        'LoanPurpose': 'Purpose of the loan',
        'HasCoSigner': 'Whether loan has a co-signer',
        'LoanIncomeRatio': 'Ratio of loan amount to income',
        'CreditScoreIncome': 'Credit score multiplied by income',
        'MonthsEmployedCreditScore': 'Months employed multiplied by credit score',
        'InterestRateLoanTerm': 'Interest rate multiplied by loan term'
    }
