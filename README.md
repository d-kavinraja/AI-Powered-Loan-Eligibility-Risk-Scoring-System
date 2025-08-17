<div align="center">
  <h1 align="center">🤖 AI-Powered Loan Eligibility & Risk Scoring System</h1>
  <p align="center">
    An end-to-end machine learning system that trains and serves a robust model to predict loan default risk, exposed via a high-performance FastAPI backend.
  </p>
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-%23F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-8C267B?style=for-the-badge&logo=lightgbm&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

</div>

---

## 📖 Table of Contents

- [🎯 Project Objective](#-project-objective)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [📂 Repository Structure](#-repository-structure)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
  - [Running the Application](#running-the-application)
- [⚙️ API Endpoints Guide](#️-api-endpoints-guide)
  - [POST /api/predict](#post-apipredict)
  - [GET /api/insights](#get-apiinsights)
  - [GET /api/charts/{chart_name}](#get-apichartschart_name)
  - [GET /health](#get-health)
- [🤖 Model Training & Retraining](#-model-training--retraining)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🎯 Project Objective

The primary goal of this project is to develop a reliable, scalable, and end-to-end system for assessing loan default risk. By leveraging a machine learning model trained on historical data, the system provides instant risk scores and actionable recommendations. The entire solution is served through a well-documented RESTful API, making it easy to integrate into existing financial workflows and applications.

---

## ✨ Key Features

-   **Advanced ML Model**: Utilizes a **LightGBM Classifier** with `GridSearchCV` for hyperparameter tuning to ensure high accuracy and robustness in predictions.
-   **Comprehensive Feature Engineering**: Creates powerful interaction features (e.g., `LoanIncomeRatio`, `MonthsEmployedCreditScore`) to capture complex borrower behaviors.
-   **High-Performance API**: Built with **FastAPI** for asynchronous, high-speed request handling, making it suitable for production environments.
-   **Robust Input Validation**: Employs **Pydantic** schemas for strict, type-safe validation of all incoming request data, preventing common errors.
-   **Model Insights Endpoint**: Offers transparency by providing detailed model performance metrics, feature importances, and the best hyperparameters used.
-   **Production-Ready Codebase**: A modular, clean, and well-organized project structure that simplifies maintenance and future development.
-   **Interactive Documentation**: Automatically generates interactive API documentation (via Swagger UI and ReDoc) for easy testing and exploration.

---

## 🏗️ System Architecture

The system follows a standard machine learning model deployment architecture. The core components are decoupled for maintainability and scalability.



1.  **Client**: A user or service sends a POST request with borrower data in JSON format.
2.  **FastAPI Backend**:
    -   Receives and validates the incoming data using Pydantic models.
    -   Passes the validated data to the feature engineering module.
3.  **ML Pipeline (`.joblib` artifact)**:
    -   The loaded Scikit-learn pipeline preprocesses the data (scaling, encoding).
    -   The trained LightGBM model predicts the probability of default.
4.  **Response Generation**: The API formats the prediction into a clear JSON response, including a risk score, category, and recommendation, and sends it back to the client.

---

## 📂 Repository Structure

The project is organized into distinct modules, each with a specific responsibility.

```
LOAN-RISK-SYSTEM/
│
├── models/                     # Pydantic schemas and ML model artifacts
│   ├── artifacts/
│   │   ├── charts/             # Generated performance charts
│   │   │   ├── confusion_matrix.png
│   │   │   └── roc_curve.png
│   │   ├── loan_default_pipeline.joblib  # The serialized ML pipeline
│   │   └── model_insights.json         # Performance metrics & feature importance
│   └── schemas.py                # Pydantic models for API validation
│
├── static/                     # Simple frontend files
│
├── utils/                      # Helper modules for the application
│   ├── data_validation.py      # Data validation logic
│   └── feature_engineering.py  # Feature engineering functions
│
├── .gitignore
├── main.py                     # Main FastAPI application file
├── README.md                   # This file
└── requirements.txt            # Project dependencies

```


## 🚀 Getting Started

Follow these steps to get the application running on your local machine.

### Prerequisites

-   Python 3.8 or higher
-   `pip` package manager
-   A `git` client

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/d-kavinraja/AI-Powered-Loan-Eligibility-Risk-Scoring-System.git](https://github.com/d-kavinraja/AI-Powered-Loan-Eligibility-Risk-Scoring-System.git)
    cd AI-Powered-Loan-Eligibility-Risk-Scoring-System
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the FastAPI server using Uvicorn:**
    ```bash
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    ```
    The `--reload` flag enables hot-reloading for development.

2.  **Access the API:**
    -   **Frontend**: `http://127.0.0.1:8000`

---

## ⚙️ API Endpoints Guide

The API provides the following endpoints for interaction.

### `POST /api/predict`

Predicts the loan default risk based on borrower data.

-   **Request Body**:
    ```json
    {
      "Age": 30,
      "Income": 55000,
      "LoanAmount": 25000,
      "CreditScore": 650,
      "MonthsEmployed": 60,
      "NumCreditLines": 4,
      "InterestRate": 12.5,
      "LoanTerm": 36,
      "DTIRatio": 0.4,
      "Education": "Bachelor's",
      "EmploymentType": "Full-time",
      "MaritalStatus": "Married",
      "HasMortgage": "Yes",
      "HasDependents": "Yes",
      "LoanPurpose": "Business",
      "HasCoSigner": "No"
    }
    ```

-   **Success Response (200 OK)**:
    ```json
    {
      "prediction": 0,
      "risk_score": 0.253,
      "risk_category": "Low Risk",
      "recommendation": "Approved"
    }
    ```

-   **Error Response (422 Unprocessable Entity)**:
    ```json
    {
      "detail": [
        {
          "loc": ["body", "CreditScore"],
          "msg": "ensure this value is greater than or equal to 300",
          "type": "value_error.number.not_ge",
          "ctx": { "limit_value": 300 }
        }
      ]
    }
    ```

### `GET /api/insights`

Retrieves the performance metrics, feature importances, and parameters of the trained model.

-   **Success Response (200 OK)**:
    ```json
    {
      "performance_metrics": {
        "accuracy": 0.887,
        "roc_auc": 0.921,
        "classification_report": {
          "0": { "precision": 0.9, "recall": 0.95, "f1-score": 0.92, "support": 1000 },
          "1": { "precision": 0.8, "recall": 0.7, "f1-score": 0.75, "support": 200 },
          "accuracy": 0.887,
          ...
        }
      },
      "feature_importance": [
        { "feature": "CreditScore", "importance": 0.154 },
        { "feature": "LoanIncomeRatio", "importance": 0.121 },
        ...
      ],
      "model_parameters": {
        "classifier__learning_rate": 0.05,
        "classifier__n_estimators": 200,
        "classifier__num_leaves": 40,
        "classifier__scale_pos_weight": 5
      }
    }
    ```

### `GET /api/charts/{chart_name}`

Serves static image files of the model's performance charts.

-   **URL Parameters**:
    -   `chart_name`: `confusion_matrix.png` or `roc_curve.png`
-   **Example Request**: `GET http://127.0.0.1:8000/api/charts/confusion_matrix.png`
-   **Success Response (200 OK)**: Returns the requested image file.

### `GET /health`

A simple health check endpoint to verify that the API is running and artifacts are loaded.

-   **Success Response (200 OK)**:
    ```json
    {
      "status": "healthy",
      "model_loaded": true,
      "insights_loaded": true
    }
    ```

---

## 🤖 Model Training & Retraining

The model can be retrained with new data to improve its performance or adapt to new patterns. The complete training pipeline is documented in the training script (e.g., `train_model.py`).

**To retrain the model:**

1.  **Prepare Your Data**: Place your updated dataset (e.g., `Loan_default.xlsx`) in the designated data directory.
2.  **Run the Training Script**: Execute the training script from the root of the project.
    ```bash
    python path/to/your/training_script.py
    ```
3.  **Verify Artifacts**: The script will automatically overwrite the existing artifacts in the `models/artifacts/` directory with the new:
    -   `loan_default_pipeline.joblib`
    -   `model_insights.json`
    -   Performance charts in `models/artifacts/charts/`
4.  **Restart the API**: Restart the Uvicorn server to load the newly trained model and its associated insights.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, please open an issue or submit a pull request.

1.  **Fork** the repository.
2.  Create your **Feature Branch** (`git checkout -b feature/AmazingFeature`).
3.  **Commit** your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  **Push** to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a **Pull Request**.

---

## 📜 License

This project is distributed under the MIT License. See `LICENSE` for more information.
