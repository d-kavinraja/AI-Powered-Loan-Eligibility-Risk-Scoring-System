// AI-Powered Loan Risk Scoring System JavaScript

// Tab functionality
function showTab(tabName) {
  // Hide all tab contents
  const tabContents = document.querySelectorAll(".tab-content")
  tabContents.forEach((content) => {
    content.classList.remove("active")
  })

  // Remove active class from all tab buttons
  const tabButtons = document.querySelectorAll(".tab-button")
  tabButtons.forEach((button) => {
    button.classList.remove("active")
  })

  // Show selected tab content
  document.getElementById(tabName).classList.add("active")

  // Add active class to clicked button
  event.target.classList.add("active")
}

// Form handling
document.getElementById("loanForm").addEventListener("submit", async function (e) {
  e.preventDefault()

  const formData = new FormData(this)
  const data = {}

  // Convert form data to object
  for (const [key, value] of formData.entries()) {
    // Convert numeric fields
    if (
      [
        "Age",
        "Income",
        "LoanAmount",
        "CreditScore",
        "MonthsEmployed",
        "NumCreditLines",
        "InterestRate",
        "LoanTerm",
      ].includes(key)
    ) {
      data[key] = Number.parseFloat(value)
    } else if (key === "DTIRatio") {
      data[key] = Number.parseFloat(value)
    } else {
      data[key] = value
    }
  }

  let originalText // Declare originalText variable
  try {
    // Show loading state
    const submitButton = this.querySelector('button[type="submit"]')
    originalText = submitButton.textContent // Assign value to originalText
    submitButton.innerHTML = '<span class="loading"></span> Assessing...'
    submitButton.disabled = true

    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })

    const result = await response.json()

    if (response.ok) {
      displayPredictionResult(result)
    } else {
      displayError(result.detail || "Prediction failed")
    }
  } catch (error) {
    displayError("Network error: " + error.message)
  } finally {
    // Reset button
    const submitButton = this.querySelector('button[type="submit"]')
    submitButton.textContent = originalText
    submitButton.disabled = false
  }
})

function displayPredictionResult(result) {
  const resultDiv = document.getElementById("predictionResult")
  const contentDiv = document.getElementById("resultContent")

  // Determine risk class for styling
  let riskClass = "risk-low"
  if (result.risk_category === "Medium Risk") riskClass = "risk-medium"
  if (result.risk_category === "High Risk") riskClass = "risk-high"

  resultDiv.className = `result-card ${riskClass}`

  contentDiv.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
            <div>
                <h4>Prediction</h4>
                <p style="font-size: 1.2rem; font-weight: 600;">
                    ${result.prediction === 0 ? "✅ No Default" : "❌ Default Risk"}
                </p>
            </div>
            <div>
                <h4>Risk Score</h4>
                <p style="font-size: 1.2rem; font-weight: 600;">
                    ${(result.risk_score * 100).toFixed(1)}%
                </p>
            </div>
            <div>
                <h4>Risk Category</h4>
                <p style="font-size: 1.2rem; font-weight: 600;">
                    ${result.risk_category}
                </p>
            </div>
            <div>
                <h4>Recommendation</h4>
                <p style="font-size: 1.2rem; font-weight: 600;">
                    ${result.recommendation}
                </p>
            </div>
        </div>
    `

  resultDiv.style.display = "block"
  resultDiv.scrollIntoView({ behavior: "smooth" })
}

function displayError(message) {
  const resultDiv = document.getElementById("predictionResult")
  const contentDiv = document.getElementById("resultContent")

  resultDiv.className = "result-card"
  contentDiv.innerHTML = `<div class="error-message">${message}</div>`
  resultDiv.style.display = "block"
}

// Load model insights
async function loadInsights() {
  try {
    const button = event.target
    button.innerHTML = '<span class="loading"></span> Loading...'
    button.disabled = true

    const response = await fetch("/api/insights")
    const insights = await response.json()

    if (response.ok) {
      displayInsights(insights)
    } else {
      displayError("Failed to load insights: " + insights.detail)
    }
  } catch (error) {
    displayError("Network error: " + error.message)
  } finally {
    const button = event.target
    button.textContent = "Refresh Insights"
    button.disabled = false
  }
}

function displayInsights(insights) {
  // Display performance metrics
  const metricsDiv = document.getElementById("performanceMetrics")
  const metrics = insights.performance_metrics

  metricsDiv.innerHTML = `
        <div style="display: grid; gap: 10px;">
            <div><strong>Accuracy:</strong> ${(metrics.accuracy * 100).toFixed(2)}%</div>
            <div><strong>ROC AUC:</strong> ${metrics.roc_auc.toFixed(4)}</div>
            <div><strong>Precision (Class 1):</strong> ${(metrics.classification_report["1"]["precision"] * 100).toFixed(2)}%</div>
            <div><strong>Recall (Class 1):</strong> ${(metrics.classification_report["1"]["recall"] * 100).toFixed(2)}%</div>
            <div><strong>F1-Score (Class 1):</strong> ${(metrics.classification_report["1"]["f1-score"] * 100).toFixed(2)}%</div>
        </div>
    `

  // Display feature importance
  const importanceDiv = document.getElementById("featureImportance")
  const features = insights.feature_importance.slice(0, 10) // Top 10 features

  const featureHtml = features
    .map(
      (feature) => `
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span>${feature.feature}</span>
            <span style="font-weight: 600;">${feature.importance.toFixed(4)}</span>
        </div>
    `,
    )
    .join("")

  importanceDiv.innerHTML = `<div>${featureHtml}</div>`

  // Show insights content
  document.getElementById("insightsContent").style.display = "block"
}

// Utility functions
function clearForm() {
  document.getElementById("loanForm").reset()
  document.getElementById("predictionResult").style.display = "none"
}

function fillSampleData() {
  // Fill form with sample data for testing
  const sampleData = {
    Age: 35,
    Income: 75000,
    LoanAmount: 250000,
    CreditScore: 720,
    MonthsEmployed: 60,
    NumCreditLines: 5,
    InterestRate: 4.5,
    LoanTerm: 360,
    DTIRatio: 0.35,
    Education: "Bachelor's",
    EmploymentType: "Full-time",
    MaritalStatus: "Married",
    HasMortgage: "Yes",
    HasDependents: "Yes",
    LoanPurpose: "Home",
    HasCoSigner: "No",
  }

  Object.keys(sampleData).forEach((key) => {
    const element = document.getElementById(key.toLowerCase()) || document.querySelector(`[name="${key}"]`)
    if (element) {
      element.value = sampleData[key]
    }
  })
}

// Initialize page
document.addEventListener("DOMContentLoaded", function() {
    console.log('Loan Risk Scoring System initialized');
});
