# 🩺 GlucoCare AI
### An ML-Based Diabetes Risk Screening and Prediction System

<p align="center">
  <strong>Artificial Intelligence and Machine Learning in Healthcare</strong>
</p>

<p align="center">
  A Machine Learning-powered educational healthcare application that analyzes patient health parameters and estimates diabetes risk using classification algorithms.
</p>

---

## 📌 Project Overview

**GlucoCare AI** is an end-to-end Machine Learning project designed to demonstrate how Artificial Intelligence can be applied in healthcare for diabetes risk screening.

The application analyzes eight health-related parameters using trained Machine Learning models to generate an estimated diabetes risk prediction.

It uses the Pima Indians Diabetes Dataset and implements Logistic Regression and Random Forest classifiers. The models are evaluated and compared using multiple performance metrics, with Random Forest selected based on its cross-validated F2-score.

An interactive Streamlit dashboard allows users to enter health parameters, view predictions, explore the dataset, and understand model performance.

> ⚠️ **Disclaimer:** GlucoCare AI is an educational screening prototype and is NOT a medical diagnostic system. Its predictions are statistical estimates and must not replace professional medical advice, laboratory testing, or clinical diagnosis.

## ✨ Key Features

- **Diabetes Risk Prediction:** Estimates diabetes risk using eight patient health parameters.
- **Machine Learning Models:** Implements Logistic Regression and Random Forest.
- **Data Preprocessing:** Handles invalid zero values and missing measurements using appropriate preprocessing techniques.
- **Model Comparison:** Evaluates multiple classifiers and selects a model using cross-validation.
- **Performance Evaluation:** Displays accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- **Interactive Dashboard:** Provides a user-friendly Streamlit interface.
- **Dataset Exploration:** Visualizes feature distributions, correlations, and class distributions.
- **Healthcare Awareness:** Provides general information about diabetes prevention and healthy lifestyle practices.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualization | Matplotlib, Seaborn |
| Frontend | Streamlit |
| Model Persistence | Joblib |
| Dataset | Pima Indians Diabetes Dataset |

## 📊 Dataset Description

The project uses the **Pima Indians Diabetes Dataset**, which contains 768 patient records and eight input features.

The dataset is commonly used for educational research and binary classification experiments involving diabetes prediction.

### Input Features

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| Blood Pressure | Diastolic blood pressure |
| Skin Thickness | Triceps skin-fold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body Mass Index |
| Diabetes Pedigree Function | Family-history-related diabetes score |
| Age | Age in years |

**Target Variable:**
- `0` – Non-diabetic
- `1` – Diabetic

### Dataset Distribution

| Category | Records |
|---|---:|
| Total Records | 768 |
| Non-diabetic (0) | 500 |
| Diabetic (1) | 268 |
| Input Features | 8 |

**Dataset Source:** [Pima Indians Diabetes Database – Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

The training script attempts to download the dataset automatically. If downloading fails, the CSV can be placed manually inside the `data/` directory.

## ⚙️ Methodology

The project follows an end-to-end Machine Learning workflow.

### 1. Data Collection
The Pima Indians Diabetes Dataset is loaded using Pandas.

### 2. Data Preprocessing
- Dataset inspection and quality checks.
- Identification of missing and invalid values.
- Treatment of zero values in selected medical measurements.
- Median imputation.
- Feature scaling where required.
- Stratified training and testing split.

### 3. Model Training

Two classification algorithms are implemented:

**Logistic Regression**

A supervised classification algorithm that estimates the probability of a binary outcome using a logistic function. It serves as an interpretable baseline model.

**Random Forest Classifier**

An ensemble learning algorithm that combines multiple decision trees to generate predictions. It can capture more complex relationships between input features.

### 4. Model Selection

GridSearchCV with stratified cross-validation is used for hyperparameter tuning.

The final model is selected based on the highest cross-validated F2-score. F2 gives greater importance to recall, which is particularly relevant when evaluating screening models.

### 5. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### 6. Prediction

The selected trained model is saved using Joblib and loaded by the Streamlit application to generate predictions from user inputs.

## 🏗️ System Architecture

```text
       Pima Indians Diabetes Dataset
                    |
                    v
             Data Collection
                    |
                    v
             Data Preprocessing
                    |
                    v
              Train-Test Split
                    |
                    v
              Model Training
                    |
           +--------+--------+
           |                 |
           v                 v
     Logistic Regression  Random Forest
           |                 |
           +--------+--------+
                    |
                    v
             Model Evaluation
                    |
                    v
              Model Selection
                    |
                    v
           Save Trained Model
                    |
                    v
             Streamlit App
                    |
                    v
             Patient Inputs
                    |
                    v
             Risk Prediction
                    |
                    v
          Prediction Dashboard
```

## 📈 Model Performance and Results

Both models were trained and evaluated using the same stratified training and testing split.

The dataset was divided into:
- Training set: 614 records
- Testing set: 154 records

### Model Comparison

The following results were obtained from the actual training and evaluation run:

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 70.8% | 72.7% |
| Precision | 57.1% | 59.1% |
| Recall | 66.7% | 72.2% |
| F1-score | 61.5% | 65.0% |
| ROC-AUC | 0.810 | 0.818 |
| False Negatives | 18 | 15 |
| Cross-validated F2-score | 0.704 | 0.739 |

### Selected Model: Random Forest

Random Forest was selected because it achieved the highest mean cross-validated F2-score of **0.739**, compared with 0.704 for Logistic Regression.

It also achieved higher recall on the held-out test dataset, identifying more of the actual positive cases in this particular evaluation.

### Understanding the Evaluation Metrics

- **Accuracy:** Measures the proportion of correctly classified records.
- **Precision:** Measures how many predicted positive cases were actually positive.
- **Recall:** Measures how many actual positive cases were correctly identified.
- **F1-score:** Represents the harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between the two classes.
- **Confusion Matrix:** Shows correct and incorrect classification counts.

In healthcare screening, recall is especially important because false negatives represent positive cases that the model fails to identify.

*Note: These results are specific to this dataset split and evaluation procedure. They do not establish clinical accuracy or suitability for real-world diagnosis.*

## 🖥️ Application Modules

The Streamlit application includes the following sections:

1. **Home Dashboard:** Introduction, project overview, and system workflow.
2. **Diabetes Risk Prediction:** Patient input form and ML-generated prediction.
3. **Model Performance:** Evaluation metrics and model comparison.
4. **Dataset Explorer:** Dataset preview and visual analytics.
5. **Healthcare Awareness:** General diabetes awareness and preventive information.
6. **About Project:** Project details, methodology, limitations, and future scope.

## 📂 Project Structure

```text
GlucoCare-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── VIVA_QUESTIONS.md
│
├── data/
│   └── diabetes.csv
│
├── models/
│   ├── diabetes_model.pkl
│   ├── metrics.json
│   └── results_summary.md
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluation.py
│   └── prediction.py
│
└── assets/
    ├── logo.png
    └── figures/
```

*The model and evaluation files are generated after running the training script.*

## 🚀 Installation and Setup

### Prerequisites

- Python 3.9 or newer
- Git
- VS Code or another Python IDE
- Internet connection for the initial dataset download

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/GlucoCare-AI.git
```

### Step 2: Navigate to the Project Directory

```bash
cd GlucoCare-AI
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## ▶️ Running the Project

### Step 1: Train the Machine Learning Model

Run this command from the project root directory:

```bash
python -m src.train_model
```

This will:
- Load the dataset.
- Perform data preprocessing.
- Train and tune the classification models.
- Evaluate model performance.
- Select the final model.
- Save the trained model and evaluation results.

### Step 2: Launch the Streamlit Application

```bash
python -m streamlit run app.py
```

### Step 3: Open the Application

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 📸 Application Screenshots

Add actual screenshots of the running application in this section.

| Module | Screenshot |
|---|---|
| Home Dashboard | Add home dashboard screenshot |
| Risk Prediction | Add prediction screenshot |
| Model Performance | Add evaluation dashboard screenshot |
| Dataset Explorer | Add dataset visualization screenshot |

You can save screenshots inside `assets/screenshots/` and embed them using Markdown image links.

## ⚠️ Limitations

- The dataset contains only 768 records.
- The dataset represents female patients of Pima Indian heritage aged 21 years and older.
- The model may not generalize to other demographic populations.
- Several medical measurements contain invalid zero values that require imputation.
- Model probabilities may not be clinically calibrated.
- The model has not undergone external clinical validation.
- The system cannot replace professional medical consultation or laboratory testing.

## 🔮 Future Scope

Possible improvements include:

- Training with larger and more diverse healthcare datasets.
- Exploring additional algorithms such as SVM and Gradient Boosting.
- Integrating Explainable AI techniques such as SHAP and LIME.
- Performing fairness and subgroup performance analysis.
- Improving probability calibration.
- Developing secure healthcare data integration.
- Exploring multilingual support.
- Conducting external validation with qualified healthcare professionals.

## 🎯 Project Objective and Educational Significance

GlucoCare AI demonstrates the practical application of supervised Machine Learning in healthcare.

It provides hands-on experience with dataset preprocessing, classification algorithms, model evaluation, model persistence, and interactive application development.

The project highlights both the potential and limitations of using AI for healthcare screening.

## 📚 References

1. Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.* Proceedings of the Annual Symposium on Computer Application in Medical Care, 261–265.
2. [Pima Indians Diabetes Database – Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
3. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
4. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
5. [Streamlit Documentation](https://docs.streamlit.io/)
6. [Pandas Documentation](https://pandas.pydata.org/docs/)
7. [NumPy Documentation](https://numpy.org/doc/)

---

## 👩‍💻 Author

**Maithili Pandey**

BE Computer Engineering  
Specialization: Artificial Intelligence and Machine Learning

**Academic Year:** 2026–2027

---

⭐ If you find this project interesting, consider giving the repository a star!

**GlucoCare AI – Exploring the potential of Machine Learning in Healthcare.**
