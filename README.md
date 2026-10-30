# GlucoCare AI – An ML-Based Diabetes Risk Screening and Prediction System

> **Educational screening prototype – NOT a medical diagnostic system.** Predictions are statistical estimates
> and must not replace professional medical advice.

**Subject:** Artificial Intelligence and Machine Learning in Healthcare (BE Computer Engineering)

---

## 1. Abstract
GlucoCare AI is an end-to-end Machine Learning project that estimates the likelihood of diabetes from eight routine
health measurements. It trains and compares Logistic Regression and Random Forest classifiers on the Pima Indians
Diabetes Dataset, selects the final model using cross-validation (with emphasis on recall), and serves it through a
Streamlit dashboard. The project demonstrates data cleaning, leakage-free preprocessing, model evaluation, and
responsible communication of ML results in healthcare.

## 2. Introduction
Diabetes mellitus is a chronic condition in which the body cannot properly regulate blood glucose. Many people
remain undiagnosed for years. Machine Learning can learn patterns from historical patient records and give an early,
low-cost *screening-style* estimate, prompting people to seek proper medical testing.

## 3. Problem Statement
Manual screening requires clinical visits and laboratory tests. There is a need for a simple, data-driven tool that
estimates diabetes likelihood from basic measurements, while being transparent about its limitations.

## 4. Objectives
1. Build a complete ML pipeline (load → clean → split → train → evaluate → save).
2. Compare at least two classification algorithms fairly on the same held-out data.
3. Choose the final model using healthcare-relevant criteria (recall, not accuracy alone).
4. Provide an interactive dashboard for prediction, performance review and data exploration.
5. Communicate results with neutral language and clear disclaimers.

## 5. Existing System
Traditional screening relies on questionnaires (e.g. risk scores) and laboratory tests (fasting glucose, OGTT, HbA1c)
interpreted by clinicians. These are accurate but need appointments, equipment and trained staff.

## 6. Proposed System
A lightweight ML-based screening prototype: the user enters eight measurements, the saved scikit-learn pipeline
cleans the input exactly as in training, and the model returns a predicted class with an estimated probability.

## 7. Hardware and Software Requirements
| Item | Requirement |
|---|---|
| Hardware | Any laptop/PC, 4 GB RAM (8 GB recommended), ~1 GB free disk |
| OS | Windows 10/11, macOS, or Linux |
| Python | 3.9 or newer (3.10+ recommended) |
| Libraries | pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib, Pillow |
| Browser | Any modern browser (Chrome, Edge, Firefox) |

## 8. Dataset Description
**Pima Indians Diabetes Database** – originally from the National Institute of Diabetes and Digestive and Kidney
Diseases (NIDDK), distributed by the UCI Machine Learning Repository and mirrored on Kaggle.
It is commonly documented as 768 records of female patients of Pima Indian heritage, aged 21 or older; the training
script prints the exact counts it finds.

| Feature | Meaning |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2 h oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin-fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Score summarising family history of diabetes |
| Age | Age in years |
| **Outcome** (target) | 0 = non-diabetic, 1 = diabetic |

**Source links:**
- Kaggle mirror: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Original paper: Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., Johannes, R.S. (1988).
  *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.* Proc. Annual Symposium on
  Computer Application in Medical Care, 261–265.

`python -m src.train_model` tries to download the CSV automatically from public mirrors. If you are offline or the
download fails, download `diabetes.csv` manually from the Kaggle link and place it at `data/diabetes.csv`.
**No data in this project is generated or fabricated.**

## 9. Methodology
1. **Load & inspect** the dataset; check NaNs and duplicates.
2. **Invalid zeros:** zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are impossible → treated as
   missing and filled with the **median** (robust to outliers).
3. **EDA:** distributions, class balance, correlation heatmap (notebook + dashboard).
4. **Stratified 80/20 split** with `random_state=42`.
5. **Pipeline** = zeros→NaN → median imputer → (StandardScaler for Logistic Regression) → classifier.
   Everything is fitted on training data only → **no data leakage**.
6. **Tuning:** `GridSearchCV`, 5-fold stratified CV, on the training set only.
7. **Model selection:** highest cross-validated **F2-score** (recall weighted twice as much as precision), chosen
   **without looking at the test set**.
8. **Final evaluation:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report on the
   untouched test set.
9. **Persistence:** pipeline + metadata saved with Joblib; the app loads the same file.

## 10. Algorithms Used
- **Logistic Regression** – linear model that outputs a probability with the sigmoid function; interpretable baseline.
- **Random Forest** – ensemble of many decision trees trained on random subsets; captures non-linear relationships and
  gives feature importance.
Both use `class_weight="balanced"` because diabetic cases are the minority class, so missed cases are penalised more.

## 11. System Architecture
```
 diabetes.csv ─► preprocessing.py ─► train_model.py ─► models/diabetes_model.pkl
                  (load, clean,        (CV tuning,     models/metrics.json
                   Pipeline)            evaluation)     assets/figures/*.png
                                                              │
 User ► Streamlit form (app.py) ► prediction.py ► saved Pipeline ► class + probability ► dashboard
```

## 12. Results and Evaluation
Results are **produced by your own training run** – none are hard-coded or copied from elsewhere.
After running `python -m src.train_model`:
- the full console output shows data checks, CV scores and test metrics;
- `models/results_summary.md` contains a ready-made results table and the model-selection explanation;
- `models/metrics.json` feeds the **Model Performance** page;
- `assets/figures/` contains the confusion matrices, model comparison, ROC curves and feature importance PNGs.

**Paste the table from `models/results_summary.md` here before submission:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | False negatives |
|---|---|---|---|---|---|---|
| _(paste after training)_ | | | | | | |

**Metrics in healthcare context**
- *Accuracy* – overall correctness; can hide missed diabetic cases when classes are imbalanced.
- *Precision* – of people flagged, how many truly had diabetes (low → false alarms).
- *Recall* – of people who truly had diabetes, how many were caught (low → missed cases = **false negatives**,
  the costliest error in screening).
- *F1* – balance of precision and recall. *F2* (used for selection) favours recall.
- *ROC-AUC* – threshold-independent ability to separate the classes.

## 13. Screenshots
Add screenshots after running the app (e.g. into `assets/screenshots/`):
1. Home dashboard 2. Risk Prediction result 3. Model Performance 4. Dataset Explorer

## 14. Installation Instructions
```bash
# 1. Open a terminal in the project folder (GlucoCare-AI/)
python -m venv venv

# 2. Activate the virtual environment
#    Windows (PowerShell):   venv\Scripts\Activate.ps1
#    Windows (cmd):          venv\Scripts\activate.bat
#    macOS / Linux:          source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 15. Execution Instructions
```bash
# Step 1 – get the dataset (automatic attempt happens inside the training script)
#          If it fails, place diabetes.csv in the data/ folder manually.

# Step 2 – train, evaluate and save the model  (run from the project root)
python -m src.train_model

# Step 3 – launch the dashboard
streamlit run app.py
```
Optional EDA notebook: `pip install notebook` then `cd notebooks && jupyter notebook`.

> Always run the training command from the project root using `python -m src.train_model`
> so that the saved model can be loaded by the app.

## 16. Limitations
- Small dataset (a few hundred records) → results have noticeable uncertainty.
- Only female patients of Pima Indian heritage, age 21+ → may **not generalise** to other groups.
- Several measurements (Insulin, SkinThickness) had many missing values replaced by medians.
- Probabilities from the models may not be well calibrated.
- No external validation; no clinical testing; outcome definition comes from the original study.
- The 0.5 decision boundary is an algorithmic default, **not** a medically validated threshold.

## 17. Future Scope
Larger, more diverse and recent datasets; more models (gradient boosting, SVM, neural networks); probability
calibration; explainability with SHAP/LIME; fairness analysis across subgroups; external clinical validation;
secure cloud deployment; multilingual interface.

## 18. Project Structure
```
GlucoCare-AI/
├── app.py                  # Streamlit dashboard (6 pages)
├── requirements.txt
├── README.md
├── VIVA_QUESTIONS.md
├── data/diabetes.csv       # you place / download this
├── models/                 # diabetes_model.pkl, metrics.json, results_summary.md (created by training)
├── notebooks/exploratory_data_analysis.ipynb
├── src/
│   ├── preprocessing.py    # loading, quality checks, pipeline builder, split
│   ├── train_model.py      # tuning, comparison, selection, saving
│   ├── evaluation.py       # metrics + all plots
│   └── prediction.py       # model loading, input validation, prediction
└── assets/logo.png, figures/
```

## 19. References
1. Smith et al. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.*
2. UCI / Kaggle – Pima Indians Diabetes Database: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
3. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.
4. WHO – Diabetes fact sheet: https://www.who.int/news-room/fact-sheets/detail/diabetes
5. Streamlit documentation: https://docs.streamlit.io
