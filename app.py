"""
GlucoCare AI - AI-Powered Diabetes Risk Screening System (Streamlit dashboard)

Run:  streamlit run app.py
Educational prototype - NOT a medical diagnostic tool.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))   # so `import src...` works (also needed to unpickle the model)

from src import evaluation as ev
from src.prediction import INPUT_RULES, load_model_bundle, predict_risk, validate_inputs
from src.preprocessing import (DATA_PATH, FEATURE_COLUMNS, MANUAL_INSTRUCTIONS, METRICS_PATH,
                               TARGET_COLUMN, ZERO_AS_MISSING_COLUMNS, data_quality_report,
                               load_dataset)

st.set_page_config(page_title="GlucoCare AI", page_icon="🩺", layout="wide")

# ----------------------------------------------------------------------------
# Styling (white / teal / light-blue / soft-green healthcare palette)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(180deg, #F4FBFC 0%, #FFFFFF 40%); }
.hero { background: linear-gradient(120deg, #0B5C63 0%, #0F9D9A 60%, #4CAF93 100%);
        padding: 28px 34px; border-radius: 18px; color: white; margin-bottom: 18px; }
.hero h1 { margin: 0; font-size: 2.2rem; color: white; }
.hero p { margin: 4px 0 0 0; font-size: 1.05rem; opacity: .92; }
.card { background: white; border: 1px solid #D9EEF2; border-left: 5px solid #0F9D9A;
        border-radius: 14px; padding: 16px 20px; margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(15,157,154,.08); }
.card h4 { margin: 0 0 6px 0; color: #0B5C63; }
.card p, .card li { color: #33474D; font-size: .95rem; margin: 0; }
.step { background: #E6F4F8; border-radius: 12px; padding: 12px; text-align: center;
        color: #0B5C63; font-weight: 500; font-size: .9rem; min-height: 78px; }
.disclaimer { background: #FFF8E6; border: 1px solid #F2D98D; color: #6B5300;
        border-radius: 10px; padding: 10px 14px; font-size: .88rem; margin-bottom: 14px; }
.result { background: #E9F7F1; border: 1px solid #A8DCC6; border-radius: 14px; padding: 18px 22px; }
.result h3 { margin: 0 0 6px 0; color: #0B5C63; }
div[data-testid="stMetric"] { background: white; border: 1px solid #D9EEF2; border-radius: 12px; padding: 10px 14px; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(90deg, #0F9D9A, #4CAF93); color: white; border: 0;
    border-radius: 10px; padding: .6rem 1.4rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def card(title: str, body: str):
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def disclaimer():
    st.markdown('<div class="disclaimer"><b>Educational prototype - not a medical diagnosis.</b> '
                'Results are statistical estimates from a small historical dataset and must not replace '
                'professional medical advice. Please consult a qualified healthcare professional '
                'about your health.</div>', unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Cached loaders (return an error string instead of raising, so pages stay friendly)
# ----------------------------------------------------------------------------
@st.cache_resource
def get_model():
    try:
        return load_model_bundle(), None
    except Exception as exc:
        return None, str(exc)


@st.cache_data
def get_metrics():
    if not METRICS_PATH.exists():
        return None, "Metrics file not found. Train the model first: `python -m src.train_model`"
    try:
        return json.loads(METRICS_PATH.read_text()), None
    except Exception as exc:
        return None, f"Could not read metrics file: {exc}"


@st.cache_data
def get_data():
    try:
        return load_dataset(), None
    except Exception as exc:
        return None, str(exc)


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_home():
    hero("🩺 GlucoCare AI", "AI-Powered Diabetes Risk Screening System")
    disclaimer()
    c1, c2 = st.columns(2)
    with c1:
        card("What GlucoCare AI does",
             "It takes eight routine health measurements and uses a trained Machine Learning model to "
             "estimate how closely the pattern resembles people labelled diabetic in a historical dataset.")
    with c2:
        card("How Machine Learning is used",
             "This is supervised classification. The model learns from labelled examples (diabetic / "
             "non-diabetic) and then estimates a probability for new, unseen inputs.")

    st.subheader("Project workflow")
    steps = ["1. Load Pima dataset", "2. Clean data & impute", "3. Stratified train/test split",
             "4. Train Logistic Regression & Random Forest", "5. Evaluate & select model", "6. Predict in this app"]
    for col, s in zip(st.columns(len(steps)), steps):
        col.markdown(f'<div class="step">{s}</div>', unsafe_allow_html=True)

    st.subheader("Dataset information")
    df, err = get_data()
    if err:
        st.warning(err)
    else:
        rep = data_quality_report(df)
        a, b, c = st.columns(3)
        a.metric("Records", rep["n_records"])
        b.metric("Input features", rep["n_features"])
        c.metric("Diabetic cases", f"{rep['class_counts'].get(1, 0)} "
                 f"({rep['class_counts'].get(1, 0) / rep['n_records'] * 100:.1f}%)")
    st.caption("Source: Pima Indians Diabetes Database (NIDDK / UCI) - female patients of Pima Indian "
               "heritage, aged 21+.")

    st.subheader("Technologies used")
    st.markdown("**Python** · **Pandas / NumPy** · **Scikit-learn** · **Matplotlib / Seaborn** · "
                "**Streamlit** · **Joblib**")


def page_prediction():
    hero("Diabetes Risk Prediction", "Enter the patient parameters and analyse the estimated risk")
    disclaimer()
    bundle, err = get_model()
    if err:
        st.error(err)
        return

    with st.form("patient_form"):
        c1, c2 = st.columns(2)
        with c1:
            preg = st.number_input("Pregnancies", 0, 20, 2, 1, help="Number of times pregnant")
            glucose = st.number_input("Glucose (mg/dL)", 40, 300, 120, 1,
                                      help="Plasma glucose concentration (2-hour oral glucose tolerance test)")
            bp = st.number_input("Diastolic blood pressure (mm Hg)", 20, 200, 70, 1)
            skin = st.number_input("Skin-fold thickness (mm)", 0, 100, 20, 1,
                                   help="Triceps skin-fold thickness. Tick 'not measured' below if unknown.")
            skin_unknown = st.checkbox("Skin thickness not measured")
        with c2:
            insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80, 1,
                                      help="2-hour serum insulin. Tick 'not measured' below if unknown.")
            insulin_unknown = st.checkbox("Insulin not measured")
            bmi = st.number_input("BMI (kg/m²)", 10.0, 70.0, 28.0, 0.1, format="%.1f")
            dpf = st.number_input("Diabetes pedigree function", 0.05, 2.5, 0.40, 0.01, format="%.2f",
                                  help="Score summarising diabetes family history (higher = stronger history)")
            age = st.number_input("Age (years)", 21, 90, 35, 1)
        submitted = st.form_submit_button("Analyze Diabetes Risk")

    if not submitted:
        st.info("Fill in the form and press **Analyze Diabetes Risk**. "
                "Values for 'not measured' fields are estimated by the model's preprocessing (median imputation).")
        return

    values = {"Pregnancies": preg, "Glucose": glucose, "BloodPressure": bp,
              "SkinThickness": 0 if skin_unknown else skin, "Insulin": 0 if insulin_unknown else insulin,
              "BMI": bmi, "DiabetesPedigreeFunction": dpf, "Age": age}
    problems = validate_inputs(values)
    if problems:
        st.error("Please correct the inputs: " + " ".join(problems))
        return
    try:
        res = predict_risk(bundle, values)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    p = res["probability_positive"]
    category = ("Pattern similar to the **diabetic** group in the dataset" if res["predicted_class"] == 1
                else "Pattern similar to the **non-diabetic** group in the dataset")
    st.markdown(f'<div class="result"><h3>Screening result</h3>'
                f'<p style="margin:0;color:#33474D">Model: {res["model_name"]}</p></div>', unsafe_allow_html=True)
    st.markdown(f"#### {category}")
    m1, m2 = st.columns(2)
    m1.metric("Estimated probability of diabetic class", f"{p * 100:.1f}%")
    m2.metric("Estimated probability of non-diabetic class", f"{(1 - p) * 100:.1f}%")
    st.pyplot(ev.plot_probability(p))

    st.markdown(
        f"**How to read this:** based on the patterns in the training data, the model estimates a "
        f"{p * 100:.1f}% probability that this combination of values belongs to the diabetic class. "
        "The category above uses the algorithm's default decision boundary (probability ≥ 0.5); it is a "
        "statistical convention, **not** a medically validated risk threshold. The probability is an "
        "estimate and may not be perfectly calibrated.")
    if res["outside_training_range"]:
        st.warning("Some inputs lie outside the range seen during training "
                   f"({', '.join(res['outside_training_range'])}); the estimate may be less reliable.")
    st.info("This tool cannot diagnose diabetes. Only laboratory tests interpreted by a healthcare "
            "professional can do that. If you are concerned about your health, please see a doctor.")
    with st.expander("Inputs used"):
        st.dataframe(pd.DataFrame([values]))


def page_performance():
    hero("Model Performance", "Every number below was produced by the actual training run")
    disclaimer()
    m, err = get_metrics()
    if err:
        st.error(err)
        return
    models = m["models"]
    ds = m["dataset"]
    st.caption(f"Held-out test set: {ds['n_test']} records · training set: {ds['n_train']} records · "
               f"stratified split · seed {ds['random_state']}")
    st.success(f"Final model: **{m['selected_model']}** - {m['selection_reason']}")

    rows = []
    for name, r in models.items():
        t = r["test"]
        rows.append({"Model": name, "Accuracy": t["accuracy"], "Precision": t["precision"],
                     "Recall": t["recall"], "F1-score": t["f1"], "ROC-AUC": t["roc_auc"],
                     "False negatives": t["fn"]})
    table = pd.DataFrame(rows).set_index("Model")
    st.subheader("Test-set metrics")
    st.dataframe(table.round(3))

    sel = models[m["selected_model"]]["test"]
    k = st.columns(5)
    for col, (lab, key) in zip(k, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                                   ("F1-score", "f1"), ("ROC-AUC", "roc_auc")]):
        col.metric(lab, f"{sel[key]:.3f}")
    st.caption(f"Metrics shown above are for the selected model ({m['selected_model']}).")

    t1, t2, t3, t4 = st.tabs(["Comparison", "Confusion matrices", "ROC & AUC", "Feature importance"])
    with t1:
        st.pyplot(ev.plot_model_comparison({n: r["test"] for n, r in models.items()}))
        cv_rows = [{"Model": n, **{k.upper(): round(v, 3) for k, v in r["cv"].items()}} for n, r in models.items()]
        st.markdown("**5-fold cross-validation on the training set** (used to select the final model)")
        st.dataframe(pd.DataFrame(cv_rows).set_index("Model"))
    with t2:
        cols = st.columns(len(models))
        for col, (n, r) in zip(cols, models.items()):
            col.pyplot(ev.plot_confusion_matrix(r["test"]["confusion_matrix"], n))
            col.caption(f"False negatives (diabetic predicted as non-diabetic): {r['test']['fn']}")
    with t3:
        st.pyplot(ev.plot_roc_curves({n: r["roc"] for n, r in models.items()}))
    with t4:
        st.pyplot(ev.plot_feature_importance(m["feature_importance"]))

    with st.expander("Classification reports"):
        for n, r in models.items():
            st.markdown(f"**{n}**")
            st.code(r["test"]["classification_report"])
    with st.expander("What do these metrics mean in healthcare?"):
        st.markdown("""
- **Accuracy** - share of all patients classified correctly. Can look good even when the rarer diabetic class is missed.
- **Precision** - of those flagged as diabetic, how many truly were. Low precision = many false alarms (unnecessary worry/tests).
- **Recall (sensitivity)** - of the truly diabetic patients, how many were caught. **Most important for screening**, because a false negative (missed case) can delay care.
- **F1-score** - balance of precision and recall.
- **ROC-AUC** - how well the model separates the two classes across all thresholds (0.5 = random, 1.0 = perfect).
- **Confusion matrix** - counts of true/false positives and negatives; the bottom-left cell holds the false negatives.
""")


def page_explorer():
    hero("Dataset Explorer", "Pima Indians Diabetes Database")
    df, err = get_data()
    if err:
        st.error(err)
        st.code(MANUAL_INSTRUCTIONS.format(path=DATA_PATH))
        return
    rep = data_quality_report(df)
    a, b, c, d = st.columns(4)
    a.metric("Records", rep["n_records"]); b.metric("Features", rep["n_features"])
    c.metric("Duplicate rows", rep["duplicate_rows"])
    d.metric("Hidden missing (zeros)", sum(rep["invalid_zero_counts"].values()))

    t1, t2, t3, t4 = st.tabs(["Preview", "Class balance & missing values", "Distributions", "Correlation"])
    with t1:
        st.dataframe(df.head(50))
        st.dataframe(df.describe().T)
    with t2:
        l, r = st.columns(2)
        l.pyplot(ev.plot_class_distribution(rep["class_counts"]))
        with r:
            st.markdown("**Missing-value analysis**")
            st.write(f"Explicit NaN values: {sum(rep['missing_values'].values())}")
            zeros = pd.DataFrame({"Zeros (invalid)": rep["invalid_zero_counts"]})
            zeros["% of rows"] = (zeros["Zeros (invalid)"] / rep["n_records"] * 100).round(1)
            st.dataframe(zeros)
            st.caption("In these columns a zero is physiologically impossible, so it is treated as a missing "
                       "value and filled by median imputation inside the training pipeline.")
    with t3:
        feat = st.selectbox("Choose a feature", FEATURE_COLUMNS)
        st.pyplot(ev.plot_feature_distribution(df, feat, TARGET_COLUMN))
        st.caption("Green = non-diabetic (0), teal = diabetic (1). Impossible zeros are hidden in medical columns.")
    with t4:
        st.pyplot(ev.plot_correlation_heatmap(df))


def page_awareness():
    hero("Healthcare Awareness", "General educational information about diabetes")
    disclaimer()
    c1, c2 = st.columns(2)
    with c1:
        card("🥗 Healthy eating",
             "Health organisations generally encourage a balanced diet rich in vegetables, fruit, whole grains, "
             "legumes and lean proteins, with less sugary drinks, refined carbohydrates and highly processed food.")
        card("🏃 Physical activity",
             "The WHO recommends adults aim for at least 150 minutes of moderate-intensity physical activity per "
             "week, plus muscle-strengthening activity, if their health allows. Reducing long periods of sitting also helps.")
        card("🩺 Regular health checkups",
             "Periodic checkups can include blood glucose, blood pressure, weight and cholesterol. Early detection "
             "makes diabetes and related conditions easier to manage.")
    with c2:
        card("⚠️ Common risk factors",
             "Overweight or obesity, family history of diabetes, older age, physical inactivity, high blood pressure, "
             "a history of gestational diabetes, and certain ethnic backgrounds.")
        card("👩‍⚕️ Why professional consultation matters",
             "Diabetes is diagnosed by laboratory tests (such as fasting glucose or HbA1c) interpreted by a clinician. "
             "A doctor considers your full history - something no screening app can do. Treatment decisions should "
             "always come from a qualified professional.")
        card("📚 Learn more",
             "World Health Organization (who.int/health-topics/diabetes), International Diabetes Federation "
             "(idf.org), American Diabetes Association (diabetes.org), CDC (cdc.gov/diabetes).")
    st.caption("This page offers general information only. It is not personal medical advice.")


def page_about():
    hero("About the Project", "GlucoCare AI - Artificial Intelligence & Machine Learning in Healthcare")
    sections = {
        "Objective": "Build an educational, end-to-end ML system that estimates diabetes likelihood from basic health measurements.",
        "Problem statement": "Early diabetes screening is often delayed. A simple data-driven tool can help illustrate how ML supports early risk awareness - while recognising that it can never replace clinical testing.",
        "Proposed solution": "A Scikit-learn pipeline (cleaning, imputation, scaling, classifier) trained on the Pima Indians dataset, compared across two algorithms, and delivered through an interactive Streamlit dashboard.",
        "Methodology": "Data inspection → invalid-zero handling → EDA → stratified split → pipeline with cross-validated tuning → evaluation (accuracy, precision, recall, F1, ROC-AUC) → model saving with Joblib → web interface.",
        "Technology stack": "Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib.",
        "Limitations": "Small dataset (768 records); only female patients of Pima Indian heritage aged 21+, so results may not generalise; some measurements were missing and imputed; probabilities may be poorly calibrated; no external validation; not a clinical tool.",
        "Future scope": "Larger and more diverse datasets, additional algorithms (e.g. gradient boosting), probability calibration, explainability (SHAP), external validation with clinicians, multilingual interface, and secure deployment.",
    }
    for title, text in sections.items():
        card(title, text)


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
PAGES = {
    "🏠 Home": page_home,
    "🩺 Risk Prediction": page_prediction,
    "📊 Model Performance": page_performance,
    "🔍 Dataset Explorer": page_explorer,
    "🥗 Healthcare Awareness": page_awareness,
    "ℹ️ About Project": page_about,
}


def main():
    logo = ROOT / "assets" / "logo.png"
    if logo.exists():
        st.sidebar.image(str(logo), width=90)
    st.sidebar.markdown("### GlucoCare AI")
    st.sidebar.caption("AI-Powered Diabetes Risk Screening System")
    choice = st.sidebar.radio("Navigation", list(PAGES))
    st.sidebar.markdown("---")
    st.sidebar.caption("Educational prototype. Not a medical diagnostic system.")
    PAGES[choice]()


main()
