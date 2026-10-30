# GlucoCare AI – Viva Preparation (28 Questions)

> Tip: answer in this order – *definition → how it applies in our project → one limitation.*

## A. Fundamentals

**1. What is Artificial Intelligence?**
AI is the field of building computer systems that perform tasks normally needing human intelligence – learning, reasoning, perception, decision-making. Machine Learning is one approach within AI.

**2. What is Machine Learning?**
ML lets computers learn patterns from data instead of being explicitly programmed with rules. In our project the model learns from historical patient records how measurements like glucose and BMI relate to diabetes.

**3. What is diabetes?**
A chronic condition where blood glucose stays too high because the body doesn't make enough insulin or can't use it effectively. Complications can affect the heart, kidneys, eyes and nerves. It is diagnosed with laboratory tests (fasting glucose, OGTT, HbA1c) by clinicians – not by our app.

**4. What is supervised learning?**
Learning from **labelled** examples (inputs + known correct output). Our data has a label `Outcome` (0/1), so this is supervised learning.

**5. What is classification?**
Supervised learning where the output is a **category**, not a number. Ours is binary classification: diabetic vs non-diabetic.

## B. Dataset and preprocessing

**6. Why the Pima Indians Diabetes Dataset?**
It is a well-known, publicly available, real clinical dataset with a clear binary target and eight understandable features – ideal for learning. Its weakness is that it is small and limited to one population.

**7. What are the eight input features?**
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction (family-history score), Age.

**8. Why is preprocessing necessary?**
Raw data has problems: hidden missing values, different scales, outliers. Models trained on unclean data give unreliable results. We handle impossible zeros, impute, and scale where needed.

**9. How did you handle missing/zero values?**
Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are physiologically impossible, so we treat them as missing and fill them with the **median** (less sensitive to outliers than the mean). This is done inside a Pipeline.

**10. What is data leakage and how did you avoid it?**
Leakage is when information from the test set influences training (e.g. computing medians on the whole dataset before splitting). We split first and put imputation/scaling inside a `Pipeline`, so these statistics are learned from training data only (and from training folds only during cross-validation).

**11. What is train–test splitting? Why stratified?**
We keep a portion (20%) unseen during training to estimate performance on new data. *Stratified* splitting keeps the same diabetic/non-diabetic ratio in both parts, which matters because the classes are imbalanced. A fixed `random_state=42` makes it reproducible.

**12. Why scale features for Logistic Regression but not Random Forest?**
Logistic Regression (gradient-based, regularised) is sensitive to feature scale. Trees split on thresholds, so scale doesn't matter.

## C. Models

**13. What is Logistic Regression?**
A linear classifier that combines features with weights and passes the result through a sigmoid to produce a probability between 0 and 1. It is simple, fast and interpretable (coefficients show direction of influence).

**14. What is Random Forest?**
An ensemble of many decision trees, each trained on a random sample of rows and features; their votes are combined. It reduces overfitting compared with a single tree and captures non-linear relationships.

**15. Why compare two algorithms?**
No algorithm is best for every dataset. Comparing a simple interpretable baseline with a more flexible model lets us justify our choice with evidence instead of assumption.

**16. What is overfitting? How did you reduce it?**
Overfitting = the model memorises training data and performs poorly on new data. We reduced it using a held-out test set, cross-validation, regularisation (`C` in Logistic Regression) and limiting tree depth / minimum samples per leaf in Random Forest via grid search.

**17. What is hyper-parameter tuning / GridSearchCV?**
Hyper-parameters (e.g. tree depth) are settings chosen before training. GridSearchCV tries combinations, evaluates each with k-fold cross-validation on training data, and keeps the best.

**18. Why did you use `class_weight="balanced"`?**
Diabetic cases are the minority. Without weighting, models tend to favour the majority class and miss positives. Balanced weights penalise missing a diabetic case more.

## D. Evaluation

**19. What is a confusion matrix?**
A 2×2 table of actual vs predicted classes: true negatives, false positives, false negatives, true positives. It shows *what kind* of mistakes the model makes.

**20. What are precision and recall? Which matters more here?**
Precision = TP / (TP + FP): how many flagged people truly had diabetes. Recall = TP / (TP + FN): how many people with diabetes we caught. In screening, a **false negative** (telling a diabetic person they're low-risk) is usually costlier than a false alarm, so recall gets special attention.

**21. What is the F1-score? What is F2?**
F1 is the harmonic mean of precision and recall. F2 is a variant that weights recall twice as much. We select the final model with cross-validated F2 because of the cost of false negatives.

**22. What is ROC-AUC?**
The ROC curve plots true-positive rate vs false-positive rate across all thresholds. AUC is the area under it: 0.5 = random guessing, 1.0 = perfect separation. It measures ranking quality independent of any one threshold.

**23. Why not rely on accuracy alone?**
With imbalanced classes, a model can have decent accuracy while missing many diabetic patients. Recall, precision, F1 and AUC reveal this.

**24. How did you select the final model without cheating?**
By cross-validated F2 on the **training** data. The test set was used only once at the end for an honest estimate, so it did not influence the choice. (The exact numbers come from our own run – see `models/results_summary.md`.)

**25. What is feature importance?**
A measure of how much each feature contributes to the model's decisions. In Random Forest it is the average reduction in impurity from splits on that feature. Caution: it shows association in this dataset, not medical causation, and can be biased toward features with many distinct values.

## E. Application and ethics

**26. What is Streamlit and how does the app generate predictions?**
Streamlit is a Python library for building web apps without HTML/JS. When the user clicks *Analyze Diabetes Risk*: inputs are validated → placed in a one-row DataFrame in the **same column order as training** → passed through the saved Pipeline (same zero handling, imputation, scaling) → the model gives a class and `predict_proba` gives the probability → we show it with neutral wording and a disclaimer.

**27. What are the limitations of the dataset?**
Small (768 records); only women of Pima heritage aged 21+ (poor generalisation); many hidden missing values replaced by medians; single hospital study from the 1980s; no external validation.

**28. What are the ethical concerns of AI in healthcare, and why can't this system replace a doctor?**
Concerns: bias and fairness across populations, privacy of health data, transparency/explainability, over-reliance on automated output, accountability when errors occur, and regulatory approval. Our system uses 8 numbers only; a doctor considers symptoms, history, lab confirmation and context, and takes responsibility for care. Hence our app is an educational screening aid with disclaimers – never a diagnosis.

## F. Improvements

**29. What improvements can be made in the future?**
Larger and more diverse data, more models (gradient boosting, SVM, neural nets), probability calibration, SHAP/LIME explanations, fairness analysis, external clinical validation, secure deployment and multilingual support.

**30. (Bonus) Why do predictions use the same Pipeline object as training?**
To avoid *training–serving skew*: if preprocessing at prediction time differed even slightly (feature order, imputation values, scaling), predictions would be wrong. Saving the whole pipeline with Joblib guarantees identical preprocessing.
