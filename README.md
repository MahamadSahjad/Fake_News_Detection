📰 Fake News Detection using Machine Learning

Short description
A Machine Learning system that detects fake news using NLP preprocessing, TF‑IDF feature extraction, and three supervised classifiers: Logistic Regression, Random Forest, and Gradient Boosting.

Table of contents

Project Overview

Repository link

Dataset

Preprocessing & Feature Extraction

Models Used

Evaluation Metrics

Results

How to run

Project structure

Save & load trained models

Future improvements

Contributing

License

Author

Project Overview

This repository contains an implementation of a Fake News Detection pipeline built in Python. The pipeline performs text cleaning and preprocessing, converts text into TF‑IDF vectors, trains three machine learning models (Logistic Regression, Random Forest, and Gradient Boosting), and evaluates them using standard classification metrics.

The goal is to provide a clear, reproducible baseline for detecting misinformation in news text data.

Repository link

Notebook: Fake_News_Detections.ipynb

If you want to open the notebook directly, use:

https://github.com/MahamadSahjad/Fake_News_Detection/blob/main/Fake_News_Detections.ipynb

Dataset

Use a labeled dataset containing text (or title + text) and label columns.

Typical sources: Kaggle fake news datasets, News datasets, or any labeled CSV.

Expected CSV format:

id,title,text,label
1,Some headline,Full article text,0
2,Another headline,Full article text,1

Note: label should be encoded as 0 = Real, 1 = Fake (or vice versa — be consistent throughout the notebook).

Preprocessing & Feature Extraction

Typical preprocessing steps included in the notebook:

Lowercasing

Removing punctuation, URLs, HTML tags

Tokenization

Stopword removal

Lemmatization or stemming

Feature extraction:

TF‑IDF Vectorization (scikit‑learn TfidfVectorizer) — recommended to use ngram_range=(1,2) and max_features (e.g., 50k) depending on dataset size.

Example (snippet)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
X = vectorizer.fit_transform(df['text'])

Models Used

The notebook trains and compares the following three classifiers. Names and code examples are provided so the repo is explicit and accurate.

1) Logistic Regression

Fast, interpretable linear classifier.

Useful baseline for text classification.

Example:

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)

2) Random Forest

Ensemble of decision trees; robust to overfitting when tuned.

Good for non-linear patterns and feature importance inspection.

Example:

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

3) Gradient Boosting (Scikit‑learn's GradientBoostingClassifier)

Sequential ensemble that often gives better accuracy but can be slower.

If you used XGBoost/LightGBM in the notebook, update the import accordingly.

Example (scikit-learn):

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)

Evaluation Metrics

Use these to compare models:

Accuracy — overall correctness

Precision — correctness of positive predictions

Recall — coverage of actual positives

F1‑score — harmonic mean of precision & recall

ROC AUC — ranking quality

Confusion matrix — TP/FP/TN/FN breakdown

Example (scikit-learn):

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

Results

Replace the placeholder values below with the actual metrics from your notebook.

Model

Accuracy

Precision

Recall

F1-score

ROC AUC

Logistic Regression

REPLACE_WITH_LR_ACC

REPLACE_WITH_LR_PREC

REPLACE_WITH_LR_REC

REPLACE_WITH_LR_F1

REPLACE_WITH_LR_AUC

Random Forest

REPLACE_WITH_RF_ACC

REPLACE_WITH_RF_PREC

REPLACE_WITH_RF_REC

REPLACE_WITH_RF_F1

REPLACE_WITH_RF_AUC

Gradient Boosting

REPLACE_WITH_GB_ACC

REPLACE_WITH_GB_PREC

REPLACE_WITH_GB_REC

REPLACE_WITH_GB_F1

REPLACE_WITH_GB_AUC

Tip: After running classification_report in your notebook, copy the values into the table above for a clean presentation.

How to run (reproduce results)

Clone the repo

git clone https://github.com/MahamadSahjad/Fake_News_Detection.git
cd Fake_News_Detection


Author

Mahamad Sahjad Dewan

GitHub: https://github.com/MahamadSahjad

Notebook: Fake_News_Detections.ipynb

