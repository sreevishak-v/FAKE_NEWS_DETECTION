import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from joblib import dump

true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')

true_news['label'] = 1
fake_news['label'] = 0

combined_data = pd.concat([true_news, fake_news], ignore_index=True)

combined_data = combined_data.sample(frac=1).reset_index(drop=True)

X = combined_data['text']
y = combined_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

model_pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression())
])

model_pipeline_lr.fit(X_train, y_train)

y_pred_lr = model_pipeline_lr.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_score_lr = f1_score(y_test, y_pred_lr)
classification_report_lr = classification_report(y_test, y_pred_lr)

print("Evaluation Metrics for Logistic Regression Model")
print("------------------------------------------------")
print("Accuracy:", accuracy_lr)
print("Precision:", precision_lr)
print("Recall:", recall_lr)
print("F1 Score:", f1_score_lr)
print("Classification Report:")
print(classification_report_lr)

dump(model_pipeline_lr, 'fake_news_detection_model.pkl')