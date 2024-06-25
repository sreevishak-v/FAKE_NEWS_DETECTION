import pandas as pd
import numpy as np
import spacy
import string
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import gensim.downloader as api

from google.colab import drive
drive.mount('/content/drive')

# Load datasets
fake = pd.read_csv("/content/drive/MyDrive/NLP FAKE NEWS DETECTION/nlp dfnds/Fake.csv")
true = pd.read_csv("/content/drive/MyDrive/NLP FAKE NEWS DETECTION/nlp dfnds/True.csv")

# Add category labels
fake['category'] = 1
true['category'] = 0

# Concatenate into one dataframe
df = pd.concat([fake, true]).reset_index(drop=True)

# Balancing the dataset
df_majority = df[df['category'] == 1].sample(n=3000, random_state=42)
df_minority = df[df['category'] == 0].sample(n=3000, random_state=42)
df_balanced = pd.concat([df_majority, df_minority])

# Tokenization with SpaCy
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
punctuations = string.punctuation

def spacy_tokenizer(sentence):
    doc = nlp(sentence)
    mytokens = [word.lemma_.lower().strip() for word in doc]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    return mytokens

df['tokens'] = df['text'].apply(spacy_tokenizer)

# Generate document vectors using pre-trained GloVe embeddings
model = api.load('glove-twitter-100')

def document_vector(tokens, embeddings, dim):
    token_vectors = [embeddings[token] for token in tokens if token in embeddings]
    if not token_vectors:
        return np.zeros(dim)
    return np.mean(token_vectors, axis=0)

df['vec'] = df['tokens'].apply(lambda x: document_vector(x, model, model.vector_size))

# Prepare X (features) and y (labels)
X = np.vstack(df['vec'])
y = df['category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define RandomForestClassifier pipeline
model_pipeline_rf = Pipeline([
    ('rf', RandomForestClassifier(random_state=1))  # No need to set max_iter for RandomForestClassifier
])

# Train the model
model_pipeline_rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = model_pipeline_rf.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_score_rf = f1_score(y_test, y_pred_rf, average='weighted')
classification_report_rf = classification_report(y_test, y_pred_rf)

# Print evaluation metrics
print("Evaluation Metrics for RandomForestClassifier Model")
print("---------------------------------------------------")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-score: {f1_score_rf:.4f}")
print("Classification Report:")
print(classification_report_rf)
