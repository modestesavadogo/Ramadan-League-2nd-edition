# -*- coding: utf-8 -*-
"""
Sentiment Analysis of Tweets - Complete Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support

# ---------------------------
# 1. Load Data
# ---------------------------
# Replace 'tweets.csv' with your actual file path
df = pd.read_csv('tweets.csv')
print("Data loaded. Shape:", df.shape)
print(df.head())

# ---------------------------
# 2. Exploratory Data Analysis
# ---------------------------
# Map polarity to readable labels
sentiment_map = {0: 'negative', 2: 'neutral', 4: 'positive'}
df['sentiment'] = df['polarity'].map(sentiment_map)

# Class distribution
sns.countplot(x='sentiment', data=df, order=['negative', 'neutral', 'positive'])
plt.title('Sentiment Class Distribution')
plt.show()
print(df['sentiment'].value_counts())

# Text length analysis
df['text_length'] = df['text'].apply(len)
print(df.groupby('sentiment')['text_length'].describe())
sns.boxplot(x='sentiment', y='text_length', data=df)
plt.title('Text Length by Sentiment')
plt.show()

# Optional: Word clouds (uncomment if you have wordcloud installed)
# from wordcloud import WordCloud
# for sentiment in ['negative', 'neutral', 'positive']:
#     text = ' '.join(df[df['sentiment'] == sentiment]['text'])
#     wordcloud = WordCloud(width=800, height=400).generate(text)
#     plt.figure(figsize=(10,5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.title(f'Word Cloud - {sentiment}')
#     plt.axis('off')
#     plt.show()

# ---------------------------
# 3. Text Preprocessing
# ---------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_tweet(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (keep the word without #)
    text = re.sub(r'#', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase
    text = text.lower()
    # Tokenize, remove stopwords, lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(clean_tweet)

# ---------------------------
# 4. Train / Validation / Test Split (Stratified)
# ---------------------------
# First split: 70% train, 30% temp (validation+test)
df_train, df_temp = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df['polarity']
)

# Second split: split temp into 15% validation, 15% test
df_val, df_test = train_test_split(
    df_temp, test_size=0.5, random_state=42, stratify=df_temp['polarity']
)

print(f"Train size: {df_train.shape[0]}")
print(f"Validation size: {df_val.shape[0]}")
print(f"Test size: {df_test.shape[0]}")

# ---------------------------
# 5. Feature Extraction with TF-IDF (fit only on training)
# ---------------------------
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')

X_train = vectorizer.fit_transform(df_train['clean_text'])
X_val   = vectorizer.transform(df_val['clean_text'])
X_test  = vectorizer.transform(df_test['clean_text'])

y_train = df_train['polarity'].values
y_val   = df_val['polarity'].values
y_test  = df_test['polarity'].values

# ---------------------------
# 6. Baseline Models & Selection
# ---------------------------
# We'll try a few classifiers and pick the best based on validation F1-macro
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': LinearSVC(max_iter=2000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', use_label_encoder=False)
}

best_val_f1 = 0
best_model_name = None
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    f1_macro = f1_score(y_val, y_pred_val, average='macro')
    print(f"{name} Validation F1-macro: {f1_macro:.4f}")
    if f1_macro > best_val_f1:
        best_val_f1 = f1_macro
        best_model_name = name
        best_model = model

print(f"\nBest baseline model: {best_model_name} with F1-macro = {best_val_f1:.4f}")

# ---------------------------
# 7. Hyperparameter Tuning (on the best model)
# ---------------------------
# Example: tune Logistic Regression if it was best
if best_model_name == 'LogisticRegression':
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }
    grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                        param_grid, cv=5, scoring='f1_macro')
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    tuned_model = grid.best_estimator_
else:
    # For other models, you could define similar grids
    tuned_model = best_model  # fallback

# Evaluate tuned model on validation
y_pred_val_tuned = tuned_model.predict(X_val)
val_f1_tuned = f1_score(y_val, y_pred_val_tuned, average='macro')
print(f"Tuned model validation F1-macro: {val_f1_tuned:.4f}")

# ---------------------------
# 8. Final Evaluation on Test Set
# ---------------------------
y_pred_test = tuned_model.predict(X_test)

# Compute all requested metrics
accuracy = accuracy_score(y_test, y_pred_test)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred_test, average='macro')
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')

print("\n" + "="*50)
print("FINAL TEST SET EVALUATION")
print("="*50)
print(f"Accuracy:          {accuracy:.4f}")
print(f"Macro Precision:   {precision_macro:.4f}")
print(f"Macro Recall:      {recall_macro:.4f}")
print(f"Macro F1:          {f1_macro:.4f}")
print(f"Weighted Precision:{precision_weighted:.4f}")
print(f"Weighted Recall:   {recall_weighted:.4f}")
print(f"Weighted F1:       {f1_weighted:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=['negative','neutral','positive']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['neg','neu','pos'], yticklabels=['neg','neu','pos'])
plt.title('Confusion Matrix - Test Set')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()

# ---------------------------
# 9. Discussion and Next Steps
# ---------------------------
# (Add comments or markdown in a notebook)
print("\n" + "="*50)
print("DISCUSSION")
print("="*50)
print("""
- The pipeline strictly avoids data leakage by fitting the TF‑IDF vectorizer only on the training set.
- Stratified splits ensure class proportions are preserved.
- Multiple baseline models were compared; the best was tuned further.
- Final metrics on the test set provide an unbiased estimate of real‑world performance.
- If classes are imbalanced, macro‑F1 is a more informative metric than accuracy.
- Potential improvements: 
    * Use word embeddings (GloVe, Word2Vec) or transformer models (BERT) for better semantic understanding.
    * Incorporate metadata (date, user) if they are available at inference time and do not leak future information.
    * Handle class imbalance with class weights or oversampling.
    * Monitor concept drift if deployed in production.
""")

