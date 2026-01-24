# -------------------------------
# Broadcast AI Model Training - Character-level TF-IDF + RandomForest
# -------------------------------

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Step 1: Loading dataset...")

base_path = 'Datasets'  # Change to your dataset folder path
data_rows = []

region_map = {
    'TAMIL': 'Tamil Nadu',
    'HINDI': 'Delhi',
    'TELUGU': 'Andhra Pradesh',
    'MALAYALAM': 'Kerala',
    'KANNADA': 'Karnataka',
    'PUNJAB': 'Punjab',
    'ASSAM': 'Assam',
    'GUJARATI': 'Gujarat',
    'BENGALI': 'West Bengal',
    'ENGLISH': 'All'
}

for language in os.listdir(base_path):
    lang_path = os.path.join(base_path, language, 'split')
    if os.path.exists(lang_path):
        files = os.listdir(lang_path)
        print(f"  Found {len(files)} files for language {language}")
        for file_name in files:
            file_path = os.path.join(lang_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                region = region_map.get(language.upper(), 'All')
                data_rows.append({
                    'language': language,
                    'region': region,
                    'content': text,
                    'file_name': file_name
                })

df_content = pd.DataFrame(data_rows)
print(f"Loaded {len(df_content)} content items.\n")

# -------------------------------
# Step 2: Train/Test Split
# -------------------------------
X = df_content['content']
y = df_content['language']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------------
# Step 3: Vectorization (Char-level TF-IDF)
# -------------------------------
print("Training character-level TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# Step 4: Random Forest Classifier
# -------------------------------
print("Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train_vec, y_train)

# -------------------------------
# Step 5: Evaluation
# -------------------------------
y_pred = model.predict(X_test_vec)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Step 6: Save Model and Vectorizer
# -------------------------------
MODEL_FILE = 'language_model_rf.pkl'
VECTORIZER_FILE = 'vectorizer_rf.pkl'

joblib.dump(model, MODEL_FILE)
joblib.dump(vectorizer, VECTORIZER_FILE)
print("Model and vectorizer saved for future use.")
