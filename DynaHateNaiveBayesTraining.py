import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib

# Load the dataset
df = pd.read_csv("Balanced_DynaHate_Sample.csv")

# Filter training data
train_df = df[df['split'] == 'train']

# Group text and labels
X_train = train_df['text'].astype(str)
y_train = train_df['label']

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_df=0.95,
        min_df=5,
        ngram_range=(1, 2)
    )),
    ('nb', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, "DynaHate_trained_naive_Bayes.joblib")

# Succes message
print("Training complete. Model saved")

test_df = df[df['split'] == 'test']

# Prepare test inputs and labels
X_test = test_df['text'].astype(str)
y_test = test_df['label']

# Load trained model
pipeline = joblib.load("DynaHate_trained_naive_Bayes.joblib")

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print metrics
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))