import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from preprocess import clean_text


df = pd.read_csv("data/raw/social_media_data.csv")


df['clean_text'] = df['text'].apply(clean_text)


X = df['clean_text']
y = df['sentiment']


vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)


model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy}")

report = classification_report(y_test, predictions)

print(report)


with open("outputs/accuracy.txt", "w") as file:
    file.write(f"Accuracy: {accuracy}")


with open("outputs/classification_report.txt", "w") as file:
    file.write(report)


joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model saved successfully")