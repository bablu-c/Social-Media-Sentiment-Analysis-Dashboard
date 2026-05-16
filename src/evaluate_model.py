import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from preprocess import clean_text


df = pd.read_csv("data/raw/social_media_data.csv")

df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text']
y = df['sentiment']


vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/sentiment_model.pkl")


X_vectorized = vectorizer.transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("images/confusion_matrix.png")

print("Confusion matrix saved successfully")