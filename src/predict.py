import joblib
import pandas as pd
from datetime import datetime

from preprocess import clean_text


model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


text = input("Enter text: ")


cleaned = clean_text(text)

vectorized = vectorizer.transform([cleaned])

prediction = model.predict(vectorized)

print("Predicted Sentiment:", prediction[0])


history = {
    'text': [text],
    'prediction': [prediction[0]],
    'timestamp': [datetime.now()]
}

history_df = pd.DataFrame(history)


history_df.to_csv(
    "outputs/prediction_history.csv",
    mode='a',
    index=False,
    header=False
)