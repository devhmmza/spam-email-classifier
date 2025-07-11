from src.preprocess import clean_text

def predict_message(model, vectorizer, message):
    """
    Predict whether a single message is spam or ham.
    """

    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    return "SPAM" if prediction == 1 else "HAM"