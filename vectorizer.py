from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer():
    """
    Initialize a TF-IDF Vectorizer with basic settings.
    """
    return TfidfVectorizer(stop_words='english', max_df=0.7)

def vectorize_text(vectorizer, train_texts, test_texts):
    """
    Fit vectorizer on training text, transform both train and test.
    """

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test

