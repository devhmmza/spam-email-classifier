# 📧 Spam Email Classifier

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and a **Naive Bayes classifier**.

---

## 🚀 Features

- Cleaned and preprocessed real-world dataset
- Text vectorization using **TF-IDF**
- ML model training using **Multinomial Naive Bayes**
- Model evaluation with accuracy, precision, recall, F1-score
- CLI for testing custom input messages
- Modular codebase (preprocessing, model, prediction)

---

## 📂 Project Structure

spam-email-classifier/
├── data/
│ └── SMSSpamCollection
├── src/
│ ├── preprocess.py
│ ├── vectorizer.py
│ ├── model.py
│ └── predict.py
├── run.py
├── extract_dataset.py
├── requirements.txt
└── README.md


💬 Example Predictions
Enter a message to check (or 'q' to quit): Win a free trip to Dubai!
-> SPAM

Enter a message to check (or 'q' to quit): Let's meet at 5 today.
-> HAM

