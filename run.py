import pandas as pd
from src.preprocess import clean_text
from sklearn.model_selection import train_test_split
from src.vectorizer import get_vectorizer, vectorize_text
from src.model import train_model, evaluate_model
from src.predict import predict_message

# Step 1: Load the dataset
df = pd.read_csv("data/SMSSpamCollection", sep='\t', header=None, names=['label', 'text'])

# Step 2: Encode labels
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Clean the text
df['clean_text'] = df['text'].apply(clean_text)

# Step 4: Split the dataset
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['clean_text'], df['label_num'], test_size=0.2, random_state=42)

# Step 5: Vectorize
vectorizer = get_vectorizer()
X_train, X_test = vectorize_text(vectorizer, X_train_text, X_test_text)

# Step 6: Train the model
model = train_model(X_train, y_train)

# Step 7: Evaluate the model
results = evaluate_model(model, X_test, y_test)

test_msgs = [
    "Congratulations! You've won a free ticket. Call now to claim.",
    "Hey, are we still meeting for lunch?",
    "URGENT! You have won a $1000 Walmart gift card. Click to claim.",
    "Dad, I’ll be home in 10 minutes."
]



# Step 8: Show results
print("\nVectorization complete!")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print("\nModel Evaluation Results:")
for metric, value in results.items():
    if metric != "confusion_matrix":
        print(f"{metric}: {value:.4f}")
    else:
        print("Confusion Matrix:")
        print(value)

print("\n Spam Predictions on Custom Messages:")
for msg in test_msgs:
    result = predict_message(model, vectorizer, msg)
    print(f"> {msg[:50]}... -> {result}")   

while True:
    user_msg = input("\n Enter a message to check (or 'q' to quit): ")
    if user_msg.lower() == 'q':
        break
    result = predict_message(model, vectorizer, user_msg)
    print(f"-> {result}")
  
