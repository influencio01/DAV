import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('spam.csv')  # Make sure the file has 'Category' and 'Message' columns

# Rename columns if needed
df.columns = ['label', 'message']  # Category -> label, Message -> message

# Convert 'ham' to 0 and 'spam' to 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorize the text (Bag of Words)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Try a custom message
custom_message = ["Win cash now! Just send WIN to 12345"]
custom_vector = vectorizer.transform(custom_message)
prediction = model.predict(custom_vector)[0]
print("Custom Message Prediction (1 = Spam, 0 = Ham):", prediction)
