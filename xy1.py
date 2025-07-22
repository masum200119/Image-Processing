import numpy as np
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, GRU, Dense, Concatenate
from tensorflow.keras.utils import to_categorical

nltk.download('stopwords')

# Sample data (replace with your actual data)
texts = ["This is an example sentence.", "Another sample input text!", "More data here..."]
labels = [0, 1, 2]  # Example categorical labels

# Step 1: Text Cleaning (Remove punctuations and lowercase)
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

texts_cleaned = [clean_text(text) for text in texts]

# Step 2: Data Augmentation (Example: Synonym replacement, here it's a placeholder)
def synonym_replacement(text):
    # Placeholder: Implement real synonym replacement if needed
    return text

texts_augmented = [synonym_replacement(text) for text in texts_cleaned]

# Step 3: Tokenization & Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_augmented)
sequences = tokenizer.texts_to_sequences(texts_augmented)
padded_sequences = pad_sequences(sequences, maxlen=50)

# Step 4: Word Embeddings
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

# Step 5: Split Dataset
X_train, X_temp, y_train, y_temp = train_test_split(padded_sequences, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

# Convert labels to categorical
y_train_cat = to_categorical(y_train)
y_val_cat = to_categorical(y_val)
y_test_cat = to_categorical(y_test)

# Step 6: Hybrid Model (BiLSTM + BiGRU)
input_layer = Input(shape=(50,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=50)(input_layer)

bilstm = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
bigru = Bidirectional(GRU(64))(bilstm)

dense_relu = Dense(64, activation='relu')(bigru)
output_layer = Dense(y_train_cat.shape[1], activation='softmax')(dense_relu)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat), epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
