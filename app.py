import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import re
import random


def load_text(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    return text


def tokenize_text(text, seq_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    sequences = []
    words = text.split()
    for i in range(seq_length, len(words)):
        seq = words[i-seq_length:i+1]
        sequences.append(tokenizer.texts_to_sequences([seq])[0])
    
    sequences = np.array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y, num_classes=total_words)
    return X, y, tokenizer, total_words


def build_model(input_length, vocab_size):
    model = Sequential([
        Embedding(vocab_size, 100, input_length=input_length),
        LSTM(150, return_sequences=True),
        LSTM(100),
        Dense(100, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def generate_text(model, tokenizer, seed_text, seq_length, num_words):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


filename = "text.txt"  
text_data = load_text(filename)


seq_length = 10  
X, y, tokenizer, vocab_size = tokenize_text(text_data, seq_length)


model = build_model(seq_length, vocab_size)
model.fit(X, y, epochs=50, batch_size=64, verbose=1)


seed_text = "what is this text about"
generated_text = generate_text(model, tokenizer, seed_text, seq_length, 50)
print("Generated Text:", generated_text)
