import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, MaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
import os

# Load the dataset
df = pd.read_csv('datasets/emotion_dataset.csv')

# Create a character dictionary from all unique characters in the dataset
characters = sorted(set(''.join(df['text'])))
char_dict = {char: idx + 1 for idx, char in enumerate(characters)}  # +1 because 0 is reserved for padding
num_characters = len(char_dict) + 1  # Including padding character

# Save the character dictionary for later use in prediction
if not os.path.exists('saved_model'):
    os.makedirs('saved_model')
with open('saved_model/char_dict.pickle', 'wb') as f:
    pickle.dump(char_dict, f)

# Function to encode text into integer sequences using the character dictionary
def encode_text(text):
    return [char_dict.get(char, 0) for char in text]  # Using 0 for characters not found in the dictionary

# Apply encoding function to the text column
df['encoded'] = df['text'].apply(encode_text)

# Maximum length for padding sequences
max_length = 250

# Pad sequences for uniform input size
X = pad_sequences(df['encoded'].values, maxlen=max_length, padding='post')

num_classes = df['label'].nunique()
y = to_categorical(df['label'], num_classes=num_classes)

# Define the model architecture
model = Sequential([
    Embedding(input_dim=num_characters, output_dim=100, input_length=max_length),
    Dropout(0.2),
    Conv1D(128, 7, activation='relu', padding='same'),
    MaxPooling1D(2),
    Conv1D(128, 5, activation='relu', padding='same'),
    GlobalMaxPooling1D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.build(input_shape=(None, max_length))  # None indicates the batch size can be variable

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=64, validation_split=0.3)

# Save the trained model
model.save('saved_model/character_level_cnn.keras')

print("Model and character dictionary saved successfully.")
