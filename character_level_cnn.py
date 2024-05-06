import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('datasets/emotion_dataset.csv')

# Create a character dictionary from all unique characters in the dataset
characters = sorted(set(''.join(df['text'])))
char_dict = {char: idx + 1 for idx, char in enumerate(characters)}  # +1 because 0 is reserved for padding
num_characters = len(char_dict) + 1  # Including padding character

# Function to encode text into integer sequences using the character dictionary
def encode_text(text):
    return [char_dict.get(char, 0) for char in text]

# Apply encoding function to the text column
df['encoded'] = df['text'].apply(encode_text)

# Maximum length for padding sequences
max_length = 250

# Pad sequences for uniform input size
X = pad_sequences(df['encoded'].values, maxlen=max_length, padding='post')

# Assuming labels are categorical and range from 0 to the number of classes - 1
num_classes = df['label'].nunique()
y = to_categorical(df['label'], num_classes=num_classes)

# Define the model architecture
model = Sequential([
    Embedding(input_dim=num_characters, output_dim=50, input_length=max_length),  # Embedding layer for character-level input
    Conv1D(64, 7, activation='relu'),  # Convolutional layer
    GlobalMaxPooling1D(),  # Global max pooling to reduce dimensionality
    Dense(num_classes, activation='softmax')  # Dense layer for classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
model.fit(X, y, epochs=50, batch_size=32)

model.save('saved_model/character_level_cnn.keras')

