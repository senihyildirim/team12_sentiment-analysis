# Import necessary libraries
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle  # For saving tokenizer and label encoder

# Ensure the 'punkt' tokenizer model is available
nltk.download('punkt')

# Load the dataset
df = pd.read_csv('datasets/emotion_dataset.csv')

# Tokenize the text
df['tokens'] = df['text'].apply(word_tokenize)

# Prepare the tokenizer and text sequences
MAX_NB_WORDS = 50000  # maximum number of words to keep based on frequency
MAX_SEQUENCE_LENGTH = 250  # maximum number of words in each text
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# Save tokenizer
with open('saved_model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Encoding categorical labels
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(df['label'])
y = to_categorical(integer_encoded)

# Save label encoder
with open('saved_model/labelencoder.pickle', 'wb') as handle:
    pickle.dump(labelencoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Model configuration
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

# Build the model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Define the checkpoint path and filenames
filepath = "saved_model/model-{epoch:02d}-{val_accuracy:.2f}.keras"  
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train the model
epochs = 50
batch_size = 64
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_acc)

# Save the final model
model.save('saved_model/final_model.keras')
