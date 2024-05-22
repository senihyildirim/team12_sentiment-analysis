import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, LSTM, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

# Ensure the 'punkt' tokenizer model is available
nltk.download('punkt')

# Load the dataset
df = pd.read_csv('datasets/emotion_dataset.csv')

# Prepare the tokenizer and text sequences
MAX_NB_WORDS = 364392
MAX_SEQUENCE_LENGTH = 200
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

# Define the model using Sequential API
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=MAX_SEQUENCE_LENGTH),
    SpatialDropout1D(0.2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    Conv1D(filters=128, kernel_size=4, activation='relu'),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

# Build the model with a sample input
model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the checkpoint path and filenames
filepath = "saved_model/model-{epoch:02d}-{val_accuracy:.2f}.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train the model
epochs = 1
batch_size = 64
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_acc)

# Save the final model
model.save('saved_model/final_model.keras')
