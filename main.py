import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_utilities(model_choice):
    if model_choice == '1':
        # Load the word-based model
        model = load_model('saved_model/model-01-0.93.keras')
        with open('saved_model/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('saved_model/labelencoder.pickle', 'rb') as handle:
            labelencoder = pickle.load(handle)
    elif model_choice == '2':
        # Load the character-level model
        model = load_model('saved_model/character_level_cnn.keras')
        tokenizer = None  # Assuming not used for character-level model
        labelencoder = pickle.load(open('saved_model/labelencoder_char.pickle', 'rb'))
    
    return model, tokenizer, labelencoder

def preprocess_text(text, tokenizer, max_sequence_length=250, model_choice='1'):
    if model_choice == '1':
        # Preprocess for word-based model
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
        return padded_sequence
    elif model_choice == '2':
        # Preprocess for character-level model
        encoded = np.zeros((1, max_sequence_length), dtype=int)
        for i, char in enumerate(text.lower()):
            if i < max_sequence_length:
                encoded[0, i] = CHAR_DICT.get(char, 0)  # Use 0 for unknown characters
        return encoded

def predict_emotion(text, model, tokenizer, labelencoder, model_choice):
    processed_text = preprocess_text(text, tokenizer, model_choice=model_choice)
    predictions = model.predict(processed_text)
    predicted_index = np.argmax(predictions, axis=1)
    predicted_emotion = labelencoder.inverse_transform(predicted_index)
    return predicted_emotion[0]

def main():
    print("Choose a model type to use:")
    print("1: Word-based CNN")
    print("2: Character-level CNN")
    model_choice = input("Enter choice (1 or 2): ")

    model, tokenizer, labelencoder = load_utilities(model_choice)

    emotionsEnum = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}

    print("Type a sentence to analyze emotion or 'exit' to quit:")
    while True:
        input_text = input("Enter text: ")
        if input_text.lower() == 'exit':
            print("Exiting the program.")
            break
        emotion = predict_emotion(input_text, model, tokenizer, labelencoder, model_choice)
        text_emotion = list(emotionsEnum.keys())[list(emotionsEnum.values()).index(emotion)]
        print(f"Predicted Emotion: {text_emotion}")

if __name__ == "__main__":
    main()
