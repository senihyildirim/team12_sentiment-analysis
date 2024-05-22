import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import shap

def load_utilities(model_choice):
    if model_choice == '1':
        # Load the word-based model
        model = load_model('final_models/cnn_best_model.keras')
        with open('saved_model/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('saved_model/labelencoder.pickle', 'rb') as handle:
            labelencoder = pickle.load(handle)
        char_dict = None
    elif model_choice == '2':
        # Load the character-level model
        model = load_model('saved_model/character_level_cnn.keras')
        with open('saved_model/char_dict.pickle', 'rb') as handle:
            char_dict = pickle.load(handle)
        tokenizer = None  # Assuming not used for character-level model
        with open('saved_model/labelencoder.pickle', 'rb') as handle:
            labelencoder = pickle.load(handle)
    
    return model, tokenizer, labelencoder, char_dict

def preprocess_text(text, tokenizer, char_dict, max_sequence_length=250, model_choice='1'):
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
                encoded[0, i] = char_dict.get(char, 0)  # Use 0 for unknown characters
        return encoded


def predict_emotion(text, model, tokenizer, labelencoder, char_dict, model_choice):
    processed_text = preprocess_text(text, tokenizer, char_dict, model_choice=model_choice)
    predictions = model.predict(processed_text)
    predicted_index = np.argmax(predictions, axis=1)
    predicted_emotion = labelencoder.inverse_transform(predicted_index)



    text_input = ["Example text to predict dangerous emotion", "Another goood example text", "And another example text"]
    processed_texts = np.array([preprocess_text(text, tokenizer, char_dict, model_choice='1') for text in text_input])
    predictions = model.predict(processed_texts)
   
    explainer = shap.KernelExplainer(predictions, np.zeros((1, model.input_shape[1])))  # set according to input shape

    # Example text input
   
    shap_values = explainer.shap_values(text_input)
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], text_input[0])
    
    return predicted_emotion[0]

def main():
    print("Choose a model type to use:")
    print("1: Word-based CNN")
    print("2: Character-level CNN")
    model_choice = input("Enter choice (1 or 2): ")

    model, tokenizer, labelencoder, char_dict = load_utilities(model_choice)

    emotionsEnum = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}

    print("Type a sentence to analyze emotion or 'exit' to quit:")
    while True:
        input_text = input("Enter text: ")
        if input_text.lower() == 'exit':
            print("Exiting the program.")
            break
        emotion = predict_emotion(input_text, model, tokenizer, labelencoder, char_dict, model_choice)
        text_emotion = list(emotionsEnum.keys())[list(emotionsEnum.values()).index(emotion)]
        print(f"Predicted Emotion: {text_emotion}")

if __name__ == "__main__":
    main()
