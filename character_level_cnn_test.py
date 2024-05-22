import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
import pickle

def load_model_and_components():
    # Load the model and tokenizer
    model = load_model('saved_model/character_level_cnn.keras')
    with open('saved_model/char_dict.pickle', 'rb') as handle:
        char_dict = pickle.load(handle)
    return model, char_dict

def preprocess_data(texts, char_dict, max_length=200):
    # Encode and pad text data
    encoded_texts = [encode_text(text, char_dict) for text in texts]
    return pad_sequences(encoded_texts, maxlen=max_length, padding='post')

def encode_text(text, char_dict):
    # Convert text to integer sequences using the character dictionary
    return [char_dict.get(char, 0) for char in text]

def evaluate_and_plot(model, X_test, y_test):
    # Model evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test Loss: {test_loss:.2f}")

    # Predictions
    predictions = model.predict(X_test)
    predicted_classes = predictions.argmax(axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # Classification report
    print(classification_report(y_test_labels, predicted_classes))

    # Confusion Matrix
    cm = confusion_matrix(y_test_labels, predicted_classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Classes')
    plt.xlabel('Predicted Classes')
    plt.show()

    # ROC and AUC
    n_classes = y_test.shape[1]
    y_test_bin = label_binarize(y_test_labels, classes=[*range(n_classes)])
    plot_roc_curve(y_test_bin, predictions, n_classes)
    plot_precision_recall_curve(y_test_bin, predictions, n_classes)

def plot_roc_curve(y_test_bin, predictions, n_classes):
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(7, 7))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in enumerate(colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(y_test_bin, predictions, n_classes):
    precision, recall, average_precision = {}, {}, {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], predictions[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], predictions[:, i])
    
    plt.figure(figsize=(7, 7))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in enumerate(colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Precision-Recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.show()

def main():
    df_test = pd.read_csv('datasets/test.csv')
    model, char_dict = load_model_and_components()
    X_test = preprocess_data(df_test['text'].values, char_dict)
    y_test = to_categorical(df_test['label'], num_classes=6)  # Make sure to adjust num_classes if necessary

    evaluate_and_plot(model, X_test, y_test)

if __name__ == '__main__':
    main()
