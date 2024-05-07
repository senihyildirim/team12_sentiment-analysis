import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
import pickle

# Load the model
model_path = 'saved_model/model-02-0.93.keras'
model = load_model(model_path)

# Load tokenizer and label encoder
with open('saved_model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('saved_model/labelencoder.pickle', 'rb') as handle:
    labelencoder = pickle.load(handle)

# Load and prepare test data
df_test = pd.read_csv('datasets/text.csv')
texts = df_test['text'].values
sequences = tokenizer.texts_to_sequences(texts)
X_test = pad_sequences(sequences, maxlen=250)

y_test = labelencoder.transform(df_test['label'])
y_test = to_categorical(y_test)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test Loss: {test_loss:.2f}")

# Make predictions
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Classification report and confusion matrix
print(classification_report(y_test_labels, predicted_classes))
cm = confusion_matrix(y_test_labels, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()

# Convert labels to binary format for multi-class ROC and Precision-Recall
n_classes = y_test.shape[1]
y_test_bin = label_binarize(y_test_labels, classes=[*range(n_classes)])

# Calculate ROC Curve and ROC Area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
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

# Calculate and plot Precision-Recall curves
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], predictions[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], predictions[:, i])

plt.figure(figsize=(7, 7))
for i, color in enumerate(colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label='Precision-Recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show()
