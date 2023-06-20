# Import necessary libraries
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Symptom2Disease.csv')

# Split the dataset into input (text) and output (label) columns
text_data = df['text'].values
labels = df['label'].values

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
text_train, text_test, labels_train, labels_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)

# Create a tokenizer and fit it on the training text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_train)

# Convert the text data to sequences of tokens
sequences_train = tokenizer.texts_to_sequences(text_train)
sequences_test = tokenizer.texts_to_sequences(text_test)

# Pad the sequences to have the same length
max_sequence_length = max(len(seq) for seq in sequences_train)

# Create a function for prediction
def predict_disease(text):
    # Load the trained model
    model = tf.keras.models.load_model('disease_model.h5')

    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_sequence_length)

    # Make the prediction
    predicted_probs = model.predict(sequence)[0]
    predicted_class = tf.argmax(predicted_probs).numpy()
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    # Calculate the confidence score
    predicted_prob = predicted_probs[predicted_class]

    # Return the predicted class and confidence score
    return predicted_label, str(round(predicted_prob*100, 2)) + "%"