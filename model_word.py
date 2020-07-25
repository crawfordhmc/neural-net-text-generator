import doc_methods
import tensorflow as tf
from pickle import dump
import numpy as np


in_filename = 'ss_words.txt'
doc = doc_methods.load_doc(in_filename)
lines = doc.split("\n")
# integer encode sequences of words
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
vocab_size = len(tokenizer.word_index) + 1
# separate I/O
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# define model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 50, input_length=seq_length))
model.add(tf.keras.layers.LSTM(100, return_sequences=True))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))
# compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# fit model
model.fit(X, y, batch_size=64, epochs=50)
# save model to file
model.save("ss_words.h5")
# save the tokenizer
dump(tokenizer, open("tokenizer.pkl", "wb"))
