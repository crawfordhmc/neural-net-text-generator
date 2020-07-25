import numpy as np
from pickle import dump
import tensorflow as tf
import doc_methods

in_filename = "ss_char.txt"
raw_text = doc_methods.load_doc(in_filename)
lines = raw_text.split("\n")

# integer encode
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
    encoded_seq = [mapping[char] for char in line]
    sequences.append(encoded_seq)
vocab_size = len(mapping)
print("Vocab size: %d" % vocab_size)

# separate I/O
sequences = np.array(sequences)
X, y = sequences[:, : -1], sequences[:, -1]
sequences = [tf.keras.utils.to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(sequences)
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# define model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=50)
model.save("ss_char.h5")
dump(mapping, open('mapping.pkl', 'wb'))
