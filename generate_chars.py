from pickle import load
import tensorflow as tf


# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    for _ in range(n_chars):
        # encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # truncate sequences to a fixed length
        encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=seq_length, truncating="pre")
        # one hot encode
        encoded = tf.keras.utils.to_categorical(encoded, num_classes=len(mapping))
        # predict character
        yhat = model.predict_classes(encoded, verbose=0)
        # reverse map integer to character
        out_char = ""
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        # append to input
        in_text += char
    return in_text


model = tf.keras.models.load_model("ss_char.h5")
mapping = load(open("mapping.pkl", "rb"))
# test start of line
print(generate_seq(model, mapping, 20, "Romeo: ", 50))
# test start of sentence
print(generate_seq(model, mapping, 20, ". Consider", 50))
# test mid-line
print(generate_seq(model, mapping, 20, ", but that", 50))
# test not in original
print(generate_seq(model, mapping, 20, "Hello worl", 50))
