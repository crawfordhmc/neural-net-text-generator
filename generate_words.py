import doc_methods
import tensorflow as tf
from random import randint
from pickle import load


def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += " " + out_word
        result.append(out_word)
    return " ".join(result)

# load cleaned text sequences
in_filename = 'ss_words.txt'
doc = doc_methods.load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1
# load model
model = tf.keras.models.load_model("ss_words.h5")
tokenizer = load(open("tokenizer.pkl", "rb"))
# select a seed text
seed_text = lines[randint(0, len(lines))]
print(seed_text + "\n")
# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed_text, 10)
print(generated)
