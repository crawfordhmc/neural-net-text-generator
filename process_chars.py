import doc_methods


raw_text = doc_methods.load_doc("shakespeare_input.txt")
# CUT FILE TO 1/4 SIZE
raw_text = raw_text[:int(len(raw_text) / 4)]
# to clean or not to clean (newlines), that is the question
# yes, because the context windows are indicated by them.
# retain all other punctuation and cases, however
tokens = raw_text.split()
raw_text = ' '.join(tokens)
# organize into sequences of characters
n = 10
sequences = list()
for i in range(n, len(raw_text)):
    seq = raw_text[i - n: i + 1]
    sequences.append(seq)
print("Total Sequences: %d" % len(sequences))
# save sequences to file
out_filename = "ss_char.txt"
doc_methods.save_doc(sequences, out_filename)
