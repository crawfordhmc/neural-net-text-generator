# Extension: use complete sentences + padding as input size
import doc_methods


doc = doc_methods.load_doc("shakespeare_input.txt")
# clean document and cut down size
tokens = doc_methods.clean_doc(doc[:int(len(doc)/4)])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))
# organize into sequences of tokens
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i - length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)
print('Total Sequences: %d' % len(sequences))

# save sequences to file
out_filename = 'ss_words.txt'
doc_methods.save_doc(sequences, out_filename)
