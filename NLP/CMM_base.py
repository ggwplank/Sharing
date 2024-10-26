import numpy as np

### From our lesson, this is the text.txt that we first used
text = "the cat is on the table"

### We define our vocabulary, composed of unique terms in the text.txt
vocabulary = ["cat", "is", "the", "on", "table"]
print("Vocabulary size:", len(vocabulary))

### We define the window_size, that is the length of the sequence we want to consider in order to predict the newt word
window_size = 4
print("Window size:", window_size, "\n")

### We create one-hot vectors for each word in our vocabulary (so its size is based on our vocabulary's size)
onehot_vectors = np.identity(len(vocabulary))
print("One-hot vectors for each word in the vocabulary:\n", onehot_vectors, "\n")


### This return the index of a specific word in our pre-defined vocabulary
def index(word):
    return vocabulary.index(word)


print(f"We combine our one-hot vectors from {window_size}x{len(vocabulary)} to 1x{window_size * len(vocabulary)}\n")
### We transform the one-hot vectors sequence from shape (4, len(vocabulary)) {a matrix 4 x len(vocabulary)}
### to a single array with window_size*len(vocabulary) {a matrix 1 x window_size*len(vocabulary)}.
### In other words, we are combining our one-hot vectors into a single one.
def seq(s1, s2, s3, s4):
    combined_vector = onehot_vectors[[index(s1), index(s2), index(s3), index(s4)]].reshape(1, window_size * len(vocabulary))
    return combined_vector


### We define the correlation matrix between the words of our text.txt, in order to predict the next word based on a sequence.
### - We combine one-hot vectors of a specific sequence, passing from {4 x len(vocabulary)} to {1 x window_size*len(vocabulary)}
### - We transpose our matrix {1 x window_size * len(vocabulary)} into a matrix {window_size * len(vocabulary) x 1}
###     in order to multiply the two matrix
### - We can now multiply our matrix --> {window_size * len(vocabulary) x 1} * {1 x len(vocabulary)}
### - We repeat for each sequence and target and sum the results
CMM = np.matmul(np.transpose(seq("the", "cat", "is", "on")), onehot_vectors[[index("the")]]) + \
      np.matmul(np.transpose(seq("cat", "is", "on", "the")), onehot_vectors[[index("table")]])
print("Combined one-hot vector transposed x one-hot vector of the target")
print(np.shape(np.transpose(seq("the", "cat", "is", "on"))), "x", np.shape(onehot_vectors[[index("the")]]))
print("\nCorrelation Matrix (CMM):", np.shape(CMM), "\n", CMM, "\n")

### WARNING: It is important to use [index(i)] instead of simply index(i) in order to obtain a 2D vector and allow
### us to multiply the result with another matrix!
### print(onehot_vectors[index("the")]) -->  [1. 0. 0. 0. 0.]
### print(onehot_vectors[[index("the")]]) --> [[1. 0. 0. 0. 0.]]


(s1, s2, s3, s4) = ("the", "cat", "is", "on")

### We calculate the probability for the next word using a specific sequence.
### We are multiplying {1 x window_size * len(vocabulary)} * {window_size * len(vocabulary) x len(vocabulary)}
print("Combined one-hot vector x CMM:", np.shape(seq(s1, s2, s3, s4)), "*", np.shape(CMM))
output = np.matmul(seq(s1, s2, s3, s4), CMM)
print("The output matrix is", np.shape(output), "\n", output)


### We find and print the word which index in the matrix has the biggest correlation
print("The index of the argmax of this matrix is", np.argmax(output), "which in our dictionary corresponds to \"",
      vocabulary[np.argmax(output)], "\"")
print(f"Predicted word: {vocabulary[np.argmax(output)]}")
