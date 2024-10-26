import numpy as np


def generate_vocabulary(text):
    return list(dict.fromkeys(text.lower().split()))


### We retrieve ALL the words (even the replicated) from our text.txt
def generate_words(text):
    return text.lower().split()


with open("text.txt","r") as f:
    text = f.read()

### We define our vocabulary, composed of unique terms in the text.txt
vocabulary = generate_vocabulary(text)
print("Vocabulary size:", len(vocabulary))

words = generate_words(text)

### We define the window_size, that is the length of the sequence we want to consider in order to predict the newt word
window_size = 12
print("Window size:", window_size, "\n")

### We create one-hot vectors for each word in our vocabulary (so its size is based on our vocabulary's size)
onehot_vectors = np.identity(len(vocabulary))
print(f"There are {len(onehot_vectors)} one-hot vectors for each word in the vocabulary:\n", onehot_vectors, "\n")


### This return the index of a specific word in our pre-defined vocabulary
def index(word):
    return vocabulary.index(word)


print(f"We combine our one-hot vectors from {window_size}x{len(vocabulary)} to 1x{window_size * len(vocabulary)}\n")
### We are combining our one-hot vectors into a single one.
def seq(vector,RM):
    # Get the indices of the words in the sequence
    indices = [index(word) for word in vector]

    # Extract the one-hot vectors corresponding to these indices
    combined_vector = onehot_vectors[indices]

    reduced_sequence = np.matmul(combined_vector, RM)

    reduced_sequence = np.reshape(reduced_sequence, (1, len(vocabulary)))

    return reduced_sequence


def create_CMM():
    CMM = np.zeros((len(vocabulary), len(vocabulary)))

    for s in range(len(words) - window_size):
        sequence = words[s:s + window_size]
        target = words[s + window_size]

        m1 = np.transpose(seq(sequence,RM))

        m2 = onehot_vectors[[index(target)]]

        CMM += np.matmul(m1, m2)

    print("Combined one-hot vector transposed and reduced x one-hot vector of the target")
    print(np.shape(m1), "x", np.shape(m2))

    return CMM


def generate_random_matrix(rows, cols, mean=0, std_dev=1):
    return np.random.normal(loc=mean, scale=std_dev, size=(rows, cols))

RM = generate_random_matrix(len(vocabulary), int(len(vocabulary)/window_size))
print("\nRandom Matrix (RM):", np.shape(RM), "\n", RM, "\n")

CMM = create_CMM()
print("\nCorrelation Matrix (CMM):", np.shape(CMM), "\n", CMM, "\n")

current_text = words[:window_size]

textout = ' '.join(current_text)

for i in range(window_size, len(words)):
    vector = words[i - window_size:i]

    sequence = seq(vector, RM)

    output = np.matmul(sequence, CMM)
    predicted_word = vocabulary[np.argmax(output)]
    textout = textout + " " + predicted_word
    #print(predicted_word)

print("\nGenerated text:\n", textout)


