import numpy as np


def generate_vocabulary(text):
    return list(dict.fromkeys(text.lower().split()))


### We retrieve ALL the words (even the replicated) from our text.txt
def generate_words(text):
    return text.lower().split()


with open("text.txt", "r") as f:
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
def seq(vector, RM):
    # Get the indices of the words in the sequence
    indices = [index(word) for word in vector]

    # Extract the one-hot vectors corresponding to these indices
    combined_vector = RM[indices]

    reduced_sequence = np.reshape(combined_vector, (1, window_size * int(len(vocabulary) / window_size)))

    return reduced_sequence


def create_CMM():
    CMM = np.zeros((len(vocabulary), len(vocabulary)))

    for s in range(len(words) - window_size):
        sequence = words[s:s + window_size]
        target = words[s + window_size]

        m1 = np.transpose(seq(sequence, RM))

        CMM[:, index(target)] += m1[:, 0]

    print("Combined one-hot vector transposed and reduced x one-hot vector of the target")

    return CMM


def generate_random_matrix(rows, cols, mean=0, std_dev=1):
    return np.random.normal(loc=mean, scale=std_dev, size=(rows, cols))


RM = generate_random_matrix(len(vocabulary), int(len(vocabulary) / window_size))
RM2 = generate_random_matrix(len(vocabulary), len(vocabulary))
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
    # print(predicted_word)

print("\nGenerated text:\n", textout)

# check text
to_check = generate_words(textout)
count = 0
for i in range(len(words)):
    if words[i] == to_check[i]:
        count += 1
print("\n\nAccuracy:", count / len(words) * 100, "%\n")


