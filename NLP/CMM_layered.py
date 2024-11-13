import numpy as np

# IL CANE GUARDA IL GATTO GIOCARE A PALLONE NEL PRATO
# IL CANE GUARDA --> RM --> CMM -> {IL}
# CANE GUARDA IL --> {GATTO}
# GUARDA IL GATTO --> {GIOCARE}

# {IL}, {GATTO}, {GIOCARE} --> RM2 --> CMM2 --> {A}

with open("text.txt", "r") as f:
    text = f.read()


def generate_vocabulary(text):
    return list(dict.fromkeys(text.lower().split()))


def generate_words(text):
    return text.lower().split()

vocabulary = generate_vocabulary(text)
D = len(vocabulary)
words = generate_words(text)
print("Vocabulary size:", D)

H = 12  # window_size
print("Window size:", H, "\n")

def index(word):
    return vocabulary.index(word)


def combiner(vector, RM):
    # Get the indices of the words in the sequence
    indices = [index(word) for word in vector]

    # Extract the vectors corresponding to these indices
    combined_vector = RM[indices]

    reduced_sequence = np.reshape(combined_vector, (1, H * int(D/H)))

    return reduced_sequence


def create_CMM(D, RM):
    CMM = np.zeros((D, D))

    for w in range(len(words) - H):
        tokens = words[w:w+H]
        target = words[w+H]

        combined_words = np.transpose(combiner(tokens, RM))

        CMM[:, index(target)] += combined_words[:, 0]

    return CMM


def generate_random_matrix(rows, cols, mean=0, std_dev=1):
    return np.random.normal(loc=mean, scale=std_dev, size=(rows, cols))

RM = generate_random_matrix(D, int(D/H))
RM2 = generate_random_matrix(D, int(D/H))
print("\nRandom Matrix (RM):", np.shape(RM), "\n", RM, "\n")

CMM = create_CMM(D, RM)
CMM2 = create_CMM(D, RM2)
print("\nCorrelation Matrix (CMM):", np.shape(CMM), "\n", CMM, "\n")

current_text = words[:2*H]
textout = ' '.join(current_text)

output_vector = []
j = 0

for i in range(H, len(words)):
    print(j)

    tokens = words[i - H:i]
    combined_tokens = combiner(tokens, RM)

    output = np.matmul(combined_tokens, CMM)

    output_vector.append(vocabulary[np.argmax(output)])
    j += 1

    if j >= H:
        #print("output_vector:", output_vector)
        combined_output_tokens = combiner(output_vector, RM2)
        final_output = np.matmul(combined_output_tokens, CMM2)
        predicted_word = vocabulary[np.argmax(final_output)]
        #print("predicted_word:", predicted_word)
        textout = textout + " " + predicted_word


print("\nGenerated text:\n", textout)

# check text
to_check = generate_words(textout)
count = 0
for i in range(len(words)):
    if words[i] == to_check[i]:
        count += 1
print("\n\nAccuracy:", count / len(words) * 100, "%\n")
