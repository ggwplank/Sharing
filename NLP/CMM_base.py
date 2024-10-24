import numpy as np

VOC = ["the", "cat", "is", "on", "table"]
onehot_vectors = np.identity(len(VOC))
len_seq = 4


def i(word):
    return VOC.index(word)


def seq(s1, s2, s3, s4):
    return onehot_vectors[[i(s1), i(s2), i(s3), i(s4)]].reshape(1, len_seq * len(VOC))


CMM = np.matmul(np.transpose(seq("the", "cat", "is", "on")), onehot_vectors[[i("the")]]) + np.matmul(
    np.transpose(seq("cat", "is", "on", "the")), onehot_vectors[[i("table")]])

(s1, s2, s3, s4) = ("cat", "is", "on","the")
out = np.matmul(seq(s1, s2, s3, s4), CMM)
print(VOC[np.argmax(out)])
print("#########################")