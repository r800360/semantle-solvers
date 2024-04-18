import sqlite3
import numpy as np
from numpy.linalg import norm

word2vec = {}
for letter_range in ("a-c", "d-h", "i-o", "p-r", "s-z"):
    with sqlite3.connect(f"./data/word2vec_{letter_range}.db") as con:
        cur = con.execute("SELECT * FROM word2vec")
        for word, vec in cur:
            vec = np.frombuffer(vec, dtype=np.float32)
            word2vec[word] = vec / norm(vec)

def similarity(first_word, second_word):
    x = word2vec[first_word]
    y = word2vec[second_word]
    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y) )


print(len(word2vec))
print(similarity("apple","orange"))