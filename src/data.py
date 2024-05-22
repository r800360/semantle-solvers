
import sqlite3

import numpy as np
from numpy.linalg import norm

word2vec = {}
def load_w2vec():
    for letter_range in ("a-c", "d-h", "i-o", "p-r", "s-z"):
        with sqlite3.connect(f"./data/word2vec_{letter_range}.db") as con:
            cur = con.execute("SELECT * FROM word2vec")
            for word, vec in cur:
                vec = np.frombuffer(vec, dtype=np.float32)
                word2vec[word] = vec / norm(vec)
    return word2vec