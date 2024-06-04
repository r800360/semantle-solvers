
import sqlite3

import numpy as np
from numpy.linalg import norm

word2vec = {}
def load_data(dataset: str):
    if len(dataset) > 3 and dataset[-3:] == ".db":        
        for letter_range in ("a-c", "d-h", "i-o", "p-r", "s-z"):
            with sqlite3.connect(f"./{dataset[:-3]}_{letter_range}.db") as con:
                cur = con.execute("SELECT * FROM word2vec")
                for word, vec in cur:
                    vec = np.frombuffer(vec, dtype=np.float32)
                    word2vec[word] = vec / norm(vec)
    else:
        raise ValueError("Invalid dataset file type")
    
    return word2vec