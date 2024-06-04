import sqlite3
import numpy as np
from numpy.linalg import norm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


word2vec = {}
word_list = []  # To preserve order of words for later annotation
vec_list = []   # To store the vectors for dimensionality reduction


for letter_range in ("a-c", "d-h", "i-o", "p-r", "s-z"):
    with sqlite3.connect(f"./data/word2vec_{letter_range}.db") as con:
        cur = con.execute("SELECT * FROM word2vec")
        for word, vec in cur:
            vec = np.frombuffer(vec, dtype=np.float32)
            word2vec[word] = vec / norm(vec)
            word_list.append(word)
            vec_list.append(vec)

def similarity(first_word, second_word):
    x = word2vec[first_word]
    y = word2vec[second_word]
    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y) )

def query(word):
    return word2vec[word]

# Convert the list of vectors to a numpy array
X = np.array(vec_list)

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)

# Plot the reduced vectors
plt.figure(figsize=(10, 8))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5)

# Annotate some points with word labels
for i in range(len(word_list)):
    if i % 1000 == 0:  # Annotate every 1000th point
        plt.annotate(word_list[i], (X_reduced[i, 0], X_reduced[i, 1]))

plt.title('t-SNE Visualization of Word Vectors')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

# print(len(word2vec))
# print(query("apple"))
# print(query("orange"))
# print(similarity("apple","orange"))
# print(len(query("apple")))
# print(np.sum(query("apple")))
# print(np.sum(query("apple")**2))

