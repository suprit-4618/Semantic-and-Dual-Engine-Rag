import numpy as np
import subprocess

docs = open("data/docs.txt").read().split("\n")
docs = [d for d in docs if d.strip()]

embeddings = np.load("embeddings.npy")

for i, emb in enumerate(embeddings):

    vec = ",".join(map(str, emb))

    cmd = f'../build/ndd insert test_index {i} "{vec}"'

    subprocess.run(cmd, shell=True)

print("Vectors inserted successfully.")