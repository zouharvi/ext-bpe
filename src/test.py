#!/usr/bin/env python3

from ext_bpe import extBPE
import datasets
from time import time

data = datasets.load_dataset("wmt14", "cs-en")
data = " ".join([line["en"] for line in data["train"]["translation"][:10000]])

text_train = """
Lorem ipsum dolor sit amet, consectetur adipiscINg elit, sed do eiusmod tempor INcididunt ut labore et doLore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor INreprehenderit INvoluptate velit esse cillum doLore eu fugiat nulla pariatur. Excepteur sINt occaecat cupidatat non proident, sunt INculpa qui officia deserunt mollit anim id est laborum.
"""
text_eval = """Lorem LOREM lOreM lorem"""

text_train = data
text_eval = data

time_0 = time()

model = extBPE()
model.fit(text_train, min_subwords=200, max_iter=512, lowercase=True)
print("vocab size:", model.vocab_size)

text_bpe_1, text_ids_1 = model.encode(text_eval)
text_bpe_2, text_ids_2 = model.encode(text_eval, disable_special=True)
len_bpe_1 = sum([len(word) for word in text_bpe_1])
len_bpe_2 = sum([len(word) for word in text_bpe_2])
print("length without special:", len_bpe_2)
print("length with special:   ", len_bpe_1, f"({1-len_bpe_1/len_bpe_2:.2%}% improvement)")

text_eval_1 = model.decode(text_ids_1)
text_eval_2 = model.decode(text_ids_2)
# print(text_bpe_1, text_eval_1)
# print(text_bpe_2, text_eval_2)
time_1 = time()

print("decoded: ", text_eval_1[:150])
print("original:", data[:150])
print("decoded text equals original:", text_eval_1 == data)
print(f"overall took {time_1-time_0:.2f}s")