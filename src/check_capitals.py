#!/usr/bin/env python3

from ext_bpe import extBPE
import datasets
from time import time

data = datasets.load_dataset("wmt14", "cs-en")
data = " ".join([line["en"] for line in data["train"]["translation"][:10000]])

text_train = data
text_eval = data

words = data.split()
print(len(words), "words")
print(len([x for x in words if x.isupper() and len(x) > 1]), "capitalized words")
print(len([x for x in words if x[0].isupper()]), "initial capital letter words")