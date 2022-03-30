#!/usr/bin/env python3

from ext_bpe import extBPE

text_train = """
Lorem ipsum dolor sit amet, consectetur adipiscINg elit, sed do eiusmod tempor INcididunt ut labore et doLore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor INreprehenderit INvoluptate velit esse cillum doLore eu fugiat nulla pariatur. Excepteur sINt occaecat cupidatat non proident, sunt INculpa qui officia deserunt mollit anim id est laborum.
"""

text_eval = """Lorem LOREM lOreM lorem"""

model = extBPE()
model.fit(text_train, min_subwords=200, max_iter=70, lowercase=True)

# text, text_ids = model.encode("lowest highest low")
text_bpe_1, text_ids_1 = model.encode(text_eval)
text_bpe_2, text_ids_2 = model.encode(text_eval, disable_special=True)
print(text_bpe_1, sum([len(word) for word in text_bpe_1]))
print(text_bpe_2, sum([len(word) for word in text_bpe_2]))

text_eval_1 = model.decode(text_ids_1)
text_eval_2 = model.decode(text_ids_2)
print(text_eval_1)
print(text_eval_2)