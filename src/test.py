#!/usr/bin/env python3

from ext_bpe import extBPE

text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

model = extBPE()
model.fit(text, min_subwords=200, max_iter=100)

text, text_ids = model.encode("lowest highest low")

text = model.decode(text_ids)
print(text)