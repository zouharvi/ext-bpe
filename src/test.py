#!/usr/bin/env python3

from ext_bpe import extBPE

text = "low low low low low lower lower newest newest newest newest newest newest widest widest widest happier happier"

model = extBPE()
model.fit(text)

text, text_ids = model.encode("lowest highest low")

text = model.decode(text_ids)
print(text)