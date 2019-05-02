# To Do

- text preprocessing
  - possibly eliminate things inside << >> (e.g., <<sounds like she's washing dishes>>)
  - possibly eliminate things inside < > (e.g., <noise>; <baby_crying>)
  - better tokenization (currently string.split); can't use pos_words in SWDA because it eliminates disfluencies and laughter, and we should have something that works both in SWDA and the larger corpus. It can probably be a simple regex tokenizer since the transcription is very regular
- data generator that sequences and batches utterances appropriately (use torchtext?)
- loading word vectors 
  - possibly use: https://torchtext.readthedocs.io/en/latest/vocab.html#glove
  - reset embedding weights for tokens that are in the word vector vocab but have a specific annotation meaning ('[', /', --', '+', etc.)
