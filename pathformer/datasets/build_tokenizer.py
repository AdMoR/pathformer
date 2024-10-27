import os
from tokenizers import BPETokenizer, normalizers, pre_tokenizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.pre_tokenizers import Whitespace, Punctuation


normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation()])
tokenizer = BPETokenizer(normalizer=normalizer, pre_tokenizer=pre_tokenizer)

fs = [os.path.join(path, f, name) for f in os.listdir(path) for name in os.listdir(os.path.join(path, f))]

tokenizer.train(fs)
