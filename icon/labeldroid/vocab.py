import pickle
from typing import Dict
from typing_extensions import Self


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    word2idx: Dict[str, int]
    idx2word: Dict[str, str]
    idx: int

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[str(self.idx)] = word
            self.idx += 1

    @classmethod
    def load(cls, path: str) -> Self:
        with open(path, "rb") as f:
            state_dict = pickle.load(f)
        vocab = cls()
        vocab.word2idx = state_dict["word2idx"]
        vocab.idx2word = state_dict["idx2word"]
        vocab.idx = max(int(idx) for idx in vocab.idx2word.keys())
        return vocab

    def save(self, path: str):
        with open(path, "wb") as f:
            state_dict = {
                "word2idx": self.word2idx,
                "idx2word": self.idx2word,
            }
            pickle.dump(state_dict, f)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
