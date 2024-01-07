import pickle
from typing import Dict
from typing_extensions import Self


class Vocabulary(object):
    """
    Simple vocabulary wrapper.

    Attributes
    ----------
    word2idx : Dict[str, int]
        Mapping from words to indices.
    idx2word : Dict[str, str]
        Reverse mapping from indices to words.
    idx : int
        Maximum index in the vocabulary.
    """

    word2idx: Dict[str, int]
    idx2word: Dict[str, str]
    idx: int

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word: str):
        """Adds a new word to the vocabulary

        Parameters
        ----------
        word : str
            Word to be added.
        """
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[str(self.idx)] = word
            self.idx += 1

    @classmethod
    def load(cls, path: str) -> Self:
        """Loads the vocabulary from a file

        The file must be saved by the `Vocabulary.save()` method.

        Parameters
        ----------
        path : str
            Path to the vocabulary file.

        Returns
        -------
        Vocabulary
            Loaded vocabulary.
        """
        with open(path, "rb") as f:
            state_dict = pickle.load(f)
        vocab = cls()
        vocab.word2idx = state_dict["word2idx"]
        vocab.idx2word = state_dict["idx2word"]
        vocab.idx = max(int(idx) for idx in vocab.idx2word.keys())
        return vocab

    def save(self, path: str):
        """Saves the vocabulary to a file

        Parameters
        ----------
        path : str
            Path to save the vocabulary file.
        """
        with open(path, "wb") as f:
            state_dict = {
                "word2idx": self.word2idx,
                "idx2word": self.idx2word,
            }
            pickle.dump(state_dict, f)

    def __call__(self, word: str) -> int:
        """Returns the index of the word

        Parameters
        ----------
        word : str
            Word to get the index.

        Returns
        -------
        int
            Index of the input word.
        """
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
