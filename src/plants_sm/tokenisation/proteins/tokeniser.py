from typing import List

from plants_sm.tokenisation.tokeniser import Tokenizer


class KmerTokenizer(Tokenizer):
    """
    Tokenizer class that is used to encode and decode protein sequences.

    Attributes
    ----------
    alphabet: str
        alphabet used to encode and decode protein sequences
    size: int
        size of the alphabet
    """

    alphabet: str = 'ARNDCQEGHILKMFPSTWYVXOUBZ-'
    size: int = len(alphabet)

    def tokenize(self, sequence: str) -> List[str]:
        """
        Tokenize a protein sequence.

        Parameters
        ----------
        sequence: str
            protein sequence to tokenize

        Returns
        -------
        List[str]
            tokenized protein sequence
        """
        if all([aa in self.alphabet for aa in sequence]):
            return list(sequence)

        else:
            raise ValueError(f'Invalid amino acid in sequence: {sequence}')


