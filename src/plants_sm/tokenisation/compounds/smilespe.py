import os
from typing import List

from plants_sm.tokenisation.tokeniser import Tokenizer
from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.pretokenizer import kmer_tokenizer

import codecs
from SmilesPE.tokenizer import SPE_Tokenizer


class AtomLevelTokenizer(Tokenizer):
    """
    Tokenizer class that is used to tokenize SMILES sequences at the atom level.
    """

    def tokenize(self, sequence: str) -> List[str]:
        """
        Tokenize a SMILES sequence at the atom level.

        Parameters
        ----------
        sequence: str
            SMILES sequence to tokenize

        Returns
        -------
        List[str]
            tokenized SMILES sequence
        """
        tokens = atomwise_tokenizer(sequence)

        return tokens


class KmerTokenizer(Tokenizer):
    """
    Tokenizer class that is used to tokenize SMILES sequences at the kmer level.

    Attributes
    ----------
    kmer_size: int
        size of the kmer

    """
    kmer_size: int = 4

    def tokenize(self, sequence: str) -> List[str]:
        """
        Tokenize a SMILES sequence at the kmer level.

        Parameters
        ----------
        sequence: str
            SMILES sequence to tokenize

        Returns
        -------
        List[str]
            tokenized SMILES sequence at the kmer level
        """
        tokens = kmer_tokenizer(sequence, ngram=self.kmer_size)

        return tokens


class SPETokenizer(Tokenizer):
    """
    Tokenizer class that is used to tokenize SMILES sequences with SPE_Tokenizer.
    """

    def tokenize(self, sequence: str) -> List[str]:
        """
        Tokenize a SMILES sequence with SPE_Tokenizer.
        """

        spe_vob = codecs.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'SPE_ChEMBL.txt'))
        spe = SPE_Tokenizer(spe_vob)
        tokens = spe.tokenize(sequence)
        tokens = tokens.split(" ")
        return tokens
