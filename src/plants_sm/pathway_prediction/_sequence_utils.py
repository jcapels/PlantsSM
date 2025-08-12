import re

class SequenceUtils:

    @staticmethod
    def is_valid_protein_sequence(seq, allow_extended=False):
        """
        Validates whether a given sequence is a valid protein sequence.

        Parameters:
        - seq (str): The protein sequence to validate.
        - allow_extended (bool): Whether to allow extended amino acids (X, B, Z, U, O).

        Returns:
        - bool: True if valid, False otherwise.
        """
        seq = seq.upper()
        if allow_extended:
            valid_aa = "ACDEFGHIKLMNPQRSTVWYXBZUO"
        else:
            valid_aa = "ACDEFGHIKLMNPQRSTVWY"

        return re.fullmatch(f"[{valid_aa}]+", seq) is not None