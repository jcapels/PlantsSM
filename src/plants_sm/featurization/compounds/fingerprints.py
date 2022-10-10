class AtomPairFingerprintCallbackHash():
    """
    Atom pair fingerprints

    Returns the atom-pair fingerprint for a molecule as an ExplicitBitVect
    """

    def __init__(self, nBits: int = 2048, minLength: int = 1, maxLength: int = 30,
                 includeChirality: bool = False, use2D: bool = True, confId: int = -1):
        """
        Parameters
        ----------
        nBits: (optional)
            ...
        """

        super().__init__()
        self.nBits = nBits
        self.minLength = minLength
        self.maxLength = maxLength
        self.includeChirality = includeChirality
        self.use2D = use2D
        self.confId = confId

    @staticmethod
    def hash_function(bit, value):
        bit = hash(value) + 0x9e3779b9 + (bit * (2 ** 6)) + (bit / (2 ** 2))
        return bit

    def _featurize(self, mol: Any) -> np.ndarray:
        """Calculate atom pair fingerprint for a single molecule.
        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
          RDKit Mol object
        Returns
        -------
        np.ndarray
          A numpy array of layered fingerprints.
        """

        try:
            matrix = rdmolops.GetDistanceMatrix(mol)
            fp = [0] * self.nBits
            for at1 in range(mol.GetNumAtoms()):
                for at2 in range(at1 + 1, mol.GetNumAtoms()):
                    atom1 = mol.GetAtomWithIdx(at1)
                    atom2 = mol.GetAtomWithIdx(at2)
                    at1_hash_code = GetAtomPairAtomCode(atom1, includeChirality=self.includeChirality)
                    at2_hash_code = GetAtomPairAtomCode(atom2, includeChirality=self.includeChirality)

                    if self.minLength <= int(matrix[at1][at2]) <= self.maxLength:
                        bit = self.hash_function(0, min(at1_hash_code, at2_hash_code))
                        bit = self.hash_function(bit, matrix[at1][at2])
                        bit = self.hash_function(bit, max(at1_hash_code, at2_hash_code))
                        index = int(bit % self.nBits)
                        fp[index] = 1

        except Exception as e:
            print('error in smile: ' + str(mol))
            fp = np.empty(self.nBits, dtype=float)
            fp[:] = np.NaN
        fp = np.asarray(fp, dtype=np.float)

        return fp