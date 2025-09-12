import pandas as pd
from rdkit import Chem

def _validate_proteins(unique_proteins: pd.DataFrame) -> set:
    """ Validate proteins.
    Be aware that it assumes the ids are in the 1st column and the sequences in 2nd!

    Parameters
    ----------
    unique_proteins : pd.DataFrame
        Dataframe with proteins. It assumes the ids are in the 1st column and the sequences in 2nd!

    Returns
    -------
    set
        Set of the IDs of the valid proteins.
    """

    ALLOWED_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWYBZUOJX')
    protein_ids = unique_proteins.columns[0]
    protein_sequences = unique_proteins.columns[1]
    
    valid_protein_ids = set()
    for _, row in unique_proteins.iterrows():
        seq = str(row[protein_sequences]).upper()
        if all(aa in ALLOWED_AMINO_ACIDS for aa in seq):
            valid_protein_ids.add(row[protein_ids])

    return valid_protein_ids

def _validate_compounds(unique_compounds: pd.DataFrame) -> set:
    """ Validate compounds.
    Be aware that it assumes the ids are in the 1st column and the SMILES in 2nd!

    Parameters
    ----------
    unique_proteins : pd.DataFrame
        Dataframe with compounds. It assumes the ids are in the 1st column and the SMILES in 2nd!

    Returns
    -------
    set
        Set of the IDs of the valid compounds.
    """

    compound_ids = unique_compounds.columns[0]
    compound_smiles = unique_compounds.columns[1]

    valid_compound_ids = set()
    for _, row in unique_compounds.iterrows():
        mol = Chem.MolFromSmiles(str(row[compound_smiles]))
        if mol is not None:
            valid_compound_ids.add(row[compound_ids])

    return valid_compound_ids