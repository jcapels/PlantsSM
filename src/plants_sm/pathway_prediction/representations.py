from abc import abstractmethod
from typing import Any
from pydantic import BaseModel, validator

from plants_sm.pathway_prediction._chem_utils import ChemUtils
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction

class Representation(BaseModel):
    """Abstract base class for chemical representations.

    This class provides a common interface for different types of chemical representations,
    requiring a string representation property.

    Attributes
    ----------
    representation : Any
        The underlying representation of the chemical entity.
    """

    representation: Any

    @property
    @abstractmethod
    def _str_representation(self):
        """Abstract property for the string representation of the chemical entity.

        Returns
        -------
        str
            The string representation of the chemical entity.
        """
        pass

    def __str__(self):
        """Return the string representation of the chemical entity.

        Returns
        -------
        str
            The string representation of the chemical entity.
        """
        return self._str_representation

class Smiles(Representation):
    """A class representing a chemical structure using SMILES notation.

    This class encapsulates a RDKit Mol object and its canonical SMILES string.

    Attributes
    ----------
    representation : str
        The canonical SMILES string.
    mol : Mol
        The RDKit Mol object.
    """

    mol: Mol

    class Config:
        arbitrary_types_allowed = True

    @validator('mol', pre=True, always=True, allow_reuse=True)
    def validate_mol(cls, value):
        """Validate that the provided value is a RDKit Mol instance.

        Parameters
        ----------
        value : Any
            The value to validate.

        Returns
        -------
        Mol
            The validated RDKit Mol instance.

        Raises
        ------
        ValueError
            If the value is not a RDKit Mol instance.
        """
        if not isinstance(value, Mol):
            raise ValueError("mol must be a RDKit Mol instance")
        return value

    @property
    def _str_representation(self):
        """Return the SMILES string representation.

        Returns
        -------
        str
            The SMILES string.
        """
        return self.representation

    @classmethod
    def from_smiles(cls, smiles: str) -> 'Smiles':
        """Create a Smiles instance from a SMILES string.

        Parameters
        ----------
        smiles : str
            The SMILES string to convert.

        Returns
        -------
        Smiles
            A new Smiles instance.

        Raises
        ------
        ValueError
            If the SMILES string is not valid.
        """
        mol = ChemUtils.smiles_to_mol(smiles)
        smiles = ChemUtils.mol_to_cannonical_smiles(mol)
        if mol is not None:
            return cls(representation=smiles, mol=mol)
        else:
            raise ValueError("SMILES is not valid")

    def to_mol(self) -> Mol:
        """Return the RDKit Mol object.

        Returns
        -------
        Mol
            The RDKit Mol object.
        """
        return self.mol

class ReactionSmarts(Representation):
    """A class representing a chemical reaction using SMARTS notation.

    This class encapsulates a RDKit ChemicalReaction object and its SMARTS string.

    Attributes
    ----------
    representation : str
        The SMARTS string.
    reaction : ChemicalReaction
        The RDKit ChemicalReaction object.
    """

    reaction: ChemicalReaction

    class Config:
        arbitrary_types_allowed = True

    @validator('reaction', pre=True, always=True, allow_reuse=True)
    def validate_reaction(cls, value):
        """Validate that the provided value is a ChemicalReaction instance.

        Parameters
        ----------
        value : Any
            The value to validate.

        Returns
        -------
        ChemicalReaction
            The validated ChemicalReaction instance.

        Raises
        ------
        ValueError
            If the value is not a ChemicalReaction instance.
        """
        if not isinstance(value, ChemicalReaction):
            raise ValueError("reaction must be a ChemicalReaction instance")
        return value

    @property
    def _str_representation(self):
        """Return the SMARTS string representation.

        Returns
        -------
        str
            The SMARTS string.
        """
        return self.representation

    @classmethod
    def from_smarts(cls, smarts: str) -> 'ReactionSmarts':
        """Create a ReactionSmarts instance from a SMARTS string.

        Parameters
        ----------
        smarts : str
            The SMARTS string to convert.

        Returns
        -------
        ReactionSmarts
            A new ReactionSmarts instance.

        Raises
        ------
        ValueError
            If the SMARTS string is not valid.
        """
        reaction = ChemUtils._smarts_to_reaction(smarts)
        if reaction is None:
            raise ValueError("SMARTS not valid")
        return cls(reaction=reaction, representation=smarts)

    def to_chemical_reaction(self):
        """Return the RDKit ChemicalReaction object.

        Returns
        -------
        ChemicalReaction
            The RDKit ChemicalReaction object.
        """
        return self.reaction
