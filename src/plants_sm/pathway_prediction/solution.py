from pydantic import BaseModel, validator
from typing import Any, Dict, List, Tuple, Union
from plants_sm.pathway_prediction.entities import BiologicalEntity, Molecule, Protein, Reaction

def sort_solutions(property_name: str):
    """
    Decorator to sort solution dictionaries by score in descending order.

    Parameters
    ----------
    property_name : str
        The name of the property to sort.

    Returns
    -------
    function
        A decorator function for the Solution class.
    """
    def decorator(cls: 'Solution') -> 'Solution':
        original_init = cls.__init__
        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Sort the dictionary by score in descending order
            solutions = getattr(self, property_name)
            setattr(self, property_name, {k: sorted(v, key=lambda x: x[1], reverse=True) for k, v in solutions.items()})
        cls.__init__ = __init__
        return cls
    return decorator

class Solution(BaseModel):
    """
    Base class for pathway prediction solutions.

    Attributes
    ----------
    score : Any, optional
        The score associated with the solution.
    """
    score: Any = None

class ReactionSolution(Solution):
    """
    A solution representing a chemical reaction.

    Attributes
    ----------
    reactants : List[Molecule]
        List of reactant molecules.
    products : List[Molecule]
        List of product molecules.
    reaction : Reaction
        The reaction object.
    ec_numbers : List[str], optional
        List of EC numbers associated with the reaction.
    """

    reactants: List[Molecule]
    products: List[Molecule]
    reaction: Reaction
    ec_numbers: List[str] = None

    def get_reactants_smiles(self) -> List[str]:
        """
        Get the SMILES strings of the reactants.

        Returns
        -------
        List[str]
            A list of SMILES strings for the reactants.
        """
        return [reactant.smiles for reactant in self.reactants]

    def get_products_smiles(self) -> List[str]:
        """
        Get the SMILES strings of the products.

        Returns
        -------
        List[str]
            A list of SMILES strings for the products.
        """
        return [product.smiles for product in self.products]

@sort_solutions("substrate_protein_solutions")
class ESISolution(Solution):
    """
    A solution representing enzyme-substrate interactions.

    Attributes
    ----------
    substrate_protein_solutions : Dict[Union[str, int], List[Tuple[Union[str, int], float]]]
        Dictionary mapping substrate IDs to lists of (protein ID, score) tuples, sorted by score.
    """

    substrate_protein_solutions: Dict[Union[str, int], List[Tuple[Union[str, int], float]]]

    def get_score(self, compound_id: str, highest: bool = False) -> Union[List[Tuple[str, float]], Tuple[str, float]]:
        """
        Get the score(s) for a specific substrate.

        Parameters
        ----------
        compound_id : str
            The ID of the substrate.
        highest : bool, default=False
            If True, return only the highest score. Otherwise, return all scores.

        Returns
        -------
        Union[List[Tuple[str, float]], Tuple[str, float]]
            The score(s) for the specified substrate ID and respective protein(s).

        Raises
        ------
        ValueError
            If the substrate does not exist in the solution set.
        """
        if compound_id in self.substrate_protein_solutions:
            scores = self.substrate_protein_solutions[compound_id]
            if highest:
                return scores[0]
            return scores
        else:
            raise ValueError("This substrate does not exist in our set of solutions")

class ECSolution(Solution):
    """
    A solution representing EC number annotations for biological entities.

    Attributes
    ----------
    entity_ec_1 : Dict[str, List[Tuple[str, float]]]
        Dictionary mapping entity IDs to lists of (EC1 number, score) tuples.
    entity_ec_2 : Dict[str, List[Tuple[str, float]]]
        Dictionary mapping entity IDs to lists of (EC2 number, score) tuples.
    entity_ec_3 : Dict[str, List[Tuple[str, float]]]
        Dictionary mapping entity IDs to lists of (EC3 number, score) tuples.
    entity_ec_4 : Dict[str, List[Tuple[str, float]]]
        Dictionary mapping entity IDs to lists of (EC4 number, score) tuples.
    entities : Dict[str, BiologicalEntity]
        Dictionary mapping entity IDs to BiologicalEntity objects.
    """

    entity_ec_1: Dict[str, List[Tuple[str, float]]]
    entity_ec_2: Dict[str, List[Tuple[str, float]]]
    entity_ec_3: Dict[str, List[Tuple[str, float]]]
    entity_ec_4: Dict[str, List[Tuple[str, float]]]
    entities: Dict[str, BiologicalEntity]

    def get_score(self, entity_id: str, ec_number: str) -> List[Tuple[str, float]]:
        """
        Get the score(s) for a specific EC number and entity.

        Parameters
        ----------
        entity_id : str
            The ID of the biological entity.
        ec_number : str
            The EC number level to get the score for (EC1, EC2, EC3, or EC4).

        Returns
        -------
        List[Tuple[str, float]]
            The list of (EC number, score) tuples for the specified entity and EC level.

        Raises
        ------
        ValueError
            If the EC number is not one of EC1, EC2, EC3, or EC4.
        """
        if ec_number == "EC1":
            return self.entity_ec_1.get(entity_id, [])
        elif ec_number == "EC2":
            return self.entity_ec_2.get(entity_id, [])
        elif ec_number == "EC3":
            return self.entity_ec_3.get(entity_id, [])
        elif ec_number == "EC4":
            return self.entity_ec_4.get(entity_id, [])
        else:
            raise ValueError(f"Unknown EC number: {ec_number}, available options are EC1, EC2, EC3, EC4.")
