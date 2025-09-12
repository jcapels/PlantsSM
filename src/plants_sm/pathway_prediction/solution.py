
from pydantic import BaseModel, validator
from typing import Any, Dict, List, Tuple, Union

from plants_sm.pathway_prediction.entities import BiologicalEntity, Molecule, Protein, Reaction

def sort_solutions(property_name: str):
    def decorator(cls: 'Solution') -> 'Solution':
        original_init = cls.__init__
        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Sort the dictionary by key (or value, etc.)
            solutions = getattr(self, property_name)
            setattr(self, property_name, {k: sorted(v, key=lambda x: x[1], reverse=True) for k, v in solutions.items()})

        cls.__init__ = __init__
        return cls
    return decorator


class Solution(BaseModel):

    score : Any = None


class ReactionSolution(Solution):

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

    substrate_protein_solutions: Dict[Union[str, int], List[Tuple[Union[str, int], float]]] # substrates as keys and proteins as values sorted

    def get_score(self, compound_id: str, highest: bool = False) -> Union[List[Tuple[str, float]], Tuple[str, float]]:
        """
        Get the score for a specific substrate.

        Parameters
        ----------
        compound_id : str
            The ID of the substrate.
        highest : bool
            If True, return the highest score. Otherwise, return all scores.
        
        Returns
        -------
        List[Tuple[str, float]]
            The list of scores for specified substrate ID and respective protein

        """

        if compound_id in self.substrate_protein_solutions:
            scores = self.substrate_protein_solutions[compound_id]

            if highest:
                return scores[0]
            
            return scores
        else:
            raise ValueError("This substrate does not exist in our set of solutions")

class ECSolution(Solution):

    entity_ec_1 : Dict[str, List[Tuple[str, float]]]
    entity_ec_2 : Dict[str, List[Tuple[str, float]]]
    entity_ec_3 : Dict[str, List[Tuple[str, float]]]
    entity_ec_4 : Dict[str, List[Tuple[str, float]]]
    entities : Dict[str, BiologicalEntity] # dictionary to trace the sequence by the ID

    def get_score(self, entity_id: str, ec_number: str) -> List[Tuple[str, float]]:
        """
        Get the score for a specific EC number.

        Parameters
        ----------
        entity_id : str
            The ID of the biological entity.
        ec_number : str
            The EC number to get the score for.
        
        Returns
        -------
        List[Tuple[str, float]]
            The list of scores for the specified EC number and the EC number

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
        

    