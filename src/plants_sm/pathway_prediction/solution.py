
from pydantic import BaseModel, validator
from typing import Any, Dict, List, Tuple

from plants_sm.pathway_prediction.entities import BiologicalEntity, Molecule, Protein, Reaction

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
        float
            The score for the specified EC number.

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
        

    