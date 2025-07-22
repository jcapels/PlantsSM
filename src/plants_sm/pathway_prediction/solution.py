
from pydantic import BaseModel, validator
from typing import Dict, List, Tuple

from plants_sm.pathway_prediction.entities import Molecule, Protein, Reaction

class Solution(BaseModel):

    score : float = None


class ReactionSolution(Solution):

    reactants: List[Molecule]
    products: List[Molecule]
    reaction: Reaction
    ec_numbers: List[str] = None


class ECSolution(Solution):

    enzymes_ec_1 : Dict[str, List[Tuple[str, float]]]
    enzymes_ec_2 : Dict[str, List[Tuple[str, float]]]
    enzymes_ec_3 : Dict[str, List[Tuple[str, float]]]
    enzymes_ec_4 : Dict[str, List[Tuple[str, float]]]
    enzymes : Dict[str, Protein] # dictionary to trace the sequence by the ID

    def get_score(self, protein_id: str, ec_number: str) -> List[Tuple[str, float]]:
        """
        Get the score for a specific EC number.

        Parameters
        ----------
        protein_id : str
            The ID of the protein.
        ec_number : str
            The EC number to get the score for.
        
        Returns
        -------
        float
            The score for the specified EC number.

        """
        if ec_number == "EC1":
            return self.enzymes_ec_1.get(protein_id, [])
        elif ec_number == "EC2":
            return self.enzymes_ec_2.get(protein_id, [])
        elif ec_number == "EC3":
            return self.enzymes_ec_3.get(protein_id, [])
        elif ec_number == "EC4":
            return self.enzymes_ec_4.get(protein_id, [])
        else:
            raise ValueError(f"Unknown EC number: {ec_number}, available options are EC1, EC2, EC3, EC4.")
        

    