from abc import ABC
from typing import List, Union
import pandas as pd
from plants_sm.pathway_prediction.annotator import Annotator
from plants_sm.pathway_prediction.reactions_ec_esi_linker import ReactionEnzymeSubstratePairsLinker
from plants_sm.pathway_prediction.solution import ReactionSolution, Solution

class ProteomeCompass(ABC):
    """A class for linking reactions to enzymes and substrates, and scoring the results.

    This class integrates enzyme, reaction, and enzyme-substrate-interaction (ESI) annotators
    to provide a comprehensive scoring mechanism for reaction solutions based on enzyme annotations.

    Attributes
    ----------
    enzyme_annotator_solution : Union[Solution, Annotator]
        Annotator or solution for enzymes.
    reaction_annotator : Annotator
        Annotator for reactions.
    esi_annotator : Annotator
        Annotator for enzyme-substrate interactions.
    device : str
        Device to run computations on (e.g., "cpu" or "cuda").
    annotations_linker : ReactionEnzymeSubstratePairsLinker
        Links reactions to enzymes and substrates.
    """

    def __init__(
        self,
        enzyme_annotator_solution: Union[Solution, Annotator],
        reaction_annotator: Annotator,
        esi_annotator: Annotator,
        device: str = "cpu"
    ):
        """Initialize the ProteomeCompass with the provided annotators and device.

        Parameters
        ----------
        enzyme_annotator_solution : Union[Solution, Annotator]
            Annotator or solution for enzymes.
        reaction_annotator : Annotator
            Annotator for reactions.
        esi_annotator : Annotator
            Annotator for enzyme-substrate interactions.
        device : str, optional
            Device to run computations on. Defaults to "cpu".
        """
        self.enzyme_annotator_solution = enzyme_annotator_solution
        if isinstance(enzyme_annotator_solution, Annotator):
            self.enzyme_annotator_solution.device = device

        self.reaction_annotator = reaction_annotator
        self.reaction_annotator.device = device
        self.esi_annotator = esi_annotator
        self.esi_annotator.device = device
        self.device = device
        self.annotations_linker = ReactionEnzymeSubstratePairsLinker(
            enzyme_annotator_solution=self.enzyme_annotator_solution,
            reaction_annotator=self.reaction_annotator,
            esi_annotator=self.esi_annotator
        )

    def direct(self, solutions: List[ReactionSolution]) -> List[ReactionSolution]:
        """Score and annotate a list of reaction solutions by linking them to enzymes and substrates.

        For each reaction solution, this method:
        1. Links reactions to enzymes and substrates.
        2. Computes a combined score based on enzyme, reaction, and ESI annotations.
        3. Updates the score and enzyme_scores attributes of each reaction solution.

        Parameters
        ----------
        solutions : List[ReactionSolution]
            List of reaction solutions to be scored and annotated.

        Returns
        -------
        List[ReactionSolution]
            The updated list of reaction solutions with new scores and enzyme scores.
        """
        reaction_id = 0
        reaction_ids = []
        reaction_rxn = []
        for solution in solutions:
            reaction = solution.reaction.representation
            reaction_ids.append(reaction_id)
            reaction_rxn.append(reaction)
            reaction_id += 1

        entities_reactions = pd.DataFrame({
            "ids": reaction_ids,
            "rxn_smiles": reaction_rxn
        })

        reaction_protein_solution, enzyme_solution, reaction_solution = self.annotations_linker.link_reactions_to_enzymes(
            reactions=entities_reactions, export_suffix_path="1"
        )

        for i, reaction in enumerate(solutions):
            id_ = str(i)
            if id_ in reaction_protein_solution:
                score = reaction_protein_solution[id_][0][1]
                enzyme_id = reaction_protein_solution[id_][0][0]
                reaction_score = reaction_solution.get_score(id_, "EC3")[0][1]
                reaction_ec = reaction_solution.get_ecs(id_, "EC3")[0]
                enzyme_ecs = enzyme_solution.get_ecs(enzyme_id, "EC3")
                enzyme_scores = enzyme_solution.get_score(enzyme_id, "EC3")
                enzyme_ec_index = enzyme_ecs.index(reaction_ec)
                # combined score here is in reality a cost function
                combined_score = reaction.score + (1 - score) + (1 - enzyme_scores[enzyme_ec_index][1]) + (1 - reaction_score)
                reaction.score = combined_score / 4
                for j, enzyme in enumerate(reaction_protein_solution[id_]):
                    enzyme_id = reaction_protein_solution[id_][j][0]
                    enzyme_ecs = enzyme_solution.get_ecs(enzyme_id, "EC3")
                    enzyme_scores = enzyme_solution.get_score(enzyme_id, "EC3")
                    enzyme_ec_index = enzyme_ecs.index(reaction_ec)
                    score = reaction_protein_solution[id_][j][1]
                    reaction.enzymes_scores[enzyme[0]] = (score + enzyme_scores[enzyme_ec_index][1] + reaction_score) / 3
            else:
                reaction.score /= 4

        return solutions
