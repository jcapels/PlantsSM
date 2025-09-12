import os
from typing import List
from plants_sm.pathway_prediction.entities import Molecule, Reaction
from plants_sm.pathway_prediction.reactor import Reactor
from plants_sm.pathway_prediction.retroformer.translate import prepare_retroformer, run_retroformer
from plants_sm.pathway_prediction.solution import ReactionSolution
from rdkit import Chem
from rdkit.Chem import Mol

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Retroformer(Reactor):
    """
    A reactor that uses the Retroformer model to predict retro-synthetic pathways.

    Attributes
    ----------
    model_retroformer : object
        The loaded Retroformer model for reaction prediction.
    args : object
        Configuration arguments for the Retroformer model.
    """

    def __init__(self):
        """
        Initialize the Retroformer reactor.

        Loads the Retroformer model and vocabulary from saved files.
        """
        exp_topk = 10
        beam_size = 10
        device = 'cuda'
        retroformer_path = os.path.join(
            BASE_DIR, "pathway_prediction", "retroformer", "saved_models", "model.pt"
        )
        vocab_path = os.path.join(
            BASE_DIR, "pathway_prediction", "retroformer", "saved_models", "vocab_share.pk"
        )
        self.model_retroformer, self.args = prepare_retroformer(
            device, beam_size, exp_topk, path=retroformer_path, vocab_path=vocab_path
        )

    def _react(self, reactants: List[Mol]) -> List[ReactionSolution]:
        """
        Predict retro-synthetic pathways for a list of reactants using Retroformer.

        Parameters
        ----------
        reactants : List[Mol]
            List of RDKit Mol objects representing the reactants.

        Returns
        -------
        List[ReactionSolution]
            List of predicted reaction solutions, each containing reactants, products, reaction, and score.
        """
        expansion_handler = lambda x: run_retroformer(self.model_retroformer, self.args, x)
        new_reactants = []
        for reactant in reactants:
            target_mol = Chem.MolToSmiles(reactant)
            new_reactants.append(target_mol)

        result = expansion_handler(new_reactants)
        solution_products_list, reactions_scores_list = result
        solutions = []

        for solutions_products, reactions_scores in zip(solution_products_list, reactions_scores_list):
            for solution_products, reaction_score in zip(solutions_products, reactions_scores):
                try:
                    reactant_molecules = []
                    for reactant in new_reactants:
                        reactant_molecules.append(Molecule.from_smiles(reactant))

                    product_molecules = []
                    products = solution_products.split(".")
                    for product_ in products:
                        product_molecules.append(Molecule.from_smiles(product_))

                    reaction = solution_products + ">>" + ".".join(new_reactants)
                    reaction = Reaction.from_smiles(reaction)
                    solutions.append(
                        ReactionSolution(
                            products=reactant_molecules,
                            reactants=product_molecules,
                            reaction=reaction,
                            ec_numbers=None,
                            score=1 - reaction_score
                        )
                    )
                except ValueError as e:
                    if str(e) != "SMILES is not valid":
                        raise ValueError
                    else:
                        print(solution_products)
        return solutions
