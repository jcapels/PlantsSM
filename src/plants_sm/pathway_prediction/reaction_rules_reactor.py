import os
from typing import List, Union
from plants_sm.pathway_prediction._chem_utils import ChemUtils
from plants_sm.pathway_prediction.entities import Molecule, Reaction, ReactionSmarts
from plants_sm.pathway_prediction.reactor import Reactor
from plants_sm.pathway_prediction.solution import ReactionSolution

from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts
from rdkit.Chem import AllChem
from rdkit.Chem import Mol

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ReactionRulesReactor(Reactor):
    
    rules_path: str = os.path.join(BASE_DIR,
                                "pathway_prediction",
                                "retrorules_MINE.tsv")
    reactions: List[Reaction] = []

    def __init__(self):
        self._convert_tsv_to_reaction_rules()

    def _convert_tsv_to_reaction_rules(self):
        import pandas as pd

        rules = pd.read_csv(self.rules_path, sep="\t")

        for _, row in rules.iterrows():
            reaction_id = row["RuleID"]
            smarts = row["SMARTS"]
            ec_numbers = row["EC_Numbers"]
            if isinstance(ec_numbers, str):
                ec_numbers = ec_numbers.split(",")
            else:
                ec_numbers = None

            reaction = ReactionSmarts.from_smarts(smarts)
            self.reactions.append(Reaction(representation=smarts, id=reaction_id, ec_numbers=ec_numbers, reaction=reaction))

    def _predict_solutions(self, reactants, reaction):

        solutions = []

        try:
            ps = reaction.reaction.to_chemical_reaction().RunReactants(reactants)
        
            res = []
            for pset in ps:
                pset = [ChemUtils._sanitize_mol(pset_i) for pset_i in pset]
                if None not in pset:
                    tres = ChemicalReaction()
                    for p in pset:
                        tres.AddProductTemplate(ChemUtils._remove_hs(p))
                    for reactant in reactants:
                        tres.AddReactantTemplate(ChemUtils._remove_hs(reactant))

                    res.append(tres)
            
            solution_reactions = list(set([AllChem.ReactionToSmiles(entry, canonical=True) for entry in res]))

            for solution_reaction in solution_reactions:
                solution_reaction = ReactionFromSmarts(solution_reaction)
                ps = solution_reaction.RunReactants(reactants)
                reactant_molecules = []
                for reactant in reactants:
                    reactant_molecules.append(Molecule.from_mol(reactant))
                
                product_molecules = []
                for product_tuple in ps:
                    smiles = Molecule.from_mol(product_tuple[0]).smiles
                    products = smiles.split(".")
                    for product_ in products:
                        product_molecules.append(Molecule.from_smiles(product_))
                
                solutions.append(ReactionSolution(products=product_molecules, 
                                reactants=reactant_molecules, 
                                reaction=reaction,
                                ec_numbers=reaction.ec_numbers
                                ))
        except ValueError:
            pass

        return solutions

    def _react(self, reactants: List[Union[str, Mol]]) -> List[ReactionSolution]:

        solutions = []

        assert len(reactants) > 0

        if isinstance(reactants[0], str):
            for i in range(len(reactants)):
                reactants[i] = Molecule.from_smiles(reactants[i]).mol

        for reaction in self.reactions:

            solutions.extend(self._predict_solutions(reactants, reaction))

        unique_solutions_product_smiles = set()
        unique_solutions = []
        for solution in solutions:
            product_smiles = ".".join([product.smiles for product in solution.products])
            if product_smiles not in unique_solutions_product_smiles:
                unique_solutions_product_smiles.add(product_smiles)
                unique_solutions.append(solution)

        return unique_solutions
    