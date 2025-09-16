

from typing import List, Union

import pandas as pd
from plants_sm.pathway_prediction.annotator import Annotator, AnnotatorLinker
from plants_sm.pathway_prediction.ec_numbers_annotator import ECAnnotator, ProtBertECAnnotator
from plants_sm.pathway_prediction.entities import BiologicalEntity
from plants_sm.pathway_prediction.reaction_ec_number_annotator import ReactionECNumberAnnotator
from plants_sm.pathway_prediction.solution import ECSolution, Solution


class ReactionEnzymeLinker(AnnotatorLinker):

    def _link(self):
        ec_solutions = [solution for solution in self.solutions if isinstance(solution, ECSolution)]

        assert len(ec_solutions) > 1
        # check the one with less solutions as the len(solution.entity_ec3)
        self.solution_1 = ec_solutions[0]
        self.solution_2 = ec_solutions[1]

        linked_solutions = {}
        for i, entry_1 in enumerate(self.solution_1.entity_ec_3.keys()):
            scores = self.solution_1.get_score(entry_1, "EC3")
            ec_numbers_1 = set([ec for ec, score in scores])
            linked_solutions[entry_1] = []

            for j, entry_2 in enumerate(self.solution_2.entity_ec_3.keys()):
                scores = self.solution_2.get_score(entry_2, "EC3")
                ec_numbers_2 = set([ec for ec, score in scores])
                if len(ec_numbers_1.intersection(ec_numbers_2)) > 0:
                    linked_solutions[entry_1].append(entry_2)

        return linked_solutions
    
class ReactionEnzymeSubstratePairsLinker(object):


    def __init__(self, reaction_annotator: Annotator, enzyme_annotator_solution: Union[ECAnnotator, ECSolution], 
                 esi_annotator: Annotator):

        self.reaction_annotator = reaction_annotator
        self.enzyme_annotator_solution = enzyme_annotator_solution
        self.esi_annotator = esi_annotator

    def link_reactions_to_enzymes(self, reactions, proteins = None, **kwargs):

        if isinstance(self.enzyme_annotator_solution, Solution):
            solutions = [self.enzyme_annotator_solution]
            annotators = [self.reaction_annotator]
        else:
            if proteins is None:
                raise ValueError("If you do not provide a solution for proteins, then provide the proteins so they can be annotated")
            solutions = None
            annotators = [self.enzyme_annotator_solution, self.reaction_annotator]

        linker = ReactionEnzymeLinker(annotators=annotators, solutions=solutions)

        if proteins is None:
            entities = [reactions]
        else:
            entities = [proteins, reactions]

        annotations = linker.link_annotations(entities, **kwargs)
        protein_ids = []
        protein_sequences = []
        compound_ids = []
        compound_smiles = []
        report_reactions = []

        compound_id_total = 0
        compounds_in_dataset = {}

        for annotation_entry in annotations:
            reactions = annotations[annotation_entry]
            if len(reactions) > 0:
                protein_id = annotation_entry
                protein_sequence = linker.solutions[0].entities[protein_id].sequence
                for reaction in reactions:
                    substrates_retrobiosynthesis = linker.solutions[1].entities[reaction].get_reactants_smiles()
                    for substrate_retrobiosynthesis in substrates_retrobiosynthesis:
                        if substrate_retrobiosynthesis not in compounds_in_dataset:
                            compound_id_total += 1
                            compounds_in_dataset[substrate_retrobiosynthesis] = compound_id_total

                        compound_id = compounds_in_dataset[substrate_retrobiosynthesis]
                        compound_ids.append(compound_id)
                        compound_smiles.append(substrate_retrobiosynthesis)
                        report_reactions.append(reaction)
                        protein_ids.append(protein_id)
                        protein_sequences.append(protein_sequence)
        
        entities = pd.DataFrame({
            'protein_ids': protein_ids,
            'protein sequence': protein_sequences,
            'compound_ids': compound_ids,
            'compound smiles': compound_smiles,
            'report_reactions': report_reactions
            })
        
        results = self.esi_annotator.annotate(entities)
        solutions = results.dataframe_with_solutions
        # Sort by 'proba' in descending order
        solutions = solutions.sort_values(by="proba", ascending=False)

        # Group by 'report_reactions' and aggregate both 'protein_ids' and 'proba' into lists of tuples
        reaction_protein_map = (
            solutions
            .groupby('report_reactions')
            .apply(lambda x: list(zip(x['protein_ids'], x['proba'])))
            .to_dict()
        )

        return reaction_protein_map, linker.solution_1, linker.solution_2


                            
                        

                        

                



            

                    



            


