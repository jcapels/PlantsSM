import os
from unittest import TestCase

import pandas as pd

from plants_sm.pathway_prediction.ec_numbers_annotator import ProtBertECAnnotator
from plants_sm.pathway_prediction.ec_numbers_annotator_utils._utils import fasta_to_dataframe
from plants_sm.pathway_prediction.esi_annotator import ProtBertESIAnnotator
from plants_sm.pathway_prediction.reaction_ec_number_annotator import ReactionECNumberAnnotator
from plants_sm.pathway_prediction.reactions_ec_esi_linker import ReactionEnzymeLinker, ReactionEnzymeSubstratePairsLinker
from tests import TEST_DIR

class TestReactionEnzymeLinker(TestCase):
    
    def setUp(self):

        self.test_csv = os.path.join(TEST_DIR, "data", "reaction_ec_data.csv")
        self.plant_data_faa = os.path.join(TEST_DIR, "data", "test_linker.fasta")

    def test_link_annotations(self):

        linker = ReactionEnzymeLinker(annotators=[ReactionECNumberAnnotator(), ProtBertECAnnotator()])

        entities_reactions = pd.read_csv(self.test_csv)
        entities_enzymes = fasta_to_dataframe(self.plant_data_faa)
 
        annotations = linker.link_annotations([entities_reactions, entities_enzymes])

        self.assertEqual(len(annotations), entities_reactions.shape[0]-1) # one of the reactions is taken out because it is invalid

    def test_link_annotations_reverse(self):
        linker = ReactionEnzymeLinker(annotators=[ProtBertECAnnotator(), ReactionECNumberAnnotator()])

        entities_reactions = pd.read_csv(self.test_csv)
        entities_enzymes = fasta_to_dataframe(self.plant_data_faa)

        annotations = linker.link_annotations([entities_enzymes, entities_reactions])

        self.assertEqual(len(annotations), entities_enzymes.shape[0]) # one of the reactions is taken out because it is invalid

    def test_link_reactions_to_enzymes_with_esi(self):
        linker = ReactionEnzymeSubstratePairsLinker(enzyme_annotator_solution=ProtBertECAnnotator(), 
                                     reaction_annotator=ReactionECNumberAnnotator(),
                                     esi_annotator=ProtBertESIAnnotator()
        )
        entities_reactions = pd.read_csv(self.test_csv)
        entities_enzymes = fasta_to_dataframe(self.plant_data_faa)
        print(linker.link_reactions_to_enzymes(reactions=entities_reactions, proteins=entities_enzymes))
    





