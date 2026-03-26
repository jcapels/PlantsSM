import os
from unittest import TestCase, skip

from plants_sm.pathway_prediction.esi_annotator import ProtBertESIAnnotator
from plants_sm.pathway_prediction.gsm_pathway_integration import GSMPathwayAnnotation
from plants_sm.pathway_prediction.solution import ECSolution
from tests import TEST_DIR

@skip
class TestPathwayPrediction(TestCase):

    def setUp(self):
        self.tomato_genome_annotation = os.path.join(TEST_DIR, "data", "tomato_ec_annotation.csv")
        self.tomato_genome = os.path.join(TEST_DIR, "data", "tomato_proteins.fasta")
        self.ecoli_model = os.path.join(TEST_DIR, "data", "iAF1260b.xml")

    def test_integrate_in_pathway_integration(self):
        # ec_solution = ECSolution.from_csv_and_fasta(self.tomato_genome_annotation, self.tomato_genome, "accession")
        reactions = ['R05855', 'R05876', 'R07513', 'R05825', 'R03230', 'R05880', 'R05857']

        genes = {'R05857': ['Solyc09g089710.3.1', 'Solyc09g010010.1.1', 'Solyc09g089700.4.1', 'Solyc03g095903.1.1', 'Solyc09g089740.3.1', 'Solyc09g089830.4.1', 'Solyc04g009860.4.1', 'Solyc04g007980.3.1', 'Solyc03g095907.1.1', 'Solyc09g089760.1.1', 'Solyc12g006390.2.1', 'Solyc12g006380.2.1', 'Solyc09g089800.2.1', 'Solyc09g010000.5.1', 'Solyc04g009850.4.1', 'Solyc09g089790.3.1', 'Solyc09g089690.3.1', 'Solyc09g089780.3.1', 'Solyc09g010040.3.1', 'Solyc09g089580.4.1', 'Solyc09g089820.2.1', 'Solyc09g010020.2.1', 'Solyc09g057930.3.1', 'Solyc03g095900.5.1', 'Solyc09g089810.1.1', 'Solyc01g044550.3.1', 'Solyc09g089680.4.1', 'Solyc09g089770.3.1', 'Solyc09g089720.4.1', 'Solyc03g116280.3.1'], 'R05855': ['Solyc08g150142.1.1'], 'R03230': ['Solyc11g020630.1.1'], 'R05876': ['Solyc04g082350.3.1', 'Solyc10g079570.3.1', 'Solyc05g014330.1.1', 'Solyc07g006670.1.1', 'Solyc01g008300.2.1', 'Solyc04g012020.1.1', 'Solyc07g052060.2.1'], 'R05825': ['Solyc03g044800.2.1', 'Solyc02g065260.4.1', 'Solyc03g044820.4.1', 'Solyc01g108820.1.1', 'Solyc01g108750.3.1', 'Solyc02g065250.2.1', 'Solyc01g108810.3.1', 'Solyc03g044790.3.1', 'Solyc01g108680.5.1', 'Solyc09g014970.5.1', 'Solyc03g044740.5.1', 'Solyc01g108780.4.1', 'Solyc03g070380.3.1', 'Solyc01g108740.3.1', 'Solyc02g089060.3.1', 'Solyc02g065280.3.1', 'Solyc02g065240.3.1'], 'R07513': ['Solyc11g021040.3.1', 'Solyc01g109020.3.1'], 'R05880': ['Solyc11g021040.3.1', 'Solyc01g109020.3.1']}
        
        annotator = GSMPathwayAnnotation(self.ecoli_model)

        for reaction_id in reactions:
            kegg_reaction_data = annotator._fetch_kegg_reaction(reaction_id)
            if kegg_reaction_data:
                annotator._add_reaction_to_model(reaction_id, kegg_reaction_data, genes[reaction_id])

        print(annotator.gsm_model.reactions.get_by_id('R05855'))
        print(annotator.gsm_model.reactions.get_by_id('R05855').genes)

    def test_gem_pathway_integration(self):
        ec_solution = ECSolution.from_csv_and_fasta(self.tomato_genome_annotation, self.tomato_genome, "accession")
        
        GSMPathwayAnnotation(self.ecoli_model, esi_annotator=ProtBertESIAnnotator(device="cuda:1")).integrate_pathway_annotations("C07201", 
                                                                             ec_solution, 
                                                                             solutions_path="solutions_kegg_C07201")

    def test_gem_plantcyc_integration(self):
        ec_solution = ECSolution.from_csv_and_fasta(self.tomato_genome_annotation, self.tomato_genome, "accession")

        annotator = GSMPathwayAnnotation(self.ecoli_model, model_type="PlantCyc", esi_annotator=ProtBertESIAnnotator(device="cuda:1"))
        
        annotator.integrate_pathway_annotations("CPD-18641", ec_solution)
        print(annotator.gsm_model.reactions.get_by_id('RXN18C3-10').genes)

    def test_get_entities_by_incomplete_ec_number(self):
        ec_solution = ECSolution.from_csv_and_fasta(self.tomato_genome_annotation, self.tomato_genome, "accession")
        ec_solution.create_ec_to_entities()
        entities = ec_solution.get_entities("2.4.1.-")
        self.assertIn("Solyc07g043500.1.1", [entity[0] for entity in entities])
