import pickle
import pandas as pd
from plants_sm.pathway_prediction.MCTS_A import MCTS_A
from plants_sm.pathway_prediction.ec_numbers_annotator import ProtBertECAnnotator
from plants_sm.pathway_prediction.entities import Molecule
from plants_sm.pathway_prediction.esi_annotator import ProtBertESIAnnotator
from plants_sm.pathway_prediction.proteome_compass import ProteomeCompass
from plants_sm.pathway_prediction.proteome_compassed_reactor import ProteomeCompassedReactor
from plants_sm.pathway_prediction.reaction_ec_number_annotator import ReactionECNumberAnnotator
from plants_sm.pathway_prediction.reactions_ec_esi_linker import ReactionEnzymeSubstratePairsLinker
from plants_sm.pathway_prediction.retroformer_reactor import Retroformer
from plants_sm.pathway_prediction.solution import ECSolution


# ProtBertECAnnotator().annotate_from_file("ITAG4.1_proteins.fasta", "fasta", output_path="tomato_genome.csv", device="cuda:2")


ec_solution = ECSolution.from_csv_and_fasta("tomato_genome.csv", "ITAG4.1_proteins.fasta", "accession")

reactor = Retroformer(topk=50, beam_size=50, device="cuda")
compound = "[H]O[C@@H]1CC2=C(C([H])([H])[C@@H](CC2)C(=C([H])[H])C([H])([H])[H])[C@@]([H])([C@H]1O[H])C([H])([H])[H]"
linker = ReactionEnzymeSubstratePairsLinker(
                                     enzyme_annotator_solution=ec_solution, 
                                     reaction_annotator=ReactionECNumberAnnotator(),
                                     esi_annotator=ProtBertESIAnnotator()
        )
compass = ProteomeCompass(enzyme_annotator_solution=ec_solution, 
                reaction_annotator=ReactionECNumberAnnotator(),
                esi_annotator=ProtBertESIAnnotator(),
                device = "cuda:0")

searcher = MCTS_A(reactors=[ProteomeCompassedReactor(compass, Retroformer(device="cuda:0"))], device='cuda:0', 
                  simulations=100, cpuct=4.0, times=500)
solutions = searcher.search(molecule=Molecule.from_smiles(compound))

with open('solution_retrosynthesis.pkl','wb') as f:
    pickle.dump(solutions,f)

