from plants_sm.pathway_prediction.ec_numbers_annotator import ProtBertECAnnotator
from plants_sm.pathway_prediction.retroformer_reactor import Retroformer
from plants_sm.pathway_prediction.solution import ECSolution


# ProtBertECAnnotator().annotate_from_file("ITAG4.1_proteins.fasta", "fasta", output_path="tomato_genome.csv", device="cuda:2")


# solution = ECSolution.from_csv_and_fasta("tomato_genome.csv", "ITAG4.1_proteins.fasta", "accession")

reactor = Retroformer()
compound = "C[C@H]1[C@H]2[C@H](C[C@@H]3[C@@]2(CC[C@H]4[C@H]3CC[C@@H]5[C@@]4(CC[C@@H](C5)O[C@H]6[C@@H]([C@H]([C@H]([C@H](O6)CO)O[C@H]7[C@@H]([C@H]([C@@H]([C@H](O7)CO)O)O[C@H]8[C@@H]([C@H]([C@@H](CO8)O)O)O)O[C@H]9[C@@H]([C@H]([C@@H]([C@H](O9)CO)O)O)O)O)O)C)C)O[C@]11[C@H](C[C@@H](CN1)CO[C@H]1[C@@H]([C@H]([C@@H]([C@H](O1)CO)O)O)O)OC(=O)C"
print(reactor.react([compound]))