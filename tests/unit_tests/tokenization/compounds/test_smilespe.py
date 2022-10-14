from unittest import TestCase

from plants_sm.tokenisation.compounds.smilespe import AtomLevelTokenizer, KmerTokenizer, SPETokenizer


class TestSmilesPETokenizers(TestCase):

    def setUp(self) -> None:
        self.compounds_to_tokenize = ["AAAAAACC[C@@]1(C[C@H]2C[C@@](C3=C(CCN(C2)C1)C4=CC=CC=C4N3)(C5=C(C=C6C(=C5)[C@]78CCN9"
                                      "[C@H]7[C@@](C=CC9)([C@H]([C@@]([C@@H]8N6C)(C(=O)OC)O)OC(=O)C)CC)OC)C(=O)OC)O",
                                      "C[C@H]1CC[C@]2([C@H]([C@H]3[C@@H](O2)C[C@@H]4[C@@]3(CC[C@H]5[C@H]4CC[C@@H]6"
                                      "[C@@]5(CC[C@@H](C6)O[C@H]7[C@@H]([C@H]([C@H]([C@H](O7)CO)O[C@H]8[C@@H]([C@H]"
                                      "([C@@H]([C@H](O8)CO)O)O[C@H]9[C@@H]([C@H]([C@@H](CO9)O)O)O)O[C@H]2[C@@H]"
                                      "([C@H]([C@@H]([C@H](O2)CO)O)O)O)O)O)C)C)C)NC1"]

    def test_atom_level_tokenizer(self):
        tokenizer = AtomLevelTokenizer()
        tokens = tokenizer.tokenize(self.compounds_to_tokenize[0])
        self.assertEqual(118, len(tokens))

        tokens = tokenizer.tokenize(self.compounds_to_tokenize[1])
        self.assertEqual(134, len(tokens))

    def test_kmer_level_tokenizer(self):
        tokenizer = KmerTokenizer()
        tokens = tokenizer.tokenize(self.compounds_to_tokenize[0])
        self.assertEqual(115, len(tokens))

        tokens = tokenizer.tokenize(self.compounds_to_tokenize[1])
        self.assertEqual(131, len(tokens))

    def test_spe_tokenizer(self):
        tokenizer = SPETokenizer()
        tokens = tokenizer.tokenize(self.compounds_to_tokenize[0])
        self.assertEqual(51, len(tokens))

        tokens = tokenizer.tokenize(self.compounds_to_tokenize[1])
        self.assertEqual(67, len(tokens))
