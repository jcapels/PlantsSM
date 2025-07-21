from unittest import TestCase

from plants_sm.pathway_prediction.entities import Molecule, Protein, Reaction


class TestCoreClasses(TestCase):

    def test_protein(self):
        protein = Protein(representation="MPISSSSSSSTKSMRRAASELERSDSVTSPRFIGRRQSLIEDARKEREAAAAAAEAAEAT")
        self.assertEqual(protein.representation, "MPISSSSSSSTKSMRRAASELERSDSVTSPRFIGRRQSLIEDARKEREAAAAAAEAAEAT")
        self.assertEqual(protein.sequence, "MPISSSSSSSTKSMRRAASELERSDSVTSPRFIGRRQSLIEDARKEREAAAAAAEAAEAT")

        protein = Protein.from_sequence("MPISSSSSSSTKSMRRAASELERSDSVTSPRFIGRRQSLIEDARKEREAAAAAAEAAEAT")
        self.assertEqual(protein.representation, "MPISSSSSSSTKSMRRAASELERSDSVTSPRFIGRRQSLIEDARKEREAAAAAAEAAEAT")
        self.assertEqual(protein.sequence, "MPISSSSSSSTKSMRRAASELERSDSVTSPRFIGRRQSLIEDARKEREAAAAAAEAAEAT")

    def test_molecule(self):
        molecule = Molecule.from_smiles(smiles="C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N")
        self.assertEqual(str(molecule.representation), "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N")
        self.assertEqual(molecule.smiles, "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N")

    def test_reaction_smarts(self):
        smarts = '([#6&v4&H3:1]-[#8&v2&H0:2]-[#6&v4&H0:3])>>([#6&v4&H0:3]-[#8&v2&H1:2].[#8&v2&H1]-[#6&v4&H0](=[#8&v2&H0])-[#6&v4&H1](-[#7&v3&H2])-[#6&v4&H2]-[#6&v4&H2]-[#16&+&v3&H0](-[#6&v4&H3:1])-[#6&v4&H2]-[#6&v4&H1]1-[#6&v4&H1](-[#8&v2&H1])-[#6&v4&H1](-[#8&v2&H1])-[#6&v4&H1](-[#7&v3&H0]2:[#6&v4&H1]:[#7&v3&H0]:[#6&v4&H0]3:[#6&v4&H0](-[#7&v3&H2]):[#7&v3&H0]:[#6&v4&H1]:[#7&v3&H0]:[#6&v4&H0]:2:3)-[#8&v2&H0]-1)'
        reaction = Reaction.from_smarts(smarts)
        self.assertEqual(reaction.smarts, smarts)
        self.assertEqual(reaction.representation, smarts)

    def test_reaction_smiles(self):

        smiles = '[CH3:11][CH:12]1[N:13]([CH2:6][C:5]2=[CH:8][N:9]=[C:2]([CH3:1])[N:3]=[C:4]2[NH2:10])[CH:14]=[CH:15][CH:16]=[C:17]1[CH2:18][CH2:19][OH:20].[OH2:7]>>[CH3:1][C:2]1=[N:9][CH:8]=[C:5]([CH2:6][OH:7])[C:4]([NH2:10])=[N:3]1.[CH3:11][C:12]1=[C:17]([CH2:18][CH2:19][OH:20])[CH:16]=[CH:15][CH:14]=[N:13]1'
        reaction = Reaction.from_smiles(smiles)
        self.assertEqual(reaction.smiles, smiles)
        self.assertEqual(reaction.representation, smiles)
        self.assertEqual(reaction.reactants[0].smiles, "[CH3:11][CH:12]1[N:13]([CH2:6][C:5]2=[CH:8][N:9]=[C:2]([CH3:1])[N:3]=[C:4]2[NH2:10])[CH:14]=[CH:15][CH:16]=[C:17]1[CH2:18][CH2:19][OH:20]")
        self.assertEqual(reaction.reactants[1].smiles, "[OH2:7]")
        self.assertEqual(reaction.products[0].smiles, "[CH3:1][C:2]1=[N:9][CH:8]=[C:5]([CH2:6][OH:7])[C:4]([NH2:10])=[N:3]1")
        self.assertEqual(reaction.products[1].smiles, "[CH3:11][C:12]1=[C:17]([CH2:18][CH2:19][OH:20])[CH:16]=[CH:15][CH:14]=[N:13]1")

