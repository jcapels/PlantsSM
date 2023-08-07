import os
import unittest
from unittest import skip

from plants_sm.alignments.alignment import BLAST
from tests import TEST_DIR


@skip("BLAST and diamond not being installed in CI")
class TestAlignments(unittest.TestCase):

    def test_blast_create_database(self):
        BLAST_DB = "data/blast_db"
        blast = BLAST(BLAST_DB)
        blast.create_database(os.path.join(TEST_DIR, "data", "test.fasta"), BLAST_DB)

    def test_blast_run_alignment(self):
        BLAST_DB = "data/blast_db"
        blast = BLAST(BLAST_DB)
        blast.create_database(os.path.join(TEST_DIR, "data", "test.fasta"), BLAST_DB)
        blast.run(os.path.join(TEST_DIR, "data", "query.fasta"),
                  os.path.join(TEST_DIR, "data", "out.tsv"), evalue=1e-5, num_hits=2)

        self.assertTrue(blast.results.shape[0] == 6)

    def test_diamond_run_alignment(self):
        DIAMOND_DB = "data/DIAMOND_DB"
        blast = BLAST(DIAMOND_DB)
        blast.create_database(os.path.join(TEST_DIR, "data", "test.fasta"), DIAMOND_DB)
        blast.run(os.path.join(TEST_DIR, "data", "query.fasta"),
                  os.path.join(TEST_DIR, "data", "out.tsv"), evalue=1e-5, num_hits=2)

        self.assertTrue(blast.results.shape[0] == 6)
