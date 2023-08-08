import os
import unittest
from unittest import skip

from plants_sm.alignments.alignment import BLAST, Diamond
from tests import TEST_DIR


#@skip("BLAST and diamond not being installed in CI")
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
        blast = Diamond(DIAMOND_DB)
        blast.create_database(os.path.join(TEST_DIR, "data", "test.fasta"), DIAMOND_DB)
        blast.run(os.path.join(TEST_DIR, "data", "query.fasta"),
                  os.path.join(TEST_DIR, "data", "out.csv"), evalue=1e-5, num_hits=2)

        self.assertTrue(blast.results.shape[0] == 6)

    def test_diamond_associate_to_ec(self):
        DIAMOND_DB = "data/DIAMOND_DB"
        blast = Diamond(DIAMOND_DB)
        blast.create_database(os.path.join(TEST_DIR, "data", "test.fasta"), DIAMOND_DB)
        blast.run(os.path.join(TEST_DIR, "data", "query.fasta"),
                  os.path.join(TEST_DIR, "data", "out.csv"), evalue=1e-5, num_hits=1)

        blast.associate_to_ec(os.path.join(TEST_DIR, "data", "test_database_ec_numbers.csv"),
                              os.path.join(TEST_DIR, "data", "out_ec.csv"))

    def test_blast_associate_to_ec(self):
        blast_DB = "data/blast_db"
        blast = BLAST(blast_DB)
        blast.create_database(os.path.join(TEST_DIR, "data", "test.fasta"), blast_DB)
        blast.run(os.path.join(TEST_DIR, "data", "query.fasta"),
                  os.path.join(TEST_DIR, "data", "out.csv"), evalue=1e-5, num_hits=1)

        blast.associate_to_ec(os.path.join(TEST_DIR, "data", "test_database_ec_numbers.csv"),
                              os.path.join(TEST_DIR, "data", "out_ec.csv"))
