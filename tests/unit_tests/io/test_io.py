import os
from unittest import TestCase

from plants_sm.io import read_csv
from tests import TEST_DIR


class TestIO(TestCase):

    def setUp(self) -> None:
        self.test_read_csv = os.path.join(TEST_DIR, "data", "example1.csv")

    def test_read_csv(self):
        df = read_csv(self.test_read_csv)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 1026)

