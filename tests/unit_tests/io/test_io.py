from plants_sm.io import read_csv, write_csv

import os
from unittest import TestCase

from plants_sm.io.excel import read_excel, write_excel
from tests import TEST_DIR


class TestIO(TestCase):

    def setUp(self) -> None:
        self.test_read_csv = os.path.join(TEST_DIR, "data", "example1.csv")
        self.test_read_excel = os.path.join(TEST_DIR, "data", "drug_list.xlsx")
        self.df_path_to_write_csv = os.path.join(TEST_DIR, "data", "test.csv")
        self.df_path_to_write_xlsx = os.path.join(TEST_DIR, "data", "test.xlsx")

    def tearDown(self) -> None:

        paths_to_remove = [self.df_path_to_write_csv,
                           self.df_path_to_write_xlsx]

        for path in paths_to_remove:
            if os.path.exists(path):
                os.remove(path)

    def test_read_csv(self):
        df = read_csv(self.test_read_csv)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 1026)

    def test_write_csv(self):
        df = read_csv(self.test_read_csv)

        written = write_csv(self.df_path_to_write_csv, df, index=False)

        self.assertTrue(written)
        self.assertTrue(os.path.exists(self.df_path_to_write_csv))

        df = read_csv(self.df_path_to_write_csv)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 1026)

    def test_read_excel(self):
        df = read_excel(self.test_read_excel, sheet_name="DrugInfo")
        self.assertEqual(df.shape[0], 271)
        self.assertEqual(df.shape[1], 117)

    def test_write_excel(self):
        df1 = read_excel(self.test_read_excel, sheet_name="DrugInfo")

        write_excel(self.df_path_to_write_xlsx, df1, index=False)

        df2 = read_excel(self.df_path_to_write_xlsx)

        self.assertEqual(df1.shape, df2.shape)
        self.assertEqual(df1.iloc[0, 0], df1.iloc[0, 0])


