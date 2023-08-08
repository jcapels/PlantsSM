import subprocess
from abc import ABC, abstractmethod

import pandas as pd
from tqdm import tqdm


class Alignment(ABC):

    def __init__(self, database):
        self.results = None
        self.database = database

    @abstractmethod
    def create_database(self, fasta_file):
        pass

    def run(self, query_file, output_file, evalue, num_hits, output_options=None):
        if output_options is None:
            output_options = ["qseqid", "sseqid", "pident", "length", "mismatch",
                              "gapopen", "qstart", "qend", "sstart", "evalue", "bitscore"]

        output_options_str = " ".join(output_options)
        self._run(query_file, output_file, evalue, num_hits, output_options_str)
        self.results = pd.read_csv(output_file, sep="\t", header=None, names=output_options)
        self.results.to_csv(output_file, sep="\t", index=False)

    @abstractmethod
    def _run(self, query_file, output_file, evalue, num_hits, output_options_str):
        pass

    def associate_to_ec(self, database_ec_dataframe, output_file):
        database_reduced = database_ec_dataframe[
            database_ec_dataframe.loc[:, "accession"].isin(self.results.loc[:, "sseqid"])]
        del database_ec_dataframe
        self.results.columns = ["qseqid", "accession", "pident", "length", "mismatch", "gapopen", "qstart", "qend",
                                "sstart", "evalue", "bitscore"]
        database_reduced.drop(columns=["sequence"], inplace=True)
        merged_df = pd.merge(self.results, database_reduced, on='accession', how='inner')
        merged_df.to_csv(output_file, index=False)


class Diamond(Alignment):

    def __init__(self, database):
        super().__init__(database)

    def create_database(self, fasta_file):
        subprocess.run(["diamond", "makedb", "--in", fasta_file, "--db", self.database])

    def _run(self, query_file, output_file, evalue, num_hits, output_options_str):
        subprocess.call(f"diamond blastp -d {self.database} -q {query_file} -o {output_file} --outfmt 6 "
                        f"{output_options_str} --evalue {evalue} --max-target-seqs {num_hits}", shell=True)


class BLAST(Alignment):

    def __init__(self, database):
        super().__init__(database)

    def create_database(self, fasta_file):
        subprocess.run(["makeblastdb", "-in", fasta_file, "-dbtype", "prot", "-out", self.database])

    def _run(self, query_file, output_file, evalue, num_hits, output_options_str):
        subprocess.run(["blastp", "-query", query_file, "-db", self.database, "-out", output_file, "-outfmt",
                        f"6 {output_options_str}", "-evalue", str(evalue), "-max_target_seqs", str(num_hits)])
