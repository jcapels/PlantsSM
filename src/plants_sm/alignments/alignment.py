import subprocess
from abc import ABC, abstractmethod

import pandas as pd


class Alignment(ABC):

    @abstractmethod
    def create_database(self, fasta_file, db_name):
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


class Diamond(Alignment):

    def __init__(self, database):
        self.database = database

    def create_database(self, fasta_file, db_name):
        subprocess.run(["diamond", "makedb", "--in", fasta_file, "--db", db_name])

    def _run(self, query_file, output_file, evalue, num_hits, output_options_str):

        subprocess.run(["diamond", "blastp", "-d", self.database, "-q", query_file, "-o", output_file,
                            "-f", f"6 {output_options_str}", "-evalue", str(evalue), "-max_target_seqs", str(num_hits)])


class BLAST(Alignment):

    def __init__(self, database):
        self.database = database
        self.results = None

    def create_database(self, fasta_file, db_name):
        subprocess.run(["makeblastdb", "-in", fasta_file, "-dbtype", "prot", "-out", db_name])

    def _run(self, query_file, output_file, evalue, num_hits, output_options_str):

        subprocess.run(["blastp", "-query", query_file, "-db", self.database, "-out", output_file, "-outfmt",
                    f"6 {output_options_str}", "-evalue", str(evalue), "-max_target_seqs", str(num_hits)])
