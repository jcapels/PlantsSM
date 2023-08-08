import subprocess
from abc import ABC, abstractmethod

import pandas as pd


class Alignment(ABC):

    def __init__(self, database):
        self.results = None
        self.database = database

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

    def associate_to_ec(self, database_ec_file, output_file):
        database = pd.read_csv(database_ec_file)

        new_results = pd.DataFrame(columns=database.columns)
        for i, row in self.results.iterrows():
            accession = row["qseqid"]
            database_accession = row["qseqid"]

            hit_row = database[database.loc[:, "accession"] == database_accession]

            if not hit_row.empty:
                new_results = pd.concat((new_results, hit_row))
                new_results.at[i, "accession"] = accession
                new_results.at[i, "name"] = accession
                new_results.at[i, "sequence"] = pd.NA

            else:
                new_results.at[i, :] = pd.NA
                new_results.at[i, "accession"] = accession
                new_results.at[i, "name"] = accession
                new_results.at[i, "sequence"] = pd.NA
                new_results.at[i, :].fillna(0, inplace=True)

        new_results.to_csv(output_file, index=False)


class Diamond(Alignment):

    def __init__(self, database):
        super().__init__(database)

    def create_database(self, fasta_file, db_name):
        subprocess.run(["diamond", "makedb", "--in", fasta_file, "--db", db_name])

    def _run(self, query_file, output_file, evalue, num_hits, output_options_str):
        subprocess.call(f"diamond blastp -d {self.database} -q {query_file} -o {output_file} --outfmt 6 "
                       f"{output_options_str} --evalue {evalue} --max-target-seqs {num_hits}", shell=True)


class BLAST(Alignment):

    def __init__(self, database):
        super().__init__(database)

    def create_database(self, fasta_file, db_name):
        subprocess.run(["makeblastdb", "-in", fasta_file, "-dbtype", "prot", "-out", db_name])

    def _run(self, query_file, output_file, evalue, num_hits, output_options_str):
        subprocess.run(["blastp", "-query", query_file, "-db", self.database, "-out", output_file, "-outfmt",
                        f"6 {output_options_str}", "-evalue", str(evalue), "-max_target_seqs", str(num_hits)])



