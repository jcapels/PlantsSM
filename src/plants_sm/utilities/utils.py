import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def convert_csv_to_fasta(csv_file, sequence_column, header_column, output_file):
    df = pd.read_csv(csv_file)

    records = []
    for index, row in df.iterrows():
        sequence = row[sequence_column]
        header = row[header_column] if header_column else f"sequence_{index}"
        record = SeqRecord(Seq(sequence), id=header, description="")
        records.append(record)

    with open(output_file, 'w') as output_handle:
        SeqIO.write(records, output_handle, "fasta")