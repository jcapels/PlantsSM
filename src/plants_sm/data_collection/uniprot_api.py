from typing import Any

from typing import Any, Union, Dict

from io import StringIO
from urllib.request import Request

import pandas as pd
from pandas.errors import EmptyDataError
import urllib
import urllib.request


from plants_sm.data_collection.api import AbstractAPIAccessor
from plants_sm.data_structures.biological_entities import Protein

class UniProtAPI(AbstractAPIAccessor):
    """
        Class to get Protein Sequences from Uniprot.
        """
    def __init__(self):
        self._data = None
        self.url = "https://www.uniprot.org/uniprot/"

    def make_request(self, params: Union[str, Dict]) -> Request:
        """
            Make a request at Uniprot url.

            Parameters
            ----------
            params: str or dict
                Data to use in the search

            Returns
            -------
            Request
        """
        if isinstance(params, str):

            params_url = urllib.parse.quote_plus(params)
            params_url = params_url.encode('ascii')
            # final_url = self.url + params_url
            request = urllib.request.Request(self.url, params_url)
            return request

        elif isinstance(params, Dict):

            data = urllib.parse.urlencode(params)
            data = data.encode('ascii')
            request = urllib.request.Request(self.url, data)

            return request

        else:
            raise ValueError("The parameters should be a string or a dictionary.")

    @property
    def data(self):
        """
            Data as a property.

            Parameters
            ----------
            Returns
            -------
        """
        return self._data

    @data.setter
    def data(self, value):
        """
            Data as a property.

            Parameters
            ----------
            value : list of the labels names to then be retrieved from the dataframe

            Returns
            -------
        """
        self._data = value

    def get_potential_uniprot_id_by_name(self, protein_name: str) -> Union[str, None]:
        """
            Data as a property.

            Parameters
            ----------
            protein_name : string of protein names to be used in the search

            Returns
            -------
        """

        params = {
            'format': 'tab',
            'query': f"gene:{protein_name}",
            'sort': 'score'
        }

        request = self.make_request(params)
        with urllib.request.urlopen(request) as response:
            res = response.read()
        try:
            df_uniprot_query_res = pd.read_csv(StringIO(res.decode("utf-8")), sep="\t")
        except EmptyDataError:
            return
        if df_uniprot_query_res.shape[0] > 0:
            return df_uniprot_query_res.at[0, "Entry"]

    def get_protein_sequences(self, uniprot_ids_list: list) -> list:
        """
            Data as a property.

            Parameters
            ----------
            uniprot_ids_list : list of ids to be used in the search for protein sequences

            Returns
            -------
            list of protein sequences for each id
        """

        checkpoint = 0
        division_list_i = len(uniprot_ids_list) // 400
        if division_list_i < 1:
            division_list_i = 1
            checkpoint = len(uniprot_ids_list)

        last_i = 0
        fasta_result = ""
        for i in range(division_list_i):
            checkpoint += 400
            uniprot_ids_batch = uniprot_ids_list[last_i:checkpoint]
            last_i = checkpoint
            uniprot_ids_list_for_url = ['id:' + uniprot_id for uniprot_id in uniprot_ids_batch]
            line = '+OR+'.join(uniprot_ids_list_for_url)
            params = f'query={line}&format=fasta'
            request = self.make_request(params)
            with request as f:
                fasta = f.read().decode('utf-8').strip()
                fasta_result = fasta_result + "\n" + fasta

        return fasta_result

    @staticmethod
    def convert_data_to_entities(self, fasta_result):
        """
            Convert data to entities

            Parameters
            ----------
            fasta_result
            Returns
            -------
        """
        protein = Protein()
        protein.convert()
        
        pr = pln(fasta_result)

        for entitie in pr.ents:
           #print(entitie.text, entitie.label_)
            entt = entitie.text, entitie.label_
        return entt    





def test():
    

    gr = UniProtAPI()
    request = gr.make_request({'AT4G35420','AT4G23100', 'AT3G58990', 'AT1G08250' })
    print(request)

    data = gr.data()
    print(data)

    name = gr.get_potential_uniprot_id_by_name(request)
    print(name)

    sequences = gr.get_protein_sequences(name)
    print(sequences)

    convert = gr.convert_data_to_entities()
    print(convert)
   

test()
