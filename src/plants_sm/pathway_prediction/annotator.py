from abc import abstractmethod
import pandas as pd
from typing import Dict, List, Union
from pydantic import BaseModel

from plants_sm.pathway_prediction.entities import BiologicalEntity, Protein
from plants_sm.pathway_prediction.solution import ECSolution, Solution

from Bio import SeqIO


class Annotator(BaseModel):

    solution: Solution = None

    @abstractmethod
    def _annotate(self, entities: Union[List[BiologicalEntity], pd.DataFrame]) -> List[Solution]:
        pass

    def annotate(self, entities: Union[List[BiologicalEntity], pd.DataFrame]) -> List[Solution]:
        self._annotate(entities)

    def annotate_from_file(self, file: str, format: str, **kwargs) -> Solution:

        self.solution = self._annotate_from_file(file, format, **kwargs)

        return self.solution
    
    @abstractmethod
    def _annotate_from_file(self, file: str, format: str, **kwargs) -> List[BiologicalEntity]:
        pass


