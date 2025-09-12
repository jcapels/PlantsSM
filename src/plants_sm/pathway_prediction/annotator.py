from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Union

from plants_sm.pathway_prediction.entities import BiologicalEntity
from plants_sm.pathway_prediction.solution import Solution



class Annotator(ABC):

    solution: Solution = None
    invalid_entities: Union[List[BiologicalEntity], pd.DataFrame] = None

    @abstractmethod
    def validate_input(self, entities: Union[List[BiologicalEntity], pd.DataFrame]):
        pass

    @abstractmethod
    def _annotate(self, entities: Union[List[BiologicalEntity], pd.DataFrame]) -> List[Solution]:
        pass

    def annotate(self, entities: Union[List[BiologicalEntity], pd.DataFrame]) -> List[Solution]:
        
        valid_entities, self.invalid_entities = self.validate_input(entities)
        
        self.solution = self._annotate(valid_entities)

        return self.solution

    def annotate_from_file(self, file: str, format: str, **kwargs) -> Solution:

        entities = self._convert_to_readable_format(file, format, **kwargs)

        self.annotate(entities)

        return self.solution
    
    def _dataframe_from_csv(self, file: str, **kwargs) -> pd.DataFrame:

        dataset = pd.read_csv(file, **kwargs)

        return dataset

    @abstractmethod
    def _convert_to_readable_format(self, file: str, format: str, **kwargs) -> Union[List[BiologicalEntity], pd.DataFrame]:
        pass


