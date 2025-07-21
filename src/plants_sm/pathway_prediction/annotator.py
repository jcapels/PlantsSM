from abc import abstractmethod
from typing import List
from pydantic import BaseModel

from plants_sm.pathway_prediction.entities import BiologicalEntity
from plants_sm.pathway_prediction.solution import ECSolution, Solution


class Annotator(BaseModel):

    solution: Solution = None

    @abstractmethod
    def _annotate(self, entities: List[BiologicalEntity]) -> List[Solution]:
        pass

    def annotate(self, entities: List[BiologicalEntity]) -> List[Solution]:
        self._annotate(entities)

class ECAnnotator(Annotator):

    solution: ECSolution = None

