from typing import List
from plants_sm.pathway_prediction.annotator import ECAnnotator
from plants_sm.pathway_prediction.entities import Protein


class ProtBertECAnnotator(ECAnnotator):


    def _annotate(self, entities: List[Protein]):
        return super()._annotate()