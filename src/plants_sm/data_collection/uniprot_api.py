from typing import Any

from plants_sm.data_collection.api import AbstractAPIAccessor


class UniProtAPI(AbstractAPIAccessor):

    def make_request(self, data: Any) -> int:
        pass

    @property
    def data(self):
        pass

    def convert_data_to_entities(self):
        pass
