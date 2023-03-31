import sys

import nni

from plants_sm.models.enumerators import ModelFileEnumeratorsUtils

if __name__ == "__main__":
    model_path = sys.argv[0]
    RECEIVED_PARAMS = nni.get_next_parameter()
    PARAMS = {}
    PARAMS.update(RECEIVED_PARAMS)
    model_type = ModelFileEnumeratorsUtils.get_class_to_load_from_directory(model_path)
    loaded_model = model_type.load(model_path)
    loaded_model.mode.__init__(PARAMS)
    loaded_model.fit()
