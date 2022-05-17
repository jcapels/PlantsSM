from plants_sm.featurization.featurizer import FeaturesGenerator
from propythia.descriptors import Descriptor


class PropythiaWrapper(FeaturesGenerator):

    def __init__(self, descriptor: str, **kwargs):
        self.descriptor = descriptor
        self.general_descriptor = Descriptor("")
        self.kwargs = kwargs
        super().__init__()

    def _featurize(self, protein_sequence: str):
        self.general_descriptor.ProteinSequence = protein_sequence
        func = getattr(self.general_descriptor, self.descriptor)
        features = func(**self.kwargs)
        return features
