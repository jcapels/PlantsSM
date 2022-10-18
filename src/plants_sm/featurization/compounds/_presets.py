from deepmol.compound_featurization import MorganFingerprint, MACCSkeysFingerprint, LayeredFingerprint, RDKFingerprint, \
    AtomPairFingerprint, TwoDimensionDescriptors

DEEPMOL_PRESETS = {
    "morgan_fingerprints": MorganFingerprint,
    "maccs_keys_fingerprints": MACCSkeysFingerprint,
    "layered_fingerprints": LayeredFingerprint,
    "rdk_fingerprints": RDKFingerprint,
    "atompair_fingerprints": AtomPairFingerprint,
    "2d_descriptors": TwoDimensionDescriptors,
}
