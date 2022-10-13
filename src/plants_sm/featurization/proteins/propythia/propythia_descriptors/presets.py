from .psycho_chemical import (LengthDescriptor,
                              ChargeDescriptor,
                              ChargeDensityDescriptor,
                              FormulaDescriptor,
                              BondDescriptor,
                              MolecularWeightDescriptor,
                              GravyDescriptor,
                              AromacityDescriptor,
                              IsoelectricPointDescriptor,
                              InstabilityIndexDescriptor,
                              SecondaryStructureDescriptor,
                              MolarExtinctionCoefficientDescriptor,
                              FlexibilityDescriptor,
                              AliphaticIndexDescriptor,
                              BomanIndexDescriptor,
                              HydrophobicRatioDescriptor,
                              ModlAMPMomentDescriptor,
                              ModlAMPGlobalDescriptor,
                              ModlAMPProfileDescriptor,
                              ModlAMPArcDescriptor)
from .amino_acid_composition import (AminoAcidCompositionDescriptor,
                                     DipeptideCompositionDescriptor,
                                     TripeptideCompositionDescriptor)
from .pseudo_amino_acid_composition import (PseudoAminoAcidCompositionDescriptor,
                                            AmphiphilicPseudoAminoAcidCompositionDescriptor)
from .correlation import (NormalizedMoreauBrotoDescriptor,
                          MoranAutoDescriptor,
                          GearyAutoDescriptor,
                          ModlAMPAutoCorrelationDescriptor,
                          ModlAMPCrossCorrelationDescriptor)
from .composition_transition_distribution import CompositionTransitionDistributionDescriptor
from .sequence_order import (SequenceOrderCouplingNumbersDescriptor,
                             QuasiSequenceOrderDescriptor)


PHYSICO_CHEMICAL_DESCRIPTORS = [
    LengthDescriptor,
    ChargeDescriptor,
    ChargeDensityDescriptor,
    FormulaDescriptor,
    BondDescriptor,
    MolecularWeightDescriptor,
    GravyDescriptor,
    AromacityDescriptor,
    IsoelectricPointDescriptor,
    InstabilityIndexDescriptor,
    SecondaryStructureDescriptor,
    MolarExtinctionCoefficientDescriptor,
    FlexibilityDescriptor,
    AliphaticIndexDescriptor,
    BomanIndexDescriptor,
    HydrophobicRatioDescriptor
]


AMINO_ACID_COMPOSITION_DESCRIPTORS = [AminoAcidCompositionDescriptor,
                                      DipeptideCompositionDescriptor,
                                      TripeptideCompositionDescriptor]


PSEUDO_AMINO_ACID_COMPOSITION_DESCRIPTORS = [PseudoAminoAcidCompositionDescriptor,
                                             AmphiphilicPseudoAminoAcidCompositionDescriptor]


AUTO_CORRELATION_DESCRIPTORS = [NormalizedMoreauBrotoDescriptor,
                                MoranAutoDescriptor,
                                GearyAutoDescriptor]

CTD_DESCRIPTORS = [CompositionTransitionDistributionDescriptor]


SEQUENCE_ORDER_DESCRIPTORS = [SequenceOrderCouplingNumbersDescriptor,
                              QuasiSequenceOrderDescriptor]


MODLAMP_CORRELATION_DESCRIPTORS = [ModlAMPAutoCorrelationDescriptor,
                                   ModlAMPCrossCorrelationDescriptor]


MODLAM_PHYSICO_CHEMICAL_DESCRIPTORS = [ModlAMPMomentDescriptor,
                                       ModlAMPGlobalDescriptor,
                                       ModlAMPProfileDescriptor,
                                       ModlAMPArcDescriptor]


DESCRIPTORS_PRESETS = {
    'all': [*PHYSICO_CHEMICAL_DESCRIPTORS,
            *AMINO_ACID_COMPOSITION_DESCRIPTORS,
            *PSEUDO_AMINO_ACID_COMPOSITION_DESCRIPTORS,
            *AUTO_CORRELATION_DESCRIPTORS,
            *CTD_DESCRIPTORS,
            *SEQUENCE_ORDER_DESCRIPTORS,
            *MODLAMP_CORRELATION_DESCRIPTORS,
            *MODLAM_PHYSICO_CHEMICAL_DESCRIPTORS],
    'performance': [*PHYSICO_CHEMICAL_DESCRIPTORS,
                    *AMINO_ACID_COMPOSITION_DESCRIPTORS,
                    *PSEUDO_AMINO_ACID_COMPOSITION_DESCRIPTORS,
                    *CTD_DESCRIPTORS,
                    *MODLAMP_CORRELATION_DESCRIPTORS],
    'physicochemical': PHYSICO_CHEMICAL_DESCRIPTORS,
    'aac': AMINO_ACID_COMPOSITION_DESCRIPTORS,
    'paac': PSEUDO_AMINO_ACID_COMPOSITION_DESCRIPTORS,
    'auto-correlation': AUTO_CORRELATION_DESCRIPTORS,
    'composition-transition-distribution': CTD_DESCRIPTORS,
    'seq-order': SEQUENCE_ORDER_DESCRIPTORS,
    'modlamp-correlation': MODLAMP_CORRELATION_DESCRIPTORS,
    'modlamp-all': [*MODLAMP_CORRELATION_DESCRIPTORS,
                    *MODLAM_PHYSICO_CHEMICAL_DESCRIPTORS]
}
