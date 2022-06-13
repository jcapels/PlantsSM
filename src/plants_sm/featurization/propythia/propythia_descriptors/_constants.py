import math

from ._utils import normalize_aap

BONDS = {'G': {'total_bounds': 9, 'hydrogen_bonds': 5, 'single_bounds': 8, 'double_bounds': 1},
         'S': {'total_bounds': 13, 'hydrogen_bonds': 7, 'single_bounds': 12, 'double_bounds': 1},
         'A': {'total_bounds': 12, 'hydrogen_bonds': 7, 'single_bounds': 11, 'double_bounds': 1},
         'D': {'total_bounds': 15, 'hydrogen_bonds': 7, 'single_bounds': 13, 'double_bounds': 2},
         'N': {'total_bounds': 16, 'hydrogen_bonds': 8, 'single_bounds': 14, 'double_bounds': 2},
         'T': {'total_bounds': 16, 'hydrogen_bonds': 9, 'single_bounds': 15, 'double_bounds': 1},
         'P': {'total_bounds': 17, 'hydrogen_bonds': 9, 'single_bounds': 16, 'double_bounds': 1},
         'E': {'total_bounds': 18, 'hydrogen_bonds': 9, 'single_bounds': 16, 'double_bounds': 2},
         'V': {'total_bounds': 18, 'hydrogen_bonds': 11, 'single_bounds': 17, 'double_bounds': 1},
         'Q': {'total_bounds': 19, 'hydrogen_bonds': 10, 'single_bounds': 17, 'double_bounds': 2},
         'M': {'total_bounds': 19, 'hydrogen_bonds': 11, 'single_bounds': 18, 'double_bounds': 1},
         'H': {'total_bounds': 20, 'hydrogen_bonds': 9, 'single_bounds': 17, 'double_bounds': 3},
         'I': {'total_bounds': 21, 'hydrogen_bonds': 13, 'single_bounds': 20, 'double_bounds': 1},
         'Y': {'total_bounds': 24, 'hydrogen_bonds': 11, 'single_bounds': 20, 'double_bounds': 4},
         'L': {'total_bounds': 21, 'hydrogen_bonds': 13, 'single_bounds': 20, 'double_bounds': 1},
         'K': {'total_bounds': 23, 'hydrogen_bonds': 14, 'single_bounds': 22, 'double_bounds': 1},
         'W': {'total_bounds': 28, 'hydrogen_bonds': 12, 'single_bounds': 23, 'double_bounds': 5},
         'F': {'total_bounds': 23, 'hydrogen_bonds': 11, 'single_bounds': 19, 'double_bounds': 4},
         'C': {'total_bounds': 25, 'hydrogen_bonds': 12, 'single_bounds': 23, 'double_bounds': 2},
         'R': {'total_bounds': 25, 'hydrogen_bonds': 14, 'single_bounds': 23, 'double_bounds': 2}}

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

HYDROPHOBICITY = {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85, "E": -0.74, "G": 0.48,
                  "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12, "S": -0.18,
                  "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08}
HYDROPHOBICITY_NORMALIZED = normalize_aap(HYDROPHOBICITY)


HYDROPHILICITY = {"A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0, "Q": 0.2, "E": 3.0, "G": 0.0, "H": -0.5,
                  "I": -1.8, "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5, "P": 0.0, "S": 0.3, "T": -0.4, "W": -3.4,
                  "Y": -2.3, "V": -1.5}
HYDROPHILICITY_NORMALIZED = normalize_aap(HYDROPHILICITY)


RESIDUE_MASS = {"A": 15.0, "R": 101.0, "N": 58.0, "D": 59.0, "C": 47.0, "Q": 72.0, "E": 73.0, "G": 1.000, "H": 82.0,
                "I": 57.0, "L": 57.0, "K": 73.0, "M": 75.0, "F": 91.0, "P": 42.0, "S": 31.0, "T": 45.0, "W": 130.0,
                "Y": 107.0, "V": 43.0}
RESIDUE_MASS_NORMALIZED = normalize_aap(RESIDUE_MASS)


PK1 = {"A": 2.35, "C": 1.71, "D": 1.88, "E": 2.19, "F": 2.58, "G": 2.34, "H": 1.78, "I": 2.32, "K": 2.20, "L": 2.36,
       "M": 2.28, "N": 2.18, "P": 1.99, "Q": 2.17, "R": 2.18, "S": 2.21, "T": 2.15, "V": 2.29, "W": 2.38, "Y": 2.20}
PK1_NORMALIZED = normalize_aap(PK1)


PK2 = {"A": 9.87, "C": 10.78, "D": 9.60, "E": 9.67, "F": 9.24, "G": 9.60, "H": 8.97, "I": 9.76, "K": 8.90, "L": 9.60,
       "M": 9.21, "N": 9.09, "P": 10.6, "Q": 9.13, "R": 9.09, "S": 9.15, "T": 9.12, "V": 9.74, "W": 9.39, "Y": 9.11}
PK2_NORMALIZED = normalize_aap(PK2)


PI = {"A": 6.11, "C": 5.02, "D": 2.98, "E": 3.08, "F": 5.91, "G": 6.06, "H": 7.64, "I": 6.04, "K": 9.47, "L": 6.04,
      "M": 5.74, "N": 10.76, "P": 6.30, "Q": 5.65, "R": 10.76, "S": 5.68, "T": 5.60, "V": 6.02, "W": 5.88, "Y": 5.63}
PI_NORMALIZED = normalize_aap(PI)


COR_HYDROPHOBICITY = {"A": 0.02, "R": -0.42, "N": -0.77, "D": -1.04, "C": 0.77, "Q": -1.10, "E": -1.14, "G": -0.80,
                      "H": 0.26, "I": 1.81, "L": 1.14, "K": -0.41, "M": 1.00, "F": 1.35, "P": -0.09, "S": -0.97,
                      "T": -0.77, "W": 1.71, "Y": 1.11, "V": 1.13}
COR_HYDROPHOBICITY_NORMALIZED = normalize_aap(COR_HYDROPHOBICITY)


COR_AV_FLEXIBILITY = {"A": 0.357, "R": 0.529, "N": 0.463, "D": 0.511, "C": 0.346, "Q": 0.493, "E": 0.497, "G": 0.544,
                      "H": 0.323, "I": 0.462, "L": 0.365, "K": 0.466, "M": 0.295, "F": 0.314, "P": 0.509, "S": 0.507,
                      "T": 0.444, "W": 0.305, "Y": 0.420, "V": 0.386}
COR_AV_FLEXIBILITY_NORMALIZED = normalize_aap(COR_AV_FLEXIBILITY)


COR_POLARIZABILITY = {"A": 0.046, "R": 0.291, "N": 0.134, "D": 0.105, "C": 0.128, "Q": 0.180, "E": 0.151, "G": 0.000,
                      "H": 0.230, "I": 0.186, "L": 0.186, "K": 0.219, "M": 0.221, "F": 0.290, "P": 0.131, "S": 0.062,
                      "T": 0.108, "W": 0.409, "Y": 0.298, "V": 0.140}
COR_POLARIZABILITY_NORMALIZED = normalize_aap(COR_POLARIZABILITY)


COR_FREE_ENERGY = {"A": -0.368, "R": -1.03, "N": 0.0, "D": 2.06, "C": 4.53, "Q": 0.731, "E": 1.77, "G": -0.525,
                   "H": 0.0,
                   "I": 0.791, "L": 1.07, "K": 0.0, "M": 0.656, "F": 1.06, "P": -2.24, "S": -0.524, "T": 0.0, "W": 1.60,
                   "Y": 4.91, "V": 0.401}
COR_FREE_ENERGY_NORMALIZED = normalize_aap(COR_FREE_ENERGY)


COR_RESIDUE_ASA = {"A": 115.0, "R": 225.0, "N": 160.0, "D": 150.0, "C": 135.0, "Q": 180.0, "E": 190.0, "G": 75.0,
                   "H": 195.0, "I": 175.0, "L": 170.0, "K": 200.0, "M": 185.0, "F": 210.0, "P": 145.0, "S": 115.0,
                   "T": 140.0, "W": 255.0, "Y": 230.0, "V": 155.0}
COR_RESIDUE_ASA_NORMALIZED = normalize_aap(COR_RESIDUE_ASA)


COR_RESIDUE_VOL = {"A": 52.6, "R": 109.1, "N": 75.7, "D": 68.4, "C": 68.3, "Q": 89.7, "E": 84.7, "G": 36.3, "H": 91.9,
                   "I": 102.0, "L": 102.0, "K": 105.1, "M": 97.7, "F": 113.9, "P": 73.6, "S": 54.9, "T": 71.2,
                   "W": 135.4,
                   "Y": 116.2, "V": 85.1}
COR_RESIDUE_VOL_NORMALIZED = normalize_aap(COR_RESIDUE_VOL)


COR_STERIC = {"A": 0.52, "R": 0.68, "N": 0.76, "D": 0.76, "C": 0.62, "Q": 0.68, "E": 0.68, "G": 0.00, "H": 0.70,
              "I": 1.02,
              "L": 0.98, "K": 0.68, "M": 0.78, "F": 0.70, "P": 0.36, "S": 0.53, "T": 0.50, "W": 0.70, "Y": 0.70,
              "V": 0.76}
COR_STERIC_NORMALIZED = normalize_aap(COR_STERIC)


COR_MUTABILITY = {"A": 100.0, "R": 65.0, "N": 134.0, "D": 106.0, "C": 20.0, "Q": 93.0, "E": 102.0, "G": 49.0, "H": 66.0,
                  "I": 96.0, "L": 40.0, "K": -56.0, "M": 94.0, "F": 41.0, "P": 56.0, "S": 120.0, "T": 97.0, "W": 18.0,
                  "Y": 41.0, "V": 74.0}
COR_MUTABILITY_NORMALIZED = normalize_aap(COR_MUTABILITY)


# '1'stand for Polar; '2'stand for Neutral, '3' stand for Hydrophobicity
CTD_HYDROPHOBICITY = {'R': '1', 'K': '1', 'E': '1', 'D': '1', 'Q': '1', 'N': '1', 'G': '2', 'A': '2', 'S': '2',
                      'T': '2', 'P': '2', 'H': '2', 'Y': '2', 'C': '3', 'L': '3', 'V': '3', 'I': '3', 'M': '3',
                      'F': '3', 'W': '3'}
CTD_HYDROPHOBICITY_TABLE = str.maketrans(CTD_HYDROPHOBICITY)

# '1'stand for (0-2.78); '2'stand for (2.95-4.0), '3' stand for (4.03-8.08)
CTD_NORMALIZED_VDWV = {'G': '1', 'A': '1', 'S': '1', 'T': '1', 'P': '1', 'D': '1', 'N': '2', 'V': '2', 'E': '2',
                       'Q': '2', 'I': '2', 'L': '2', 'M': '3', 'H': '3', 'K': '3', 'F': '3', 'R': '3', 'Y': '3',
                       'W': '3'}
CTD_NORMALIZED_VDWV_TABLE = str.maketrans(CTD_NORMALIZED_VDWV)

# '1'stand for (4.9-6.2); '2'stand for (8.0-9.2), '3' stand for (10.4-13.0)
CTD_POLARITY = {'L': '2', 'I': '2', 'F': '3', 'W': '3', 'C': '2', 'M': '3', 'V': '2', 'Y': '3', 'P': '2', 'N': '2',
                'E': '2', 'Q': '2', 'K': '3', 'H': '3', 'R': '3'}
CTD_POLARITY_TABLE = str.maketrans(CTD_POLARITY)

# '1'stand for Positive; '2'stand for Neutral, '3' stand for Negative
CTD_CHARGE = {'K': '1', 'R': '1', 'A': '2', 'N': '2', 'C': '2', 'Q': '2', 'G': '2', 'H': '2', 'I': '2', 'L': '2',
              'M': '2', 'F': '2', 'P': '2', 'S': '2', 'T': '2', 'W': '2', 'Y': '2', 'V': '2', 'D': '3', 'E': '3'}
CTD_CHARGE_TABLE = str.maketrans(CTD_CHARGE)

# '1'stand for Helix; '2'stand for Strand, '3' stand for coil
CTD_SECONDARY_STRUCTURE = {'E': '1', 'A': '1', 'L': '1', 'M': '1', 'Q': '1', 'K': '1', 'R': '1', 'H': '1', 'V': '2',
                           'I': '2', 'Y': '2', 'C': '2', 'W': '2', 'F': '2', 'T': '2', 'G': '3', 'N': '3', 'P': '3',
                           'S': '3', 'D': '3'}
CTD_SECONDARY_STRUCTURE_TABLE = str.maketrans(CTD_SECONDARY_STRUCTURE)

# '1'stand for Buried; '2'stand for Exposed, '3' stand for Intermediate
CTD_SOLVENT_ACCESSIBILITY = {'A': '1', 'L': '1', 'F': '1', 'C': '1', 'G': '1', 'I': '1', 'V': '1', 'W': '1', 'R': '2',
                             'K': '2', 'Q': '2', 'E': '2', 'N': '2', 'D': '2', 'M': '3', 'P': '3', 'S': '3', 'T': '3',
                             'H': '3', 'Y': '3'}
CTD_SOLVENT_ACCESSIBILITY_TABLE = str.maketrans(CTD_SOLVENT_ACCESSIBILITY)

# '1'stand for (0-0.108); '2'stand for (0.128-0.186), '3' stand for (0.219-0.409)
POLARIZABILITY = {'G': '1', 'A': '1', 'S': '1', 'D': '1', 'T': '1', 'C': '2', 'P': '2', 'N': '2', 'V': '2', 'E': '2',
                  'Q': '2', 'I': '2', 'L': '2', 'K': '3', 'M': '3', 'H': '3', 'F': '3', 'R': '3', 'Y': '3', 'W': '3'}
POLARIZABILITY_TABLE = str.maketrans(POLARIZABILITY)


# Distance is the Schneider-Wrede physicochemical distance matrix used by Chou et. al.
DISTANCE_1 = {"GW": 0.923, "GV": 0.464, "GT": 0.272, "GS": 0.158, "GR": 1.0, "GQ": 0.467, "GP": 0.323, "GY": 0.728,
              "GG": 0.0, "GF": 0.727, "GE": 0.807, "GD": 0.776, "GC": 0.312, "GA": 0.206, "GN": 0.381, "GM": 0.557,
              "GL": 0.591, "GK": 0.894, "GI": 0.592, "GH": 0.769, "ME": 0.879, "MD": 0.932, "MG": 0.569, "MF": 0.182,
              "MA": 0.383, "MC": 0.276, "MM": 0.0, "ML": 0.062, "MN": 0.447, "MI": 0.058, "MH": 0.648, "MK": 0.884,
              "MT": 0.358, "MW": 0.391, "MV": 0.12, "MQ": 0.372, "MP": 0.285, "MS": 0.417, "MR": 1.0, "MY": 0.255,
              "FP": 0.42, "FQ": 0.459, "FR": 1.0, "FS": 0.548, "FT": 0.499, "FV": 0.252, "FW": 0.207, "FY": 0.179,
              "FA": 0.508, "FC": 0.405, "FD": 0.977, "FE": 0.918, "FF": 0.0, "FG": 0.69, "FH": 0.663, "FI": 0.128,
              "FK": 0.903, "FL": 0.131, "FM": 0.169, "FN": 0.541, "SY": 0.615, "SS": 0.0, "SR": 1.0, "SQ": 0.358,
              "SP": 0.181, "SW": 0.827, "SV": 0.342, "ST": 0.174, "SK": 0.883, "SI": 0.478, "SH": 0.718, "SN": 0.289,
              "SM": 0.44, "SL": 0.474, "SC": 0.185, "SA": 0.1, "SG": 0.17, "SF": 0.622, "SE": 0.812, "SD": 0.801,
              "YI": 0.23, "YH": 0.678, "YK": 0.904, "YM": 0.268, "YL": 0.219, "YN": 0.512, "YA": 0.587, "YC": 0.478,
              "YE": 0.932, "YD": 1.0, "YG": 0.782, "YF": 0.202, "YY": 0.0, "YQ": 0.404, "YP": 0.444, "YS": 0.612,
              "YR": 0.995, "YT": 0.557, "YW": 0.244, "YV": 0.328, "LF": 0.139, "LG": 0.596, "LD": 0.944, "LE": 0.892,
              "LC": 0.296, "LA": 0.405, "LN": 0.452, "LL": 0.0, "LM": 0.062, "LK": 0.893, "LH": 0.653, "LI": 0.013,
              "LV": 0.133, "LW": 0.341, "LT": 0.397, "LR": 1.0, "LS": 0.443, "LP": 0.309, "LQ": 0.376, "LY": 0.205,
              "RT": 0.808, "RV": 0.914, "RW": 1.0, "RP": 0.796, "RQ": 0.668, "RR": 0.0, "RS": 0.86, "RY": 0.859,
              "RD": 0.305, "RE": 0.225, "RF": 0.977, "RG": 0.928, "RA": 0.919, "RC": 0.905, "RL": 0.92, "RM": 0.908,
              "RN": 0.69, "RH": 0.498, "RI": 0.929, "RK": 0.141, "VH": 0.649, "VI": 0.135, "EM": 0.83, "EL": 0.854,
              "EN": 0.599, "EI": 0.86, "EH": 0.406, "EK": 0.143, "EE": 0.0, "ED": 0.133, "EG": 0.779, "EF": 0.932,
              "EA": 0.79, "EC": 0.788, "VM": 0.12, "EY": 0.837, "VN": 0.38, "ET": 0.682, "EW": 1.0, "EV": 0.824,
              "EQ": 0.598, "EP": 0.688, "ES": 0.726, "ER": 0.234, "VP": 0.212, "VQ": 0.339, "VR": 1.0, "VT": 0.305,
              "VW": 0.472, "KC": 0.871, "KA": 0.889, "KG": 0.9, "KF": 0.957, "KE": 0.149, "KD": 0.279, "KK": 0.0,
              "KI": 0.899, "KH": 0.438, "KN": 0.667, "KM": 0.871, "KL": 0.892, "KS": 0.825, "KR": 0.154, "KQ": 0.639,
              "KP": 0.757, "KW": 1.0, "KV": 0.882, "KT": 0.759, "KY": 0.848, "DN": 0.56, "DL": 0.841, "DM": 0.819,
              "DK": 0.249, "DH": 0.435, "DI": 0.847, "DF": 0.924, "DG": 0.697, "DD": 0.0, "DE": 0.124, "DC": 0.742,
              "DA": 0.729, "DY": 0.836, "DV": 0.797, "DW": 1.0, "DT": 0.649, "DR": 0.295, "DS": 0.667, "DP": 0.657,
              "DQ": 0.584, "QQ": 0.0, "QP": 0.272, "QS": 0.461, "QR": 1.0, "QT": 0.389, "QW": 0.831, "QV": 0.464,
              "QY": 0.522, "QA": 0.512, "QC": 0.462, "QE": 0.861, "QD": 0.903, "QG": 0.648, "QF": 0.671, "QI": 0.532,
              "QH": 0.765, "QK": 0.881, "QM": 0.505, "QL": 0.518, "QN": 0.181, "WG": 0.829, "WF": 0.196, "WE": 0.931,
              "WD": 1.0, "WC": 0.56, "WA": 0.658, "WN": 0.631, "WM": 0.344, "WL": 0.304, "WK": 0.892, "WI": 0.305,
              "WH": 0.678, "WW": 0.0, "WV": 0.418, "WT": 0.638, "WS": 0.689, "WR": 0.968, "WQ": 0.538, "WP": 0.555,
              "WY": 0.204, "PR": 1.0, "PS": 0.196, "PP": 0.0, "PQ": 0.228, "PV": 0.244, "PW": 0.72, "PT": 0.161,
              "PY": 0.481, "PC": 0.179, "PA": 0.22, "PF": 0.515, "PG": 0.376, "PD": 0.852, "PE": 0.831, "PK": 0.875,
              "PH": 0.696, "PI": 0.363, "PN": 0.231, "PL": 0.357, "PM": 0.326, "CK": 0.887, "CI": 0.304, "CH": 0.66,
              "CN": 0.324, "CM": 0.277, "CL": 0.301, "CC": 0.0, "CA": 0.114, "CG": 0.32, "CF": 0.437, "CE": 0.838,
              "CD": 0.847, "CY": 0.457, "CS": 0.176, "CR": 1.0, "CQ": 0.341, "CP": 0.157, "CW": 0.639, "CV": 0.167,
              "CT": 0.233, "IY": 0.213, "VA": 0.275, "VC": 0.165, "VD": 0.9, "VE": 0.867, "VF": 0.269, "VG": 0.471,
              "IQ": 0.383, "IP": 0.311, "IS": 0.443, "IR": 1.0, "VL": 0.134, "IT": 0.396, "IW": 0.339, "IV": 0.133,
              "II": 0.0, "IH": 0.652, "IK": 0.892, "VS": 0.322, "IM": 0.057, "IL": 0.013, "VV": 0.0, "IN": 0.457,
              "IA": 0.403, "VY": 0.31, "IC": 0.296, "IE": 0.891, "ID": 0.942, "IG": 0.592, "IF": 0.134, "HY": 0.821,
              "HR": 0.697, "HS": 0.865, "HP": 0.777, "HQ": 0.716, "HV": 0.831, "HW": 0.981, "HT": 0.834, "HK": 0.566,
              "HH": 0.0, "HI": 0.848, "HN": 0.754, "HL": 0.842, "HM": 0.825, "HC": 0.836, "HA": 0.896, "HF": 0.907,
              "HG": 1.0, "HD": 0.629, "HE": 0.547, "NH": 0.78, "NI": 0.615, "NK": 0.891, "NL": 0.603, "NM": 0.588,
              "NN": 0.0, "NA": 0.424, "NC": 0.425, "ND": 0.838, "NE": 0.835, "NF": 0.766, "NG": 0.512, "NY": 0.641,
              "NP": 0.266, "NQ": 0.175, "NR": 1.0, "NS": 0.361, "NT": 0.368, "NV": 0.503, "NW": 0.945, "TY": 0.596,
              "TV": 0.345, "TW": 0.816, "TT": 0.0, "TR": 1.0, "TS": 0.185, "TP": 0.159, "TQ": 0.322, "TN": 0.315,
              "TL": 0.453, "TM": 0.403, "TK": 0.866, "TH": 0.737, "TI": 0.455, "TF": 0.604, "TG": 0.312, "TD": 0.83,
              "TE": 0.812, "TC": 0.261, "TA": 0.251, "AA": 0.0, "AC": 0.112, "AE": 0.827, "AD": 0.819, "AG": 0.208,
              "AF": 0.54, "AI": 0.407, "AH": 0.696, "AK": 0.891, "AM": 0.379, "AL": 0.406, "AN": 0.318, "AQ": 0.372,
              "AP": 0.191, "AS": 0.094, "AR": 1.0, "AT": 0.22, "AW": 0.739, "AV": 0.273, "AY": 0.552, "VK": 0.889}


DISTANCE_1_POW_2 = {(aa_pair[0], aa_pair[1]): math.pow(distance, 2) for aa_pair, distance in DISTANCE_1.items()}


# Distance is the Grantham chemical distance matrix used by Grantham et. al.
DISTANCE_2 = {"GW": 0.923, "GV": 0.464, "GT": 0.272, "GS": 0.158, "GR": 1.0, "GQ": 0.467, "GP": 0.323, "GY": 0.728,
              "GG": 0.0, "GF": 0.727, "GE": 0.807, "GD": 0.776, "GC": 0.312, "GA": 0.206, "GN": 0.381, "GM": 0.557,
              "GL": 0.591, "GK": 0.894, "GI": 0.592, "GH": 0.769, "ME": 0.879, "MD": 0.932, "MG": 0.569, "MF": 0.182,
              "MA": 0.383, "MC": 0.276, "MM": 0.0, "ML": 0.062, "MN": 0.447, "MI": 0.058, "MH": 0.648, "MK": 0.884,
              "MT": 0.358, "MW": 0.391, "MV": 0.12, "MQ": 0.372, "MP": 0.285, "MS": 0.417, "MR": 1.0, "MY": 0.255,
              "FP": 0.42, "FQ": 0.459, "FR": 1.0, "FS": 0.548, "FT": 0.499, "FV": 0.252, "FW": 0.207, "FY": 0.179,
              "FA": 0.508, "FC": 0.405, "FD": 0.977, "FE": 0.918, "FF": 0.0, "FG": 0.69, "FH": 0.663, "FI": 0.128,
              "FK": 0.903, "FL": 0.131, "FM": 0.169, "FN": 0.541, "SY": 0.615, "SS": 0.0, "SR": 1.0, "SQ": 0.358,
              "SP": 0.181, "SW": 0.827, "SV": 0.342, "ST": 0.174, "SK": 0.883, "SI": 0.478, "SH": 0.718, "SN": 0.289,
              "SM": 0.44, "SL": 0.474, "SC": 0.185, "SA": 0.1, "SG": 0.17, "SF": 0.622, "SE": 0.812, "SD": 0.801,
              "YI": 0.23, "YH": 0.678, "YK": 0.904, "YM": 0.268, "YL": 0.219, "YN": 0.512, "YA": 0.587, "YC": 0.478,
              "YE": 0.932, "YD": 1.0, "YG": 0.782, "YF": 0.202, "YY": 0.0, "YQ": 0.404, "YP": 0.444, "YS": 0.612,
              "YR": 0.995, "YT": 0.557, "YW": 0.244, "YV": 0.328, "LF": 0.139, "LG": 0.596, "LD": 0.944, "LE": 0.892,
              "LC": 0.296, "LA": 0.405, "LN": 0.452, "LL": 0.0, "LM": 0.062, "LK": 0.893, "LH": 0.653, "LI": 0.013,
              "LV": 0.133, "LW": 0.341, "LT": 0.397, "LR": 1.0, "LS": 0.443, "LP": 0.309, "LQ": 0.376, "LY": 0.205,
              "RT": 0.808, "RV": 0.914, "RW": 1.0, "RP": 0.796, "RQ": 0.668, "RR": 0.0, "RS": 0.86, "RY": 0.859,
              "RD": 0.305, "RE": 0.225, "RF": 0.977, "RG": 0.928, "RA": 0.919, "RC": 0.905, "RL": 0.92, "RM": 0.908,
              "RN": 0.69, "RH": 0.498, "RI": 0.929, "RK": 0.141, "VH": 0.649, "VI": 0.135, "EM": 0.83, "EL": 0.854,
              "EN": 0.599, "EI": 0.86, "EH": 0.406, "EK": 0.143, "EE": 0.0, "ED": 0.133, "EG": 0.779, "EF": 0.932,
              "EA": 0.79, "EC": 0.788, "VM": 0.12, "EY": 0.837, "VN": 0.38, "ET": 0.682, "EW": 1.0, "EV": 0.824,
              "EQ": 0.598, "EP": 0.688, "ES": 0.726, "ER": 0.234, "VP": 0.212, "VQ": 0.339, "VR": 1.0, "VT": 0.305,
              "VW": 0.472, "KC": 0.871, "KA": 0.889, "KG": 0.9, "KF": 0.957, "KE": 0.149, "KD": 0.279, "KK": 0.0,
              "KI": 0.899, "KH": 0.438, "KN": 0.667, "KM": 0.871, "KL": 0.892, "KS": 0.825, "KR": 0.154, "KQ": 0.639,
              "KP": 0.757, "KW": 1.0, "KV": 0.882, "KT": 0.759, "KY": 0.848, "DN": 0.56, "DL": 0.841, "DM": 0.819,
              "DK": 0.249, "DH": 0.435, "DI": 0.847, "DF": 0.924, "DG": 0.697, "DD": 0.0, "DE": 0.124, "DC": 0.742,
              "DA": 0.729, "DY": 0.836, "DV": 0.797, "DW": 1.0, "DT": 0.649, "DR": 0.295, "DS": 0.667, "DP": 0.657,
              "DQ": 0.584, "QQ": 0.0, "QP": 0.272, "QS": 0.461, "QR": 1.0, "QT": 0.389, "QW": 0.831, "QV": 0.464,
              "QY": 0.522, "QA": 0.512, "QC": 0.462, "QE": 0.861, "QD": 0.903, "QG": 0.648, "QF": 0.671, "QI": 0.532,
              "QH": 0.765, "QK": 0.881, "QM": 0.505, "QL": 0.518, "QN": 0.181, "WG": 0.829, "WF": 0.196, "WE": 0.931,
              "WD": 1.0, "WC": 0.56, "WA": 0.658, "WN": 0.631, "WM": 0.344, "WL": 0.304, "WK": 0.892, "WI": 0.305,
              "WH": 0.678, "WW": 0.0, "WV": 0.418, "WT": 0.638, "WS": 0.689, "WR": 0.968, "WQ": 0.538, "WP": 0.555,
              "WY": 0.204, "PR": 1.0, "PS": 0.196, "PP": 0.0, "PQ": 0.228, "PV": 0.244, "PW": 0.72, "PT": 0.161,
              "PY": 0.481, "PC": 0.179, "PA": 0.22, "PF": 0.515, "PG": 0.376, "PD": 0.852, "PE": 0.831, "PK": 0.875,
              "PH": 0.696, "PI": 0.363, "PN": 0.231, "PL": 0.357, "PM": 0.326, "CK": 0.887, "CI": 0.304, "CH": 0.66,
              "CN": 0.324, "CM": 0.277, "CL": 0.301, "CC": 0.0, "CA": 0.114, "CG": 0.32, "CF": 0.437, "CE": 0.838,
              "CD": 0.847, "CY": 0.457, "CS": 0.176, "CR": 1.0, "CQ": 0.341, "CP": 0.157, "CW": 0.639, "CV": 0.167,
              "CT": 0.233, "IY": 0.213, "VA": 0.275, "VC": 0.165, "VD": 0.9, "VE": 0.867, "VF": 0.269, "VG": 0.471,
              "IQ": 0.383, "IP": 0.311, "IS": 0.443, "IR": 1.0, "VL": 0.134, "IT": 0.396, "IW": 0.339, "IV": 0.133,
              "II": 0.0, "IH": 0.652, "IK": 0.892, "VS": 0.322, "IM": 0.057, "IL": 0.013, "VV": 0.0, "IN": 0.457,
              "IA": 0.403, "VY": 0.31, "IC": 0.296, "IE": 0.891, "ID": 0.942, "IG": 0.592, "IF": 0.134, "HY": 0.821,
              "HR": 0.697, "HS": 0.865, "HP": 0.777, "HQ": 0.716, "HV": 0.831, "HW": 0.981, "HT": 0.834, "HK": 0.566,
              "HH": 0.0, "HI": 0.848, "HN": 0.754, "HL": 0.842, "HM": 0.825, "HC": 0.836, "HA": 0.896, "HF": 0.907,
              "HG": 1.0, "HD": 0.629, "HE": 0.547, "NH": 0.78, "NI": 0.615, "NK": 0.891, "NL": 0.603, "NM": 0.588,
              "NN": 0.0, "NA": 0.424, "NC": 0.425, "ND": 0.838, "NE": 0.835, "NF": 0.766, "NG": 0.512, "NY": 0.641,
              "NP": 0.266, "NQ": 0.175, "NR": 1.0, "NS": 0.361, "NT": 0.368, "NV": 0.503, "NW": 0.945, "TY": 0.596,
              "TV": 0.345, "TW": 0.816, "TT": 0.0, "TR": 1.0, "TS": 0.185, "TP": 0.159, "TQ": 0.322, "TN": 0.315,
              "TL": 0.453, "TM": 0.403, "TK": 0.866, "TH": 0.737, "TI": 0.455, "TF": 0.604, "TG": 0.312, "TD": 0.83,
              "TE": 0.812, "TC": 0.261, "TA": 0.251, "AA": 0.0, "AC": 0.112, "AE": 0.827, "AD": 0.819, "AG": 0.208,
              "AF": 0.54, "AI": 0.407, "AH": 0.696, "AK": 0.891, "AM": 0.379, "AL": 0.406, "AN": 0.318, "AQ": 0.372,
              "AP": 0.191, "AS": 0.094, "AR": 1.0, "AT": 0.22, "AW": 0.739, "AV": 0.273, "AY": 0.552, "VK": 0.889}


DISTANCE_2_POW_2 = {(aa_pair[0], aa_pair[1]): math.pow(distance, 2) for aa_pair, distance in DISTANCE_2.items()}
