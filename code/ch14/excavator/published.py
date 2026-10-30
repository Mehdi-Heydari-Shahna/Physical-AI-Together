"""
What is published about the Komatsu PC138US-8, and where.

Physical AI, Together, Chapter 14 (Section 14.1).

Every number of the excavator that the chapter takes from a source is here,
with the source's key; every other number of the model is ours (front.py,
mass.py). The sources:

  BROCHURE  Komatsu, "PC138US-8 Hydraulic Excavator", specification brochure
            CEN00183-06, printed 2014-03 (Japan); the copies on
            komatsu.jp (asia_b/download_crawlerexcavators/pc138us-8_cen00183-06.pdf)
            and komatsu.com.au are the same document. Read on 24 October 2026.
  WEBPAGE   Komatsu Australia, product page of the PC138US-8,
            https://www.komatsu.com/en-au/products/equipment/excavators/crawler-excavators/pc138us-8,
            read on 24 October 2026. It gives an operating weight of 14,700 kg
            without its configuration; the brochure's 13,480 to 13,850 kg
            (with the shoes and the front of the lifting tables) is the one
            used here. Only the arm cylinder's rod diameter is taken from it.
  TAMPERE   M. Ferlibas and R. Ghabcheloo, "Load weight estimation on an
            excavator in static and dynamic motions", Proceedings of the 17th
            Scandinavian International Conference on Fluid Power (SICFP'21),
            Linkoping, 2021: gravitational parameters identified on a
            PC138US-8 fitted with a tiltrotator (Table 1).
  ISO10567  ISO 10567:2007, Earth-moving machinery - Hydraulic excavators -
            Lift capacity (the sample pages of the standard): the tipping
            lines of a crawler machine, the lift point.
  ISO6015   ISO 6015:2006, Earth-moving machinery - Hydraulic excavators and
            backhoe loaders - Methods of determining tool forces (the sample
            pages): tool forces at the bucket lip, the force radii.
"""

# ----------------------------------------------------------------------------
# The machine of the brochure's lifting tables: 4.6 m one-piece boom, 2.5 m
# arm, 0.50 m^3 bucket, 500 mm shoes, the standard counterweight
# ----------------------------------------------------------------------------
OPERATING_WEIGHT = 13480.0     # [kg] BROCHURE, 500 mm shoes, with the front above, full
                               # fuel, lubricants, coolant, operator and standard equipment
COUNTERWEIGHT = 3250.0         # [kg] BROCHURE, standard
BOOM_LENGTH = 4.600            # [m] BROCHURE, one-piece boom
ARM_LENGTH = 2.500             # [m] BROCHURE
BUCKET_CAPACITY = 0.50         # [m^3] BROCHURE, SAE heaped
BUCKET_MASS = 399.0            # [kg] BROCHURE, the 0.50 m^3 bucket
BUCKET_WIDTH = 0.859           # [m] BROCHURE, without side cutters

TRACK_GAUGE = 1.990            # [m] BROCHURE
TRACK_ON_GROUND = 2.880        # [m] BROCHURE, length of track on ground
TRACK_LENGTH = 3.610           # [m] BROCHURE
SHOE_WIDTH = 0.500             # [m] BROCHURE, standard
OVERALL_WIDTH = 2.490          # [m] BROCHURE
TAIL_SWING_RADIUS = 1.480      # [m] BROCHURE
COUNTERWEIGHT_CLEARANCE = 0.900  # [m] BROCHURE, ground clearance of the counterweight
CAB_HEIGHT = 2.815             # [m] BROCHURE, overall height to the top of the cab

# working range, 2.5 m arm (BROCHURE)
MAX_DIGGING_HEIGHT = 9.340
MAX_DUMPING_HEIGHT = 6.840
MAX_DIGGING_DEPTH = 5.480
MAX_VERTICAL_WALL_DEPTH = 4.900
MAX_DIGGING_REACH = 8.300
MAX_REACH_AT_GROUND = 8.180
MIN_SWING_RADIUS = 1.980

# hydraulics (BROCHURE; the arm's rod: WEBPAGE)
CYLINDERS = {                  # number, bore [m], stroke [m], rod [m] or None
    "boom": (2, 0.105, 1.055, None),
    "arm": (1, 0.110, 1.175, 0.075),
    "bucket": (1, 0.095, 0.885, None),
}
RELIEF_IMPLEMENT = 34.8e6      # [Pa] implement and travel circuits
RELIEF_SWING = 27.1e6          # [Pa]
PUMP_MAX_FLOW = 241.5 / 60000.0  # [m^3/s] 241.5 L/min
SWING_SPEED = 11.0 * 2.0 * 3.141592653589793 / 60.0   # [rad/s] 11.0 rpm
TRAVEL_SPEED_HIGH = 5.1 / 3.6  # [m/s]
TRAVEL_SPEED_LOW = 2.9 / 3.6   # [m/s]

# tool forces, 2.5 m arm (BROCHURE): ISO and SAE ratings
BUCKET_FORCE_ISO = 93.2e3      # [N]
ARM_FORCE_ISO = 61.8e3         # [N]
BUCKET_FORCE_SAE = 81.4e3      # [N]
ARM_FORCE_SAE = 60.8e3         # [N]

# lifting capacity, 2.5 m arm (BROCHURE): ratings over the front (Cf) and over
# the side (Cs) [kg], by bucket hook height B [m] and reach from the swing
# centre A [m]; "*" (hydraulic=True) where the load is limited by the
# hydraulic capacity rather than by tipping. Ratings do not exceed 87 % of the
# hydraulic lift capacity or 75 % of the tipping load (SAE J/ISO 10567). The
# text extracted from the brochure does not show which cells of the rows at
# 6.1 m and -3.0 m are empty; their columns are those of the printed table as
# the book's author read it (24 October 2026): at B = 6.1 m the 3.0 m and 6.1 m
# cells are empty, at B = -3.0 m the 6.1 m cell. The column at the maximum
# reach does not give the reach and is kept apart (LIFT_TABLE_MAX). The model
# is fitted to the rows at 3.0 m and 0.0 m ("fit"); the rows at 6.1 m and
# -3.0 m test it ("test").
LIFT_TABLE = {
    # (B, A): (Cf, hydraulic, Cs, hydraulic, role)
    (6.1, 4.6): (3060, True, 3060, True, "test"),
    (3.0, 3.0): (5770, True, 5770, True, "fit"),
    (3.0, 4.6): (4320, True, 2990, False, "fit"),
    (3.0, 6.1): (2880, False, 1830, False, "fit"),
    (0.0, 3.0): (5630, True, 4840, False, "fit"),
    (0.0, 4.6): (4260, False, 2600, False, "fit"),
    (0.0, 6.1): (2690, False, 1660, False, "fit"),
    (-3.0, 3.0): (6040, True, 4820, False, "test"),
    (-3.0, 4.6): (4180, False, 2540, False, "test"),
}
LIFT_TABLE_MAX = {             # B: (Cf, hydraulic, Cs, hydraulic) at the maximum reach,
    6.1: (1690, True, 1690, True),       # whose reach the brochure does not give
    3.0: (1580, True, 1370, False),
    0.0: (1940, True, 1290, False),
    -3.0: (3000, False, 1850, False),
}
TIPPING_SHARE = 0.75           # ISO 10567 via BROCHURE
HYDRAULIC_SHARE = 0.87

# identified gravitational parameters of the PC138US-8 with a tiltrotator
# (TAMPERE, Table 1, static estimation) [kg m]; the bucket and tiltrotator
# together 750 kg. pi_s1 = M_bu r4 cos(a4), pi_s2 = M_bu r4 sin(a4),
# pi_s3 = M_bu a3 + M_st r3 cos(a3), pi_s4 = M_st r3 sin(a3),
# pi_s5 = (M_bu + M_st) a2 + M_bo r2 cos(a2), pi_s6 = M_bo r2 sin(a2);
# the paper's Table 5 (dynamic estimation of tau23) gives other values.
TAMPERE_ATTACHMENT = 750.0
TAMPERE_STATIC = {"pi_s1": 189.85, "pi_s2": 318.80, "pi_s3": 2753.27, "pi_s4": 132.33,
                  "pi_s5": 8297.89, "pi_s6": 1156.44}
TAMPERE_DYNAMIC = {"pi_s1": 298.04, "pi_s2": 592.86, "pi_s3": 8228.37, "pi_s4": 2571.05,
                   "pi_s5": 8927.32, "pi_s6": 1386.54}
