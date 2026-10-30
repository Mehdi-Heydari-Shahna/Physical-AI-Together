"""
One Apptronik Apollo carrying a beam: the two-loop controller of the
Team-Impedance-Control repository (Heydari Shahna, 2026) with one robot.

Physical AI, Together, Chapter 15 (Section 15.3).

  one_robot.py  the repository's code loaded from a checkout, and what one
                robot needs that the repository's team of two to four does
                not: a station for one robot, an aluminium beam of the HEA
                200 section, and the beam's reference following the
                pelvis's sideways shift in each weight shift
  qp.py         a solver of the load-sharing QP with daqp's interface, from
                NNLS (SciPy), for an environment without daqp
"""
