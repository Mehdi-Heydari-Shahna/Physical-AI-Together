"""
The yard of Chapter 13: the route, its lane, its grounds and its obstacles.

Physical AI, Together, Chapter 13 (Section 13.1).

World frame: x east, y north, z up; the ground's surface is z = 0. The route
is a centre line of three pieces: a straight towards the east on asphalt, a
quarter turn to the left on gravel, and a straight towards the north that
crosses from gravel onto wet grass and ends at the face of a loading dock.
Its arc length s is the coordinate along the route; the dock's face is at
s = 25 m, as the wall of Chapter 3 is at 25 m. The lane is 1.4 m wide,
centred on the route. Beside the lane stand three stacks of pallets; the
dock closes it.

The yard is one flat surface, x from -4 to 17 m and y from -4 to 16 m:
asphalt west of x = 10 m, gravel east of it and south of the line where the
route enters the wet grass, wet grass north of that line. The friction
coefficients of the grounds are the simulator's, chosen for the example;
they are not measurements of real surfaces. What a robot may assume about
them is the business of labels (Section 4.1.4).

The robot is in the lane while every point of its footprint, a rectangle of
the A300's length and width (user manual) centred on base_link and aligned
with its heading, is within the lane's half width of the centre line
(footprint_offset).
"""
import dataclasses
import math

STRAIGHT_1 = 10.0             # m, asphalt, towards the east
RADIUS = 3.0                  # m, the quarter turn to the left
TURN = 0.5 * math.pi * RADIUS
DOCK_S = 25.0                 # m, the face of the dock along the route
STRAIGHT_2 = DOCK_S - STRAIGHT_1 - TURN
LANE_HALF_WIDTH = 0.7         # m
FOOTPRINT = (0.990, 0.698)    # m, the A300's length and width (user manual)
GRASS_S = 18.0                # m, where the gravel gives way to wet grass
GRASS_Y = RADIUS + (GRASS_S - STRAIGHT_1 - TURN)


@dataclasses.dataclass(frozen=True)
class Ground:
    name: str
    friction: float           # Coulomb coefficient of the simulator's contacts
    s_from: float             # m, along the route
    s_to: float
    rect: tuple               # (x0, x1, y0, y1) of the slab that carries it [m]
    rgba: tuple


YARD = (-4.0, 17.0, -4.0, 16.0)  # (x0, x1, y0, y1) [m]
GROUNDS = (
    Ground("asphalt", 0.8, 0.0, STRAIGHT_1, (-4.0, STRAIGHT_1, -4.0, 16.0),
           (0.30, 0.30, 0.32, 1.0)),
    Ground("gravel", 0.6, STRAIGHT_1, GRASS_S, (STRAIGHT_1, 17.0, -4.0, GRASS_Y),
           (0.62, 0.58, 0.50, 1.0)),
    Ground("wet grass", 0.35, GRASS_S, DOCK_S, (STRAIGHT_1, 17.0, GRASS_Y, 16.0),
           (0.30, 0.48, 0.25, 1.0)),
)


def point(s):
    """(x, y, heading) of the centre line at arc length s; beyond the ends the
    straights continue."""
    if s <= STRAIGHT_1:
        return s, 0.0, 0.0
    if s <= STRAIGHT_1 + TURN:
        a = (s - STRAIGHT_1) / RADIUS
        return STRAIGHT_1 + RADIUS * math.sin(a), RADIUS - RADIUS * math.cos(a), a
    return STRAIGHT_1 + RADIUS, RADIUS + (s - STRAIGHT_1 - TURN), 0.5 * math.pi


def curvature(s):
    return 1.0 / RADIUS if STRAIGHT_1 < s < STRAIGHT_1 + TURN else 0.0


def project(x, y):
    """(s, d): the arc length of the nearest point of the centre line, and the
    signed distance from it, positive to the left of the route."""
    candidates = []
    # first straight
    s1 = min(max(x, -50.0), STRAIGHT_1)
    candidates.append((s1, y, math.hypot(x - s1, y)))
    # the turn, centred at (STRAIGHT_1, RADIUS)
    cx, cy = STRAIGHT_1, RADIUS
    a = math.atan2(x - cx, -(y - cy))          # 0 at the start of the turn
    a = min(max(a, 0.0), 0.5 * math.pi)
    px, py = cx + RADIUS * math.sin(a), cy - RADIUS * math.cos(a)
    r = math.hypot(x - cx, y - cy)
    candidates.append((STRAIGHT_1 + RADIUS * a, RADIUS - r, math.hypot(x - px, y - py)))
    # second straight, x = STRAIGHT_1 + RADIUS
    s3 = min(max(y - RADIUS, 0.0), 50.0)
    candidates.append((STRAIGHT_1 + TURN + s3, -(x - STRAIGHT_1 - RADIUS),
                       math.hypot(x - STRAIGHT_1 - RADIUS, y - RADIUS - s3)))
    s, d, _ = min(candidates, key=lambda c: c[2])
    return s, d


def footprint_offset(x, y, heading, spacing=0.1):
    """The largest distance from the centre line of a point of the robot's
    footprint (FOOTPRINT, centred at (x, y), along the heading), sampled
    along its edges at most spacing apart; between two samples the lane's
    edges, straight or of radius 2.3 m and more, depart from a chord by
    less than 1 mm."""
    length, width = FOOTPRINT
    c, s = math.cos(heading), math.sin(heading)
    n_l, n_w = math.ceil(length / spacing), math.ceil(width / spacing)
    pts = [(-length / 2 + length * i / n_l, side * width / 2) for i in range(n_l + 1) for side in (1, -1)]
    pts += [(side * length / 2, -width / 2 + width * j / n_w) for j in range(1, n_w) for side in (1, -1)]
    return max(abs(project(x + c * a - s * b, y + s * a + c * b)[1]) for a, b in pts)


def ground_at(s):
    """The ground under the centre line at arc length s."""
    for g in GROUNDS:
        if g.s_from <= s < g.s_to:
            return g
    return GROUNDS[-1] if s >= GROUNDS[-1].s_to else GROUNDS[0]


def ground_at_xy(x, y):
    """The ground at the point (x, y) of the yard (asphalt outside it)."""
    if x < STRAIGHT_1:
        return GROUNDS[0]
    return GROUNDS[1] if y < GRASS_Y else GROUNDS[2]


@dataclasses.dataclass(frozen=True)
class Obstacle:
    name: str
    center: tuple             # (x, y, z) [m]
    half: tuple               # half sizes (x, y, z) [m]
    yaw: float = 0.0          # rad, about z


DOCK_X, DOCK_Y = STRAIGHT_1 + RADIUS, RADIUS + STRAIGHT_2
OBSTACLES = (
    # the loading dock, whose face is at s = 25 m
    Obstacle("dock", (DOCK_X, DOCK_Y + 0.5, 0.6), (2.5, 0.5, 0.6)),
    # pallets beside the lane, 0.1 m outside its edge
    Obstacle("pallets on the first straight", (6.0, -(LANE_HALF_WIDTH + 0.1 + 0.6), 0.5),
             (1.0, 0.6, 0.5)),
    Obstacle("pallets outside the turn",
             (STRAIGHT_1 + (RADIUS + LANE_HALF_WIDTH + 0.1 + 0.4) * math.sin(math.pi / 4),
              RADIUS - (RADIUS + LANE_HALF_WIDTH + 0.1 + 0.4) * math.cos(math.pi / 4), 0.5),
             (0.4, 0.4, 0.5), math.pi / 4),
    Obstacle("pallets on the second straight",
             (DOCK_X + LANE_HALF_WIDTH + 0.1 + 0.6, 20.0 - STRAIGHT_1 - TURN + RADIUS, 0.5),
             (0.6, 1.0, 0.5)),
)
