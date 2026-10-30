"""Support polygons on the floor: signed distances, the incentre of a triangle."""
import numpy as np


def ccw(points):
    """The points of a convex polygon in counter-clockwise order (about their mean)."""
    P = np.asarray(points, float)
    c = P.mean(axis=0)
    ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    return P[np.argsort(ang)]


def signed_distance(poly, x):
    """Signed distance of point x from the boundary of the convex polygon poly
    (counter-clockwise): the distance to the boundary inside, minus the
    Euclidean distance to the polygon outside. It is concave and has slope at
    most 1 in every direction."""
    P = np.asarray(poly, float)
    x = np.asarray(x, float)
    n = len(P)
    inside = np.inf
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        e = b - a
        nrm = np.array([-e[1], e[0]]) / np.hypot(e[0], e[1])    # inward normal for ccw
        inside = min(inside, float(np.dot(x - a, nrm)))
    if inside >= 0.0:
        return inside
    out = np.inf
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        e = b - a
        t = min(1.0, max(0.0, float(np.dot(x - a, e) / np.dot(e, e))))
        out = min(out, float(np.linalg.norm(x - (a + t * e))))
    return -out


def inner_distance_box(poly, centre, half):
    """Least signed distance over the axis-aligned box centre +- half
    (its corners bind for a convex polygon's edge lines)."""
    cx, cy = centre
    hx, hy = half
    return min(signed_distance(poly, (cx + sx * hx, cy + sy * hy)) for sx in (-1, 1) for sy in (-1, 1))


def incentre(A, B, C):
    A, B, C = (np.asarray(p, float) for p in (A, B, C))
    a, b, c = np.linalg.norm(B - C), np.linalg.norm(A - C), np.linalg.norm(A - B)
    I = (a * A + b * B + c * C) / (a + b + c)
    s = 0.5 * (a + b + c)
    area = 0.5 * abs((B - A)[0] * (C - A)[1] - (B - A)[1] * (C - A)[0])
    return I, area / s
