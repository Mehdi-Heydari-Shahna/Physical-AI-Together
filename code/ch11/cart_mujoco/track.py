"""The profile of the simulated track and the track coordinate (Section 11.1).

The track is a vertical profile in the x-z plane of the simulator, x towards
the wall and z up: level firm ground up to the start of the ramp, a circular
crest of radius R that turns the ground down to the slope angle theta, and the
straight downhill slope. The ground coordinate s is the arc length along this
profile from the start. The axle of a wheel of radius r rolls along the
parallel curve at distance r above the ground, which over the crest is a
circle of radius R + r. The axle coordinate of a point, the arc length along
that curve to the point of it above the point's foot, equals s on the firm
ground and exceeds it by r * theta beyond the crest. The track coordinate p
of the cart is the axle coordinate of its front axle plus the distance by
which the bumper leads that axle.

Why the axle coordinate: where the curvature of the ground jumps, at both
ends of the crest, the rate of the ground coordinate of a moving axle jumps
by the factor R / (R + r), although the axle moves smoothly. A drive loop
that followed the ground coordinate would see a jump of 1 % in the speed of
the cart (Section 11.1.3). A loaded axle runs lower than r, by the
penetration of its wheel, so that the rate of its axle coordinate still
jumps there, by the factor (R + r) / (R + d), d its distance from the
ground: 49.3 um/s where the front axle reaches the crest in the run of
Chapter 3 (Section 11.1.4).
"""
import math
from collections import namedtuple

Foot = namedtuple("Foot", "s x z angle distance part")


class Profile:
    """Firm ground for s <= s0, a crest of radius R from s0 to s1, and a
    slope of angle theta beyond s1; axles at height r. Lengths in metres,
    angles in radians, the angle of the ground positive downhill."""

    def __init__(self, s0, radius, theta, r):
        self.s0, self.R, self.theta, self.r = s0, radius, theta, r
        self.s1 = s0 + radius * theta                    # end of the crest
        self.cx, self.cz = s0, -radius                   # centre of the crest
        self.bx = self.cx + radius * math.sin(theta)     # start of the slope
        self.bz = self.cz + radius * math.cos(theta)

    # --- the ground ----------------------------------------------------------
    def point(self, s):
        """(x, z, angle) of the ground at ground coordinate s."""
        if s <= self.s0:
            return s, 0.0, 0.0
        if s <= self.s1:
            phi = (s - self.s0) / self.R
            return (self.cx + self.R * math.sin(phi), self.cz + self.R * math.cos(phi), phi)
        t = s - self.s1
        return (self.bx + t * math.cos(self.theta), self.bz - t * math.sin(self.theta),
                self.theta)

    def normal(self, s):
        """Unit normal (x, z) of the ground at s, pointing out of the ground."""
        a = self.point(s)[2]
        return math.sin(a), math.cos(a)

    def foot(self, x, z):
        """The closest point of the ground to (x, z), which lies above it:
        ground coordinate, position, angle, distance and part."""
        cands = []
        if x <= self.s0:
            cands.append(Foot(x, x, 0.0, 0.0, z, "firm"))
        dx, dz = x - self.cx, z - self.cz
        phi = math.atan2(dx, dz)
        if 0.0 <= phi <= self.theta:
            cands.append(Foot(self.s0 + self.R * phi, self.cx + self.R * math.sin(phi),
                              self.cz + self.R * math.cos(phi), phi,
                              math.hypot(dx, dz) - self.R, "crest"))
        c, sn = math.cos(self.theta), math.sin(self.theta)
        t = (x - self.bx) * c - (z - self.bz) * sn
        if t >= 0.0:
            fx, fz = self.bx + t * c, self.bz - t * sn
            cands.append(Foot(self.s1 + t, fx, fz, self.theta,
                              (x - fx) * sn + (z - fz) * c, "slope"))
        if not cands:
            raise ValueError(f"({x}, {z}) has no foot on the ground")
        return min(cands, key=lambda f: abs(f.distance))

    # --- the axle coordinate -------------------------------------------------------
    def axle(self, s):
        """Axle coordinate above ground coordinate s."""
        if s <= self.s0:
            return s
        if s <= self.s1:
            return self.s0 + (s - self.s0) * (self.R + self.r) / self.R
        return s + self.r * self.theta

    def ground(self, a):
        """Ground coordinate below axle coordinate a (the inverse of axle)."""
        if a <= self.s0:
            return a
        if a <= self.s0 + (self.R + self.r) * self.theta:
            return self.s0 + (a - self.s0) * self.R / (self.R + self.r)
        return a - self.r * self.theta

    def axle_rate(self, foot, vx, vz):
        """Rate of the axle coordinate of a point with foot 'foot' that moves
        at (vx, vz): its velocity along the ground's tangent, scaled on the
        crest by (R + r) / (R + d), d its distance from the ground."""
        along = vx * math.cos(foot.angle) - vz * math.sin(foot.angle)
        if foot.part == "crest":
            return along * (self.R + self.r) / (self.R + foot.distance)
        return along

    def ground_rate(self, foot, vx, vz):
        """Rate of the ground coordinate of the foot, for the diagnostics:
        scaled on the crest by R / (R + d), it jumps where the curvature of
        the ground jumps."""
        along = vx * math.cos(foot.angle) - vz * math.sin(foot.angle)
        if foot.part == "crest":
            return along * self.R / (self.R + foot.distance)
        return along
