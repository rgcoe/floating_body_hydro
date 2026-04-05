# -*- coding: utf-8 -*-
import rhinoscriptsyntax as rs
import math

rs.EnableRedraw(False)

# -------------------------------------------------
# Parameters
# -------------------------------------------------
length = 155.0
beam = 20.0
draft = 9.5
stations = 18

flare = 0.35
mid_start = 0.30
mid_end   = 0.70

# -------------------------------------------------
# Layers
# -------------------------------------------------
curve_layer = "Hull::Stations_Starboard_BWL"
surface_layer = "Hull::Surface_Starboard_BWL"

if not rs.IsLayer(curve_layer):
    rs.AddLayer(curve_layer)

if not rs.IsLayer(surface_layer):
    rs.AddLayer(surface_layer)

# -------------------------------------------------
# Longitudinal fullness
# -------------------------------------------------
def fullness(t):
    if t < mid_start:
        return math.sin((t / mid_start) * math.pi * 0.5) ** 1.8
    elif t > mid_end:
        return math.sin(((1 - t) / (1 - mid_end)) * math.pi * 0.5) ** 1.3
    else:
        return 1.0

# -------------------------------------------------
# Starboard-only, below-waterline section
# -------------------------------------------------
def section_points(x, f, t):
    hb = 0.5 * beam * f
    d  = draft * f

    flare_y = flare * hb * (1 - f)

    return [
        (x, hb * 0.95 + flare_y, 0.0),  # waterline
        (x, hb, -d * 0.25),             # chine
        (x, hb * 0.65, -d * 0.75),
        (x, 0.0, -d)                    # keel
    ]

# -------------------------------------------------
# Build station curves
# -------------------------------------------------
curves = []
rs.CurrentLayer(curve_layer)

for i in range(stations):
    t = float(i) / (stations - 1)
    x = t * length
    f = fullness(t)

    # Fine bow
    if t < 0.04:
        f *= t / 0.04

    pts = section_points(x, f, t)
    crv = rs.AddInterpCurve(pts, degree=3)
    curves.append(crv)

# -------------------------------------------------
# Loft hull surface
# -------------------------------------------------
rs.CurrentLayer(surface_layer)
hull = rs.AddLoftSrf(curves, loft_type=1)

rs.EnableRedraw(True)
print("Curves and surface created on separate layers.")
