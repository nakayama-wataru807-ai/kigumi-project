#!/usr/bin/env python3
"""
Standalone wrapper around vis_utils.plot_force_evolution().

Each *_surf.vtu file produced by PolyFEM contains:
  - sidesets       (Float64, per node) – which boundary id the node belongs to
  - traction_force (Float64, 3 comp.)  – per-node lumped traction forces
                                         (already integrated over nodal area)

Summing traction_force over all nodes assigned to FORCE_SIDESET gives the
resultant force vector at that step.

Unit note:  when the mesh is in mm and the elastic modulus in GPa,
  1 GPa·mm² = 10⁹ N/m² × (10⁻³ m)² = 10³ N = 1 kN
So the force axis is labelled "kN" below; adjust FORCE_UNIT_LABEL if your mesh
uses different length / pressure units.

Usage:
    python scripts/plot_force_evolution.py

Output:
    Saves force_evolution.png in <SIM_DIR>/output by default.
"""

import os

from vis_utils import plot_force_evolution as save_force_evolution_plot

# ── CONFIG ────────────────────────────────────────────────────────────────────

SIM_DIR    = (
    "/Users/quentinbecker/Library/CloudStorage/"
    "GoogleDrive-quentinbecker@g.ecc.u-tokyo.ac.jp/"
    "My Drive/kigumi-project/simulations/bending/"
    "0526_1609_layered_simulation/"
)
OUTPUT_DIR = os.path.join(SIM_DIR, "output")

# All steps to consider
STEPS = list(range(11))          # 0, 1, 2, …, 10

# Boundary sideset id of the loaded (Neumann) face
FORCE_SIDESET = 1

# Human-readable unit label for the force axis
FORCE_UNIT_LABEL = "kN  (GPa·mm²)"

def main():
    save_force_evolution_plot(
        SIM_DIR,
        OUTPUT_DIR,
        STEPS,
        sideset_id=FORCE_SIDESET,
        force_unit_label=FORCE_UNIT_LABEL,
    )


if __name__ == "__main__":
    main()
