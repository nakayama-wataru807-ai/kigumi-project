#!/usr/bin/env python3
"""
Plot the total applied force on a Neumann boundary sideset as a function of
simulation step.

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
"""

import base64
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────

SIM_DIR    = (
    "/Users/quentinbecker/Library/CloudStorage/"
    "GoogleDrive-quentinbecker@g.ecc.u-tokyo.ac.jp/"
    "My Drive/kigumi-project/simulations/bending/"
    "0525_2343_layered_simulation/"
)
OUTPUT_DIR = "/Users/quentinbecker/Research/kigumi-project/output/"

# All steps to consider
STEPS = list(range(11))          # 0, 1, 2, …, 10

# Boundary sideset id of the loaded (Neumann) face
FORCE_SIDESET = 1

# Human-readable unit label for the force axis
FORCE_UNIT_LABEL = "kN  (GPa·mm²)"

# ── VTU HELPERS ───────────────────────────────────────────────────────────────

_VTK_DTYPE = {
    "Float32": np.float32, "Float64": np.float64,
    "Int32":   np.int32,   "Int64":   np.int64,
    "UInt32":  np.uint32,  "UInt64":  np.uint64,
}


def _decode_array(elem, dtype=None):
    """Decode a base64-binary VTK DataArray (header_type UInt64)."""
    if dtype is None:
        dtype = _VTK_DTYPE[elem.attrib["type"]]
    raw = base64.b64decode(elem.text.strip())[8:]   # skip 8-byte UInt64 header
    return np.frombuffer(raw, dtype=dtype)


# ── FORCE EXTRACTION ──────────────────────────────────────────────────────────

def load_force_at_step(surf_vtu_path, sideset_id):
    """
    Read the total applied force on *sideset_id* from a single surface VTU.

    Returns a (3,) float64 array [Fx, Fy, Fz], or None if the file / fields
    are missing.
    """
    if not os.path.isfile(surf_vtu_path):
        print(f"[WARN] Not found: {surf_vtu_path}")
        return None

    root  = ET.parse(surf_vtu_path).getroot()
    piece = root.find(".//Piece")
    n_pts = int(piece.attrib["NumberOfPoints"])

    ss_elem = piece.find(".//*[@Name='sidesets']")
    tf_elem = piece.find(".//*[@Name='traction_force']")
    if ss_elem is None or tf_elem is None:
        print(f"[WARN] sidesets / traction_force not found in {surf_vtu_path}")
        return None

    sidesets = _decode_array(ss_elem).astype(np.float64)
    traction = _decode_array(tf_elem).astype(np.float64).reshape(n_pts, 3)

    mask = np.round(sidesets).astype(int) == sideset_id
    if not mask.any():
        print(f"[WARN] No nodes found on sideset {sideset_id} in {surf_vtu_path}")
        return np.zeros(3)

    return traction[mask].sum(axis=0)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    steps  = []
    forces = []

    for step in STEPS:
        path = os.path.join(SIM_DIR, f"step_{step}_surf.vtu")
        f    = load_force_at_step(path, FORCE_SIDESET)
        if f is not None:
            steps.append(step)
            forces.append(f)
            print(f"Step {step:2d}:  Fx={f[0]:+9.4g}  Fy={f[1]:+9.4g}  Fz={f[2]:+9.4g}")

    if not steps:
        print("[WARN] No force data found – nothing to plot.")
        return

    steps  = np.array(steps)
    forces = np.array(forces)   # (N, 3)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(steps, forces[:, 0], "o-", color="tomato",    linewidth=2,
            markersize=6, label=r"$F_x$")
    ax.plot(steps, forces[:, 1], "s-", color="steelblue", linewidth=2,
            markersize=6, label=r"$F_y$  (applied, Neumann BC)")
    ax.plot(steps, forces[:, 2], "^-", color="seagreen",  linewidth=2,
            markersize=6, label=r"$F_z$")

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    ax.set_xlabel("Simulation step", fontsize=12)
    ax.set_ylabel(f"Force  [{FORCE_UNIT_LABEL}]", fontsize=12)
    ax.set_title(f"Applied force on sideset {FORCE_SIDESET}", fontsize=13)
    ax.set_xticks(STEPS)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "force_evolution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nForce evolution plot saved → {out_path}")


main()
