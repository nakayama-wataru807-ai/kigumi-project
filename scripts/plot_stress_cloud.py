"""
plot_stress_cloud.py
--------------------
Per-step stress point-cloud plots overlaid on failure envelopes.

One PNG per simulation step is saved to <output_dir>/stress_cloud_step_NNN.png.

Callable API
------------
    from plot_stress_cloud import generate_stress_clouds
    generate_stress_clouds(sim_dir, output_dir, fiber_dir, radial_dir,
                           strengths_mpa, steps=None,
                           normalise=True, slice_axes=(0, 1))

Standalone
----------
    conda run -n kigumi_env python scripts/plot_stress_cloud.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from plot_criteria import (
    _DEFAULT_STRENGTHS_MPa, _tw_coefficients,
    draw_failure_envelopes, save_square_figure,
)
from physics_utils import load_tet_stress_material_frame
from vis_utils import find_available_steps


def _full_tsai_wu(sl, sr, st, trt, tlt, tlr, s):
    """Full Tsai-Wu failure index from 6 material-frame stress components (MPa)."""
    f_lin, f_quad, f_cross = _tw_coefficients(s)
    f1, f2, f3 = f_lin[0], f_lin[1], f_lin[2]
    f11, f22, f33, f44, f55, f66 = f_quad
    f12, f13, f23 = f_cross[(0, 1)], f_cross[(0, 2)], f_cross[(1, 2)]
    return (
        f1*sl   + f2*sr   + f3*st
        + f11*sl**2 + f22*sr**2 + f33*st**2
        + f44*trt**2 + f55*tlt**2 + f66*tlr**2
        + 2*f12*sl*sr + 2*f13*sl*st + 2*f23*sr*st
    )


def generate_stress_clouds(sim_dir, output_dir,
                           fiber_dir, radial_dir,
                           strengths_mpa,
                           steps=None,
                           normalise=True,
                           slice_axes=(0, 1),
                           criteria=None,
                           transparent=True):
    """
    Generate one stress-cloud PNG per simulation step.

    Points are coloured by the *full* 6-component Tsai-Wu failure index.
    The failure-envelope contours show the 2-D slice specified by *slice_axes*.

    Parameters
    ----------
    sim_dir       : str  Path to folder containing step_N.vtu files.
    output_dir    : str  Destination for the PNG files.
    fiber_dir     : (3,) longitudinal (L) direction in global coords.
    radial_dir    : (3,) radial (R) direction in global coords.
    strengths_mpa : dict (MPa) with keys Xt, Xc, Yt, Yc, Zt, Zc, Slr, Slt,
                    Srt, Fstar_12, Fstar_13, Fstar_23.
    steps         : list[int] or None.  None -> detect all available steps.
    normalise     : True  -> axes in sigma/scale units.
                    False -> physical MPa axes.
    slice_axes    : (i, j) — which two of the 6 stress components to display.
                    Default (0, 1) = (sigmaL, sigmaR) plane.
    criteria      : list of envelope criteria to draw behind the cloud.
                    Default ['tsai_wu'] — only Tsai-Wu.
    transparent   : Save with a transparent background (default True).
    """
    if criteria is None:
        criteria = ['tsai_wu']
    s = {**_DEFAULT_STRENGTHS_MPa, **strengths_mpa}
    if steps is None:
        steps = find_available_steps(sim_dir)
    if not steps:
        print(f'[WARN] No steps found in {sim_dir}')
        return

    os.makedirs(output_dir, exist_ok=True)

    # Index names for human-readable printing
    _COMP_NAME = ['σL', 'σR', 'σT', 'τRT', 'τLT', 'τLR']
    i, j = int(slice_axes[0]), int(slice_axes[1])

    for step in steps:
        vol_path = os.path.join(sim_dir, f'step_{step}.vtu')
        result   = load_tet_stress_material_frame(vol_path, fiber_dir, radial_dir)

        if result[0] is None:
            print(f'  [skip] step {step}: no stress data')
            continue

        sl_gpa, sr_gpa, st_gpa, trt_gpa, tlt_gpa, tlr_gpa = result

        # PolyFEM writes stresses in GPa → convert to MPa
        sl, sr, st, trt, tlt, tlr = (v * 1000.0 for v in result)

        # Full Tsai-Wu failure index (all 6 components) for colouring
        tw_fi = _full_tsai_wu(sl, sr, st, trt, tlt, tlr, s)

        # Pick the two components for the scatter x/y axes
        comp_values = [sl, sr, st, trt, tlt, tlr]
        from plot_criteria import _hill_scale
        si_vals = comp_values[i]
        sj_vals = comp_values[j]
        scale_i = _hill_scale(i, s)
        scale_j = _hill_scale(j, s)

        x_pts = si_vals / scale_i if normalise else si_vals
        y_pts = sj_vals / scale_j if normalise else sj_vals

        fig, ax = plt.subplots(figsize=(7, 7))
        draw_failure_envelopes(ax, s, normalise=normalise, slice_axes=slice_axes,
                               criteria=criteria, show_labels=not transparent)

        safe   = tw_fi < 1.0
        failed = ~safe

        sc = ax.scatter(x_pts[safe], y_pts[safe], c=tw_fi[safe],
                        cmap='viridis', vmin=0, vmax=1,
                        s=6, alpha=0.6, zorder=4, rasterized=True)
        if failed.any():
            ax.scatter(x_pts[failed], y_pts[failed],
                       c='red', s=10, alpha=0.85, zorder=5,
                       rasterized=True, label=f'TW ≥ 1  ({failed.sum()} tets)')
            ax.legend(fontsize=8, loc='upper right')

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Tsai-Wu failure index (full 3-D)', fontsize=9)

        n_failed = int((tw_fi >= 1).sum())
        ax.set_title(
            rf'Stress cloud — step {step}  '
            rf'({_COMP_NAME[i]}–{_COMP_NAME[j]} plane, {len(sl)} tets)',
            fontsize=11, pad=10)

        outpath = os.path.join(output_dir,
                               f'stress_cloud_step_{step:03d}.png')
        save_square_figure(fig, ax, outpath, dpi=120, transparent=transparent)
        plt.close(fig)
        print(f'  step {step:3d}: {len(sl):5d} tets, {n_failed:4d} failed '
              f'(TW>=1)  ->  {os.path.basename(outpath)}')


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == '__main__':
    _SIM_DIR = (
        '/Users/quentinbecker/Library/CloudStorage/'
        'GoogleDrive-quentinbecker@g.ecc.u-tokyo.ac.jp/'
        'My Drive/kigumi-project/simulations/bending/'
        '0526_2143_layered_simulation'
    )
    _OUTPUT_DIR    = os.path.join(_SIM_DIR, 'output')
    _FIBER_DIR     = (1, 0, 0)
    _RADIAL_DIR    = (0, 1, 0)
    _NORMALISE     = True
    _SLICE_AXES    = (0, 1)   # sigmaL vs sigmaR
    _STRENGTHS_MPa = dict(Xt=90, Xc=50, Yt=4, Yc=8,
                          Zt=3, Zc=6, Slr=11, Slt=11, Srt=1.4,
                          Fstar_12=-0.5, Fstar_13=-0.5, Fstar_23=-0.5)

    print(f'Generating stress clouds for {os.path.basename(_SIM_DIR)} ...')
    generate_stress_clouds(
        _SIM_DIR, _OUTPUT_DIR,
        _FIBER_DIR, _RADIAL_DIR,
        _STRENGTHS_MPa,
        normalise=_NORMALISE,
        slice_axes=_SLICE_AXES,
    )
    print('Done.')
