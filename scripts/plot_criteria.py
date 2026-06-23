"""
plot_criteria.py
----------------
Failure-envelope comparison plots (von Mises / Hill / Tsai-Wu) for orthotropic
wood-like materials.  Supports arbitrary 2-D slices of the 6-D stress space.

Stress component indexing (Voigt, material frame):
    0: σL   1: σR   2: σT   3: τRT   4: τLT   5: τLR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Standalone CLI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    python scripts/plot_criteria.py [OPTIONS]

Options
-------
--criteria CRITERION [...]
    Which envelopes to draw.  Any subset of:  von_mises  hill  tsai_wu
    Default: all three.

--axes I J
    Stress-component indices for the 2-D slice (default: 0 1 = σL–σR).
        0=σL  1=σR  2=σT  3=τRT  4=τLT  5=τLR

--normalise / --no-normalise
    Normalised axes (σ/scale, dimensionless) or physical MPa axes.
    Omit the flag entirely to generate BOTH variants.

--lim FLOAT
    Half-range of the plot (default: 1.8).
    Normalised: ±lim in scaled units.
    Physical:   ±lim × max(scale_i, scale_j) MPa.
    This value is fixed regardless of which criteria are active so that
    multiple figures (e.g. von Mises only vs all three) are directly comparable.

--no-labels
    Suppress axis labels, title, intercept annotations, and legend.
    Useful when compositing into a larger multi-panel figure.

--out PATH
    Output file path.
    Default: output/failure_criteria_<axes>_<normalised|physical>.png
    (relative to the repository root).
    Ignored when --normalise is omitted (both variants are saved automatically).

--dpi INT
    Output resolution (default: 150).

Examples
--------
# Default: all criteria, σL–σR plane, both normalised and physical
python scripts/plot_criteria.py

# Von Mises only, normalised
python scripts/plot_criteria.py --criteria von_mises --normalise

# Hill + Tsai-Wu, σL–σT plane, physical axes, no labels
python scripts/plot_criteria.py --criteria hill tsai_wu --axes 0 2 --no-normalise --no-labels

# Custom output path and resolution
python scripts/plot_criteria.py --criteria tsai_wu --normalise --out output/tw_only.png --dpi 300

# Wider axis range for a zoomed-out view
python scripts/plot_criteria.py --lim 2.5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Callable API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from plot_criteria import draw_failure_envelopes, save_square_figure

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_failure_envelopes(ax,
        criteria=['von_mises'],          # subset, or None for all three
        normalise=True,
        slice_axes=(0, 1),               # (σL, σR) plane
        show_labels=False,               # hide annotations/legend
    )
    save_square_figure(fig, ax, 'out.png')
"""

import matplotlib; matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Component metadata ────────────────────────────────────────────────────────
_COMP_LATEX = [
    r'\sigma_L', r'\sigma_R', r'\sigma_T',
    r'\tau_{RT}', r'\tau_{LT}', r'\tau_{LR}',
]
_COMP_NORMAL = frozenset({0, 1, 2})   # indices of normal-stress components

# Per-component: (tension_key, compression_key)
# Shear components have compression_key = None (symmetric ±S)
_COMP_STRENGTH = [
    ('Xt', 'Xc'), ('Yt', 'Yc'), ('Zt', 'Zc'),
    ('Srt', None), ('Slt', None), ('Slr', None),
]

_ALL_CRITERIA = ('von_mises', 'hill', 'tsai_hill', 'tsai_wu')

_DEFAULT_STRENGTHS_MPa = dict(
    Xt=90, Xc=50,    # longitudinal (L) — along grain
    Yt=4,  Yc=8,     # radial       (R)
    Zt=3,  Zc=6,     # tangential   (T)
    Slr=11, Slt=11,  # shear parallel to grain
    Srt=1.4,         # rolling shear
    Fstar_12=-0.5,   # Tsai-Hahn interaction coefficients
    Fstar_13=-0.5,
    Fstar_23=-0.5,
)


def _hill_scale(idx, s):
    """Geometric-mean (Hill) reference strength for component *idx* (MPa)."""
    kt, kc = _COMP_STRENGTH[idx]
    return np.sqrt(s[kt] * s[kc]) if kc is not None else s[kt]


def _tw_coefficients(s):
    """
    All Tsai-Wu coefficients from strengths dict (MPa).

    Returns
    -------
    f_lin  : (6,) array — linear terms fi
    f_quad : (6,) array — quadratic terms fii
    f_cross: dict (i,j)->fij, i<j — interaction terms (normal pairs only)
    """
    f1   = 1/s['Xt']  - 1/s['Xc']
    f2   = 1/s['Yt']  - 1/s['Yc']
    f3   = 1/s['Zt']  - 1/s['Zc']
    f11  = 1/(s['Xt'] * s['Xc'])
    f22  = 1/(s['Yt'] * s['Yc'])
    f33  = 1/(s['Zt'] * s['Zc'])
    f44  = 1/s['Srt']**2
    f55  = 1/s['Slt']**2
    f66  = 1/s['Slr']**2
    f12  = s['Fstar_12'] * np.sqrt(f11 * f22)
    f13  = s['Fstar_13'] * np.sqrt(f11 * f33)
    f23  = s['Fstar_23'] * np.sqrt(f22 * f33)
    return (
        np.array([f1,  f2,  f3,  0.0, 0.0, 0.0]),
        np.array([f11, f22, f33, f44, f55, f66]),
        {(0, 1): f12, (0, 2): f13, (1, 2): f23},
    )


def _axis_label(idx, scale, normalise):
    """Axis-label string for stress component *idx*."""
    comp = _COMP_LATEX[idx]
    if normalise:
        scale_names = ['X_H', 'Y_H', 'Z_H', 'S_{RT}', 'S_{LT}', 'S_{LR}']
        sn  = scale_names[idx]
        fmt = '.0f' if scale >= 10 else '.1f'
        return rf'${comp}\,/\,{sn}$  (${sn}={scale:{fmt}}$ MPa)'
    return rf'${comp}$ (MPa)'


def _intercept_annotations(idx, s, scale, normalise, on_x_axis):
    """
    Return list of (label, plot_x, plot_y, xytext, ha, va) for the
    strength-intercept markers of component *idx*.
    """
    _SYM   = ['X', 'Y', 'Z', 'S_{RT}', 'S_{LT}', 'S_{LR}']
    _HNAME = ['X_H', 'Y_H', 'Z_H', 'S_{RT}', 'S_{LT}', 'S_{LR}']

    kt, kc = _COMP_STRENGTH[idx]
    sym  = _SYM[idx]
    hname = _HNAME[idx]

    if kc is not None:                      # normal stress — T/C asymmetry
        vp = s[kt];  vn = -s[kc]
        if normalise:
            lp = rf'${sym}_T/{hname}$'
            ln = rf'$-{sym}_C/{hname}$'
        else:
            lp = rf'${sym}_T$'
            ln = rf'$-{sym}_C$'
    else:                                   # shear — symmetric ±S
        vp = s[kt];  vn = -s[kt]
        if normalise:
            lp = r'$+1$'
            ln = r'$-1$'
        else:
            lp = rf'${sym}$'
            ln = rf'$-{sym}$'

    pp = vp / scale if normalise else vp
    pn = vn / scale if normalise else vn

    if on_x_axis:
        return [
            (lp, pp, 0,  ( 6,  0), 'left',   'center'),
            (ln, pn, 0,  (-6,  0), 'right',  'center'),
        ]
    return [
        (lp, 0,  pp, ( 0,  6), 'center', 'bottom'),
        (ln, 0,  pn, ( 0, -6), 'center', 'top'),
    ]


def draw_failure_envelopes(ax, strengths_mpa=None, normalise=True,
                           slice_axes=(0, 1), lim=1.8, grid_n=900,
                           criteria=None, show_labels=True):
    """
    Draw failure envelopes on *ax* for a chosen 2-D slice of the 6-component
    material-frame stress space.

    Parameters
    ----------
    ax            : matplotlib Axes
    strengths_mpa : dict (MPa) with keys Xt, Xc, Yt, Yc, Zt, Zc, Slr, Slt,
                    Srt, Fstar_12, Fstar_13, Fstar_23.  Wood defaults if None.
    normalise     : True  -> dimensionless axes sigma/scale.
                    False -> physical MPa axes (equal aspect).
    slice_axes    : (i, j) — component indices 0..5 as x/y axes.
                    Default (0, 1) = (sigmaL, sigmaR) plane.
                    All other components are fixed at zero.
    lim           : half-range for normalised axes (also used to derive the
                    physical margin as ``lim * max(scale_i, scale_j)``).
                    Fixed regardless of which *criteria* are active, so figures
                    are directly comparable.
    grid_n        : contour-grid resolution.
    criteria      : list of criteria to draw, any subset of
                    ``['von_mises', 'hill', 'tsai_wu']``.
                    ``None`` (default) plots all three.
                    Example — von Mises only: ``criteria=['von_mises']``
    show_labels   : If False, suppress axis labels, ticks, spines, title,
                    intercept annotations, and legend (useful for compositing
                    into larger multi-panel figures).

    Returns
    -------
    legend_handles : list[Patch]
    scale_dict     : {'scale_x': float, 'scale_y': float} — reference strengths
                     (MPa) for normalising scatter data.
    """
    active = set(_ALL_CRITERIA) if criteria is None else set(criteria)
    s = {**_DEFAULT_STRENGTHS_MPa, **(strengths_mpa or {})}
    i, j = int(slice_axes[0]), int(slice_axes[1])
    scale_i = _hill_scale(i, s)
    scale_j = _hill_scale(j, s)

    # ── Grid ──────────────────────────────────────────────────────────────────
    # SI, SJ hold physical stresses (MPa); X, Y are the plot coordinates.
    # The domain is always derived from *lim* so limits are criteria-independent.
    if normalise:
        U, V  = np.meshgrid(np.linspace(-lim, lim, grid_n),
                            np.linspace(-lim, lim, grid_n))
        SI, SJ = U * scale_i, V * scale_j
        X, Y   = U, V
        margin = None
    else:
        margin = lim * max(scale_i, scale_j)
        SI, SJ = np.meshgrid(np.linspace(-margin, margin, grid_n),
                             np.linspace(-margin, margin, grid_n))
        X, Y   = SI, SJ

    # ── Tsai-Wu ───────────────────────────────────────────────────────────────
    f_lin, f_quad, f_cross = _tw_coefficients(s)
    fij = f_cross.get((min(i, j), max(i, j)), 0.0)
    TW  = (f_lin[i]*SI  + f_lin[j]*SJ
           + f_quad[i]*SI**2 + f_quad[j]*SJ**2
           + 2*fij*SI*SJ - 1.0)

    # Tsai-Wu 2-D centre: stationary point of the quadratic part.
    # Requires solving the 2×2 gradient system
    #   [fii  fij] [cx]   [-fi/2]
    #   [fij  fjj] [cy] = [-fj/2]
    # The simpler formula cx = -fi/(2·fii) is only exact when fij = 0.
    fi_s  = f_lin[i]  if i in _COMP_NORMAL else 0.0
    fj_s  = f_lin[j]  if j in _COMP_NORMAL else 0.0
    fii_s = f_quad[i] if i in _COMP_NORMAL else 1.0   # dummy (fi_s=0 anyway)
    fjj_s = f_quad[j] if j in _COMP_NORMAL else 1.0
    _A = np.array([[fii_s, fij], [fij, fjj_s]])
    _b = np.array([-fi_s / 2, -fj_s / 2])
    try:
        cx_phys, cy_phys = np.linalg.solve(_A, _b)
    except np.linalg.LinAlgError:
        cx_phys = -fi_s / (2 * fii_s)
        cy_phys = -fj_s / (2 * fjj_s)
    cx_plot = cx_phys / scale_i if normalise else cx_phys
    cy_plot = cy_phys / scale_j if normalise else cy_phys

    # ── Hill & VM (normal-stress pairs only) ──────────────────────────────────
    both_normal = (i in _COMP_NORMAL) and (j in _COMP_NORMAL)
    if both_normal:
        Hi, Hj  = scale_i, scale_j
        SY      = np.sqrt(Hi * Hj)                   # same-area VM calibration
        ri, rj  = Hi / SY, Hj / SY                   # normalised VM scale factors

        # True Hill '48 cross-term from 3D consistency:
        # F+G=1/Xi², F+H=1/Xj², G+H=1/Xk²  →  F12 = -H = -(1/Xi²+1/Xj²-1/Xk²)/2
        # k is the third normal direction for the (i,j) pair.
        _third = {(0, 1): 2, (0, 2): 1, (1, 2): 0}
        k = _third[tuple(sorted((i, j)))]
        Hk = _hill_scale(k, s)
        F12_hill = -0.5 * (1/Hi**2 + 1/Hj**2 - 1/Hk**2)  # may be positive if Hk < Hj
        hill_convex = (1/Hi**2) * (1/Hj**2) > F12_hill**2  # det > 0 iff ellipse

        if normalise:
            f12_n = F12_hill * Hi * Hj   # cross-term in (U,V) space
            HILL  = U**2 + 2*f12_n*U*V + V**2 - 1.0
            VM    = ri**2*U**2 - U*V + rj**2*V**2 - 1.0
            # Tsai-Hill: same form but F12 = -1/(2*max(Hi,Hj)²) — Tsai (1965) ad-hoc
            F12_th = -1.0 / (2 * max(Hi, Hj)**2)
            f12_th_n = F12_th * Hi * Hj
            TH = U**2 + 2*f12_th_n*U*V + V**2 - 1.0
        else:
            HILL  = SI**2/Hi**2 + 2*F12_hill*SI*SJ + SJ**2/Hj**2 - 1.0
            VM    = SI**2 - SI*SJ + SJ**2 - SY**2
            F12_th = -1.0 / (2 * max(Hi, Hj)**2)
            TH = SI**2/Hi**2 + 2*F12_th*SI*SJ + SJ**2/Hj**2 - 1.0

    # ── Drawing ───────────────────────────────────────────────────────────────
    cv = '#4c78a8'; ch = '#f28e2b'; cth = '#76b7b2'; ct = '#e15759'
    if both_normal and 'von_mises' in active:
        ax.contour(X, Y, VM,   levels=[0], colors=[cv], linewidths=5.0)
        ax.contourf(X, Y, VM,  levels=[-1e9, 0], colors=[cv], alpha=0.10)
    if both_normal and 'hill' in active:
        ax.contour(X, Y, HILL, levels=[0], colors=[ch], linewidths=5.0)
        ax.contourf(X, Y, HILL, levels=[-1e9, 0], colors=[ch], alpha=0.10)
        if not hill_convex and show_labels:
            ax.text(0.02, 0.02, "Hill '48: non-convex\n(hyperbola for this wood)",
                    transform=ax.transAxes, fontsize=7, color=ch,
                    va='bottom', ha='left', style='italic')
    if both_normal and 'tsai_hill' in active:
        ax.contour(X, Y, TH,  levels=[0], colors=[cth], linewidths=5.0)
        ax.contourf(X, Y, TH, levels=[-1e9, 0], colors=[cth], alpha=0.10)
    if 'tsai_wu' in active:
        ax.contour(X, Y, TW,  levels=[0], colors=[ct], linewidths=5.0)
        ax.contourf(X, Y, TW, levels=[-1e9, 0], colors=[ct], alpha=0.10)

    ax.axhline(0, color='k', lw=0.6, ls='-', zorder=-1)
    ax.axvline(0, color='k', lw=0.6, ls='-', zorder=-1)

    # Strength intercept markers — only relevant when Tsai-Wu is drawn,
    # since they mark the asymmetric T/C intercepts (Xt ≠ Xc etc.)
    if show_labels and 'tsai_wu' in active:
        annot_cfg = (_intercept_annotations(i, s, scale_i, normalise, on_x_axis=True)
                     + _intercept_annotations(j, s, scale_j, normalise, on_x_axis=False))
        for lab, x, y, xytext, ha, va in annot_cfg:
            ax.plot(x, y, 'ko', ms=5, zorder=5)
            ax.annotate(lab, (x, y), textcoords='offset points',
                        fontsize=10, xytext=xytext, ha=ha, va=va)

    # Tsai-Wu centre + origin markers
    if 'tsai_wu' in active:
        ax.plot(cx_plot, cy_plot, '+', color=ct, ms=10, mew=2.0, zorder=6)

    # ── Axis labels & limits ──────────────────────────────────────────────────
    if show_labels:
        ax.set_xlabel(_axis_label(i, scale_i, normalise), fontsize=11)
        ax.set_ylabel(_axis_label(j, scale_j, normalise), fontsize=11)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    if normalise:
        ax.set_xlim(-lim, lim);  ax.set_ylim(-lim, lim)
    else:
        ax.set_xlim(-margin, margin);  ax.set_ylim(-margin, margin)
    ax.set_aspect('equal', adjustable='box')

    # ── Title ────────────────────────────────────────────────────────────────
    if show_labels:
        ci, cj    = _COMP_LATEX[i], _COMP_LATEX[j]
        fixed_str = ', '.join(f'${_COMP_LATEX[k]}=0$' for k in range(6) if k not in {i, j})
        ax.set_title(rf'Failure envelopes: ${ci}$–${cj}$ plane  ({fixed_str})',
                     fontsize=11, pad=10)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = []
    if both_normal and 'von_mises' in active:
        legend_handles.append(
            mpatches.Patch(facecolor=cv, edgecolor=cv, alpha=0.45, linewidth=2,
                           label=r'von Mises  (isotropic)'),
        )
    if both_normal and 'hill' in active:
        legend_handles.append(
            mpatches.Patch(facecolor=ch, edgecolor=ch, alpha=0.45, linewidth=2,
                           label=r'Hill  (orthotropic, symmetric)'),
        )
    if both_normal and 'tsai_hill' in active:
        legend_handles.append(
            mpatches.Patch(facecolor=cth, edgecolor=cth, alpha=0.45, linewidth=2,
                           label=r'Tsai-Hill  (orthotropic, symmetric, ad-hoc $F_{12}$)'),
        )
    if 'tsai_wu' in active:
        legend_handles.append(
            mpatches.Patch(facecolor=ct, edgecolor=ct, alpha=0.45, linewidth=2,
                           label=r'Tsai-Wu  (orthotropic, asymmetric)'),
        )
    if show_labels and legend_handles:
        ax.legend(handles=legend_handles,
                  loc='upper right' if normalise else 'upper left',
                  fontsize=9.5, framealpha=0.95)

    return legend_handles, {'scale_x': scale_i, 'scale_y': scale_j}


def save_square_figure(fig, ax, outpath, dpi=150, transparent=False):
    """
    Save *fig* so the axes box is pixel-square.

    ``set_aspect('equal')`` + ``bbox_inches='tight'`` produces non-square output
    because tight-layout adds asymmetric label padding *after* the aspect
    constraint is applied.  This function measures the padding post-draw and
    manually resizes the figure before saving without ``bbox_inches='tight'``.

    Parameters
    ----------
    transparent : bool
        If True, figure and axes backgrounds are saved with alpha=0.
    """
    if transparent:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    fig.canvas.draw()
    bbox_ax  = ax.get_window_extent()
    bbox_fig = fig.get_window_extent()
    pad_l = bbox_ax.x0
    pad_r = bbox_fig.width  - bbox_ax.x1
    pad_b = bbox_ax.y0
    pad_t = bbox_fig.height - bbox_ax.y1

    ax_size = bbox_ax.width
    fig.set_size_inches((ax_size + pad_l + pad_r) / fig.dpi,
                        (ax_size + pad_b + pad_t) / fig.dpi)

    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    fig.savefig(outpath, dpi=dpi, transparent=transparent)
    print(f'Saved -> {os.path.normpath(outpath)}')


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    p = argparse.ArgumentParser(
        description='Plot failure-envelope slices (von Mises / Hill / Tsai-Hill / Tsai-Wu).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--criteria', nargs='+',
        choices=list(_ALL_CRITERIA), default=None, metavar='CRITERION',
        help='Criteria to draw. Any subset of: von_mises hill tsai_hill tsai_wu. '
             'Default: all four.',
    )
    p.add_argument(
        '--axes', nargs=2, type=int, default=[0, 1], metavar=('I', 'J'),
        help='Two stress-component indices for the slice (0=σL 1=σR 2=σT '
             '3=τRT 4=τLT 5=τLR).',
    )
    p.add_argument(
        '--normalise', action=argparse.BooleanOptionalAction, default=None,
        help='--normalise (default): dimensionless axes. '
             '--no-normalise: physical MPa axes. '
             'Omit to generate both.',
    )
    p.add_argument(
        '--lim', type=float, default=1.8,
        help='Half-range for the normalised axes (physical margin = lim * max_scale).',
    )
    p.add_argument(
        '--no-labels', action='store_true',
        help='Suppress axis labels, ticks, spines, title, annotations, and legend.',
    )
    p.add_argument(
        '--transparent', action='store_true',
        help='Save with a transparent background.  Implied by --no-labels.',
    )
    p.add_argument(
        '--out', default=None,
        help='Output file path.  Default: output/failure_criteria_<axes>_<suffix>.png '
             'inside the repo root.  When --normalise is omitted both variants are saved '
             'and --out is ignored.',
    )
    p.add_argument(
        '--dpi', type=int, default=150,
        help='Output resolution in DPI.',
    )
    args = p.parse_args()

    _slice_axes    = tuple(args.axes)
    _show_labels   = not args.no_labels
    _transparent   = args.transparent or args.no_labels
    _AXES_SHORT = ['sL', 'sR', 'sT', 'tRT', 'tLT', 'tLR']
    _axes_tag = '_'.join(_AXES_SHORT[k] for k in _slice_axes)

    # Decide which normalise variants to produce
    _variants = (
        [(args.normalise, args.out)]
        if args.normalise is not None
        else [(True, None), (False, None)]
    )

    for _norm, _out in _variants:
        _suffix = 'normalised' if _norm else 'physical'
        if _out is None:
            _out = os.path.join(_repo_root, 'output',
                                f'failure_criteria_{_axes_tag}_{_suffix}.png')

        _fig, _ax = plt.subplots(figsize=(7, 7))
        draw_failure_envelopes(
            _ax,
            normalise=_norm,
            slice_axes=_slice_axes,
            criteria=args.criteria,
            lim=args.lim,
            show_labels=_show_labels,
        )
        save_square_figure(_fig, _ax, _out, dpi=args.dpi, transparent=_transparent)
        plt.close(_fig)
