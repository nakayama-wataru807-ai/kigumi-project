"""
Blender script: render deformed surface geometry from PolyFEM simulation steps.

USAGE
-----
Basic (solid wood colors, steps defined in CONFIG):
    blender --background --python scripts/render_simulation.py

Plot-only mode (no PNG renders, only evolution plots if enabled):
    set RENDER_IMAGES = False in CONFIG, then run as usual.

With scalar colorization:
    blender --background --python scripts/render_simulation.py -- --field <name>

With colormap override:
    blender --background --python scripts/render_simulation.py -- --field <name> --cmap <cmap>

CLI OPTIONS
-----------
--field <name>
    Colorize the mesh by the named scalar field.  Recognized values:

    displacement_magnitude
        Euclidean norm of the nodal displacement vector.  Always available.

    von_mises
        Von Mises equivalent stress.  Loaded from the surface VTU if present;
        falls back to the sibling volume VTU (step_N.vtu) mapped to surface
        faces via nearest-tet-centroid.  Requires  "tensor_values": true  in
        the PolyFEM paraview output config.

    tsai_wu
        Tsai-Wu failure index (FI).  FI ≥ 1 indicates predicted failure.
        Loaded from the volume VTU (step_N.vtu); requires the full Cauchy
        stress tensor output ("tensor_values": true).
        When TSAI_WU_HIGHLIGHT_FAILURE is True (default):
          - colormap is clamped to [0, 1] so the safe range is fully resolved
          - faces with FI ≥ 1 are overridden to solid red
        When TSAI_WU_PLOT_EVOLUTION is True (default):
          - a matplotlib summary plot is saved to OUTPUT_DIR/tsai_wu_evolution.png

    <any PointData field>
        Any scalar (or vector → magnitude) field present in the VTU, e.g.
        "rho", "discr", "C_00".  Falls back to the volume VTU if absent from
        the surface file.

--cmap <name>
    Any matplotlib colormap name (default: viridis).

CONFIGURATION
-------------
All paths, step list, strength parameters, render settings, and feature flags
are in the CONFIG section at the top of this file.  Key variables:

    SIM_DIR               – folder containing step_N_surf.vtu / step_N.vtu
    STEPS                 – list of time steps to render, e.g. [0, 5, 10]
    COLORIZE_FIELD        – default field (None → solid wood colors)
    COLORIZE_CLIM         – fixed [vmin, vmax], or None to auto-scale globally
    TSAI_WU_STRENGTHS     – wood strength parameters (GPa)
    TSAI_WU_FIBER_DIR     – longitudinal (grain) axis in global coordinates
    TSAI_WU_HIGHLIGHT_FAILURE – paint failed faces red (bool)
    TSAI_WU_PLOT_EVOLUTION    – save tsai_wu_evolution.png (bool)
    OUTPUT_DIR            – directory for rendered PNGs and .blend file
    RENDER_IMAGES         – if False, skip rendering and save .blend only
    RENDER_ENGINE         – "CYCLES" or "BLENDER_EEVEE_NEXT"
    RENDER_SAMPLES        – samples per pixel (higher = cleaner, slower)

EXAMPLES
--------
    # Default render, solid wood colors
    blender --background --python scripts/render_simulation.py

    # Tsai-Wu failure map with red highlight and evolution plot
    blender --background --python scripts/render_simulation.py -- --field tsai_wu

    # Displacement magnitude, plasma colormap
    blender --background --python scripts/render_simulation.py -- --field displacement_magnitude --cmap plasma

    # Von Mises stress
    blender --background --python scripts/render_simulation.py -- --field von_mises

    # Tsai-Wu evolution plot only (no renders)
    #   1) set COLORIZE_FIELD = "tsai_wu"
    #   2) set TSAI_WU_PLOT_EVOLUTION = True
    #   3) set RENDER_IMAGES = False
    blender --background --python scripts/render_simulation.py
"""

import bpy
import base64
import struct
import xml.etree.ElementTree as ET
import numpy as np
import os
import sys
import glob

# ── Inject conda env site-packages so matplotlib is available inside Blender ──
_conda_prefix = os.environ.get('CONDA_PREFIX', '')
if _conda_prefix:
    for _sp in glob.glob(os.path.join(_conda_prefix, 'lib', 'python3.*', 'site-packages')):
        if _sp not in sys.path:
            sys.path.insert(0, _sp)

# ── CONFIG ────────────────────────────────────────────────────────────────────

SIM_DIR = "/Users/quentinbecker/Library/CloudStorage/GoogleDrive-quentinbecker@g.ecc.u-tokyo.ac.jp/My Drive/kigumi-project/simulations/bending/0525_2343_layered_simulation/"   # folder with step_N_surf.vtu files
STEPS   = [0, 5, 10]                     # which time steps to import

# Vertex colorization
# Set to None to use per-body solid colors (default).
# Set to a PointData field name to colorize by that scalar, e.g.:
#   "rho", "discr", "C_00"
# Special derived quantities:
#   "displacement_magnitude"  – magnitude of the solution vector
#   "von_mises"               – Von Mises stress (requires tensor_values output;
#                               enable with  "tensor_values": true  in paraview options)
# Override at runtime:  blender --background --python render_simulation.py -- --field von_mises
COLORIZE_FIELD = None
COLORIZE_CMAP  = "viridis"   # any matplotlib colormap name
COLORIZE_CLIM  = None        # [vmin, vmax], or None to auto-scale per step

# ── CLI overrides ─────────────────────────────────────────────────────────────
_argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
for _i, _a in enumerate(_argv):
    if _a == '--field' and _i + 1 < len(_argv):
        COLORIZE_FIELD = _argv[_i + 1]
    elif _a == '--cmap' and _i + 1 < len(_argv):
        COLORIZE_CMAP = _argv[_i + 1]

# Materials per body id (body 1 = Male piece, body 2 = Female piece)
BODY_COLORS = {
    1: (0.80, 0.55, 0.25, 1.0),   # warm wood – male
    # 2: (0.65, 0.42, 0.18, 1.0),   # dark wood  – female
    2: (117.0 / 255.0, 76.0 / 255.0, 33.0 / 255.0, 1.0),   # dark wood  – female
}

# Output paths
OUTPUT_DIR   = "/Users/quentinbecker/Research/kigumi-project/output/"
OUTPUT_BLEND = "/Users/quentinbecker/Research/kigumi-project/output/scene.blend"

# Render settings
RENDER_ENGINE    = "CYCLES"  # "CYCLES", or "BLENDER_EEVEE_NEXT" for viewport-like rendering with good performance
RENDER_RES_X     = 1920
RENDER_RES_Y     = 1080
RENDER_SAMPLES   = 64
# When False, skip image rendering and only run data/plot generation + .blend save.
RENDER_IMAGES    = True

# ── TSAI-WU FAILURE CRITERION ─────────────────────────────────────────────────
# Grain / fiber (L) and radial (R) directions in global model coordinates.
# The tangential axis T = cross(L, R) is derived automatically.
# Adjust to match how the joint is oriented in your mesh.
TSAI_WU_FIBER_DIR  = (1.0, 0.0, 0.0)   # longitudinal (L) – along grain
TSAI_WU_RADIAL_DIR = (0.0, 1.0, 0.0)   # radial (R)

# Strength parameters in the same stress units as E (GPa when E = 8 GPa).
# Source: Wood Handbook – Wood as an Engineering Material, FPL-GTR-282 (2021),
#         Ch. 5, Senalik & Farber; Table 5–3, clear-wood values at 12 % MC.
# Defaults below represent a medium-density temperate hardwood (SG ≈ 0.55,
# e.g. ash, cherry, or a comparable Asian species used in kigumi joinery).
#
#   Xt / Xc  – tensile / compressive strength parallel to grain (L)
#   Yt / Yc  – tensile / compressive strength, radial direction (R)
#   Zt / Zc  – tensile / compressive strength, tangential direction (T)
#   Slr/Slt  – shear strength in the LR and LT planes
#   Srt      – rolling-shear strength in the RT plane (≈ Fv / 8 for most species)
#
# Interaction coefficients (dimensionless) follow the Tsai-Hahn approximation:
#   F*ij = Fij / sqrt(Fii · Fjj) = –0.5   (Tsai & Hahn, 1980; ADA144274 p. 14)
# This value is recommended when biaxial test data are unavailable.
TSAI_WU_STRENGTHS = {
    'Xt':       0.090,   # GPa  (= 90 MPa)
    'Xc':       0.050,   # GPa  (= 50 MPa)
    'Yt':       0.004,   # GPa  (=  4 MPa)
    'Yc':       0.008,   # GPa  (=  8 MPa)
    'Zt':       0.003,   # GPa  (=  3 MPa)  – tangential tensile, typically lower than radial
    'Zc':       0.006,   # GPa  (=  6 MPa)
    'Slr':      0.011,   # GPa  (= 11 MPa)  – shear parallel to grain, LR plane
    'Slt':      0.011,   # GPa  (= 11 MPa)  – shear parallel to grain, LT plane
    'Srt':      0.0014,  # GPa  (=  1.4 MPa) – rolling shear, RT plane
    'Fstar_12': -0.5,    # Tsai-Hahn interaction, L-R
    'Fstar_13': -0.5,    # Tsai-Hahn interaction, L-T
    'Fstar_23': -0.5,    # Tsai-Hahn interaction, R-T
}

# When True, faces where FI ≥ 1 are overridden to solid red in the render.
# The colormap is also clamped to [0, 1] so the safe range is fully resolved.
# Set to False to keep the raw continuous colormap.
TSAI_WU_HIGHLIGHT_FAILURE = True

# When True, save a matplotlib plot of the failure criterion evolution across
# the loaded steps to OUTPUT_DIR/tsai_wu_evolution.png.
TSAI_WU_PLOT_EVOLUTION = True

# ── VTU PARSER ────────────────────────────────────────────────────────────────

# Map VTK XML type names → numpy dtypes
_VTK_DTYPE = {
    "Float32": np.float32, "Float64": np.float64,
    "Int8":  np.int8,   "Int16":  np.int16,
    "Int32": np.int32,  "Int64":  np.int64,
    "UInt8": np.uint8,  "UInt16": np.uint16,
    "UInt32": np.uint32, "UInt64": np.uint64,
}


def _decode_array(data_array_elem, dtype=None):
    """Decode a base64 binary DataArray (VTK XML format="binary", header_type=UInt64).

    If *dtype* is None the type declared in the element's ``type`` attribute is used.
    """
    if dtype is None:
        dtype = _VTK_DTYPE[data_array_elem.attrib["type"]]
    b64 = data_array_elem.text.strip()
    raw = base64.b64decode(b64)
    # First 8 bytes = UInt64 data length in bytes
    raw = raw[8:]
    return np.frombuffer(raw, dtype=dtype)


def load_surface_vtu(path):
    """
    Parse a *_surf.vtu file.

    Returns
    -------
    verts : (N, 3) float64 – deformed vertex positions
    faces : list of lists – polygon face indices (triangles / quads)
    body_ids : (N,) int64  – per-vertex body id (1 or 2)
    """
    tree = ET.parse(path)
    root = tree.getroot()
    piece = root.find(".//Piece")
    n_pts   = int(piece.attrib["NumberOfPoints"])
    n_cells = int(piece.attrib["NumberOfCells"])

    # ---------- vertices (reference positions) ----------
    pts_elem = piece.find("Points/DataArray")
    pts_flat = _decode_array(pts_elem)
    verts = pts_flat.reshape(n_pts, 3).astype(np.float64)

    # ---------- cells -------------------------------------------------------
    cells_elem  = piece.find("Cells")
    conn_elem   = cells_elem.find("DataArray[@Name='connectivity']")
    off_elem    = cells_elem.find("DataArray[@Name='offsets']")

    connectivity = _decode_array(conn_elem)
    offsets      = _decode_array(off_elem)

    faces = []
    prev  = 0
    for off in offsets:
        vids = connectivity[prev:int(off)].tolist()
        # keep triangles and quads; skip degenerate cells
        if len(vids) >= 3:
            faces.append(vids)
        prev = int(off)

    # ---------- solution (displacement) – keep raw copy then add to verts ----
    sol_elem = piece.find(".//*[@Name='solution']")
    raw_solution = None
    if sol_elem is not None:
        ncomp        = int(sol_elem.attrib.get("NumberOfComponents", 3))
        raw_solution = _decode_array(sol_elem).reshape(n_pts, ncomp).astype(np.float64)
        verts        = verts + raw_solution[:, :3]

    # ---------- body ids (per vertex) – stored as Float64 in this output ----
    body_elem = piece.find(".//*[@Name='body_ids']")
    if body_elem is not None:
        body_ids = np.round(_decode_array(body_elem).astype(np.float64)).astype(np.int64)
    else:
        body_ids = np.ones(n_pts, dtype=np.int64)

    # ---------- requested scalar field (optional) ----------------------------
    scalars       = None
    scalar_domain = 'POINT'
    if COLORIZE_FIELD:
        scalars, scalar_domain = extract_scalar_field(
            piece, COLORIZE_FIELD, n_pts, verts, raw_solution,
            vtu_path=path, surf_faces=faces)

    return verts, faces, body_ids, scalars, scalar_domain


# ── SCALAR FIELD EXTRACTION ───────────────────────────────────────────────────

# Stress field name conventions across PolyFEM versions.
# PolyFEM writes tensor components as {name}_{i}{j} (0-indexed row/col).
# In 3D: stress_00=σxx, stress_11=σyy, stress_22=σzz,
#        stress_01=σxy, stress_02=σxz, stress_12=σyz
_STRESS_VOIGT = [
    # PolyFEM tensor_values output (primary)
    ('stress_00', 'stress_11', 'stress_22', 'stress_01', 'stress_02', 'stress_12'),
    # Alternate names used by some assembler implementations
    ('cauchy_stess_00', 'cauchy_stess_11', 'cauchy_stess_22',
     'cauchy_stess_01', 'cauchy_stess_02', 'cauchy_stess_12'),
    # Legacy / user-defined names
    ('sigma_xx', 'sigma_yy', 'sigma_zz', 'sigma_xy', 'sigma_xz', 'sigma_yz'),
    ('stress_xx', 'stress_yy', 'stress_zz', 'stress_xy', 'stress_xz', 'stress_yz'),
]


def _load_volume_element_scalar(surf_vtu_path, field_name, surf_verts, surf_faces):
    """
    Load a scalar field from the sibling volume VTU (step_N.vtu) and assign
    one value per surface *face* by mapping each face centroid to the nearest
    element (tetrahedron) centroid.  No per-vertex smoothing is performed.

    Returns (per_face_scalars, 'FACE') or (None, 'FACE').
    """
    vol_path = surf_vtu_path.replace('_surf.vtu', '.vtu')
    if not os.path.isfile(vol_path):
        print(f'[WARN] Volume VTU not found: {vol_path}')
        return None, 'FACE'

    print(f'  Loading volume VTU for per-element "{field_name}": {os.path.basename(vol_path)}')
    root  = ET.parse(vol_path).getroot()
    piece = root.find('.//Piece')
    n_vol = int(piece.attrib['NumberOfPoints'])

    # Deformed node positions
    vol_pts = _decode_array(piece.find('Points/DataArray')).reshape(n_vol, 3).astype(np.float64)
    sol_elem = piece.find(".//*[@Name='solution']")
    if sol_elem is not None:
        ncomp   = int(sol_elem.attrib.get('NumberOfComponents', 3))
        sol     = _decode_array(sol_elem).reshape(n_vol, ncomp).astype(np.float64)
        vol_pts = vol_pts + sol[:, :3]

    # Per-node scalar field
    sc_elem = piece.find(f".//*[@Name='{field_name}']")
    if sc_elem is None:
        print(f'[WARN] Field "{field_name}" not found in volume VTU.')
        return None, 'FACE'
    node_scalars = _decode_array(sc_elem).astype(np.float64)

    # Build per-element centroid and scalar by averaging over element nodes
    cells_elem = piece.find('Cells')
    conn = _decode_array(cells_elem.find("DataArray[@Name='connectivity']")).astype(np.int64)
    offs = _decode_array(cells_elem.find("DataArray[@Name='offsets']")).astype(np.int64)

    tet_centroids = []
    tet_scalars   = []
    prev = 0
    for off in offs:
        nids = conn[prev:int(off)]
        tet_centroids.append(vol_pts[nids].mean(axis=0))
        tet_scalars.append(node_scalars[nids].mean())
        prev = int(off)
    tet_centroids = np.array(tet_centroids)
    tet_scalars   = np.array(tet_scalars)

    # Per-surface-face centroid
    face_centroids = np.array([surf_verts[np.array(f)].mean(axis=0) for f in surf_faces])

    # Map each surface face to the nearest element centroid
    try:
        from scipy.spatial import KDTree
        _, idx = KDTree(tet_centroids).query(face_centroids, k=1)
    except ImportError:
        idx = np.argmin(
            np.sum((tet_centroids[np.newaxis] - face_centroids[:, np.newaxis]) ** 2, axis=2),
            axis=1)
    return tet_scalars[idx], 'FACE'


# ── TSAI-WU IMPLEMENTATION ────────────────────────────────────────────────────

def _material_rotation(fiber_dir, radial_dir):
    """
    Build a (3, 3) rotation matrix Q whose rows are the orthonormal material
    axes [L, R, T] expressed in global coordinates.  Applying Q to a global
    stress tensor gives the tensor in the material frame:
        S_mat = Q @ S_global @ Q.T
    """
    L = np.asarray(fiber_dir,  dtype=np.float64); L /= np.linalg.norm(L)
    R = np.asarray(radial_dir, dtype=np.float64)
    R -= np.dot(R, L) * L                          # Gram-Schmidt
    R /= np.linalg.norm(R)
    T = np.cross(L, R)
    return np.array([L, R, T])                      # (3, 3)


def _tsai_wu_coefficients(s):
    """
    Pre-compute the Tsai-Wu tensor components from a strength dictionary.

    3D criterion (Tsai & Wu, 1971; AFWAL-TR-85-3014):
        FI = F_i σ_i  +  F_ij σ_i σ_j   (failure when FI ≥ 1)

    In Voigt notation with axes ordered (L, R, T, RT, LT, LR):
        Linear (distinguish tension from compression):
            F1 = 1/Xt – 1/Xc,  F2 = 1/Yt – 1/Yc,  F3 = 1/Zt – 1/Zc
            F4 = F5 = F6 = 0   (shear response is symmetric)
        Quadratic diagonal:
            F11 = 1/(Xt Xc),  F22 = 1/(Yt Yc),  F33 = 1/(Zt Zc)
            F44 = 1/Srt²,     F55 = 1/Slt²,     F66 = 1/Slr²
        Off-diagonal (Tsai-Hahn approximation, ADA144274 p. 14):
            Fij = F*ij √(Fii Fjj)   with F*ij = –0.5 (default)

    Returns a flat tuple in the order used by _tsai_wu_from_stress_tensors.
    """
    F1  = 1/s['Xt'] - 1/s['Xc']
    F2  = 1/s['Yt'] - 1/s['Yc']
    F3  = 1/s['Zt'] - 1/s['Zc']
    F11 = 1 / (s['Xt'] * s['Xc'])
    F22 = 1 / (s['Yt'] * s['Yc'])
    F33 = 1 / (s['Zt'] * s['Zc'])
    F44 = 1 / s['Srt'] ** 2
    F55 = 1 / s['Slt'] ** 2
    F66 = 1 / s['Slr'] ** 2
    F12 = s['Fstar_12'] * np.sqrt(F11 * F22)
    F13 = s['Fstar_13'] * np.sqrt(F11 * F33)
    F23 = s['Fstar_23'] * np.sqrt(F22 * F33)
    return F1, F2, F3, F11, F22, F33, F44, F55, F66, F12, F13, F23


def _tsai_wu_from_stress_tensors(S_global, strengths, fiber_dir, radial_dir):
    """
    Vectorised Tsai-Wu failure index for an array of Cauchy stress tensors.

    Parameters
    ----------
    S_global : (N, 3, 3)  Cauchy stress tensors in the global frame
    strengths : dict       TSAI_WU_STRENGTHS
    fiber_dir, radial_dir : (3,) global directions of L and R material axes

    Returns
    -------
    FI : (N,) float64  failure index (> 1 → predicted failure)
    """
    Q = _material_rotation(fiber_dir, radial_dir)               # (3, 3)
    # Rotate all tensors at once: S_mat[n] = Q @ S_global[n] @ Q.T
    S_mat = np.einsum('ij,njk,lk->nil', Q, S_global, Q)        # (N, 3, 3)

    sL  = S_mat[:, 0, 0];  sR  = S_mat[:, 1, 1];  sT  = S_mat[:, 2, 2]
    tRT = S_mat[:, 1, 2];  tLT = S_mat[:, 0, 2];  tLR = S_mat[:, 0, 1]

    F1, F2, F3, F11, F22, F33, F44, F55, F66, F12, F13, F23 = \
        _tsai_wu_coefficients(strengths)

    return (F1*sL + F2*sR + F3*sT
            + F11*sL**2 + F22*sR**2 + F33*sT**2
            + F44*tRT**2 + F55*tLT**2 + F66*tLR**2
            + 2*F12*sL*sR + 2*F13*sL*sT + 2*F23*sR*sT)


def _load_volume_tsai_wu(surf_vtu_path, surf_verts, surf_faces):
    """
    Load the Cauchy stress tensor from the sibling volume VTU, compute the
    Tsai-Wu failure index once per tetrahedron (using the element-average
    stress tensor), then map values to surface faces via nearest tet centroid.

    Prefers `cauchy_stess_avg_*` fields (element-averaged, nodally projected)
    over `cauchy_stess_*` (raw nodal extrapolation) when both are present.

    Returns (per_face_FI, 'FACE') or (None, 'FACE').
    """
    vol_path = surf_vtu_path.replace('_surf.vtu', '.vtu')
    if not os.path.isfile(vol_path):
        print(f'[WARN] Volume VTU not found for Tsai-Wu: {vol_path}')
        return None, 'FACE'

    print(f'  Loading volume VTU for Tsai-Wu: {os.path.basename(vol_path)}')
    root  = ET.parse(vol_path).getroot()
    piece = root.find('.//Piece')
    n_vol = int(piece.attrib['NumberOfPoints'])

    # Deformed node positions
    vol_pts  = _decode_array(piece.find('Points/DataArray')).reshape(n_vol, 3).astype(np.float64)
    sol_elem = piece.find(".//*[@Name='solution']")
    if sol_elem is not None:
        ncomp   = int(sol_elem.attrib.get('NumberOfComponents', 3))
        sol     = _decode_array(sol_elem).reshape(n_vol, ncomp).astype(np.float64)
        vol_pts = vol_pts + sol[:, :3]

    # Load the three rows of the Cauchy stress tensor (prefer _avg variants)
    def _row(name_avg, name_plain):
        e = piece.find(f".//*[@Name='{name_avg}']")
        if e is None:
            e = piece.find(f".//*[@Name='{name_plain}']")
        if e is None:
            return None
        return _decode_array(e).reshape(n_vol, 3).astype(np.float64)

    row1 = _row('cauchy_stess_avg_1', 'cauchy_stess_1')
    row2 = _row('cauchy_stess_avg_2', 'cauchy_stess_2')
    row3 = _row('cauchy_stess_avg_3', 'cauchy_stess_3')

    if row1 is None or row2 is None or row3 is None:
        print('[WARN] Cauchy stress tensor rows not found in volume VTU.\n'
              '       Enable them with  "tensor_values": true  in the PolyFEM config.')
        return None, 'FACE'

    # Per-node stress tensor: (n_vol, 3, 3)
    S_nodes = np.stack([row1, row2, row3], axis=1)   # (n_vol, 3, 3)

    # Build per-element centroid and average stress tensor
    cells_elem = piece.find('Cells')
    conn = _decode_array(cells_elem.find("DataArray[@Name='connectivity']")).astype(np.int64)
    offs = _decode_array(cells_elem.find("DataArray[@Name='offsets']")).astype(np.int64)

    tet_centroids = []
    tet_stress    = []
    prev = 0
    for off in offs:
        nids = conn[prev:int(off)]
        tet_centroids.append(vol_pts[nids].mean(axis=0))
        tet_stress.append(S_nodes[nids].mean(axis=0))   # average 3×3 tensor
        prev = int(off)

    tet_centroids = np.array(tet_centroids)   # (M, 3)
    tet_stress    = np.array(tet_stress)      # (M, 3, 3)

    # Tsai-Wu failure index per element
    FI = _tsai_wu_from_stress_tensors(
        tet_stress, TSAI_WU_STRENGTHS, TSAI_WU_FIBER_DIR, TSAI_WU_RADIAL_DIR)

    # Map to surface faces via nearest tet centroid
    face_centroids = np.array([surf_verts[np.array(f)].mean(axis=0) for f in surf_faces])
    try:
        from scipy.spatial import KDTree
        _, idx = KDTree(tet_centroids).query(face_centroids, k=1)
    except ImportError:
        idx = np.argmin(
            np.sum((tet_centroids[np.newaxis] - face_centroids[:, np.newaxis]) ** 2, axis=2),
            axis=1)

    return FI[idx], 'FACE'


def extract_scalar_field(piece, field_name, n_pts, verts, raw_solution,                         vtu_path=None, surf_faces=None):
    """
    Extract a scalar array from a VTK Piece element.

    Parameters
    ----------
    field_name : str
        A PointData field name, or one of the special keys
        ``'displacement_magnitude'`` / ``'von_mises'``.
    vtu_path : str, optional
        Path to the surface VTU file.  When provided and a field is absent
        from the surface mesh, the sibling volume VTU (step_N.vtu) is
        used as a fallback, mapping per-element values to surface faces.
    surf_faces : list of lists, optional
        Surface face connectivity, required for per-element volume fallback.

    Returns
    -------
    scalars : ndarray or None
    domain  : 'POINT' or 'FACE'  (indicates the Blender attribute domain)
    """
    if field_name == 'displacement_magnitude':
        if raw_solution is not None:
            return np.linalg.norm(raw_solution, axis=1), 'POINT'
        print('[WARN] "displacement_magnitude" requires solution field – not found.')
        return None, 'POINT'

    if field_name == 'von_mises':
        # 1. Direct per-node field in the surface VTU
        vm_elem = piece.find(".//*[@Name='von_mises']")
        if vm_elem is not None:
            return _decode_array(vm_elem).astype(np.float64), 'POINT'

        # 2. Compute from individual Cauchy-stress components in surface VTU
        for names in _STRESS_VOIGT:
            elems = [piece.find(f".//*[@Name='{n}']") for n in names]
            if all(e is not None for e in elems):
                sxx, syy, szz, sxy, sxz, syz = [_decode_array(e) for e in elems]
                vm = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2
                                    + 6.0 * (sxy**2 + sxz**2 + syz**2)))
                return vm.astype(np.float64), 'POINT'

        # 3. Fall back to the volume VTU: one value per tet, mapped to surface faces
        if vtu_path is not None and surf_faces is not None:
            result, domain = _load_volume_element_scalar(vtu_path, 'von_mises',
                                                         verts, surf_faces)
            if result is not None:
                return result, domain

        print('[WARN] Von Mises stress not found in surface VTU and no sibling '
              'volume VTU is available.\n'
              '       The volume VTU (step_N.vtu) is required alongside '
              'step_N_surf.vtu.')
        return None, 'FACE'

    if field_name == 'tsai_wu':
        # Tsai-Wu failure index requires the full 3D Cauchy stress tensor,
        # which lives in the volume VTU (step_N.vtu) as cauchy_stess_[avg_]1/2/3.
        if vtu_path is not None and surf_faces is not None:
            result, domain = _load_volume_tsai_wu(vtu_path, verts, surf_faces)
            if result is not None:
                return result, domain
        print('[WARN] Tsai-Wu requires the sibling volume VTU (step_N.vtu) with\n'
              '       Cauchy stress output ("tensor_values": true in PolyFEM config).')
        return None, 'FACE'

    # Generic named field – try surface VTU first, then volume VTU
    elem = piece.find(f".//*[@Name='{field_name}']")
    if elem is None:
        if vtu_path is not None and surf_faces is not None:
            result, domain = _load_volume_element_scalar(vtu_path, field_name,
                                                         verts, surf_faces)
            if result is not None:
                return result, domain
        print(f'[WARN] Field "{field_name}" not found in PointData.')
        return None, 'POINT'
    ncomp = int(elem.attrib.get('NumberOfComponents', 1))
    data  = _decode_array(elem).astype(np.float64)
    if ncomp > 1:
        data = np.linalg.norm(data.reshape(n_pts, ncomp), axis=1)
    return data, 'POINT'


def scalars_to_vertex_colors(scalars, cmap_name='viridis', clim=None):
    """
    Map a 1-D float array to RGBA colors using a matplotlib colormap.
    Returns an (N, 4) float32 array in [0, 1].
    """
    try:
        import matplotlib.cm as _cm
        import matplotlib.colors as _mc
    except ImportError:
        print('[WARN] matplotlib not available – falling back to grey gradient.')
        t = (scalars - scalars.min()) / (scalars.ptp() + 1e-30)
        rgb = np.stack([t, t, t, np.ones_like(t)], axis=1).astype(np.float32)
        return rgb

    vmin, vmax = (scalars.min(), scalars.max()) if clim is None else clim
    norm   = _mc.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap   = _cm.get_cmap(cmap_name)
    colors = cmap(norm(scalars)).astype(np.float32)   # (N, 4) RGBA in [0,1]
    return colors


# ── FAILURE EVOLUTION PLOT ───────────────────────────────────────────────────

def plot_failure_evolution(step_data, output_dir):
    """
        Save a matplotlib figure to output_dir/tsai_wu_evolution.png showing,
        for each provided step:
      - Max Tsai-Wu failure index (left axis, red line)
      - Mean Tsai-Wu failure index (left axis, orange dashed line)
      - Fraction of surface faces with FI ≥ 1 (right axis, blue bars)
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    steps     = sorted(step_data.keys())
    max_fi    = []
    mean_fi   = []
    frac_fail = []   # percent of faces with FI ≥ 1

    for s in steps:
        fi = step_data[s][3]   # scalars array (may be None)
        if fi is None:
            max_fi.append(np.nan)
            mean_fi.append(np.nan)
            frac_fail.append(np.nan)
        else:
            max_fi.append(float(fi.max()))
            mean_fi.append(float(fi.mean()))
            frac_fail.append(float((fi >= 1.0).mean()) * 100.0)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(steps, max_fi,  'o-',  color='crimson',    linewidth=2,   label='Max FI')
    ax1.plot(steps, mean_fi, 's--', color='darkorange',  linewidth=1.5, label='Mean FI')
    ax1.axhline(1.0, color='crimson', linestyle=':', linewidth=1.2,
                alpha=0.7, label='FI = 1 (failure threshold)')

    bar_width = max(0.3, (max(steps) - min(steps)) * 0.06) if len(steps) > 1 else 0.4
    ax2.bar(steps, frac_fail, width=bar_width, alpha=0.25,
            color='steelblue', label='% faces failed (FI ≥ 1)')

    ax1.set_xlabel('Simulation step', fontsize=11)
    ax1.set_ylabel('Tsai-Wu failure index (FI)', fontsize=11)
    ax2.set_ylabel('Faces with FI ≥ 1 (%)', color='steelblue', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f %%'))
    ax2.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    ax1.set_title('Tsai-Wu failure criterion evolution', fontsize=13)
    ax1.set_xticks(steps)
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'tsai_wu_evolution.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Tsai-Wu evolution plot saved → {out_path}')


def find_available_steps(sim_dir):
    """Return sorted step ids detected from step_*_surf.vtu files in sim_dir."""
    steps = []
    for path in glob.glob(os.path.join(sim_dir, 'step_*_surf.vtu')):
        name = os.path.basename(path)
        try:
            steps.append(int(name[len('step_'):-len('_surf.vtu')]))
        except ValueError:
            continue
    return sorted(set(steps))


# ── BLENDER HELPERS ───────────────────────────────────────────────────────────

def _get_or_create_material(name, color):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value  = 0.6
    return mat


def build_blender_mesh(step, verts, faces, body_ids, offset,
                       vertex_colors=None, scalar_domain='POINT'):
    """Create one Blender mesh object for a simulation step.

    If *vertex_colors* is provided it is an (N, 4) RGBA float32 array where
    N equals the number of vertices (scalar_domain='POINT') or the number of
    faces (scalar_domain='FACE').  A flat per-face color is used in the latter
    case, which avoids misleading interpolation across element boundaries.
    """
    obj_name  = f"step_{step:03d}"
    mesh_name = f"mesh_{step:03d}"

    # Remove existing object with same name
    if obj_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
    if mesh_name in bpy.data.meshes:
        bpy.data.meshes.remove(bpy.data.meshes[mesh_name])

    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.update()

    obj = bpy.data.objects.new(obj_name, mesh)
    obj.location = offset
    bpy.context.collection.objects.link(obj)

    if vertex_colors is not None:
        # ── color attribute (per-face or per-vertex depending on data source) ─
        col_attr = mesh.color_attributes.new(
            name="Col", type='FLOAT_COLOR', domain=scalar_domain)
        for i, rgba in enumerate(vertex_colors):
            col_attr.data[i].color = tuple(float(c) for c in rgba)

        # Single material that reads vertex colors via Attribute node
        mat_name = "VertexColor"
        if mat_name not in bpy.data.materials:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            bsdf  = nodes["Principled BSDF"]
            attr  = nodes.new("ShaderNodeAttribute")
            attr.attribute_name = "Col"
            links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
            bsdf.inputs["Roughness"].default_value = 0.5
        mesh.materials.append(bpy.data.materials[mat_name])
    else:
        # ── per-body solid materials ──────────────────────────────────────────
        unique_ids = sorted(set(body_ids.tolist()))
        body_to_mat_idx = {}
        for bid in unique_ids:
            color = BODY_COLORS.get(bid, (0.7, 0.7, 0.7, 1.0))
            mat   = _get_or_create_material(f"body_{bid}", color)
            if mat.name not in [m.name for m in mesh.materials]:
                mesh.materials.append(mat)
            body_to_mat_idx[bid] = [i for i, m in enumerate(mesh.materials)
                                     if m.name == mat.name][0]

        for poly, face_vids in zip(mesh.polygons, faces):
            ids      = body_ids[face_vids]
            majority = int(np.bincount(ids).argmax())
            poly.material_index = body_to_mat_idx.get(
                majority, body_to_mat_idx.get(unique_ids[0], 0))

    return obj


# ── SCENE SETUP ───────────────────────────────────────────────────────────────

def setup_scene():
    """Remove default objects and set up basic lighting."""
    for obj in list(bpy.data.objects):
        if obj.type in {"MESH", "LIGHT", "CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # Sun light
    import math
    light_data = bpy.data.lights.new("Sun", type="SUN")
    light_data.energy = 3.0
    light_data.angle  = math.radians(15)    # angular diameter → (smaller = sharper shadows)
    light_obj  = bpy.data.objects.new("Sun", light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.rotation_euler = (0.6, 0.2, -0.8)

    # World background
    bpy.context.scene.world.color = (0.05, 0.05, 0.05)

    # Transparent background
    scene = bpy.context.scene
    scene.render.film_transparent = True

    # Render settings
    scene = bpy.context.scene
    scene.render.engine          = RENDER_ENGINE
    scene.render.resolution_x    = RENDER_RES_X
    scene.render.resolution_y    = RENDER_RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode  = "RGBA"
    # filepath is set per-step in main()
    if RENDER_ENGINE == "CYCLES":
        scene.cycles.samples     = RENDER_SAMPLES
    else:
        scene.eevee.taa_render_samples = RENDER_SAMPLES


# ── SHADOW CATCHER ───────────────────────────────────────────────────────────

def add_shadow_catcher(z_offset=-0.01):
    """
    Add a large invisible plane just below the mesh geometry that catches
    shadows and composites them onto the transparent background.

    Works in both Cycles (native is_shadow_catcher flag) and Eevee
    (shadow_catcher object property + Holdout material mix).
    """
    import mathutils

    # Compute the lowest Z of all mesh objects
    meshes = [o for o in bpy.data.objects if o.type == 'MESH'
              and o.name != 'ShadowCatcher']
    if not meshes:
        return

    min_z = min(
        (obj.matrix_world @ mathutils.Vector(c)).z
        for obj in meshes
        for c in obj.bound_box
    )

    # Extent of the scene for plane sizing
    all_corners = [
        obj.matrix_world @ mathutils.Vector(c)
        for obj in meshes for c in obj.bound_box
    ]
    xs = [v.x for v in all_corners]
    ys = [v.y for v in all_corners]
    extent = max(max(xs) - min(xs), max(ys) - min(ys)) * 3.0

    bpy.ops.mesh.primitive_plane_add(
        size=extent,
        location=(
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
            min_z + z_offset,
        )
    )
    plane = bpy.context.active_object
    plane.name = 'ShadowCatcher'

    if RENDER_ENGINE == 'CYCLES':
        plane.is_shadow_catcher = True
    else:
        # Eevee: mark as shadow catcher
        plane.cycles.is_shadow_catcher = True
        # Create a simple shadeless material so Eevee picks up the flag
        mat = bpy.data.materials.new('ShadowCatcherMat')
        mat.use_nodes = True
        mat.blend_method = 'BLEND'
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        out   = nodes.new('ShaderNodeOutputMaterial')
        trans = nodes.new('ShaderNodeBsdfTransparent')
        links.new(trans.outputs['BSDF'], out.inputs['Surface'])
        plane.data.materials.append(mat)


# ── CAMERA AUTO-FRAMING ───────────────────────────────────────────────────────

def fit_camera_to_scene(margin=2.5):
    """Create a camera that frames all mesh objects with a 3/4 overhead view."""
    import math

    meshes = [o for o in bpy.data.objects
              if o.type == "MESH" and o.name != 'ShadowCatcher']
    if not meshes:
        return

    # World-space bounding box of all meshes combined
    all_corners = []
    for obj in meshes:
        for corner in obj.bound_box:
            all_corners.append(obj.matrix_world @ __import__('mathutils').Vector(corner))

    xs = [v.x for v in all_corners]
    ys = [v.y for v in all_corners]
    zs = [v.z for v in all_corners]
    cx, cy, cz = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2
    extent    = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    dist      = extent * margin

    # 3/4 view: camera sits in front-left-above
    elev, azim = math.radians(35), math.radians(-45)
    cam_x = cx + dist * math.cos(elev) * math.cos(azim)
    cam_y = cy + dist * math.cos(elev) * math.sin(azim)
    cam_z = cz + dist * math.sin(elev)

    cam_data = bpy.data.cameras.new("Camera")
    cam_obj  = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (cam_x, cam_y, cam_z)

    # Point the camera at the scene centre
    import mathutils
    direction = mathutils.Vector((cx, cy, cz)) - mathutils.Vector((cam_x, cam_y, cam_z))
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    bpy.context.scene.camera = cam_obj


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"

    # ── pass 1: load all steps to determine global color limits ──────────────
    step_data = {}   # step → (verts, faces, body_ids, scalars, scalar_domain)
    for step in STEPS:
        vtu_path = os.path.join(SIM_DIR, f"step_{step}_surf.vtu")
        if not os.path.isfile(vtu_path):
            print(f"[WARN] Not found: {vtu_path}")
            continue
        print(f"Loading step {step} …")
        step_data[step] = load_surface_vtu(vtu_path)
        verts, faces, body_ids, scalars, scalar_domain = step_data[step]
        print(f"  {len(verts)} verts, {len(faces)} faces")

    # Determine color limits across all steps (unless already fixed by user)
    clim = COLORIZE_CLIM
    if COLORIZE_FIELD and clim is None:
        all_scalars = [step_data[s][3] for s in step_data if step_data[s][3] is not None]
        if all_scalars:
            global_min = float(min(s.min() for s in all_scalars))
            global_max = float(max(s.max() for s in all_scalars))
            clim = [global_min, global_max]
            print(f"Global color limits for '{COLORIZE_FIELD}': "
                  f"[{global_min:.4g}, {global_max:.4g}]")

    # For Tsai-Wu with failure highlighting, clamp the colormap to [0, 1] so
    # the safe range is fully resolved and the red override is unambiguous.
    if (COLORIZE_FIELD == 'tsai_wu' and TSAI_WU_HIGHLIGHT_FAILURE
            and COLORIZE_CLIM is None):
        clim = [0.0, 1.0]
        print('Tsai-Wu failure highlighting active: colormap clamped to [0, 1]')

    # Failure evolution plot (Tsai-Wu only, independent of rendering)
    if COLORIZE_FIELD == 'tsai_wu' and TSAI_WU_PLOT_EVOLUTION:
        plot_steps = find_available_steps(SIM_DIR)
        if not plot_steps:
            print(f"[WARN] No step_*_surf.vtu files found in: {SIM_DIR}")
        else:
            print(f'Building Tsai-Wu evolution from all available steps: {plot_steps}')
            plot_data = {}
            for step in plot_steps:
                if step in step_data:
                    # Reuse already loaded rendered steps.
                    plot_data[step] = step_data[step]
                    continue

                vtu_path = os.path.join(SIM_DIR, f"step_{step}_surf.vtu")
                if not os.path.isfile(vtu_path):
                    print(f"[WARN] Not found: {vtu_path}")
                    continue

                print(f"Loading plot-only step {step} …")
                plot_data[step] = load_surface_vtu(vtu_path)

            if plot_data:
                plot_failure_evolution(plot_data, OUTPUT_DIR)

    if not RENDER_IMAGES:
        print('RENDER_IMAGES=False: skipping PNG render pass.')
        print(f"Saving .blend to {OUTPUT_BLEND} …")
        bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)
        print("Processing complete (plot-only mode).")
        return

    # ── pass 2: render each step with the shared color limits ─────────────────
    for step, (verts, faces, body_ids, scalars, scalar_domain) in step_data.items():
        setup_scene()

        vertex_colors = None
        if scalars is not None:
            vertex_colors = scalars_to_vertex_colors(scalars, COLORIZE_CMAP, clim)
            # Override failed elements (FI ≥ 1) with solid red
            if COLORIZE_FIELD == 'tsai_wu' and TSAI_WU_HIGHLIGHT_FAILURE:
                failed_mask = scalars >= 1.0
                n_failed = int(failed_mask.sum())
                if n_failed:
                    vertex_colors[failed_mask] = np.array(
                        [1.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    pct = 100.0 * n_failed / len(scalars)
                    print(f'  Step {step}: {n_failed} failed faces ({pct:.1f} %) '
                          f'highlighted red (FI ≥ 1)')
        build_blender_mesh(step, verts, faces, body_ids, (0.0, 0.0, 0.0),
                           vertex_colors=vertex_colors, scalar_domain=scalar_domain)

        add_shadow_catcher()
        fit_camera_to_scene()

        out_path = os.path.join(OUTPUT_DIR, f"render_step{step:03d}.png")
        scene.render.filepath = out_path
        print(f"Rendering step {step} → {out_path} …")
        bpy.ops.render.render(write_still=True)

    print(f"Saving .blend to {OUTPUT_BLEND} …")
    bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)
    print("All renders complete.")


main()
