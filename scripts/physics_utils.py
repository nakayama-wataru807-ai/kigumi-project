import os

import numpy as np

from vis_utils import cellwise_means
from vis_utils import decode_array
from vis_utils import load_deformed_points
from vis_utils import load_piece_cells
from vis_utils import load_vtu_piece
from vis_utils import map_surface_faces_to_element_values


# Stress field name conventions across PolyFEM versions.
# PolyFEM writes tensor components as {name}_{i}{j} (0-indexed row/col).
# In 3D: stress_00=σxx, stress_11=σyy, stress_22=σzz,
#        stress_01=σxy, stress_02=σxz, stress_12=σyz
_STRESS_VOIGT = [
    ('stress_00', 'stress_11', 'stress_22', 'stress_01', 'stress_02', 'stress_12'),
    ('cauchy_stess_00', 'cauchy_stess_11', 'cauchy_stess_22',
     'cauchy_stess_01', 'cauchy_stess_02', 'cauchy_stess_12'),
    ('sigma_xx', 'sigma_yy', 'sigma_zz', 'sigma_xy', 'sigma_xz', 'sigma_yz'),
    ('stress_xx', 'stress_yy', 'stress_zz', 'stress_xy', 'stress_xz', 'stress_yz'),
]


def point_von_mises_from_piece(piece):
    """Return pointwise von Mises stress from a VTK Piece, or None if unavailable."""
    for names in _STRESS_VOIGT:
        elems = [piece.find(f".//*[@Name='{name}']") for name in names]
        if all(elem is not None for elem in elems):
            sxx, syy, szz, sxy, sxz, syz = [decode_array(elem) for elem in elems]
            vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
                                + 6.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)))
            return vm.astype(np.float64)
    return None


def material_rotation(fiber_dir, radial_dir):
    """
    Build a (3, 3) rotation matrix Q whose rows are the orthonormal material
    axes [L, R, T] expressed in global coordinates.
    """
    longitudinal = np.asarray(fiber_dir, dtype=np.float64)
    longitudinal /= np.linalg.norm(longitudinal)

    radial = np.asarray(radial_dir, dtype=np.float64)
    radial -= np.dot(radial, longitudinal) * longitudinal
    radial /= np.linalg.norm(radial)

    tangential = np.cross(longitudinal, radial)
    return np.array([longitudinal, radial, tangential])


def tsai_wu_coefficients(strengths):
    """Pre-compute the Tsai-Wu tensor coefficients from a strength dictionary."""
    f1 = 1 / strengths['Xt'] - 1 / strengths['Xc']
    f2 = 1 / strengths['Yt'] - 1 / strengths['Yc']
    f3 = 1 / strengths['Zt'] - 1 / strengths['Zc']
    f11 = 1 / (strengths['Xt'] * strengths['Xc'])
    f22 = 1 / (strengths['Yt'] * strengths['Yc'])
    f33 = 1 / (strengths['Zt'] * strengths['Zc'])
    f44 = 1 / strengths['Srt'] ** 2
    f55 = 1 / strengths['Slt'] ** 2
    f66 = 1 / strengths['Slr'] ** 2
    f12 = strengths['Fstar_12'] * np.sqrt(f11 * f22)
    f13 = strengths['Fstar_13'] * np.sqrt(f11 * f33)
    f23 = strengths['Fstar_23'] * np.sqrt(f22 * f33)
    return f1, f2, f3, f11, f22, f33, f44, f55, f66, f12, f13, f23


def tsai_wu_from_stress_tensors(stress_tensors, strengths, fiber_dir, radial_dir):
    """Vectorised Tsai-Wu failure index for an array of global stress tensors."""
    rotation = material_rotation(fiber_dir, radial_dir)
    stress_material = np.einsum('ij,njk,lk->nil', rotation, stress_tensors, rotation)

    sigma_l = stress_material[:, 0, 0]
    sigma_r = stress_material[:, 1, 1]
    sigma_t = stress_material[:, 2, 2]
    tau_rt = stress_material[:, 1, 2]
    tau_lt = stress_material[:, 0, 2]
    tau_lr = stress_material[:, 0, 1]

    f1, f2, f3, f11, f22, f33, f44, f55, f66, f12, f13, f23 = tsai_wu_coefficients(strengths)

    return (f1 * sigma_l + f2 * sigma_r + f3 * sigma_t
            + f11 * sigma_l ** 2 + f22 * sigma_r ** 2 + f33 * sigma_t ** 2
            + f44 * tau_rt ** 2 + f55 * tau_lt ** 2 + f66 * tau_lr ** 2
            + 2 * f12 * sigma_l * sigma_r + 2 * f13 * sigma_l * sigma_t
            + 2 * f23 * sigma_r * sigma_t)


def load_tet_stress_material_frame(vol_vtu_path, fiber_dir, radial_dir):
    """
    Load a volume VTU and return per-tet Cauchy stresses in the material frame.

    Parameters
    ----------
    vol_vtu_path : str  Path to the volume VTU (e.g. step_N.vtu).
    fiber_dir    : array-like (3,)  Longitudinal (L) direction in global coords.
    radial_dir   : array-like (3,)  Radial (R) direction in global coords.

    Returns
    -------
    sigma_l, sigma_r, sigma_t : np.ndarray (n_tets,) in the simulation's stress
        units (GPa for PolyFEM). Returns (None, None, None) if the file is
        missing or the stress fields are absent.
    """
    if not os.path.isfile(vol_vtu_path):
        return (None,) * 6

    piece   = load_vtu_piece(vol_vtu_path)
    vol_pts, _ = load_deformed_points(piece)
    n_vol   = len(vol_pts)

    def _row(avg_name, plain_name):
        elem = piece.find(f".//*[@Name='{avg_name}']")
        if elem is None:
            elem = piece.find(f".//*[@Name='{plain_name}']")
        if elem is None:
            return None
        return decode_array(elem).reshape(n_vol, 3).astype(np.float64)

    row1 = _row('cauchy_stess_avg_1', 'cauchy_stess_1')
    row2 = _row('cauchy_stess_avg_2', 'cauchy_stess_2')
    row3 = _row('cauchy_stess_avg_3', 'cauchy_stess_3')

    if row1 is None or row2 is None or row3 is None:
        return (None,) * 6

    stress_nodes = np.stack([row1, row2, row3], axis=1)   # (n_nodes, 3, 3)

    conn, offs  = load_piece_cells(piece)
    tet_stress  = cellwise_means(stress_nodes, conn, offs)  # (n_tets, 3, 3)

    rotation = material_rotation(fiber_dir, radial_dir)
    stress_mat = np.einsum('ij,njk,lk->nil', rotation, tet_stress, rotation)

    return (
        stress_mat[:, 0, 0],   # σL
        stress_mat[:, 1, 1],   # σR
        stress_mat[:, 2, 2],   # σT
        stress_mat[:, 1, 2],   # τRT
        stress_mat[:, 0, 2],   # τLT
        stress_mat[:, 0, 1],   # τLR
    )


def load_volume_tsai_wu(surf_vtu_path, surf_verts, surf_faces,
                        strengths, fiber_dir, radial_dir):
    """
    Load the sibling volume VTU, compute elementwise Tsai-Wu failure indices,
    and map them to surface faces via nearest tet centroid.
    """
    vol_path = surf_vtu_path.replace('_surf.vtu', '.vtu')
    if not os.path.isfile(vol_path):
        print(f'[WARN] Volume VTU not found for Tsai-Wu: {vol_path}')
        return None, 'FACE'

    print(f'  Loading volume VTU for Tsai-Wu: {os.path.basename(vol_path)}')
    piece = load_vtu_piece(vol_path)
    vol_pts, _ = load_deformed_points(piece)
    n_vol = len(vol_pts)

    def _row(avg_name, plain_name):
        elem = piece.find(f".//*[@Name='{avg_name}']")
        if elem is None:
            elem = piece.find(f".//*[@Name='{plain_name}']")
        if elem is None:
            return None
        return decode_array(elem).reshape(n_vol, 3).astype(np.float64)

    row1 = _row('cauchy_stess_avg_1', 'cauchy_stess_1')
    row2 = _row('cauchy_stess_avg_2', 'cauchy_stess_2')
    row3 = _row('cauchy_stess_avg_3', 'cauchy_stess_3')

    if row1 is None or row2 is None or row3 is None:
        print('[WARN] Cauchy stress tensor rows not found in volume VTU.\n'
              '       Enable them with  "tensor_values": true  in the PolyFEM config.')
        return None, 'FACE'

    stress_nodes = np.stack([row1, row2, row3], axis=1)

    conn, offs = load_piece_cells(piece)
    tet_centroids = cellwise_means(vol_pts, conn, offs)
    tet_stress = cellwise_means(stress_nodes, conn, offs)

    failure_index = tsai_wu_from_stress_tensors(
        tet_stress, strengths, fiber_dir, radial_dir)

    return map_surface_faces_to_element_values(
        surf_verts, surf_faces, tet_centroids, failure_index), 'FACE'