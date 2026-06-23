import base64
import os
import xml.etree.ElementTree as ET

import numpy as np


_VTK_DTYPE = {
    "Float32": np.float32, "Float64": np.float64,
    "Int8": np.int8, "Int16": np.int16,
    "Int32": np.int32, "Int64": np.int64,
    "UInt8": np.uint8, "UInt16": np.uint16,
    "UInt32": np.uint32, "UInt64": np.uint64,
}


def decode_array(data_array_elem, dtype=None):
    """Decode a base64-binary VTK DataArray with UInt64 header."""
    if dtype is None:
        dtype = _VTK_DTYPE[data_array_elem.attrib["type"]]
    raw = base64.b64decode(data_array_elem.text.strip())[8:]
    return np.frombuffer(raw, dtype=dtype)


def load_vtu_piece(path):
    """Return the VTK XML Piece node from a .vtu file."""
    root = ET.parse(path).getroot()
    return root.find(".//Piece")


def load_deformed_points(piece):
    """Return deformed point positions and the raw solution array, if present."""
    n_pts = int(piece.attrib["NumberOfPoints"])
    verts = decode_array(piece.find("Points/DataArray")).reshape(n_pts, 3).astype(np.float64)

    raw_solution = None
    sol_elem = piece.find(".//*[@Name='solution']")
    if sol_elem is not None:
        ncomp = int(sol_elem.attrib.get("NumberOfComponents", 3))
        raw_solution = decode_array(sol_elem).reshape(n_pts, ncomp).astype(np.float64)
        verts = verts + raw_solution[:, :3]

    return verts, raw_solution


def load_piece_cells(piece):
    """Return cell connectivity and offsets arrays for a VTK Piece."""
    cells_elem = piece.find("Cells")
    conn = decode_array(cells_elem.find("DataArray[@Name='connectivity']")).astype(np.int64)
    offs = decode_array(cells_elem.find("DataArray[@Name='offsets']")).astype(np.int64)
    return conn, offs


def build_faces_from_cells(connectivity, offsets):
    """Convert VTK cell connectivity into a list of polygon faces."""
    faces = []
    prev = 0
    for off in offsets:
        vids = connectivity[prev:int(off)].tolist()
        if len(vids) >= 3:
            faces.append(vids)
        prev = int(off)
    return faces


def load_surface_mesh_data(path):
    """Return piece, deformed verts, faces, body_ids, and raw solution for a surface VTU."""
    piece = load_vtu_piece(path)
    verts, raw_solution = load_deformed_points(piece)
    conn, offs = load_piece_cells(piece)
    faces = build_faces_from_cells(conn, offs)

    body_elem = piece.find(".//*[@Name='body_ids']")
    if body_elem is not None:
        body_ids = np.round(decode_array(body_elem).astype(np.float64)).astype(np.int64)
    else:
        body_ids = np.ones(len(verts), dtype=np.int64)

    return piece, verts, faces, body_ids, raw_solution


def cellwise_means(values, connectivity, offsets):
    """Average nodal values over each cell defined by connectivity/offsets."""
    means = []
    prev = 0
    for off in offsets:
        nids = connectivity[prev:int(off)]
        means.append(values[nids].mean(axis=0))
        prev = int(off)
    return np.array(means)


def map_surface_faces_to_element_values(surf_verts, surf_faces, element_centroids,
                                        element_values):
    """Map each surface face to the nearest element centroid and return those values."""
    face_centroids = np.array([surf_verts[np.array(face)].mean(axis=0) for face in surf_faces])
    try:
        from scipy.spatial import KDTree
        _, idx = KDTree(element_centroids).query(face_centroids, k=1)
    except ImportError:
        idx = np.argmin(
            np.sum((element_centroids[np.newaxis] - face_centroids[:, np.newaxis]) ** 2, axis=2),
            axis=1)
    return np.asarray(element_values)[idx]


def load_volume_element_scalar(surf_vtu_path, field_name, surf_verts, surf_faces):
    """Load a nodal scalar from the sibling volume VTU and map its cell averages to faces."""
    vol_path = surf_vtu_path.replace('_surf.vtu', '.vtu')
    if not os.path.isfile(vol_path):
        print(f'[WARN] Volume VTU not found: {vol_path}')
        return None, 'FACE'

    print(f'  Loading volume VTU for per-element "{field_name}": {os.path.basename(vol_path)}')
    piece = load_vtu_piece(vol_path)
    vol_pts, _ = load_deformed_points(piece)

    sc_elem = piece.find(f".//*[@Name='{field_name}']")
    if sc_elem is None:
        print(f'[WARN] Field "{field_name}" not found in volume VTU.')
        return None, 'FACE'
    node_scalars = decode_array(sc_elem).astype(np.float64)

    conn, offs = load_piece_cells(piece)
    tet_centroids = cellwise_means(vol_pts, conn, offs)
    tet_scalars = cellwise_means(node_scalars, conn, offs)
    return map_surface_faces_to_element_values(
        surf_verts, surf_faces, tet_centroids, tet_scalars), 'FACE'


def find_available_steps(sim_dir):
    """Return sorted step ids detected from step_*_surf.vtu files in sim_dir."""
    import glob

    steps = []
    for path in glob.glob(os.path.join(sim_dir, 'step_*_surf.vtu')):
        name = os.path.basename(path)
        try:
            steps.append(int(name[len('step_'):-len('_surf.vtu')]))
        except ValueError:
            continue
    return sorted(set(steps))


def load_force_at_step(surf_vtu_path, sideset_id):
    """
    Read the total applied force on *sideset_id* from a single surface VTU.

    Returns a (3,) float64 array [Fx, Fy, Fz], or None if the file / fields
    are missing.
    """
    if not os.path.isfile(surf_vtu_path):
        print(f"[WARN] Not found: {surf_vtu_path}")
        return None

    root = ET.parse(surf_vtu_path).getroot()
    piece = root.find(".//Piece")
    n_pts = int(piece.attrib["NumberOfPoints"])

    ss_elem = piece.find(".//*[@Name='sidesets']")
    tf_elem = piece.find(".//*[@Name='traction_force']")
    if ss_elem is None or tf_elem is None:
        print(f"[WARN] sidesets / traction_force not found in {surf_vtu_path}")
        return None

    sidesets = decode_array(ss_elem).astype(np.float64)
    traction = decode_array(tf_elem).astype(np.float64).reshape(n_pts, 3)

    mask = np.round(sidesets).astype(int) == sideset_id
    if not mask.any():
        print(f"[WARN] No nodes found on sideset {sideset_id} in {surf_vtu_path}")
        return np.zeros(3)

    return traction[mask].sum(axis=0)


def plot_force_evolution(sim_dir, output_dir, steps, sideset_id=1,
                         force_unit_label="kN  (GPa·mm²)"):
    """Save force_evolution.png for the requested simulation steps."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    steps_found = []
    forces = []

    for step in steps:
        path = os.path.join(sim_dir, f"step_{step}_surf.vtu")
        force = load_force_at_step(path, sideset_id)
        if force is not None:
            steps_found.append(step)
            forces.append(force)
            print(
                f"Step {step:2d}:  Fx={force[0]:+9.4g}  "
                f"Fy={force[1]:+9.4g}  Fz={force[2]:+9.4g}"
            )

    if not steps_found:
        print("[WARN] No force data found – nothing to plot.")
        return None

    step_array = np.array(steps_found)
    forces = np.array(forces)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(step_array, forces[:, 0], "o-", color="tomato", linewidth=2,
            markersize=6, label=r"$F_x$")
    ax.plot(step_array, forces[:, 1], "s-", color="steelblue", linewidth=2,
            markersize=6, label=r"$F_y$  (applied, Neumann BC)")
    ax.plot(step_array, forces[:, 2], "^-", color="seagreen", linewidth=2,
            markersize=6, label=r"$F_z$")

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    ax.set_xlabel("Simulation step", fontsize=12)
    ax.set_ylabel(f"Force  [{force_unit_label}]", fontsize=12)
    ax.set_title(f"Applied force on sideset {sideset_id}", fontsize=13)
    ax.set_xticks(step_array.tolist())
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "force_evolution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nForce evolution plot saved → {out_path}")
    return out_path


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

    steps = sorted(step_data.keys())
    max_fi = []
    mean_fi = []
    frac_fail = []

    for step in steps:
        fi = step_data[step][3]
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

    ax1.plot(steps, max_fi, 'o-', color='crimson', linewidth=2, label='Max FI')
    ax1.plot(steps, mean_fi, 's--', color='darkorange', linewidth=1.5, label='Mean FI')
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
    return out_path