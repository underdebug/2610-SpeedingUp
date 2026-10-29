"""Run build_triangles after Cloth::build_triangles fills the host indices.

Use build_triangles_cc at the CPU test's printf breakpoint instead.
"""
import os
import subprocess
from pathlib import Path

# from mem import mem

# Qt otherwise warns when choosing X11 in a GNOME Wayland session.
# Preserve any display backend explicitly selected by the user.
if os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build_constraints(Agg="Agg"):
    """Export one PNG per pass from the selected build_constraints frame.

    Pause after dist_const_ids is filled, before build_constraints returns.
    The filename suffix is the zero-based pass index.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if Agg not in ("Agg", "QtAgg"):
        Agg = "Agg"
    plt.switch_backend(Agg)

    def read(name, **kwargs):
        value = mem(name, **kwargs)
        if value is None:
            raise RuntimeError(
                f"Cannot read {name}; pause after the constraint loops "
                "inside Cloth::build_constraints."
            )
        return value

    num_particles = int(read("num_particles"))
    num_constraints = int(read("num_dist_constraints"))
    positions = np.column_stack([
        read("h_pos_x", Size=num_particles),
        read("h_pos_z", Size=num_particles),
    ])
    raw_ids = read("d_dist_const_ids", Size=2 * num_constraints)
    print("Reading constraint indices from d_dist_const_ids")
    dist_const_ids = np.asarray(
        raw_ids, dtype=int
    ).reshape(-1)
    indices = dist_const_ids.reshape(-1, 2)
    sizes = np.asarray(read("pass_sizes"), dtype=int).reshape(-1)
    if len(indices) != num_constraints or sizes.sum() != num_constraints:
        raise ValueError("Constraint count does not match dist_const_ids/pass_sizes")
    if np.any(indices < 0) or np.any(indices >= num_particles):
        raise ValueError("Constraint endpoint index is outside the particle array")
    independent = np.asarray(read("pass_independent"), dtype=bool).reshape(-1)
    if len(sizes) != 5 or len(independent) != len(sizes) or np.any(sizes < 0):
        raise ValueError("Expected five passes with matching independence flags")
    pass_ends = np.cumsum(sizes)

    project_root = Path(os.environ.get("PROJECT_ROOT") or
                        Path(__file__).resolve().parent.parent).resolve()
    if project_root.name == "src":
        project_root = project_root.parent
    output_dir = project_root / "mem" / "figure"
    output_dir.mkdir(parents=True, exist_ok=True)

    pass_names = ["Structural Z, parity 0", "Structural Z, parity 1",
                  "Structural X, parity 0", "Structural X, parity 1",
                  "Shear + bending"]
    focus_id = 9 if num_particles > 9 else None
    # Filtering preserves the device-buffer sequence and global constraint IDs.
    def direction(cid):
        a, b = indices[cid]
        dx, dz = positions[b] - positions[a]
        if np.isclose(dx, 0):
            return "vertical"
        if np.isclose(dz, 0):
            return "horizontal"
        return "diagonal"

    direction_titles = {"diagonal": "Diagonal 45° + 135°",
                        "vertical": "Vertical", "horizontal": "Horizontal"}
    jobs = [(pass_nr, False, None) for pass_nr in range(4)]
    jobs.extend((4, False, kind) for kind in ("diagonal", "vertical", "horizontal"))
    if focus_id is not None:
        jobs.append((4, True, None))
    for pass_nr, focused, kind in jobs:
        start = int(pass_ends[pass_nr - 1]) if pass_nr else 0
        end = int(pass_ends[pass_nr])
        constraint_ids = [cid for cid in range(start, end)
                          if (not focused or focus_id in indices[cid])
                          and (kind is None or direction(cid) == kind)]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(*positions.T, color="0.25", s=25, zorder=3)
        for particle, point in enumerate(positions):
            ax.annotate(f"P{particle}", point, xytext=(5, 6),
                        textcoords="offset points", fontsize=9, zorder=5)
        # Give every constraint in this pass its own color and global ID.
        count = len(constraint_ids)
        colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, max(end - start, 1)))
        for constraint_id in constraint_ids:
            local_id = constraint_id - start
            a, b = indices[constraint_id]
            color = colors[local_id]
            ax.plot(*positions[[a, b]].T, color=color, linewidth=2,
                    label=f"C{constraint_id} [{2 * constraint_id}, {2 * constraint_id + 1}] "
                          f"[{a}, {b}]", zorder=2)
            # One label per constraint, at 1/3 with a perpendicular offset.
            delta = positions[b] - positions[a]
            length = float(np.linalg.norm(delta))
            normal = np.array([-delta[1], delta[0]]) / length if length else np.array([0., 1.])
            label_position = positions[a] + delta / 3
            ax.annotate(f"C{constraint_id} [{2 * constraint_id}, {2 * constraint_id + 1}] "
                        f"[{a}, {b}]",
                        label_position, xytext=tuple(7 * normal),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=7, color=color,
                        bbox=dict(facecolor="white", alpha=0.85,
                                  edgecolor="none", pad=0.5), zorder=5)
        focus_title = f" — point P{focus_id}" if focused else ""
        pass4_names = {"diagonal": "4_1 Diag", "vertical": "4_2 Vertical",
                       "horizontal": "4_3 Hor"}
        plot_title = ("Pass 4_4 P9" if focused else
                      f"Pass {pass4_names[kind]}" if pass_nr == 4 else
                      f"Pass {pass_nr}: {pass_names[pass_nr]}")
        if focused:
            ax.scatter(*positions[focus_id], color="crimson", marker="*",
                       s=220, zorder=6, label=f"Selected P{focus_id}")
        ax.set(xlabel="X", ylabel="Z",
               title=f"{plot_title}\n"
                     f"{count} constraints, independent={bool(independent[pass_nr])}\n"
                     "In d_dist_const_ids order (original C IDs and array indices)")
        ax.set_aspect("equal")
        ax.margins(0.18)
        if count:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
                      fontsize=7, ncol=max(1, (count + 23) // 24))
        fig.tight_layout()
        image_name = ("4_4_P9" if focused else
                      pass4_names[kind].replace(" ", "_") if pass_nr == 4 else
                      str(pass_nr))
        fig.savefig(output_dir / f"build_constraints_{image_name}.png", dpi=180)
        if Agg == "QtAgg":
            plt.show(block=True)
        plt.close(fig)

    print(f"Saved {len(jobs)} constraint images to {output_dir}")
    return output_dir


def simulate_step(Agg="QtAgg"):
    if Agg not in ("Agg", "QtAgg"): Agg = "Agg"
    matplotlib.use(Agg)
    import matplotlib.pyplot as plt

    """Plot the current cloth and sphere from a Cloth::simulate_step frame."""
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    num_particles = int(mem("num_particles"))
    num_tris = int(mem("num_tris"))
    # Device positions stay current even before the final device-to-host copy.
    vertices = np.column_stack([
        mem(name, Size=num_particles)
        for name in ("d_pos_x", "d_pos_y", "d_pos_z")
    ])
    triangles = np.asarray(mem("h_tri_ids", Size=3 * num_tris),
                           dtype=int).reshape(-1, 3)
    center = np.array([float(mem(f"sphere_center.{axis}"))
                       for axis in ("x", "y", "z")])
    radius = float(mem("sphere_radius"))

    # Match State::draw_sphere(), reached through main.cu's State instance.
    # The sphere is procedural; there is no stored sphere index buffer to read.
    slices, stacks = 10, 9

    def sphere_vertex(j, i):
        latitude = -0.5 * np.pi + np.pi * j / stacks
        longitude = 2.0 * np.pi * i / slices
        ring_radius = radius * np.cos(latitude)
        return center + (ring_radius * np.cos(longitude),
                         radius * np.sin(latitude),
                         ring_radius * np.sin(longitude))

    sphere_triangles = []
    for j in range(stacks):
        for i in range(slices):
            if j > 0:
                sphere_triangles.append([sphere_vertex(j, i),
                                         sphere_vertex(j + 1, i),
                                         sphere_vertex(j, i + 1)])
            if j < stacks - 1:
                sphere_triangles.append([sphere_vertex(j, i + 1),
                                         sphere_vertex(j + 1, i),
                                         sphere_vertex(j + 1, i + 1)])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Close each triangle explicitly; line collections avoid Matplotlib 3.6's
    # empty-facecolor failure when projecting unfilled Poly3DCollection objects.
    ax.add_collection3d(Line3DCollection(
        vertices[triangles][:, [0, 1, 2, 0]], colors="steelblue", linewidths=0.5))
    ax.add_collection3d(Line3DCollection(
        np.asarray(sphere_triangles)[:, [0, 1, 2, 0]],
        colors="gray", linewidths=0.7))
    # Include the full collision sphere and keep equal world-unit scales.
    bounds = np.vstack((vertices, center - radius, center + radius))
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    midpoint = (lower + upper) / 2
    half_extent = max(float((upper - lower).max()) * 0.55, 1e-6)
    ax.set(xlim=(midpoint[0] - half_extent, midpoint[0] + half_extent),
           ylim=(midpoint[1] - half_extent, midpoint[1] + half_extent),
           zlim=(midpoint[2] - half_extent, midpoint[2] + half_extent),
           xlabel="X", ylabel="Y", zlabel="Z",
           title=f"Cloth::simulate_step\n{num_particles} vertices / {num_tris} triangles")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-60, vertical_axis="y")
    fig.tight_layout()
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    if project_root.name == "src":
        project_root = project_root.parent
    output_path = project_root / "mem" / "figure" / "simulate_step.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)

    if Agg == "Agg":
        plt.close(fig)
        subprocess.Popen(["xdg-open", str(output_path)], stderr=subprocess.DEVNULL)
    else:
        plt.show(block=True)

def simulate_step_cc():
    # Read live C++ local variables from the selected GDB frame.
    x = mem("h_pos_x")
    z = mem("h_pos_z")
    indices = mem("h_tri_ids")
    vertices = {i: (px, pz) for i, (px, pz) in enumerate(zip(x, z))}
    triangles = [tuple(int(v) for v in row) for row in indices.reshape(-1, 3)]

    fig, ax = plt.subplots(figsize=(9, 7))
    for t, ids in enumerate(triangles):
        points = [vertices[i] for i in ids]
        closed = points + points[:1]
        ax.plot(*zip(*closed), color="steelblue", linewidth=1)
        # Label each triangle center with its ID and three vertex indices.
        cx = sum(p[0] for p in points) / 3
        cz = sum(p[1] for p in points) / 3
        ax.text(cx, cz, f"T{t}\n{ids}", ha="center", va="center", fontsize=8)
    for i, (x, z) in vertices.items():
        ax.plot(x, z, "o", color="crimson", markersize=4)
        ax.annotate(f"P{i}", (x, z), xytext=(4, 5), textcoords="offset points")
    ax.set(xlabel="X", ylabel="Z", title="build_triangles_TEST\n"
           f"{len(vertices)} vertices / {len(triangles)} triangles")
    ax.set_aspect("equal")
    ax.margins(0.15)
    fig.tight_layout()
    # Save only the image; no intermediate data files.
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    if project_root.name == "src":
        project_root = project_root.parent
    output_dir = project_root / "mem" / "figure"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "simulate_step.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def build_triangles(Agg="Agg"):
    if Agg not in ("Agg", "QtAgg"): Agg = "Agg"
    matplotlib.use(Agg)
    import matplotlib.pyplot as plt

    # Cloth stores raw host pointers, so mem() needs explicit array sizes.
    num_particles = int(mem("num_particles"))
    num_tris = int(mem("num_tris"))
    x = mem("h_pos_x", Size=num_particles)
    z = mem("h_pos_z", Size=num_particles)
    indices = mem("h_tri_ids", Size=3 * num_tris)
    vertices = {i: (px, pz) for i, (px, pz) in enumerate(zip(x, z))}
    triangles = [tuple(int(v) for v in row) for row in indices.reshape(-1, 3)]

    fig, ax = plt.subplots(figsize=(9, 7))
    for ids in triangles:
        points = [vertices[i] for i in ids]
        closed = points + points[:1]
        ax.plot(*zip(*closed), color="steelblue", linewidth=1)
    for i, (x, z) in vertices.items():
        ax.plot(x, z, "o", color="crimson", markersize=4)
        ax.annotate(f"P{i}", (x, z), xytext=(4, 5), textcoords="offset points")
    ax.set(xlabel="X", ylabel="Z", title="Cloth::build_triangles\n"
           f"{len(vertices)} vertices / {len(triangles)} triangles")
    ax.set_aspect("equal")
    ax.margins(0.15)
    fig.tight_layout()
    # Save only the image; no intermediate data files.
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    if project_root.name == "src":
        project_root = project_root.parent
    output_dir = project_root / "mem" / "figure"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "build_triangles.png"
    fig.savefig(output_path, dpi=180)

    if Agg == "Agg":
        plt.close(fig)
        subprocess.Popen(["xdg-open", str(output_path)], stderr=subprocess.DEVNULL)
    else:
        plt.show(block=True)


def build_triangles_cc():
    # Read live C++ local variables from the selected GDB frame.
    x = mem("h_pos_x")
    z = mem("h_pos_z")
    indices = mem("h_tri_ids")
    vertices = {i: (px, pz) for i, (px, pz) in enumerate(zip(x, z))}
    triangles = [tuple(int(v) for v in row) for row in indices.reshape(-1, 3)]

    fig, ax = plt.subplots(figsize=(9, 7))
    for t, ids in enumerate(triangles):
        points = [vertices[i] for i in ids]
        closed = points + points[:1]
        ax.plot(*zip(*closed), color="steelblue", linewidth=1)
        # Label each triangle center with its ID and three vertex indices.
        cx = sum(p[0] for p in points) / 3
        cz = sum(p[1] for p in points) / 3
        ax.text(cx, cz, f"T{t}\n{ids}", ha="center", va="center", fontsize=8)
    for i, (x, z) in vertices.items():
        ax.plot(x, z, "o", color="crimson", markersize=4)
        ax.annotate(f"P{i}", (x, z), xytext=(4, 5), textcoords="offset points")
    ax.set(xlabel="X", ylabel="Z", title="build_triangles_TEST\n"
           f"{len(vertices)} vertices / {len(triangles)} triangles")
    ax.set_aspect("equal")
    ax.margins(0.15)
    fig.tight_layout()
    # Save only the image; no intermediate data files.
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    if project_root.name == "src":
        project_root = project_root.parent
    output_dir = project_root / "mem" / "figure"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "build_triangles.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    subprocess.Popen(["xdg-open", str(output_path)], stderr=subprocess.DEVNULL)
