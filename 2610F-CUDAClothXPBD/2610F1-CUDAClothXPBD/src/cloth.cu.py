"""Run build_triangles after Cloth::build_triangles fills the host indices.

Use build_triangles_cc at the CPU test's printf breakpoint instead.
"""
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mem import mem

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
    for x, z in vertices.values():
        ax.plot(x, z, "o", color="crimson", markersize=4)
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
    output_path = output_dir / "cloth_triangles.png"
    fig.savefig(output_path, dpi=180)

    if Agg == "Agg":
        plt.close(fig)
        subprocess.Popen(["xdg-open", str(output_path)])
    else:
        plt.show(block=True)

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
    output_path = project_root / "mem" / "figure" / "cloth_simulate_step.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)

    if Agg == "Agg":
        plt.close(fig)
        subprocess.Popen(["xdg-open", str(output_path)])
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
    output_path = output_dir / "cloth_triangles.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    subprocess.Popen(["xdg-open", str(output_path)])




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
    output_path = output_dir / "cloth_triangles.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

