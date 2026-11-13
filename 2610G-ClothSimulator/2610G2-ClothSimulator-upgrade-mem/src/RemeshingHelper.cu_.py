from pathlib import Path

import numpy as np


def faceSizingInput(filename=None):
    """Capture inputs while paused in faceSizing, on either CPU or CUDA.

    Debug console: -exec source RemeshingHelper.cu_.py faceSizingInput
    Load later with np.load("mem.data", allow_pickle=False).
    The default output is mem.data in the project root; each call replaces it.
    """
    def read(expression, dtype=np.float32):
        value = mem(expression)
        if value is None:
            raise ValueError(f"Cannot read {expression}; pause inside faceSizing")
        if isinstance(value, dict):
            value = value["data"]
        return np.asarray(value, dtype=dtype)

    data = {
        "format_version": np.asarray(1, dtype=np.int32),
        "face_inverse": read("face->inverse").reshape(2, 2),
        "face_area": read("face->area"),
    }
    indices = []
    data["vertex_u"] = np.stack([
        read(f"face->vertices[{i}]->u").reshape(2) for i in range(3)
    ])
    for field in ("x", "n", "v"):
        data[f"node_{field}"] = np.stack([
            read(f"face->vertices[{i}]->node->{field}").reshape(3)
            for i in range(3)
        ])
    for i in range(3):
        index = int(read(f"face->vertices[{i}]->node->index", np.int64))
        if index < 0:
            raise ValueError(f"Invalid node index: {index}")
        indices.append(index)
    data["node_index"] = np.asarray(indices, dtype=np.int64)
    for field in ("p", "n"):
        data[f"plane_{field}"] = np.stack([
            read(f"planes[{index}].{field}").reshape(3) for index in indices
        ])

    # Face::curvature uses each edge's UV direction and dihedral angle.
    # Preserve the geometry needed by Edge::angle, without calling C++ methods.
    boundary = np.zeros(3, dtype=np.bool_)
    edge_positions = np.zeros((3, 2, 3), dtype=np.float32)
    adjacent_normals = np.zeros((3, 2, 3), dtype=np.float32)
    for i in range(3):
        adjacent = [int(read(f"face->edges[{i}]->adjacents[{j}]", np.uint64))
                    for j in range(2)]
        boundary[i] = 0 in adjacent
        if not boundary[i]:
            for j in range(2):
                edge_positions[i, j] = read(f"face->edges[{i}]->nodes[{j}]->x").reshape(3)
                adjacent_normals[i, j] = read(f"face->edges[{i}]->adjacents[{j}]->n").reshape(3)
    data["edge_boundary"] = boundary
    data["edge_node_x"] = edge_positions
    data["edge_adjacent_n"] = adjacent_normals
    for field in ("refineAngle", "refineVelocity", "refineCompression",
                  "ribStiffening", "sizeMin", "sizeMax", "aspectMin", "triangleDensity"):
        data[f"remeshing_{field}"] = read(f"remeshing->{field}")

    output = (Path(filename) if filename is not None else
              Path(__file__).resolve().parent.parent / "mem.data")
    output.parent.mkdir(parents=True, exist_ok=True)
    # A file handle prevents NumPy from appending .npz to the requested filename.
    with output.open("wb") as stream:
        np.savez(stream, **data)
    print(f"Saved faceSizing input: {output.resolve()}")
    return output


def maxTensor(Agg="Agg"):
    """Plot the C++ disks while paused on the return statement in maxTensor.

    Run with: -exec source RemeshingHelper.cu_.py maxTensor Agg
    Agg saves mem/figure/maxTensor.png; QtAgg also opens a window.
    """
    if Agg not in ("Agg", "QtAgg"):
        raise ValueError("Display argument must be Agg or QtAgg")

    n = int(mem("n").item())
    if not 0 <= n <= 5:
        raise ValueError(f"Invalid disk count {n}; pause on the return in maxTensor")
    matrices = [mem(f"M[{i}]").reshape(2, 2) for i in range(5)]
    active = [i for i, matrix in enumerate(matrices) if np.trace(matrix) != 0]
    if len(active) != n:
        raise ValueError("Disk count does not match M; pause on the return in maxTensor")
    # Read only initialized entries. d[] packs the nonzero-trace input matrices.
    disks = [(mem(f"d[{i}].o").reshape(2), mem(f"d[{i}].r").item())
             for i in range(n)]
    center = mem("disk.o", "", 2)
    radius = mem("disk.r", "", 1)
    answer = mem("ans", "", 2)

    import matplotlib
    matplotlib.use(Agg)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig = plt.figure("maxTensor", figsize=(13, 7))
    fig.clear()
    ax = fig.add_subplot(121)
    panel = fig.add_subplot(122)
    panel.axis("off")
    lines = ["C++ maxTensor values", f"Active disks: {n} / 5", ""]

    def draw_disk(origin, r, color, label, **style):
        if not np.isfinite(origin).all() or not np.isfinite(r) or r < 0:
            lines.append(f"{label}: invalid disk (not plotted)")
            return
        ax.add_patch(Circle(origin, r, fill=False, edgecolor=color,
                            label=label, **style))
        ax.plot(*origin, marker="+", color=color)

    for j, (origin, r) in enumerate(disks):
        draw_disk(origin, r, f"C{active[j]}", f"M[{active[j]}]", linewidth=1.5)
    draw_disk(center, radius, "black", "Enclosing disk", linewidth=2, linestyle="--")
    for i, matrix in enumerate(matrices):
        status = "" if i in active else " (ignored: zero trace)"
        lines.append(f"M[{i}]{status}:\n" + np.array2string(matrix, precision=5))
    lines.extend(["", f"disk.o: {center}", f"disk.r: {radius:.6g}",
                  "ans:\n" + np.array2string(answer, precision=5)])
    ax.autoscale_view()
    ax.margins(0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="(M00 - M11) / 2", ylabel="(M01 + M10) / 2",
           title="maxTensor disks (radius = trace / 2)")
    ax.grid(alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    panel.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    output = Path(__file__).resolve().parent.parent / "mem/figure/maxTensor.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved maxTensor plot: {output}")
    if Agg == "QtAgg":
        plt.show(block=True)
    else:
        plt.close(fig)
    return fig, ax


def faceSizing(Agg="Agg"):
    """Plot the current faceSizing frame: faceSizing QtAgg opens a window."""
    if Agg not in ("Agg", "QtAgg"):
        raise ValueError("Display argument must be Agg or QtAgg")
    import matplotlib
    matplotlib.use(Agg)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def read(expression):
        value = mem(expression)
        if value is None:
            raise ValueError(f"Cannot read {expression}; pause inside faceSizing")
        if isinstance(value, dict):
            value = value["data"]
        return np.asarray(value)

    positions = np.stack([read(f"face->vertices[{i}]->node->x").reshape(3)
                          for i in range(3)]).astype(float)
    normals = np.stack([read(f"face->vertices[{i}]->node->n").reshape(3)
                       for i in range(3)]).astype(float)
    indices = [int(read(f"face->vertices[{i}]->node->index")) for i in range(3)]
    if not np.isfinite(positions).all():
        raise ValueError("Face positions contain nonfinite values")
    plane_points, plane_normals = [], []
    for index in indices:
        if index < 0:
            raise ValueError(f"Invalid node index: {index}")
        plane_normals.append(read(f"planes[{index}].n").reshape(3))
        plane_points.append(read(f"planes[{index}].p").reshape(3))
    plane_points = np.asarray(plane_points, dtype=float)
    plane_normals = np.asarray(plane_normals, dtype=float)
    active = (np.isfinite(plane_points).all(axis=1)
              & np.isfinite(plane_normals).all(axis=1)
              & (np.linalg.norm(plane_normals, axis=1) > 0))

    plt.switch_backend(Agg)
    fig = plt.figure("faceSizing", figsize=(15, 9))
    fig.clear()
    ax = fig.add_subplot(121, projection="3d")
    ax.add_collection3d(Poly3DCollection(
        [positions], facecolor="lightblue", edgecolor="black", alpha=0.6))
    ax.scatter(*positions.T, color="crimson", label="Face nodes")
    for corner, (index, position) in enumerate(zip(indices, positions)):
        ax.text(*position, f" {corner}: node {index}", color="crimson")
    scale = max(float(np.max(np.linalg.norm(
        positions - np.roll(positions, 1, axis=0), axis=1))), 1e-6) * 0.2
    valid_normals = np.isfinite(normals).all(axis=1) & (np.linalg.norm(normals, axis=1) > 0)
    if valid_normals.any():
        ax.quiver(*positions[valid_normals].T, *normals[valid_normals].T,
                  length=scale, normalize=True, color="green", label="Node normals")
    for i in np.flatnonzero(active):
        n = plane_normals[i] / np.linalg.norm(plane_normals[i])
        projection = positions[i] - np.dot(positions[i] - plane_points[i], n) * n
        ax.plot(*np.stack([positions[i], projection]).T, "--", color="orange")
        ax.quiver(*projection, *n, length=scale, color="orange",
                  label="Obstacle plane normal" if i == np.flatnonzero(active)[0] else None)
    ax.auto_scale_xyz(*positions.T)
    ax.set_box_aspect(np.maximum(np.ptp(positions, axis=0), scale))
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="faceSizing input geometry")
    ax.legend()
    panel = fig.add_subplot(122)
    panel.axis("off")
    lines = ["C++ faceSizing values", "", "Remeshing settings"]

    def display(label, expression):
        try:
            value = read(expression)
            text = np.array2string(value, precision=5, suppress_small=True,
                                   max_line_width=65)
        except (ValueError, TypeError, RuntimeError, KeyError):
            text = "unavailable"
        lines.append(f"{label}: {text}")

    for field in ("refineAngle", "refineVelocity", "refineCompression",
                  "ribStiffening", "sizeMin", "sizeMax", "aspectMin", "triangleDensity"):
        display(field, f"remeshing->{field}")
    display("face area", "face->area")
    lines.extend(["", "Local metrics (at the paused C++ line)"])
    # GDB can expose stack storage before initialization. Only show a local
    # after execution has passed the statement that assigns it.
    try:
        import gdb
        location = gdb.selected_frame().find_sal()
        source_lines = Path(location.symtab.fullname()).read_text().splitlines()
        start = next(i for i, line in enumerate(source_lines)
                     if line.startswith("Matrix2x2f faceSizing("))
        executed = "\n".join(source_lines[start:location.line - 1])
    except (ImportError, AttributeError, OSError, StopIteration, RuntimeError):
        executed = ""
    for label, expression, statement in (
        ("M[0] curvature", "M[0]", "M[0] ="),
        ("M[1] normal variation", "M[1]", "M[1] ="),
        ("M[2] velocity", "M[2]", "M[2] ="),
        ("M[3] compression", "M[3]", "M[3] ="),
        ("M[4] obstacle", "M[4]", "M[4] ="),
        ("S combined", "S", "Matrix2x2f S ="),
        ("Q eigenvectors", "Q", "eigenvalueDecomposition(S, Q, l);"),
        ("l eigenvalues (current)", "l", "eigenvalueDecomposition(S, Q, l);"),
        ("lMax", "lMax", "float lMax ="),
        ("lMin aspect limit", "lMin", "float lMin ="),
        ("matrix final sizing", "matrix", "Matrix2x2f matrix ="),
    ):
        if statement in executed:
            display(label, expression)
        else:
            lines.append(f"{label}: pending")
    panel.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    output = Path(__file__).resolve().parent.parent / "mem/figure/faceSizing.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved faceSizing plot: {output}")
    if Agg == "QtAgg":
        plt.show(block=True)
    else:
        plt.close(fig)
    return fig, ax

def flipGpu(Agg="Agg"):
    """Plot the current flipGpu frame after the findEdge calls.

    Run: -exec source RemeshingHelper.cu_.py flipGpu Agg
    """
    if Agg not in ("Agg", "QtAgg"):
        raise ValueError("Display argument must be Agg or QtAgg")

    def read(expression):
        value = mem(expression)
        if value is None:
            raise ValueError(f"Cannot read {expression}; pause after the findEdge calls in flipGpu")
        if isinstance(value, dict):
            if "data" not in value:
                raise ValueError(f"Unexpected value for {expression}: {list(value)}")
            value = value["data"]
        return np.asarray(value)

    def pointer(expression):
        # mem() dereferences pointers to structs, returning their fields instead
        # of the address. Evaluate the pointer itself in the selected frame.
        import gdb
        try:
            return int(gdb.parse_and_eval(expression))
        except (gdb.error, RuntimeError, ValueError) as exc:
            raise ValueError(f"Cannot read pointer {expression}; pause after the findEdge calls in flipGpu") from exc

    statements = [
        ("Vertex* vertex0 = edge->vertices[0][0];", "vertex0"),
        ("Vertex* vertex1 = edge->vertices[1][1];", "vertex1"),
        ("Vertex* vertex2 = edge->opposites[0];", "vertex2"),
        ("Vertex* vertex3 = edge->opposites[1];", "vertex3"),
        ("Face* face0 = edge->adjacents[0];", "face0"),
        ("Face* face1 = edge->adjacents[1];", "face1"),
        ("Edge* edge0 = face0->findEdge(vertex1, vertex2);", "edge0"),
        ("Edge* edge1 = face0->findEdge(vertex2, vertex0);", "edge1"),
        ("Edge* edge2 = face1->findEdge(vertex0, vertex3);", "edge2"),
        ("Edge* edge3 = face1->findEdge(vertex3, vertex1);", "edge3"),
    ]
    values = {name: pointer(name) for _, name in statements}
    if any(not values[f"vertex{i}"] for i in range(4)) or any(
            not values[f"face{i}"] for i in range(2)):
        raise ValueError("A vertex or face is null; flipGpu requires two complete adjacent faces")

    import matplotlib
    matplotlib.use(Agg)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, (ax, panel) = plt.subplots(1, 2, figsize=(17, 9),
                                    gridspec_kw={"width_ratios": [1, 1.25]})
    panel.axis("off")
    names = {values[f"vertex{i}"]: f"vertex{i}" for i in range(4)}
    positions, indices = {}, {}
    for i in range(4):
        name = f"vertex{i}"
        positions[values[name]] = read(f"{name}->u").astype(float).reshape(2)
        indices[values[name]] = int(read(f"{name}->index").item())
    face_vertices = []
    for i in range(2):
        vertices = [pointer(f"face{i}->vertices[{j}]") for j in range(3)]
        face_vertices.append(vertices)
        for j, ptr in enumerate(vertices):
            if not ptr:
                raise ValueError(f"face{i}->vertices[{j}] is null")
            if ptr not in positions:
                positions[ptr] = read(f"face{i}->vertices[{j}]->u").astype(float).reshape(2)
                indices[ptr] = int(read(f"face{i}->vertices[{j}]->index").item())
                names[ptr] = f"face{i}.vertex{j}"
        coords = np.array([positions[ptr] for ptr in vertices])
        if not np.isfinite(coords).all():
            raise ValueError("Vertex UV coordinates contain nonfinite values")
        ax.add_patch(Polygon(coords, closed=True, facecolor=("#95c9f0", "#ffc688")[i],
                             edgecolor="none", alpha=0.55))
        ax.annotate(f"face{i}", coords.mean(axis=0), ha="center", fontsize=11, weight="bold")

    edge_names = {pointer("edge"): "edge (flip)"}
    edge_names.update({values[f"edge{i}"]: f"edge{i}" for i in range(4)
                       if values[f"edge{i}"]})
    drawn = set()
    for i, vertices in enumerate(face_vertices):
        for j, start in enumerate(vertices):
            end = vertices[(j + 1) % 3]
            ptr = pointer(f"face{i}->edges[{j}]")
            key = (ptr, min(start, end), max(start, end))
            if key in drawn:
                continue
            drawn.add(key)
            a, b = positions[start], positions[end]
            label = edge_names.get(ptr, f"face{i}.edge{j}" if ptr else "null edge")
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#333333", linewidth=2)
            ax.annotate(label, (a + b) / 2, xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=9, color="#5a2580",
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))
    for ptr, pos in positions.items():
        ax.scatter(*pos, color="crimson", zorder=3)
        ax.annotate(f"{names[ptr]} (index {indices[ptr]})", pos,
                    xytext=(5, 6), textcoords="offset points", fontsize=9, color="crimson")
    ax.autoscale_view()
    ax.margins(0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="u", ylabel="v", title="flipGpu: adjacent faces in UV space")
    ax.grid(alpha=0.25)

    lines = ["C++ statement and pointer value", ""]
    for statement, name in statements:
        value = f"0x{values[name]:x}" if values[name] else "nullptr"
        lines.extend([statement, f"    {name} = {value}"])
    lines += ["", "edge->vertices entries"]
    for row in range(2):
        for column in range(2):
            expression = f"edge->vertices[{row}][{column}]"
            ptr = pointer(expression)
            if ptr:
                index = int(read(f"{expression}->index").item())
                lines.append(f"{expression} = 0x{ptr:x} (index {index})")
            else:
                lines.append(f"{expression} = nullptr")
    lines += ["", "Vertex indices and UV coordinates"]
    for i in range(4):
        ptr = values[f"vertex{i}"]
        lines.append(f"vertex{i}: index {indices[ptr]}, u = {np.array2string(positions[ptr], precision=5)}")
    lines += ["", "Expected faces: (vertex0, vertex1, vertex2)",
              "                (vertex0, vertex1, vertex3)"]
    for i, expected in enumerate(((0, 1, 2), (0, 1, 3))):
        status = "OK" if set(face_vertices[i]) == {values[f"vertex{j}"] for j in expected} else "MISMATCH"
        lines.append(f"face{i} vertex pointers: {status}")
    if any(not values[f"edge{i}"] for i in range(4)):
        lines.append("A findEdge result is null; subsequent code dereferences it.")
    panel.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=7.5)
    fig.tight_layout()
    output = Path(__file__).resolve().parent.parent / "mem/figure/flipGpu.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved flipGpu plot: {output}")
    if Agg == "QtAgg":
        plt.show(block=True)
    else:
        plt.close(fig)
        import subprocess
        try:
            subprocess.Popen(["xdg-open", str(output)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            print(f"Could not open flipGpu plot: {exc}")
    return fig, ax