from pathlib import Path

import numpy as np


def curvature(Agg="Agg"):
    """Plot this face and its edge neighbors while paused in Face::curvature.

    Debugger extension: -exec source Face.cu_.py curvature QtAgg
    Use Agg to save mem/figure/curvature.png without opening a window.
    Each edge has two adjacent slots: this face and at most one neighbor.
    """
    if Agg not in ("Agg", "QtAgg"):
        raise ValueError("Display argument must be Agg or QtAgg")
    import matplotlib
    matplotlib.use(Agg)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def read(expression):
        value = mem(expression)
        if value is None:
            raise ValueError(f"Cannot read {expression}; pause inside Face::curvature")
        if isinstance(value, dict):
            value = value["data"]
        return np.asarray(value)

    def pointer(expression):
        # mem otherwise dereferences struct pointers, including Face*.
        return int(read(f"(unsigned long long)({expression})").item())

    def vector(expression, size):
        value = read(expression).astype(float).reshape(size)
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite geometry: {expression}")
        return value

    def face_geometry(expression):
        return np.stack([vector(f"{expression}->vertices[{j}]->node->x", 3)
                         for j in range(3)])

    current = pointer("this")
    positions = face_geometry("this")
    normal = vector("this->n", 3)
    uv = np.stack([vector(f"this->vertices[{j}]->u", 2) for j in range(3)])
    indices = [int(read(f"this->vertices[{j}]->node->index").item()) for j in range(3)]
    area = float(read("this->area").item())
    if not np.isfinite(area) or area <= 0:
        raise ValueError(f"Face area must be positive and finite, got {area}")

    neighbors = {}
    edge_data = []
    tensor = np.zeros((2, 2))
    colors = ["#d97706", "#16a34a", "#9333ea"]
    for i in range(3):
        edge = f"this->edges[{i}]"
        if pointer(edge) == 0:
            raise ValueError(f"Edge {i} is null; pause after mesh initialization")
        adjacent = [pointer(f"{edge}->adjacents[{j}]") for j in range(2)]
        if adjacent.count(current) != 1:
            raise ValueError(f"Edge {i} must contain this face in exactly one adjacent slot")
        other_slot = 1 - adjacent.index(current)
        other = adjacent[other_slot]
        neighbor = f"{edge}->adjacents[{other_slot}]"
        endpoints = np.stack([vector(f"{edge}->nodes[{j}]->x", 3) for j in range(2)])
        angle = 0.0
        if other:
            if other not in neighbors:
                neighbors[other] = (face_geometry(neighbor), vector(f"{neighbor}->n", 3), i)
            # Match Edge::angle's signed orientation; do not call inferior methods.
            direction = endpoints[0] - endpoints[1]
            length = np.linalg.norm(direction)
            if length == 0:
                raise ValueError(f"Edge {i} has zero world-space length")
            n0, n1 = [vector(f"{edge}->adjacents[{j}]->n", 3) for j in range(2)]
            angle = float(np.arctan2(np.dot(direction / length, np.cross(n0, n1)),
                                    np.dot(n0, n1)))
        e = uv[(i + 1) % 3] - uv[i]
        length = np.linalg.norm(e)
        if length == 0:
            raise ValueError(f"Edge {i} has zero material-space length")
        t = np.array([-e[1], e[0]]) / length
        tensor -= 0.5 * angle * length * np.outer(t, t) / area
        edge_data.append((endpoints, angle, other))

    plt.switch_backend(Agg)
    fig = plt.figure("Face::curvature", figsize=(15, 8))
    fig.clear()
    grid = fig.add_gridspec(2, 2, width_ratios=(1.6, 1))
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    material = fig.add_subplot(grid[0, 1])
    panel = fig.add_subplot(grid[1, 1])
    all_points = np.concatenate([positions] + [item[0] for item in neighbors.values()])
    scale = max(float(np.max(np.ptp(all_points, axis=0))), 1e-6)

    def draw_face(points, n, color, label):
        ax.add_collection3d(Poly3DCollection(
            [points], facecolor=color, edgecolor=color, alpha=0.35, linewidth=1.5))
        center = points.mean(axis=0)
        ax.text(*center, label, color=color, fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
        if np.linalg.norm(n) > 0:
            ax.quiver(*center, *n, length=scale * 0.16, normalize=True, color=color)

    draw_face(positions, normal, "#2563eb", "this face")
    for points, n, i in neighbors.values():
        draw_face(points, n, colors[i], f"neighbor across e{i}")
    ax.scatter(*positions.T, color="#2563eb", s=30)
    for j, (point, index) in enumerate(zip(positions, indices)):
        ax.text(*point, f" v{j} / node {index}", fontsize=9)

    material.add_patch(Polygon(uv, facecolor="#dbeafe", edgecolor="none"))
    lines = [f"{len(neighbors)} edge neighbor(s) for this face",
             "2 faces per interior edge = this + neighbor", ""]
    for i, (endpoints, angle, other) in enumerate(edge_data):
        label = f"e{i}: {np.degrees(angle):+.2f} deg" if other else f"e{i}: boundary (0 deg)"
        style = "-" if other else "--"
        ax.plot(*endpoints.T, style, color=colors[i], linewidth=3, label=label)
        ax.text(*endpoints.mean(axis=0), f" e{i}", color=colors[i])
        pair = uv[[i, (i + 1) % 3]]
        material.plot(*pair.T, style, color=colors[i], linewidth=2)
        material.annotate(f"e{i}", pair.mean(axis=0), xytext=(5, 5),
                          textcoords="offset points", color=colors[i])
        lines.append(f"{label}  [{angle:+.5f} rad]")
    for j, point in enumerate(uv):
        material.annotate(f"v{j}", point, xytext=(5, -12), textcoords="offset points")
    material.autoscale_view()
    material.margins(0.25)
    material.set_aspect("equal", adjustable="box")
    material.set(xlabel="u[0]", ylabel="u[1]", title="Current face in material coordinates")

    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    for set_limit, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        set_limit(value - scale * 0.65, value + scale * 0.65)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="Edge neighbors and face normals")
    ax.legend(loc="upper left", fontsize=9)
    matrix_text = np.array2string(tensor, formatter={"float_kind": lambda x: f"{x: .6g}"})
    lines.extend(["", f"Material area = {area:.6g}",
                  "Curvature (recomputed from geometry):", matrix_text,
                  "", "K = -sum(angle * |e| * outer(t,t)) / (2*area)",
                  "t = unit perpendicular to material edge e"])
    panel.axis("off")
    panel.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    output = Path(__file__).resolve().parent.parent / "mem/figure/curvature.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print("\n".join(lines[:6]))
    print(f"Saved curvature plot: {output}")
    if Agg == "QtAgg":
        plt.show(block=True)
    else:
        plt.close(fig)
    return fig, ax
