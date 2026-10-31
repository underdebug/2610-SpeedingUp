
import numpy as np
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

np.set_printoptions(suppress=True)  # FORMAT LONG G *
np.set_printoptions(linewidth=150)  # MAX WIDTH NODELETE
np.set_printoptions(threshold=1000) # MAX NUMBER NODELETE
np.set_printoptions(edgeitems=10)   # FIRST LAST NUMBER NODELETE *

def add_triangle_hover(fig, ax, positions, indices, mesh):
    """Color the hovered triangle red using the current 3D camera."""
    colors = np.tile(matplotlib.colors.to_rgba("lightblue", 0.85),
                     (len(indices), 1))
    selected = None
    corner_labels = [ax.text(0, 0, 0, "", color="darkred", fontsize=11,
                             fontweight="bold", zorder=100,
                             bbox=dict(facecolor="white", alpha=0.8,
                                       edgecolor="none", pad=1),
                             visible=False)
                     for _ in range(3)]

    def highlight(face):
        nonlocal selected
        if face == selected:
            return
        if selected is not None:
            colors[selected] = matplotlib.colors.to_rgba("lightblue", 0.85)
        if face is not None:
            colors[face] = matplotlib.colors.to_rgba("red", 0.85)
        for corner, label in enumerate(corner_labels):
            label.set_visible(face is not None)
            if face is not None:
                vertex = indices[face, corner]
                label.set_position_3d(positions[vertex])
                label.set_text(str(vertex))
        selected = face
        mesh.set_facecolor(colors)
        fig.canvas.draw_idle()

    def hover(event):
        if event.inaxes != ax or event.button is not None:
            highlight(None)
            return
        px, py, depth = proj3d.proj_transform(*positions.T, ax.get_proj())
        screen = ax.transData.transform(np.column_stack((px, py)))
        corners = screen[indices]
        a = corners[:, 0]
        b = corners[:, 1] - a
        c = corners[:, 2] - a
        p = np.array([event.x, event.y]) - a
        determinant = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
        valid = np.abs(determinant) > 1e-10
        u = np.full(len(indices), np.nan)
        v = np.full(len(indices), np.nan)
        np.divide(p[:, 0] * c[:, 1] - p[:, 1] * c[:, 0],
                  determinant, out=u, where=valid)
        np.divide(b[:, 0] * p[:, 1] - b[:, 1] * p[:, 0],
                  determinant, out=v, where=valid)
        hits = np.flatnonzero(valid & (u >= 0) & (v >= 0) & (u + v <= 1))
        if len(hits):
            z = np.asarray(depth)[indices[hits]]
            hit_depth = ((1 - u[hits] - v[hits]) * z[:, 0]
                         + u[hits] * z[:, 1] + v[hits] * z[:, 2])
            face = hits[np.argmin(hit_depth)]
            highlight(int(face))
        else:
            highlight(None)

    return fig.canvas.mpl_connect("motion_notify_event", hover)


def read_vertices(name):
    # Vector3f is this project's Vector<float, 3>, with a data[3] member.
    vertices = mem(name)
    return np.asarray([
        vertex["data"] if isinstance(vertex, dict) else vertex
        for vertex in vertices
    ], dtype=np.float32).reshape(-1, 3)


def read_mesh_tables():
    """Read CPU vertex and face pointer tables in the selected Mesh frame."""
    vertices = mem("this->vertices")
    faces = mem("this->faces")
    if vertices is None or faces is None or not len(vertices) or not len(faces):
        raise ValueError("Empty mesh tables; pause after mesh initialization")
    vertex_rows = {int(vertex): i for i, vertex in enumerate(vertices)}
    positions = []
    for vertex in vertices:
        position = mem(f"((Vertex*){int(vertex)})->node->x")
        positions.append(position["data"])
    triangles = []
    for face in faces:
        corners = mem(f"((Face*){int(face)})->vertices")
        triangles.append([vertex_rows[int(corner)] for corner in corners])
    return (np.asarray(positions, dtype=np.float32).reshape(-1, 3),
            np.asarray(triangles, dtype=np.int32).reshape(-1, 3))


def load(filename="load.png", Agg=None):
    """Export PNG; Agg opens the image, QtAgg shows an interactive window."""
    output_path = Path(filename)
    if output_path.suffix.lower() != ".png":
        raise ValueError("Output filename must end in .png")
    if Agg not in (None, "Agg", "QtAgg"):
        raise ValueError("Optional display argument must be Agg or QtAgg")
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent.parent / "mem/figure" / output_path
    plt.switch_backend("QtAgg" if Agg == "QtAgg" else "Agg")
    
    if bool(mem("gpu")):
        # Device tables use GPU pointers; use the retained host geometry.
        x = read_vertices("x")
        indices = np.asarray(mem("xIndices"), dtype=np.int32).reshape(-1, 3)
        print("GPU mesh: using host vertex/face data from Mesh::load")
    else:
        x, indices = read_mesh_tables()
    if not len(x) or not len(indices):
        raise ValueError("No mesh to plot; pause at the end of Mesh::load")
    if np.any(indices < 0) or np.any(indices >= len(x)):
        raise ValueError("xIndices contains invalid vertex indices")

    # print("Vertices:", x.shape, x)
    # print("Triangles:", indices.shape, indices)


    # Each row contains the three corners of one triangle
    triangles = x[indices]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(
        triangles,
        facecolor="lightblue",
        edgecolor="black",
        linewidth=0.2,
        alpha=0.85
    )
    ax.add_collection3d(mesh)

    # Sample vertex-table rows at a uniform interval, with at most 10 labels.
    interval = max(1, (len(x) + 9) // 10)
    marked = np.arange(0, len(x), interval)
    ax.scatter(*x[marked].T, color="crimson", s=25, depthshade=False)
    for index in marked:
        ax.text(*x[index], str(index), color="crimson")

    ax.auto_scale_xyz(x[:, 0], x[:, 1], x[:, 2])
    ax.set_box_aspect(np.maximum(np.ptp(x, axis=0), 1e-6))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Mesh::load")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    
    if Agg == "Agg":
        plt.close(fig)
        subprocess.Popen(["xdg-open", str(output_path)], stderr=subprocess.DEVNULL)
    elif Agg == "QtAgg":
        add_triangle_hover(fig, ax, x, indices, mesh)
        plt.show(block=True)
    else:
        plt.close(fig)

    print(f"Saved mesh plot: {output_path}")
    return fig, ax

def load_obj(path):
    vertices = []
    faces = []

    with open(path, "r") as file:
        for line in file:
            s = line.split()
            if not s:
                continue

            if s[0] == "v":
                vertices.append([float(v) for v in s[1:4]])

            elif s[0] == "f":
                indices = [
                    int(v.split("/")[0]) - 1
                    for v in s[1:]
                ]

                # Triangulate polygon faces
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i+1]])

    vertices = np.array(vertices)
    faces = np.array(faces, dtype=int)

    print("Vertices:", len(vertices))
    print("Triangles:", len(faces))

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(
        vertices[faces],
        facecolor="lightblue",
        edgecolor="black",
        linewidth=0.15,
        alpha=0.9
    )

    ax.add_collection3d(mesh)
    ax.auto_scale_xyz(*vertices.T)
    ax.set_box_aspect(np.maximum(np.ptp(vertices, axis=0), 1e-6))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("dress4.obj")

    plt.savefig("mem/figure/load_obj.png", dpi=180)

    plt.show()

def triangulate(polygon_name="polygon", triangles_name="triangles"):
    """Draw main's polygon test in the XY plane; call with F8 before return 0."""
    x = read_vertices("x")
    polygon = np.asarray(mem(polygon_name), dtype=np.int32).reshape(-1)
    indices = np.asarray(mem(triangles_name), dtype=np.int32).reshape(-1)
    if polygon.size < 3:
        raise ValueError(f"{polygon_name} needs at least three vertices")
    if np.any(polygon < 0) or np.any(polygon >= len(x)):
        raise ValueError(f"{polygon_name} contains invalid vertex indices")

    # During stepping, a triangle may have only one or two indices appended.
    complete = indices[:indices.size // 3 * 3].reshape(-1, 3)
    if np.any(complete < 0) or np.any(complete >= len(x)):
        raise ValueError(f"{triangles_name} contains invalid vertex indices")

    fig = plt.figure("Mesh::triangulate", figsize=(9, 8))
    fig.clear()
    ax = fig.add_subplot(111)
    if len(complete):
        ax.add_collection(PolyCollection(
            x[complete, :2], facecolor="lightblue", edgecolor="black",
            linewidth=1, alpha=0.55
        ))
    boundary = x[np.append(polygon, polygon[0])]
    ax.plot(*boundary[:, :2].T, color="crimson", linewidth=2, label="Polygon")
    points = x[polygon]
    ax.scatter(*points[:, :2].T, color="black", s=30)
    for index in polygon:
        ax.annotate(str(index), x[index, :2], xytext=(5, 5),
                    textcoords="offset points")
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.1)
    ax.set(xlabel="X", ylabel="Y",
           title=f"Mesh::triangulate: {len(complete)} complete triangles")
    ax.legend()
    fig.tight_layout()
    output = Path(__file__).resolve().parent.parent / "mem/figure/triangulate.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    subprocess.Popen(["xdg-open", str(output)], stderr=subprocess.DEVNULL)
    print(f"Saved plot: {output}")
    print(f"Polygon: {polygon.size} vertices; triangles: {len(complete)}; "
          f"pending indices: {indices.size % 3}")
    return fig, ax
