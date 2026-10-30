
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load():
    # Read C++ mesh memory through GDB
    x = np.asarray(mem("x"), dtype=np.float32).reshape(-1, 3)
    indices = np.asarray(mem("xIndices"), dtype=np.int32).reshape(-1, 3)

    print("Vertices:", x.shape)
    print("Triangles:", indices.shape)

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

    ax.auto_scale_xyz(x[:, 0], x[:, 1], x[:, 2])
    ax.set_box_aspect(np.maximum(np.ptp(x, axis=0), 1e-6))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Mesh::load")

    plt.savefig("/tmp/mesh_debug.png", dpi=150)
    plt.show(block=False)

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

load_obj('meshes/dress4.obj')