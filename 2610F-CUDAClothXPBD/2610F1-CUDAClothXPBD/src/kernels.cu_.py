
import os
import subprocess
from pathlib import Path

# Qt otherwise warns when choosing X11 in a GNOME Wayland session.
# Preserve any display backend explicitly selected by the user.
if os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mem import mem

def integrate(Agg="Agg"):
    """Plot the selected particle after the kernel's sphere projection."""
    import numpy as np
    Agg = Agg if Agg in ("Agg", "QtAgg") else "Agg"
    plt.switch_backend(Agg)

    def read(name):
        value = mem(name)
        if value is None:
            raise RuntimeError(f"Cannot read {name}; pause inside the integrate kernel.")
        return value

    particle = int(read("p_nr"))
    count = int(read("num_particles"))
    if not 0 <= particle < count:
        raise ValueError(f"Particle {particle} is outside [0, {count})")
    previous = np.array([float(read(f"prev_pos_{a}[{particle}]")) for a in "xyz"])
    current = np.array([float(read(f"pos_{a}[{particle}]")) for a in "xyz"])
    velocity = np.array([float(read(f"vel_{a}[{particle}]")) for a in "xyz"])
    center = np.array([float(read(f"sphere_center.{a}")) for a in "xyz"])
    radius = float(read("sphere_radius"))
    dt = float(read("dt"))
    # Velocity has already received gravity at this breakpoint.
    predicted = previous + velocity * dt
    points = np.vstack((previous, predicted, current))
    if not np.isfinite(np.concatenate((points.ravel(), center, [radius, dt]))).all():
        raise ValueError("Particle or sphere contains non-finite values")
    if radius < 0:
        raise ValueError("Sphere radius must be nonnegative")

    from matplotlib.patches import Circle
    relative = points - center
    # The vertical plane is spanned by world Y and the horizontal O-to-pos
    # direction. It contains O and pos (new), and is perpendicular to XOZ.
    horizontal = relative[2, [0, 2]]
    horizontal_length = float(np.linalg.norm(horizontal))
    # On the Y axis any vertical plane contains both points; choose XOY.
    direction = horizontal / horizontal_length if horizontal_length > 0 else np.array([1., 0.])
    section = np.column_stack((relative[:, [0, 2]] @ direction, relative[:, 1]))
    extent = max(float(np.abs(section).max()), radius, 1e-6) * 1.15
    root = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parent.parent)
    output_dir = root / "mem" / "figure"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    ax.add_patch(Circle((0, 0), radius, facecolor="aliceblue", edgecolor="gray",
                        linewidth=1.5, label="Sphere section"))
    ax.axhline(0, color="0.8", linewidth=0.7)
    ax.axvline(0, color="0.8", linewidth=0.7)
    ax.scatter(0, 0, color="black", marker="+", s=70)
    ax.annotate("O (sphere center)", (0, 0), xytext=(6, 6),
                textcoords="offset points", fontsize=9)
    ax.plot([0, section[2, 0]], [0, section[2, 1]], "--", color="crimson", linewidth=1)
    ax.plot(*section.T, color="steelblue", linewidth=1)
    for label, point, color in zip(
            ("pos (previous, projected)", "pos (predicted, projected)", "pos (new)"),
            section, ("steelblue", "orange", "crimson")):
        ax.scatter(*point, color=color, s=50, label=label)
        ax.annotate(label, point, xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent),
           xlabel="Horizontal distance from O in the cutting plane",
           ylabel="Y − sphere_center.y",
           title=f"integrate: P{particle} — vertical section through O and pos (new)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend()
    output = output_dir / "integrate.png"
    fig.savefig(output, dpi=180)
    print(f"Saved {output}")
    print(f"P{particle}: previous={previous}, predicted={predicted}, current={current}")
    print(f"Distance to sphere center: {np.linalg.norm(current - center):.6g}")
    if Agg == "QtAgg":
        plt.show(block=True)
    plt.close(fig)
    if Agg == "Agg":
        try:
            subprocess.Popen(["xdg-open", str(output)], stderr=subprocess.DEVNULL)
        except OSError as error:
            print(f"Image saved, but could not open the viewer: {error}")
    return output

def solve_distance_constraints(Agg="Agg"):
    """Pause at the source command after the constraint's atomic additions."""
    import numpy as np
    Agg = Agg if Agg in ("Agg", "QtAgg") else "Agg"
    plt.switch_backend(Agg)

    def read(name):
        value = mem(name)
        if value is None:
            raise RuntimeError(f"Cannot read {name}; pause at the end of solve_distance_constraints.")
        return value

    ids = [int(read("id0")), int(read("id1"))]
    old = np.array([[float(read(f"p{i}_{a}")) for a in "xyz"] for i in range(2)])
    correction = np.array([float(read(f"dP_{a}")) for a in "xyz"])
    weights = np.array([float(read("w0")), float(read("w1"))])
    proposed = old + np.array([weights[0], -weights[1]])[:, None] * correction
    deferred = int(read("solve_type")) == 1
    if deferred:
        new = proposed
        after_label = "Proposed (this constraint; applied later)"
    else:
        new = np.array([[float(read(f"pos_{a}[{i}]")) for a in "xyz"] for i in ids])
        after_label = "After (live positions)"
    if not np.isfinite(np.vstack((old, new))).all():
        raise ValueError("Constraint positions contain non-finite values")
    rest = float(read("l0"))
    constraint = int(read("cNr"))
    fig = plt.figure(figsize=(10, 8), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*old.T, "o--", color="steelblue", label="Before", linewidth=1.5)
    ax.plot(*new.T, "o-", color="crimson", label=after_label, linewidth=1.5)
    for i, particle in enumerate(ids):
        delta = new[i] - old[i]
        ax.quiver(*old[i], *delta, color="darkgreen", arrow_length_ratio=0.15)
        ax.text(*old[i], f"  P{particle} old", fontsize=9)
        ax.text(*new[i], f"  P{particle} new", fontsize=9)
        print(f"P{particle}: old={old[i]}, new={new[i]}, movement={delta}")
    bounds = np.vstack((old, new))
    middle = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    extent = max(float(np.ptp(bounds, axis=0).max()) * 0.65, 1e-6)
    ax.set(xlim=(middle[0]-extent, middle[0]+extent),
           ylim=(middle[1]-extent, middle[1]+extent),
           zlim=(middle[2]-extent, middle[2]+extent),
           xlabel="X", ylabel="Y", zlabel="Z",
           title=f"Constraint C{constraint}: P{ids[0]} ↔ P{ids[1]}\n"
                 f"Length: {np.linalg.norm(old[1]-old[0]):.6g} → "
                 f"{np.linalg.norm(new[1]-new[0]):.6g}; rest={rest:.6g}")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-60, vertical_axis="y")
    ax.legend()
    root = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parent.parent)
    output = root / "mem" / "figure" / "solve_distance_constraints.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved {output}")
    if Agg == "QtAgg":
        plt.show(block=True)
    plt.close(fig)
    if Agg == "Agg":
        try:
            subprocess.Popen(["xdg-open", str(output)], stderr=subprocess.DEVNULL)
        except OSError as error:
            print(f"Image saved, but could not open the viewer: {error}")
    return output