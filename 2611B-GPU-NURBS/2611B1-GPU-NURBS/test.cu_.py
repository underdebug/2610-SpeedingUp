def NormalTest(Agg='QtAgg'):
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use(Agg)
    from matplotlib import pyplot as plt

    cp_u, cp_v = int(mem('cp_u')), int(mem('cp_v'))
    degree_u, degree_v = int(mem('degree_u')), int(mem('degree_v'))
    count = cp_u * cp_v
    arrays = []
    for name in ('display_points', 'normal', 'control_points'):
        values = mem(name, Size=count)
        if values is None:
            raise ValueError(f'Could not read {name}; stop after the sampling loop and before delete[].')
        points = np.array([[p['x'], p['y'], p['z']] for p in values], dtype=float)
        if points.shape != (count, 3):
            raise ValueError(f'Expected {count} points for {name}, got {points.shape}.')
        arrays.append(points)

    surface = arrays[0].reshape(cp_v, cp_u, 3)
    normals = arrays[1].reshape(cp_v, cp_u, 3) * 0.05
    controls = arrays[2].reshape(cp_u, cp_v, 3).transpose(1, 0, 2)

    fig = plt.figure('NURBS surface and normals', figsize=(10, 8), clear=True)
    ax = fig.add_subplot(111, projection='3d')
    if cp_u > 1 and cp_v > 1:
        ax.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2],
                        color='skyblue', alpha=0.55, edgecolor='steelblue',
                        linewidth=0.5, rstride=1, cstride=1)
    else:
        ax.plot(surface[..., 0].ravel(), surface[..., 1].ravel(), surface[..., 2].ravel(), color='steelblue')
    ax.scatter(surface[..., 0], surface[..., 1], surface[..., 2], color='steelblue', s=12, label='Surface points')
    ax.scatter(controls[..., 0], controls[..., 1], controls[..., 2], color='darkorange', s=20, label='Control points')
    for control, point in zip(controls.reshape(-1, 3), surface.reshape(-1, 3)):
        ax.plot(*np.stack((control, point), axis=1), color='darkorange', alpha=0.55, linewidth=0.8)
    ax.quiver(surface[..., 0], surface[..., 1], surface[..., 2],
              normals[..., 0], normals[..., 1], normals[..., 2], color='crimson')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'NURBS surface and normals (degree_u={degree_u}, degree_v={degree_v}, cp_u={cp_u}, cp_v={cp_v})')
    ax.legend()

    figure_dir = Path(__file__).resolve().parent / 'mem' / 'figure'
    figure_dir.mkdir(parents=True, exist_ok=True)
    image_path = figure_dir / 'normal_test.png'
    fig.savefig(image_path, dpi=150, bbox_inches='tight')
    print(f'Saved {image_path}')
    plt.show()
    return fig
