def knots_u(Agg='Agg'):
    from pathlib import Path
    import matplotlib

    matplotlib.use(Agg)
    from matplotlib import pyplot as plt

    name = 'knots_u'
    degree_u = int(mem('degree_u'))
    degree_v = int(mem('degree_v'))
    cp_u, cp_v = int(mem('cp_u')), int(mem('cp_v'))
    size = degree_u + cp_u + 1
    values = mem(name, Size=size)
    if values is None:
        raise ValueError(f'Could not read {name}; stop after readNu() completes.')

    fig, ax = plt.subplots()
    ax.plot(range(size), values, marker='o')
    ax.set_xlabel('Knot index')
    ax.set_ylabel('Knot value')
    ax.set_title(f'NURBS U knot vector (degree_u={degree_u}, degree_v={degree_v}, '
                 f'cp_u={cp_u}, cp_v={cp_v})')
    ax.grid(True)

    figure_dir = Path(__file__).resolve().parent / 'mem' / 'figure'
    figure_dir.mkdir(parents=True, exist_ok=True)
    image_path = figure_dir / f'{name}.png'
    fig.savefig(image_path, dpi=150, bbox_inches='tight')
    print(f'Saved {image_path}')
    if Agg.lower() == 'agg':
        import subprocess
        subprocess.Popen(['xdg-open', str(image_path)])
        plt.close(fig)
    else:
        plt.show()
    return fig


def knots_v(Agg='Agg'):
    from pathlib import Path
    import matplotlib

    matplotlib.use(Agg)
    from matplotlib import pyplot as plt

    name = 'knots_v'
    degree_v = int(mem('degree_v'))
    degree_u = int(mem('degree_u'))
    cp_u, cp_v = int(mem('cp_u')), int(mem('cp_v'))
    size = degree_v + cp_v + 1
    values = mem(name, Size=size)
    if values is None:
        raise ValueError(f'Could not read {name}; stop after readNu() completes.')

    fig, ax = plt.subplots()
    ax.plot(range(size), values, marker='o')
    ax.set_xlabel('Knot index')
    ax.set_ylabel('Knot value')
    ax.set_title(f'NURBS V knot vector (degree_u={degree_u}, degree_v={degree_v}, '
                 f'cp_u={cp_u}, cp_v={cp_v})')
    ax.grid(True)

    figure_dir = Path(__file__).resolve().parent / 'mem' / 'figure'
    figure_dir.mkdir(parents=True, exist_ok=True)
    image_path = figure_dir / f'{name}.png'
    fig.savefig(image_path, dpi=150, bbox_inches='tight')
    print(f'Saved {image_path}')
    if Agg.lower() == 'agg':
        import subprocess
        subprocess.Popen(['xdg-open', str(image_path)])
        plt.close(fig)
    else:
        plt.show()
    return fig


def control_points(Agg='QtAgg'):
    from pathlib import Path
    import numpy as np
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    cp_u, cp_v = int(mem('cp_u')), int(mem('cp_v'))
    degree_u, degree_v = int(mem('degree_u')), int(mem('degree_v'))
    points = mem('control_points', Size=cp_u * cp_v)
    if points is None:
        raise ValueError('Could not read control_points; stop after readNu() completes.')
    coordinates = np.array([[point['x'], point['y'], point['z']] for point in points])
    coordinates = coordinates.reshape(cp_u, cp_v, 3)

    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates[..., 0], coordinates[..., 1], coordinates[..., 2])
    for row in coordinates:
        ax.plot(row[:, 0], row[:, 1], row[:, 2])
    for column in coordinates.transpose(1, 0, 2):
        ax.plot(column[:, 0], column[:, 1], column[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'NURBS control points (degree_u={degree_u}, degree_v={degree_v}, '
                 f'cp_u={cp_u}, cp_v={cp_v})')

    figure_dir = Path(__file__).resolve().parent / 'mem' / 'figure'
    figure_dir.mkdir(parents=True, exist_ok=True)
    image_path = figure_dir / 'control_points.png'
    fig.savefig(image_path, dpi=150, bbox_inches='tight')
    print(f'Saved {image_path}')

    if Agg.lower() == 'agg':
        import subprocess
        subprocess.Popen(['xdg-open', str(image_path)])
        return fig

    matplotlib.use(Agg)
    from matplotlib.backends.qt_compat import QtWidgets
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.backends.backend_qt import FigureManagerQT

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    canvas = FigureCanvasQTAgg(fig)
    manager = FigureManagerQT(canvas, 1)
    manager.set_window_title('Control points')
    manager.show()
    app.exec()
    return fig
