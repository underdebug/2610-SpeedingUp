"""Generate a static, upright pose for the 16-part female rigid body.

Coordinates use the original motion file's Z-up frame. The dress configuration
applies its existing world rotation and translation afterward. Two identical
samples provide a valid interpolation interval with zero initial velocity.
"""
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    # Part order matches body0000.obj ... body0015.obj and the original motion.
    x, y = -0.35, -0.185
    positions = [
        (x, y, 1.569),                  # head
        (x, y + 0.225, 1.32),           # upper arms
        (x, y - 0.225, 1.32),
        (x, y + 0.235, 1.08),           # forearms
        (x, y - 0.235, 1.08),
        (x, y, 1.311),                  # chest
        (x, y, 1.133),                  # abdomen
        (x, y, 0.956),                  # pelvis
        (x, y + 0.10, 0.652),           # thighs
        (x, y - 0.10, 0.652),
        (x, y + 0.10, 0.285),           # lower legs
        (x, y - 0.10, 0.285),
        (x, y + 0.10, 0.049),           # feet
        (x, y - 0.10, 0.049),
        (x, y - 0.235, 0.99),           # hands
        (x, y + 0.235, 0.99),
    ]
    lines = ['NumFrames: 2']
    for i, position in enumerate(positions):
        lines.append(f'body[{i}] position')
        lines.extend([' '.join(f'{value:.6f}' for value in position)] * 2)
        lines.append(f'body[{i}] orientation')
        lines.extend(['1 0 0 0'] * 2)
    path = root / 'motions/standing-debug.mot'
    path.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
