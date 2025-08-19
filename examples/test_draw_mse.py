import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.tools import visual


def main():
    np.random.seed(42)
    x = np.linspace(0, 8 * np.pi, 300)
    true = np.sin(x)
    preds = true + np.random.normal(0, 0.2, size=true.shape)

    output_dir = os.path.join(PROJECT_ROOT, 'test_outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'demo_draw_mse.pdf')

    visual(true, preds, output_path, draw_mse=True, mse_value=None)
    print(f'绘图完成: {output_path}')


if __name__ == '__main__':
    main() 