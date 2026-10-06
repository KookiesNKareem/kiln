"""Perspective-rectify an angled die photo from its four top-surface corners.
usage: rectify.py IN OUT W_MM H_MM PX_PER_MM x_tl,y_tl x_tr,y_tr x_br,y_br x_bl,y_bl"""
import sys
import numpy as np
from PIL import Image
src, out, w_mm, h_mm, ppm = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
corners = [tuple(map(float, c.split(','))) for c in sys.argv[6:10]]
W, H = int(round(w_mm * ppm)), int(round(h_mm * ppm))
dst = [(0, 0), (W, 0), (W, H), (0, H)]
A, b = [], []
for (x, y), (u, v) in zip(corners, dst):
    A.append([u, v, 1, 0, 0, 0, -x * u, -x * v]); b.append(x)
    A.append([0, 0, 0, u, v, 1, -y * u, -y * v]); b.append(y)
coef = np.linalg.solve(np.array(A), np.array(b))
Image.open(src).convert('RGB').transform((W, H), Image.PERSPECTIVE, tuple(coef), Image.BICUBIC).save(out, quality=92)
print(W, H)
