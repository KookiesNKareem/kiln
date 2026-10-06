"""Draw a mm grid (origin bottom-left) on a die image so block edges can be read off in die coordinates.
usage: grid.py IMG AREA_MM2 OUT [crop x0,y0,x1,y1 in mm] [maxpx]"""
import sys
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
img, area, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
im = Image.open(img).convert('RGB')
W, H = im.size
w_mm = (area * W / H) ** 0.5
h_mm = area / w_mm
s = W / w_mm
crop = [float(v) for v in sys.argv[4].split(',')] if len(sys.argv) > 4 and sys.argv[4] != '-' else [0, 0, w_mm, h_mm]
maxpx = int(sys.argv[5]) if len(sys.argv) > 5 else 1600
x0, y0, x1, y1 = crop
c = im.crop((int(x0 * s), int(H - y1 * s), int(x1 * s), int(H - y0 * s)))
k = min(1.0, maxpx / max(c.size))
c = c.resize((int(c.size[0] * k), int(c.size[1] * k)))
d = ImageDraw.Draw(c)
step = 1 if (x1 - x0) > 6 else 0.5
v = int(x0)
while v <= x1:
    if v >= x0:
        px = (v - x0) * s * k
        major = abs(v - round(v / 2) * 2) < 1e-6
        d.line([(px, 0), (px, c.size[1])], fill=(255, 255, 0) if major else (255, 120, 0), width=1)
        if major: d.text((px + 2, 2), f"{v:g}", fill=(255, 255, 0))
    v += step
v = int(y0)
while v <= y1:
    if v >= y0:
        py = (y1 - v) * s * k
        major = abs(v - round(v / 2) * 2) < 1e-6
        d.line([(0, py), (c.size[0], py)], fill=(0, 255, 255) if major else (0, 140, 255), width=1)
        if major: d.text((2, py + 2), f"{v:g}", fill=(0, 255, 255))
    v += step
c.save(out, quality=85)
print(f"die {w_mm:.2f} x {h_mm:.2f} mm, {s:.1f} px/mm")
