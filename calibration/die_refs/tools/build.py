"""Turn the pixel-space block readings in specs.py into per-chip JSON (die mm, origin bottom-left) and overlay PNGs.

Run from anywhere: python3 calibration/die_refs/tools/build.py
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import specs  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

COLORS = {
    "sm_array": (80, 220, 80), "l2": (60, 140, 255), "xbar": (255, 60, 220), "hbm_phy": (255, 200, 0),
    "serdes": (255, 90, 40), "pcie": (255, 140, 140), "uncore": (200, 200, 200), "mem_ctrl_edge": (0, 220, 220),
    "spine": (180, 120, 255), "io_misc": (150, 150, 90), "mxu": (60, 200, 60), "vpu_vmem": (120, 255, 120),
    "cmem": (60, 140, 255), "oci": (160, 200, 160), "hbm_ctrl_phy": (255, 200, 0), "ici": (255, 90, 40),
    "misc": (200, 200, 200), "tc_ctrl": (255, 255, 120), "interstitial": (140, 140, 140), "xlu": (180, 255, 60),
}


def to_mm(spec, box):
    x0, y0, x1, y1 = box
    dx0, dy0, dx1, dy1 = spec["die_px"]
    w, h = spec["die"]["w_mm"], spec["die"]["h_mm"]
    if spec.get("rotate"):  # image rows run along die x (west->east), image columns along die y (north->south)
        fx0, fx1 = (y0 - dy0) / (dy1 - dy0), (y1 - dy0) / (dy1 - dy0)
        fy0, fy1 = 1 - (x1 - dx0) / (dx1 - dx0), 1 - (x0 - dx0) / (dx1 - dx0)
    else:
        fx0, fx1 = (x0 - dx0) / (dx1 - dx0), (x1 - dx0) / (dx1 - dx0)
        fy0, fy1 = 1 - (y1 - dy0) / (dy1 - dy0), 1 - (y0 - dy0) / (dy1 - dy0)
    clamp = lambda v: min(1.0, max(0.0, v))
    return [round(clamp(fx0) * w, 2), round(clamp(fy0) * h, 2), round(clamp(fx1) * w, 2), round(clamp(fy1) * h, 2)]


def build(name, spec):
    die = spec["die"]
    area = die["w_mm"] * die["h_mm"] if die.get("w_mm") else die.get("area_mm2_pub")
    blocks = []
    for b in spec.get("blocks", []):
        out = {k: v for k, v in b.items() if k != "px"}
        if b.get("px"):
            boxes = [to_mm(spec, p) for p in b["px"]]
            a = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes)
            out["bbox_mm"] = boxes[0] if len(boxes) == 1 else boxes
            out["area_mm2"] = round(a, 1)
            out["frac"] = round(a / area, 4)
            out.setdefault("count", len(boxes))
            out["area_each_mm2"] = round(a / out["count"], 2)
        else:
            out.setdefault("bbox_mm", None)
            if out.get("frac") is not None and out.get("area_mm2") is None and area:
                out["area_mm2"] = round(out["frac"] * area, 1)
        blocks.append(out)
    doc = {
        "chip": spec["chip"], "die": {**die, "area_mm2": round(area, 1) if area else None}, "blocks": blocks,
        "edges": spec.get("edges", {}), "package": spec.get("package", {}), "transistors": spec["transistors"],
        "node": spec.get("node"), "images": spec.get("images", []), "notes": spec.get("notes", []),
        "kiln_disagreements": spec.get("kiln_disagreements", []),
    }
    measured = [b for b in blocks if b.get("frac") is not None and b.get("bbox_mm") is not None]
    doc["coverage_frac"] = round(sum(b["frac"] for b in measured), 4)
    with open(os.path.join(ROOT, f"{name}.json"), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    overlay(name, spec, blocks)
    return doc


def font(sz):
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def draw_boxes(img, spec, blocks, k, label=True):
    d = ImageDraw.Draw(img, "RGBA")
    f = font(max(12, int(img.size[0] / 70)))
    for b, sb in zip(blocks, spec.get("blocks", [])):
        if not sb.get("px"):
            continue
        c = COLORS.get(b["kind"], (255, 255, 255))
        for i, (x0, y0, x1, y1) in enumerate(sb["px"]):
            r = [x0 * k, y0 * k, x1 * k, y1 * k]
            d.rectangle(r, fill=c + (55,), outline=c + (255,), width=max(2, int(img.size[0] / 600)))
            if label and i == 0:
                d.text((r[0] + 4, r[1] + 3), b["id"], fill=(255, 255, 255, 255), font=f, stroke_width=2,
                       stroke_fill=(0, 0, 0, 255))
    for lab, kind, (x0, y0, x1, y1) in spec.get("overlay_extra", []):
        c = COLORS.get(kind, (255, 255, 255))
        r = [x0 * k, y0 * k, x1 * k, y1 * k]
        d.rectangle(r, outline=c + (255,), width=max(2, int(img.size[0] / 300)))
        d.text((r[0] + 4, r[1] + 3), lab, fill=(255, 255, 255, 255), font=f, stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return img


def schematic(spec, blocks, width=900):
    dx0, dy0, dx1, dy1 = spec["die_px"]
    k = width / (dx1 - dx0)
    img = Image.new("RGB", (width + 40, int((dy1 - dy0) * k) + 70), (24, 24, 28))
    shifted = dict(spec)
    shifted["blocks"] = [dict(b, px=[(x0 - dx0 + 20 / k, y0 - dy0 + 20 / k, x1 - dx0 + 20 / k, y1 - dy0 + 20 / k)
                                     for x0, y0, x1, y1 in b["px"]]) if b.get("px") else b for b in spec["blocks"]]
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 20 + (dx1 - dx0) * k, 20 + (dy1 - dy0) * k], outline=(255, 255, 255), width=2)
    draw_boxes(img, shifted, blocks, k)
    d.text((20, img.size[1] - 40), f"{spec['chip']}: schematic redraw of {spec['overlay_src_note']}", fill=(220, 220, 220),
           font=font(14))
    return img


def overlay(name, spec, blocks):
    src = spec.get("overlay_img")
    if not src:
        return
    if src.get("free"):
        im = Image.open(os.path.join(ROOT, src["path"])).convert("RGB")
        k = im.size[0] / src["px_frame_w"]
        if src.get("crop"):
            cx0, cy0, cx1, cy1 = src["crop"]
            im = im.crop([int(v * k) for v in src["crop"]])
            spec = dict(spec, blocks=[dict(b, px=[(x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0) for x0, y0, x1, y1 in b["px"]])
                                      if b.get("px") else b for b in spec["blocks"]])
        im.thumbnail((1800, 1800))
        k = im.size[0] / ((src["crop"][2] - src["crop"][0]) if src.get("crop") else src["px_frame_w"])
        draw_boxes(im, spec, blocks, k).quantize(256, dither=Image.Dither.NONE).save(
            os.path.join(ROOT, f"overlay_{name}.png"), optimize=True)
    else:
        schematic(spec, blocks).save(os.path.join(ROOT, f"overlay_{name}.png"), optimize=True)
        p = os.path.join(ROOT, src["path"])
        if os.path.exists(p):  # local-only overlay on the copyrighted source (gitignored)
            im = Image.open(p).convert("RGB")
            if src.get("crop"):
                x0, y0, x1, y1 = src["crop"]
                im = im.crop((x0, y0, x1, y1))
                if src.get("upscale"):
                    im = im.resize((im.size[0] * src["upscale"], im.size[1] * src["upscale"]), Image.LANCZOS)
            k = im.size[0] / src["px_frame_w"]
            draw_boxes(im, spec, blocks, k).save(os.path.join(ROOT, f"overlay_{name}_photo.png"))


if __name__ == "__main__":
    only = sys.argv[1:]
    for name, spec in specs.CHIPS.items():
        if only and name not in only:
            continue
        doc = build(name, spec)
        kinds = {}
        for b in doc["blocks"]:
            if b.get("frac") is not None and b.get("bbox_mm") is not None:
                kinds[b["kind"]] = kinds.get(b["kind"], 0) + b["frac"]
        print(name, f"die {doc['die']['w_mm']}x{doc['die']['h_mm']}", f"coverage {doc['coverage_frac']:.3f}",
              {k: round(v, 3) for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])})
