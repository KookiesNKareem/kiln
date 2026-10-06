"""Block readings in image pixel space. Frames:
  ga100: 1267 x 1600 downscale of img/ga100_dieshot_kurnal_2400.jpg (same image as the Commons original), die fills the frame.
  gp100: 1311 x 1600 downscale of img/gp100_rectified.jpg (tools/rectify.py on Fritz's angled photo), die fills the frame.
  gh100: 4x upscale of crop (850,300)-(1190,660) of NVIDIA's GTC22 module render; image rows run along die x (rotate).
  tpu_v4i: Jouppi ISCA 2021 Fig. 6 (921 x 698 extracted bitmap).
  tpu_v4: Jouppi ISCA 2023 Fig. 2 package photo (686 x 680 extracted bitmap).
Coordinates are (x0, y0, x1, y1) with y down, as read off the image; build.py converts them to die mm with origin bottom-left.
"""

A100_WP = "NVIDIA A100 Tensor Core GPU Architecture whitepaper (2020): 54.2 B transistors, 826 mm2, TSMC 7N"
H100_WP = "NVIDIA H100 Tensor Core GPU Architecture whitepaper (2022): 80 B transistors, 814 mm2, TSMC 4N"
V100_WP = "NVIDIA Tesla V100 GPU Architecture whitepaper (2017): 21.1 B transistors, 815 mm2, TSMC 12 nm FFN"
P100_WP = "NVIDIA Tesla P100 whitepaper (2016): 15.3 B transistors, 610 mm2, TSMC 16 nm FinFET"
V4I = "Jouppi et al., Ten Lessons From Three Generations Shaped Google's TPUv4i, ISCA 2021 (Table 1, Fig. 6)"
V4 = "Jouppi et al., TPU v4: An Optically Reconfigurable Supercomputer..., ISCA 2023, arXiv 2304.01433 (Table 4, Fig. 2), CC BY 4.0"

GA100_IMG = {
    "path": "img/ga100_dieshot_kurnal_2400.jpg",
    "url": "https://commons.wikimedia.org/wiki/File:GA100_Die_Shot.jpg (original 6065x7657, "
           "https://upload.wikimedia.org/wikipedia/commons/b/be/GA100_Die_Shot.jpg)",
    "attribution": "Wikimedia Commons, CC BY 3.0, author listed as unknown, credit https://kurnal-insights.com/en/dieshot/?id=nvidia-ga100 "
                   "(Kurnal Insights lists the die as 32.35 x 25.65 mm). Uploaded 2026-05-26, downloaded 2026-10-06.",
    "type": "die photo (top-metal/colourised; real die, not a marketing render)",
}

CHIPS = {}

# ---------------------------------------------------------------- GA100
CHIPS["ga100"] = {
    "chip": "NVIDIA GA100 (A100)",
    "node": {"value": "TSMC N7 (NVIDIA: 7N)", "source": A100_WP},
    "transistors": {"value": 54.2e9, "source": A100_WP},
    "die": {"w_mm": 25.58, "h_mm": 32.30, "orientation": "portrait; HBM PHYs on the long (W/E) edges",
            "source": "area = 826 mm2 (whitepaper); aspect 6065/7657 = 0.792 from the die photo. "
                      "Independent caliper value: 25.65 x 32.35 mm = 829.8 mm2 (Kurnal Insights)."},
    "die_px": (0, 0, 1267, 1600),
    "overlay_img": {"free": True, "path": "img/ga100_dieshot_kurnal_2400.jpg", "px_frame_w": 1267},
    "overlay_src_note": "GA100 die photo",
    "blocks": [
        {"id": "gpc", "kind": "sm_array", "count": 8,
         "px": [(88, 72, 620, 338), (645, 72, 1180, 338), (88, 560, 620, 800), (645, 560, 1180, 800),
                (88, 800, 620, 1050), (645, 800, 1180, 1050), (88, 1270, 610, 1530), (650, 1270, 1180, 1530)],
         "confidence": "medium",
         "source": "own reading: SM-tile arrays (repeated mirrored tiles with register-file/L1 SRAM bars) in 2 columns x 4 rows. "
                   "The split into 8 GPCs uses one bright marker square per array (8 markers at x~400/860, y~300/580/1025/1300 "
                   "in the frame) plus the published 8-GPC count; GPC boundaries inside the middle rows are not visible as gaps. "
                   "Boxes include GPC-level logic (raster/PE/MMU) and inter-TPC wiring."},
        {"id": "l2_band", "kind": "l2", "count": 4,
         "px": [(88, 338, 500, 560), (765, 338, 1180, 560), (88, 1050, 500, 1270), (765, 1050, 1180, 1270)],
         "confidence": "medium",
         "source": "own reading: two horizontal bands (~25% and ~70% of die height) of cross-shaped units with dense SRAM bars; "
                   "per half-band I count 8 crosses + 4 smaller units = ~20 slice-like units, x4 = ~80, matching 80 x 512 KB L2 "
                   "slices (40 MB). Bands include slice logic/tags, so this is the L2 *region*, not bitcell area. Which band is "
                   "which partition is inferred (each band sits between two GPC rows, next to one crossbar)."},
        {"id": "xbar", "kind": "xbar", "count": 2, "px": [(500, 330, 765, 575), (500, 1025, 765, 1285)],
         "confidence": "low",
         "source": "own reading: two dark, wire-dominated macros at die centre of each L2 band, joined by a vertical bus; "
                   "consistent with one GPC<->L2 crossbar per partition (whitepaper), but not annotated anywhere I found."},
        {"id": "center_spine", "kind": "spine", "count": 3,
         "px": [(620, 72, 645, 330), (620, 575, 645, 1025), (610, 1285, 650, 1460)],
         "confidence": "low",
         "source": "own reading: vertical routing channel on the die centre line (top stem, inter-crossbar bus, bottom stem)."},
        {"id": "hbm_phy_w", "kind": "hbm_phy", "count": 3, "px": [(0, 30, 75, 350), (0, 640, 75, 955), (0, 1220, 75, 1550)],
         "confidence": "high",
         "source": "own reading: three identical PHY strips on the west edge (pale macro + orange bump-fanout channel); "
                   "6 sites total matches whitepaper. Length 6.2-6.7 mm each, depth ~1.5 mm incl. fanout."},
        {"id": "hbm_phy_e", "kind": "hbm_phy", "count": 3,
         "px": [(1190, 45, 1267, 355), (1190, 640, 1267, 955), (1190, 1250, 1267, 1555)],
         "confidence": "high", "source": "own reading: mirror of the west edge."},
        {"id": "fb_edge_logic", "kind": "mem_ctrl_edge", "count": 4,
         "px": [(0, 355, 88, 640), (0, 955, 88, 1220), (1180, 355, 1267, 640), (1180, 955, 1267, 1250)],
         "confidence": "low",
         "source": "own reading: irregular logic between the HBM PHYs on each long edge; plausibly memory-controller/FBPA logic "
                   "(the whitepaper's 10+2 512-bit controllers), but not identified from structure. The controllers may also "
                   "sit inside the L2 bands next to the slices."},
        {"id": "serdes_n", "kind": "serdes", "count": 3, "px": [(80, 0, 387, 58), (478, 0, 790, 58), (880, 0, 1188, 58)],
         "confidence": "medium",
         "source": "own reading: three identical SerDes macros on the north edge, each with 2 rows x 8 lane cells and a central "
                   "PLL column (16 lanes each). 12 NVLink3 x 4 lanes = 48 lanes = 3 macros, so these are most likely NVLink."},
        {"id": "serdes_s", "kind": "serdes", "count": 1, "px": [(72, 1540, 420, 1600)],
         "confidence": "medium",
         "source": "own reading: a 4th macro identical to the north ones on the south edge (west end). 16 lanes = PCIe Gen4 x16 "
                   "is the natural fit (NVLink/PCIe share the NVHS SerDes design), but which of the four macros is PCIe is inferred."},
        {"id": "io_misc_n", "kind": "io_misc", "count": 3,
         "px": [(387, 0, 478, 58), (790, 0, 880, 58), (80, 58, 1188, 72)],
         "confidence": "low", "source": "own reading: non-SerDes I/O cells between the north macros and the logic strip under them."},
        {"id": "uncore_s", "kind": "uncore", "count": 1, "px": [(420, 1530, 1190, 1600)],
         "confidence": "low",
         "source": "own reading: irregular non-SM logic along the south edge with a small I/O strip (GPIO/misc). Front end, "
                   "copy engines, NVDEC/NVJPG are expected somewhere here but individual blocks are not identifiable."},
    ],
    "edges": {
        "w": [{"kind": "hbm_phy", "count": 3, "length_mm": [6.5, 6.4, 6.7]}],
        "e": [{"kind": "hbm_phy", "count": 3, "length_mm": [6.3, 6.4, 6.2]}],
        "n": [{"kind": "serdes (NVLink, inferred)", "count": 3, "length_mm": [6.2, 6.3, 6.2], "lanes_each": 16}],
        "s": [{"kind": "serdes (PCIe x16, inferred)", "count": 1, "length_mm": 7.0, "lanes_each": 16},
              {"kind": "uncore + small misc I/O", "length_mm": 15.6}],
        "note": "Image orientation is arbitrary; what is fixed is: HBM on both long edges (3+3), SerDes on both short edges (3+1).",
    },
    "package": {
        "hbm_sites": 6, "hbm_active": 5,
        "hbm_placement": "3 per side along the die's long (W/E) edges; stack long axis parallel to the die long edge",
        "disabled_site": "A100 40GB enables 5 of 6 HBM2 sites (whitepaper); all six sites look identical in NVIDIA's module "
                         "render, so which site is unused cannot be read from public images",
        "hbm_stack_mm": [7.6, 11.5],
        "interposer_mm": [44.7, 37.8],
        "substrate_mm": [54.4, 54.3],
        "source": "measured on NVIDIA's SXM4 module render (developer blog 'NVIDIA Ampere Architecture In-Depth', "
                  "nvidia-a100-gpu-on-sxm4.jpg), scale set by the die (350 x 442 px = 826 mm2, aspect 0.792 matches the die "
                  "photo). Cross-check: stacks come out 7.6 x 11.5 mm vs 7.75 x 11.87 mm HBM2 KGSD. Interposer = dark "
                  "mould ring around die+stacks (~1690 mm2), low confidence. Render, not photo.",
        "confidence": "medium (stack count/placement), low (interposer/substrate size)",
    },
    "images": [
        GA100_IMG,
        {"path": "img/nonfree/a100_sxm4_module_nvidia_render.jpg (local only, not committed)",
         "url": "https://developer-blogs.nvidia.com/wp-content/uploads/2020/05/nvidia-a100-gpu-on-sxm4.jpg",
         "attribution": "NVIDIA (marketing render), copyright NVIDIA", "type": "package render"},
        {"path": "img/nonfree/hc32_a100_slide3_blockdiagram.png (local only)",
         "url": "https://hc32.hotchips.org/assets/program/conference/day1/HotChips2020_GPU_NVIDIA_Choquette_v01.pdf (slide 3)",
         "attribution": "NVIDIA, Hot Chips 32", "type": "logical block diagram (not physical)"},
    ],
    "notes": [
        "Fractions are of the 826 mm2 die; boxes tile the die with small overlaps/gaps (see coverage_frac).",
        "NVIDIA's block diagram draws one central L2 band with GPCs above and below; the photo shows TWO L2 bands, each between "
        "two GPC rows, with a crossbar at the centre of each band.",
    ],
    "kiln_disagreements": [
        "a100_sxm4_40gb.json5 declares the L2 partitions as a separate [1,2] grid (placed below the GPC array, per the task "
        "brief; placer behaviour not re-checked here); the die has two full-width L2 bands at ~25% and ~70% of the height, each sandwiched between GPC rows, with the crossbar in the band centre.",
        "GPC arrangement: kiln uses grid [2,4] (2 columns x 4 rows would match; check the grid's row/column convention). The photo "
        "is 2 columns x 4 rows, with L2 bands between rows 1/2 and 3/4.",
        "Edges: kiln has 12 NVLinks on S and PCIe on N. The die has 3 NVLink-like SerDes macros on one short edge and the 4th "
        "(probably PCIe) on the opposite short edge; HBM W/E 3+3 agrees.",
        "Outline 25.6 x 32.3 mm agrees with the photo aspect (0.792) and Kurnal's caliper 25.65 x 32.35 mm.",
        "Disabled HBM site: kiln disables hbm5/hbm_if5; which physical site is unused is not determinable from public photos.",
        "uncore footprint 60 mm2 (assumed in kiln) vs ~14 mm2 of clearly non-SM, non-L2 logic on the south edge plus ~33 mm2 "
        "of edge logic between HBM PHYs; front-end/copy-engine/decoder logic may also be distributed in the bands.",
    ],
}

# ---------------------------------------------------------------- GP100
CHIPS["gp100"] = {
    "chip": "NVIDIA GP100 (Tesla P100)",
    "node": {"value": "TSMC 16 nm FinFET (16FF+)", "source": P100_WP},
    "transistors": {"value": 15.3e9, "source": P100_WP},
    "die": {"w_mm": 22.35, "h_mm": 27.29, "orientation": "portrait; HBM PHYs on the long (W/E) edges",
            "source": "area 610 mm2 (whitepaper) at Fritzchens Fritz's measured aspect 22.499 x 27.471 mm (618 mm2 caliper, "
                      "1.3% larger than published, probably includes seal ring/scribe)."},
    "die_px": (0, 0, 1311, 1600),
    "overlay_img": {"free": True, "path": "img/gp100_rectified.jpg", "px_frame_w": 1311},
    "overlay_src_note": "rectified GP100 die photo",
    "blocks": [
        {"id": "gpc", "kind": "sm_array", "count": 6,
         "px": [(95, 15, 470, 565), (490, 15, 865, 565), (865, 15, 1230, 565),
                (95, 860, 460, 1420), (490, 860, 865, 1420), (865, 860, 1230, 1420)],
         "confidence": "medium",
         "source": "own reading of the rectified photo: 3 x 2 array of SM-tile blocks separated by the north-south spine and a "
                   "horizontal band; each block shows 2 mirrored columns x 5 rows of SM-like tiles (=10 SMs = 5 TPCs, matching the "
                   "whitepaper's 6 GPC x 5 TPC). One magenta marker square per GPC (6 total)."},
        {"id": "l2_band", "kind": "l2", "count": 2, "px": [(160, 565, 590, 860), (730, 565, 1150, 860)],
         "confidence": "medium",
         "source": "own reading: central horizontal band of quad-symmetric SRAM-dominated units either side of a sparse centre "
                   "block; the only large non-SM array, so L2 (4 MB) with slice logic."},
        {"id": "xbar", "kind": "xbar", "count": 1, "px": [(590, 565, 730, 860)], "confidence": "low",
         "source": "own reading: sparse, wire-dominated block at die centre of the band (crossbar/hub, inferred)."},
        {"id": "center_spine", "kind": "spine", "count": 2, "px": [(470, 15, 490, 565), (460, 860, 490, 1420)],
         "confidence": "low", "source": "own reading: bright routing channel between GPC columns."},
        {"id": "hbm_phy_w", "kind": "hbm_phy", "count": 2, "px": [(0, 185, 45, 545), (0, 1020, 45, 1400)],
         "confidence": "high", "source": "own reading: two PHY strips on the west edge (4 HBM2 sites total, whitepaper)."},
        {"id": "hbm_phy_e", "kind": "hbm_phy", "count": 2, "px": [(1270, 190, 1311, 545), (1270, 1035, 1311, 1400)],
         "confidence": "high", "source": "own reading: mirror on the east edge."},
        {"id": "fb_edge_logic", "kind": "mem_ctrl_edge", "count": 2, "px": [(0, 565, 160, 860), (1150, 565, 1311, 860)],
         "confidence": "low",
         "source": "own reading: distinct logic between the two PHYs of each long edge, at the ends of the L2 band; "
                   "plausibly memory controllers (8 x 512-bit) but not identified."},
        {"id": "nvlink", "kind": "serdes", "count": 1, "px": [(40, 1470, 640, 1600)], "confidence": "medium",
         "source": "own reading: two rows of large orange SerDes macros on the south edge, 4 groups; NVLink1 is 4 links x 8 lanes."},
        {"id": "pcie_probable", "kind": "pcie", "count": 1, "px": [(670, 1555, 1000, 1600)], "confidence": "low",
         "source": "own reading: a different, smaller I/O macro row (blue) on the same south edge; PCIe Gen3 x16 by elimination."},
        {"id": "uncore_s", "kind": "uncore", "count": 3,
         "px": [(640, 1420, 1311, 1555), (95, 1420, 640, 1470), (1000, 1555, 1311, 1600)], "confidence": "low",
         "source": "own reading: irregular logic between the bottom GPCs and the south I/O (front end, hub, copy engines expected)."},
    ],
    "edges": {
        "w": [{"kind": "hbm_phy", "count": 2, "length_mm": [6.2, 6.5]}],
        "e": [{"kind": "hbm_phy", "count": 2, "length_mm": [6.0, 6.2]}],
        "s": [{"kind": "serdes (NVLink, 2 rows)", "count": 4, "length_mm": 10.2},
              {"kind": "pcie (probable)", "count": 1, "length_mm": 5.6}],
        "n": [{"kind": "thin misc I/O strip only", "length_mm": None}],
        "note": "NVLink and the probable PCIe both sit on the same short edge.",
    },
    "package": {
        "hbm_sites": 4, "hbm_active": 4, "hbm_placement": "2 per side along the die's long edges, stack long axis parallel to them",
        "hbm_stack_mm": [7.527, 11.490], "package_mm": [55, 55], "interposer_mm": [41, 29],
        "source": "Fritzchens Fritz (Commons, CC0): HBM stack and package dimensions from his file descriptions; interposer size "
                  "is my estimate from the SWIR package photo (scale from the die width), low confidence.",
    },
    "images": [
        {"path": "img/gp100_rectified.jpg", "url": "derived: tools/rectify.py on the photo below, corners (815,1047) (3267,96) (5255,2070) (2693,3150)",
         "attribution": "Fritzchens Fritz, CC0", "type": "die photo (perspective-rectified)"},
        {"path": "img/gp100_fritz_angled_die_2400.jpg",
         "url": "https://commons.wikimedia.org/wiki/File:Nvidia@16nm@Pascal@GP100@Tesla_P100@T_Taiwan_1912A1_PN9G70.S6W_GP100-897-A1_DSC06755-DSC06929.jpg",
         "attribution": "Fritzchens Fritz, CC0 (taken 2021-12-31)", "type": "die photo (angled, delidded die)"},
        {"path": "img/gp100_fritz_swir_die_2000.jpg",
         "url": "https://commons.wikimedia.org/wiki/File:Nvidia@16nm@Pascal@GP100@Tesla_P100@T_Taiwan_1912A1_PN9G70.S6W_GP100-897-A1_DSCx03@SWIR.jpg",
         "attribution": "Fritzchens Fritz, CC0 (2021-12-23)", "type": "die photo, short-wave IR through the backside (mirrored)"},
        {"path": "img/gp100_fritz_swir_package_2000.jpg",
         "url": "https://commons.wikimedia.org/wiki/File:Nvidia@16nm@Pascal@GP100@Tesla_P100@T_Taiwan_1912A1_PN9G70.S6W_GP100-897-A1_DSCx01@SWIR.jpg",
         "attribution": "Fritzchens Fritz, CC0 (2021-12-21)", "type": "package photo (SWIR)"},
    ],
    "notes": ["Rectification maps the four top-surface corners to the 22.499 x 27.471 mm rectangle; residual error ~1% of size."],
    "kiln_disagreements": [
        "p100_sxm2_16gb.json5 puts NVLink on S and PCIe on N; on the die both NVLink and the probable PCIe macro are on the same "
        "short edge (the other short edge has only a thin misc I/O strip).",
        "GPC layout: kiln grid [2,3]; die is 3 columns x 2 rows with ONE central L2 band between the rows (consistent if the grid "
        "is rows x columns = 2 x 3).",
        "Outline: kiln 22.1 x 27.6 mm (aspect assumed 0.80); photo aspect is 0.819 (22.35 x 27.29 at 610 mm2).",
    ],
}

# ---------------------------------------------------------------- GH100 (render only)
CHIPS["gh100"] = {
    "chip": "NVIDIA GH100 (H100)",
    "node": {"value": "TSMC 4N", "source": H100_WP},
    "transistors": {"value": 80e9, "source": H100_WP},
    "die": {"w_mm": 25.94, "h_mm": 31.38, "orientation": "portrait; HBM PHYs on the long (W/E) edges",
            "source": "area 814 mm2 (whitepaper); aspect 0.827 measured on a top-down H100 SXM5 package photo "
                      "(die 640 x 774 px). No free die photo of GH100 was found."},
    "die_px": (125, 335, 1235, 1105),
    "rotate": True,
    "overlay_img": {"free": False, "path": "img/nonfree/gh100_h100_module_nvidia_render_gtc22.png",
                    "crop": (850, 300, 1190, 660), "upscale": 4, "px_frame_w": 1360},
    "overlay_src_note": "NVIDIA GTC22 module render (stylised; low confidence)",
    "blocks": [
        {"id": "hbm_phy_w", "kind": "hbm_phy", "count": 6,
         "px": [(175, 345, 305, 380), (315, 345, 445, 380), (555, 345, 690, 380), (695, 345, 825, 380),
                (925, 345, 1055, 380), (1060, 345, 1195, 380)],
         "confidence": "low",
         "source": "NVIDIA render: 6 PHY-like blocks along the edge facing 3 HBM3 stacks (2 per stack, plausibly the two "
                   "halves of a 16-channel HBM3 PHY). Depth is stylised."},
        {"id": "hbm_phy_e", "kind": "hbm_phy", "count": 6,
         "px": [(165, 1060, 295, 1095), (305, 1060, 435, 1095), (555, 1060, 690, 1095), (695, 1060, 825, 1095),
                (930, 1060, 1060, 1095), (1070, 1060, 1205, 1095)],
         "confidence": "low", "source": "NVIDIA render: mirror on the opposite long edge."},
        {"id": "sm_tiles", "kind": "sm_array", "count": 3,
         "px": [(205, 410, 370, 1015), (525, 410, 870, 1015), (1025, 410, 1215, 1015)],
         "confidence": "low",
         "source": "NVIDIA render: 3 groups of SM-like tiles, 3 + 6 + 3 columns x 12 rows = 144 tiles = full GH100 SM count, "
                   "split by a horizontal band at mid-height. Treat as a hint of arrangement only (render, not layout)."},
        {"id": "interstitial", "kind": "interstitial", "count": 2, "px": [(380, 410, 520, 1015), (880, 410, 1020, 1015)],
         "confidence": "low",
         "source": "NVIDIA render: columns of small blocks between the SM groups, each with an orange xbar-like macro at its "
                   "mid-point. In die orientation these are two full-width bands between SM rows with a hub at the centre, i.e. "
                   "exactly where GA100's two L2 bands + crossbars sit, so most likely L2 + crossbar; unconfirmed."},
        {"id": "io_short_edges", "kind": "serdes", "count": 2, "px": [(125, 380, 195, 1050), (1200, 380, 1235, 1050)],
         "confidence": "low",
         "source": "NVIDIA render: striped I/O-like structures along both short edges (NVLink4 x18 + PCIe5 x16 expected)."},
    ],
    "edges": {
        "w": [{"kind": "hbm_phy", "count": 3, "length_mm": None, "note": "2 PHY blocks per stack in the render"}],
        "e": [{"kind": "hbm_phy", "count": 3, "length_mm": None}],
        "n": [{"kind": "serdes (render)", "count": None, "length_mm": None}],
        "s": [{"kind": "serdes (render)", "count": None, "length_mm": None}],
        "note": "Only HBM on the long edges (3+3, also confirmed by package photos) is reliable; short-edge I/O is from the render.",
    },
    "package": {
        "hbm_sites": 6, "hbm_active": 5, "hbm_placement": "3 per side along the die long edges (package photo)",
        "disabled_site": "H100 SXM5 80GB: 5 active sites of 6 (whitepaper/HC34); the 6 visible stacks look identical, the unused "
                         "one cannot be identified from photos",
        "hbm_stack_mm_visible": [9.7, 10.0],
        "source": "top-down H100 SXM5 package photo (wccftech 2022-05, origin a Chinese teardown, copyright), scale from the "
                  "die = 814 mm2 (0.0406 mm/px). Stack footprint is the visible top die incl. mould, low confidence.",
    },
    "images": [
        {"path": "img/nonfree/gh100_h100_module_nvidia_render_gtc22.png (local only)",
         "url": "https://cdn.wccftech.com/wp-content/uploads/2022/03/NVIDIA-GH100-H100-Hopper-GPU.png",
         "attribution": "NVIDIA GTC 2022 marketing render, copyright NVIDIA", "type": "marketing render (stylised die)"},
        {"path": "img/nonfree/h100_sxm5_package_photo.png (local only)",
         "url": "https://cdn.wccftech.com/wp-content/uploads/2022/05/20220429-nvidia-h100-hopper-ai-gpu-04-low_res-scale-4_00x-Custom-Custom.png",
         "attribution": "unknown photographer via wccftech, copyright", "type": "package photo"},
        {"path": "img/nonfree/hc34_h100_slide4_blockdiagram.png (local only)",
         "url": "https://hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/HC2022.NVIDIA.Choquette.vfinal01.pdf (slide 4)",
         "attribution": "NVIDIA, Hot Chips 34", "type": "logical block diagram"},
    ],
    "notes": ["TechInsights sells a GH100 floorplan (DFR-2303-801); no free annotated die shot was found. All GH100 blocks are low "
              "confidence and fractions should not be used as calibration targets beyond 'HBM on long edges, I/O on short edges'."],
    "kiln_disagreements": [
        "h100_sxm5_80gb.json5 outline 25.3 x 32.2 (aspect 0.786 assumed) vs 25.9 x 31.4 (0.827) from the package photo.",
        "NVLink placed on S and PCIe on N in kiln; the render shows I/O on both short edges, split unknown.",
        "L2: kiln [1,2] grid of partitions as a separate block; the render shows SM tiles in 3 + 6 + 3 rows (die orientation) "
        "separated by two bands with central hubs, the same arrangement as the GA100 photo (L2 bands between GPC rows).",
    ],
}

# ---------------------------------------------------------------- GV100 (no measurable die image)
CHIPS["gv100"] = {
    "chip": "NVIDIA GV100 (Tesla V100)",
    "node": {"value": "TSMC 12 nm FFN", "source": V100_WP},
    "transistors": {"value": 21.1e9, "source": V100_WP + "; Hot Chips 29 slide 2 says '21B transistors, 815 mm2'"},
    "die": {"w_mm": None, "h_mm": None, "area_mm2_pub": 815, "orientation": "HBM on two opposite (long) edges",
            "source": "no free die photo found; NVIDIA's GTC17 module render shows the die but in perspective with stylised "
                      "content, so no aspect or block boxes are claimed."},
    "die_px": None,
    "blocks": [
        {"id": "l2_bands", "kind": "l2", "count": 2, "bbox_mm": None, "frac": None, "confidence": "low",
         "source": "NVIDIA GTC17 render (rectified by eye): two horizontal bands of SRAM-like blocks, each with a central hub, "
                   "between rows of SM tiles, i.e. the same organisation as the GA100 photo. Not measured."},
        {"id": "nvlink_pcie", "kind": "serdes", "count": None, "bbox_mm": None, "frac": None, "confidence": "low",
         "source": "NVIDIA render: SerDes-like I/O rows along one short edge only."},
    ],
    "edges": {"w": [{"kind": "hbm_phy", "count": 2, "length_mm": None}], "e": [{"kind": "hbm_phy", "count": 2, "length_mm": None}],
              "s": [{"kind": "serdes (NVLink2 x6 + PCIe3, render)", "count": None, "length_mm": None}],
              "note": "HBM 2+2 on the long edges is consistent across render, package photos and the GP100 predecessor."},
    "package": {"hbm_sites": 4, "hbm_active": 4, "hbm_placement": "2 per side along the die long edges",
                "source": "NVIDIA V100 SXM2 module render (GTC17) and product photos"},
    "images": [
        {"path": "img/nonfree/v100_sxm2_module_nvidia_render_gtc17.jpg (local only)",
         "url": "https://cdn.wccftech.com/wp-content/uploads/2017/05/NVIDIA-Telsa-V100.jpg",
         "attribution": "NVIDIA GTC 2017 marketing render, copyright NVIDIA", "type": "marketing render"},
        {"path": "img/nonfree/hc29_v100_slide2_blockdiagram.png (local only)",
         "url": "https://old.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.21-Monday-Pub/HC29.21.10-GPU-Gaming-Pub/HC29.21.132-Volta-Choquette-NVIDIA-Final3.pdf (slide 2)",
         "attribution": "NVIDIA, Hot Chips 29", "type": "logical block diagram"},
    ],
    "notes": ["Searched Wikimedia Commons, Fritzchens Fritz's CC0 set, Kurnal Insights' gallery, TechPowerUp, NVIDIA blogs and "
              "Hot Chips decks: no GV100 die photo with a usable licence/geometry. Fritz/Kurnal have GP100/GA100 but not GV100."],
    "kiln_disagreements": [
        "v100_sxm2_32gb.json5: NVLink S / PCIe N; the render shows I/O only on one short edge.",
        "L2: the render suggests two L2 bands between SM rows (GA100-like), not one block below all GPCs.",
    ],
}

# ---------------------------------------------------------------- TPU v4i (published floorplan)
CHIPS["tpu_v4i"] = {
    "chip": "Google TPU v4i",
    "node": {"value": "7 nm", "source": V4I},
    "transistors": {"value": 16e9, "source": V4I + " Table 1"},
    "die": {"w_mm": 23.07, "h_mm": 17.38, "orientation": "landscape as drawn in Fig. 6",
            "source": "paper gives only '< 400 mm2'; mm values here assume exactly 400 mm2 (upper bound) at the Fig. 6 aspect "
                      "892:672 = 1.327. Fractions do not depend on this assumption."},
    "die_px": (22, 20, 914, 692),
    "overlay_img": {"free": False, "path": "img/nonfree/tpu_v4i_floorplan_jouppi2021_fig6.png", "px_frame_w": 921},
    "overlay_src_note": "Jouppi ISCA 2021 Fig. 6 floorplan",
    "blocks": [
        {"id": "mxu", "kind": "mxu", "count": 4,
         "px": [(47, 271, 148, 433), (165, 271, 266, 433), (688, 271, 788, 433), (805, 271, 912, 433)],
         "confidence": "high", "source": "Fig. 6 (paper text/Table 7: MXUs 11% of die)."},
        {"id": "xlu", "kind": "xlu", "count": 2, "px": [(266, 271, 298, 433), (657, 271, 688, 433)],
         "confidence": "high", "source": "Fig. 6"},
        {"id": "vpu_vmem", "kind": "vpu_vmem", "count": 1, "px": [(298, 271, 657, 433)], "confidence": "high",
         "source": "Fig. 6"},
        {"id": "tcs_smem_imem", "kind": "tc_ctrl", "count": 1, "px": [(690, 211, 788, 269)], "confidence": "high",
         "source": "Fig. 6"},
        {"id": "cmem", "kind": "cmem", "count": 2, "px": [(273, 74, 689, 270), (269, 434, 684, 628)],
         "confidence": "high", "source": "Fig. 6 (paper: CMEM = 28% of die, 128 MB)."},
        {"id": "oci", "kind": "oci", "count": 11,
         "px": [(220, 25, 730, 67), (220, 74, 271, 270), (150, 230, 216, 270), (691, 74, 826, 207), (790, 211, 826, 268),
                (150, 271, 163, 433), (790, 271, 803, 433), (150, 434, 216, 475), (220, 434, 265, 628),
                (691, 434, 823, 630), (220, 637, 731, 685)],
         "confidence": "high", "source": "Fig. 6 (caption: OCI blocks are stretched to fill space)."},
        {"id": "hbmc_serdes", "kind": "hbm_ctrl_phy", "count": 2, "px": [(832, 25, 911, 268), (830, 436, 911, 685)],
         "confidence": "high", "source": "Fig. 6: 'HBMC & SerDes' on the east edge (2 HBM2 stacks)."},
        {"id": "ici_icr_lst", "kind": "ici", "count": 2, "px": [(25, 71, 216, 228), (25, 25, 155, 71)],
         "confidence": "high", "source": "Fig. 6: ICR + 2x 'LST & SerDes' (ICI, 2 x 400 Gb/s) in the NW corner."},
        {"id": "pcie_uhi_mgr", "kind": "pcie", "count": 3, "px": [(25, 565, 216, 690), (150, 479, 214, 563), (70, 495, 148, 563)],
         "confidence": "high", "source": "Fig. 6: PCIe controller & SerDes, UHI, MGR in the SW corner."},
        {"id": "misc_gpio", "kind": "misc", "count": 7,
         "px": [(155, 23, 218, 71), (26, 254, 148, 273), (26, 273, 46, 433), (70, 438, 147, 494), (28, 438, 66, 605),
                (732, 25, 826, 67), (735, 637, 823, 685)],
         "confidence": "high", "source": "Fig. 6: Misc and GPIO blocks."},
    ],
    "edges": {"e": [{"kind": "hbm_ctrl_phy", "count": 2, "length_mm_at_400mm2": 6.3}],
              "w": [{"kind": "ici serdes (LST)", "count": 2}, {"kind": "pcie serdes", "count": 1}, {"kind": "gpio", "count": 1}],
              "n": [{"kind": "gpio", "count": 1}], "s": [{"kind": "gpio", "count": 1}],
              "note": "From Fig. 6 as drawn; HBM on ONE edge only."},
    "package": {"hbm_stacks": 2, "source": V4I + " (8 GB HBM2, 614 GB/s; 2 HBMC blocks in Fig. 6)"},
    "images": [{"path": "img/nonfree/tpu_v4i_floorplan_jouppi2021_fig6.png (local only)",
                "url": "https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf (page 7, Fig. 6); doi 10.1109/ISCA52012.2021.00010",
                "attribution": "Jouppi et al., ISCA 2021, copyright IEEE", "type": "published floorplan figure (schematic, to scale per paper)"}],
    "notes": ["Validation: the redrawn boxes give CMEM 27.1% (paper: 28%) and 4 MXUs 10.8% (paper: 11%).",
              "Caption: 'The TensorCore and CMEM block arrangements are derived from the TPUv4 floorplan.'"],
    "kiln_disagreements": [
        "tpu_v4i.json5 uses outline auto with no shoreline; the paper's floorplan puts both HBM controllers/PHYs on ONE edge (E), "
        "ICI and PCIe SerDes on the opposite (W) edge, and splits CMEM into two halves above and below a central TensorCore row.",
    ],
}

# ---------------------------------------------------------------- TPU v4 (package photo only)
CHIPS["tpu_v4"] = {
    "chip": "Google TPU v4",
    "node": {"value": "7 nm", "source": V4 + " Table 4"},
    "transistors": {"value": 22e9, "source": V4 + " Table 4"},
    "die": {"w_mm": 23.1, "h_mm": 25.7, "orientation": "portrait in Fig. 2; HBM on the long (W/E) edges",
            "source": "paper: '< 600 mm2', and A100 (826) is '~40% larger' (=> ~590). Measured on Fig. 2 package photo: die 262 x "
                      "292 px, scale from HBM2 stack width 88 px = 7.75 mm -> 23.1 x 25.7 mm = 593 mm2 (using the stack length "
                      "instead gives 638 mm2; so +-4%)."},
    "die_px": (213, 186, 475, 478),
    "overlay_img": {"free": True, "path": "img/tpu_v4_package_jouppi2023_fig2.png", "px_frame_w": 686},
    "overlay_src_note": "TPU v4 package photo",
    "blocks": [
        {"id": "sparsecore", "kind": "misc", "count": 4, "bbox_mm": None, "frac": 0.05, "confidence": "high (fraction only)",
         "source": V4 + ": 'SCs are ... only ~5% of the die area'. Location unpublished."},
        {"id": "cmem", "kind": "cmem", "count": 1, "bbox_mm": None, "frac": None, "confidence": "n/a",
         "source": "128 MiB CMEM (Table 4); v4i's floorplan says its CMEM/TensorCore arrangement derives from v4's, but v4's own "
                   "floorplan is unpublished."},
    ],
    "overlay_extra": [("die", "sm_array", (213, 186, 475, 478)), ("hbm", "hbm_phy", (122, 190, 210, 320)),
                      ("hbm", "hbm_phy", (122, 350, 210, 480)), ("hbm", "hbm_phy", (478, 190, 567, 320)),
                      ("hbm", "hbm_phy", (478, 350, 567, 480))],
    "edges": {"w": [{"kind": "hbm (package)", "count": 2}], "e": [{"kind": "hbm (package)", "count": 2}],
              "note": "From the package photo: 2 HBM2 stacks on each long edge of the die."},
    "package": {"hbm_stacks": 4, "hbm_placement": "2 per side, long axis parallel to the die long edges",
                "die_px": [213, 186, 475, 478], "hbm_px": [[122, 190, 210, 320], [122, 350, 210, 480], [478, 190, 567, 320],
                                                           [478, 350, 567, 480]],
                "source": V4 + " Fig. 2 (package photo)"},
    "images": [
        {"path": "img/tpu_v4_package_jouppi2023_fig2.png", "url": "https://arxiv.org/abs/2304.01433 (Fig. 2, inset bitmap)",
         "attribution": "Jouppi et al. 2023, CC BY 4.0", "type": "package photo"},
        {"path": "img/tpu_v4_board_jouppi2023_fig2_commons.jpg", "url": "https://commons.wikimedia.org/wiki/File:TPU_v4.png",
         "attribution": "Jouppi et al. 2023 via Wikimedia Commons, CC BY 4.0", "type": "board photo (packages under cold plates)"},
    ],
    "notes": ["No TPU v4 die photo or floorplan is published (ISCA 2023 has none; the v4i paper reuses its TensorCore/CMEM arrangement)."],
    "kiln_disagreements": ["tpu_v4.json5 assumes a 24 x 24 mm square (576 mm2); the package photo gives ~23.1 x 25.7 mm (aspect 0.90) "
                           "with HBM 2+2 on the long edges."],
}

# ---------------------------------------------------------------- TPU v5e / v6e (nothing to measure)
CHIPS["tpu_v5e"] = {
    "chip": "Google TPU v5e", "node": {"value": None, "source": "unpublished"},
    "transistors": {"value": None, "source": "unpublished"},
    "die": {"w_mm": None, "h_mm": None, "source": "unpublished. A '~325 mm2' SemiAnalysis estimate circulates second-hand "
                                                   "(aiwiki.ai); not verified, not used."},
    "die_px": None, "blocks": [], "edges": {},
    "package": {"hbm_capacity_GB": 16, "hbm_bw_GBps": 819, "hbm_stacks": None,
                "source": "Google Cloud TPU v5e docs (16 GB HBM, 819 GB/s per chip); stack count not published and no package "
                          "photo found (819 GB/s is not one HBM2e stack at <=3.6 Gb/s, so 'single HBM2e stack' claims look wrong)."},
    "images": [], "notes": ["No die photo or floorplan found."], "kiln_disagreements": [],
}
CHIPS["tpu_v6e"] = {
    "chip": "Google TPU v6e (Trillium)", "node": {"value": None, "source": "unpublished"},
    "transistors": {"value": None, "source": "unpublished"},
    "die": {"w_mm": None, "h_mm": None, "source": "unpublished"},
    "die_px": None, "blocks": [], "edges": {},
    "package": {"hbm_capacity_GB": 32, "hbm_stacks": "2 (low confidence)",
                "source": "ServeTheHome SC24 board photos (https://www.servethehome.com/?p=82380, copyright, not stored): the lid "
                          "window shows one large die and a column split into two pieces along one edge, read as 2 HBM stacks "
                          "on one side; resolution too low to confirm. 32 GB per chip from Google Cloud docs."},
    "images": [], "notes": ["No die photo or floorplan found."], "kiln_disagreements": [],
}
