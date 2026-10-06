import json

import pytest

from kiln_evo import config as C
from kiln_evo.archive import Archive, make_axes, write_05_archive


def test_axes_use_fixed_06_ranges():
    mb, ob, fpt = make_axes([{"name": "machine_balance", "bins": 8}, {"name": "onchip_bytes", "bins": 4},
                             {"name": "flops_per_tile", "bins": 6}])
    assert (mb.low, mb.high, mb.log) == (1.0, 4096.0, True)
    assert mb.bin(1.0) == 0 and mb.bin(4096.0) == 7 and mb.bin(1e9) == 7 and mb.bin(0.01) == 0
    assert mb.bin(64.0) == 4  # 2^6 of 2^12 -> halfway
    assert ob.bin(2**27) == 2 and fpt.low == 1e9
    assert mb.edges()[0] == pytest.approx(1.0) and mb.edges()[-1] == pytest.approx(4096.0)
    with pytest.raises(ValueError, match="neither standard"):
        make_axes([{"name": "bogus"}])
    custom = make_axes([{"name": "bogus", "low": 0, "high": 10, "bins": 5}])[0]
    assert custom.bin(5) == 2


def _rec(rid, fit, feats, island=0, status="ok"):
    return {"id": rid, "gen": 1, "island": island, "fitness": fit, "status": status, "features": feats,
            "eligible": status == "ok", "parents": [], "operator": "parametric"}


FEATS = {"machine_balance": 200.0, "onchip_bytes": 9e7, "peak_flops_bf16": 3.1e14, "compute_tiles": 432}


def test_insert_improve_migrate_snapshot():
    a = Archive(make_axes([{"name": "machine_balance", "bins": 8}, {"name": "flops_per_tile", "bins": 4}]), 2)
    r1 = _rec("a", 1.0, FEATS)
    r1["cell"] = list(a.cell(r1["features"]))
    assert a.insert(r1) == "new_cell"
    r2 = {**_rec("b", 0.9, FEATS), "cell": r1["cell"]}
    assert a.insert(r2) is None
    r3 = {**_rec("c", 1.2, FEATS, island=1), "cell": r1["cell"]}
    assert a.insert(r3) == "improved"
    assert a.islands[0][tuple(r1["cell"])] == "a" and a.islands[1][tuple(r1["cell"])] == "c"
    assert a.migrate(1, 0, 1) == ["c"]
    bad = {**_rec("d", 5.0, FEATS, status="invalid"), "cell": None}
    assert a.insert(bad) is None
    assert a.cell({"machine_balance": 3.0}) is None  # flops_per_tile missing -> no cell
    snap = json.loads(json.dumps(a.snapshot()))
    b = Archive(a.axes, 2)
    b.records = a.records
    b.restore(snap)
    assert b.grid == a.grid and b.islands == a.islands
    assert a.best()["id"] == "c" and a.qd_score() == pytest.approx(1.2)


def test_05_archive_files(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.ipc as ipc

    cfg = C.finalize({"name": "t", "out_dir": str(tmp_path)}, tmp_path)
    a = Archive(make_axes(cfg["descriptors"]), 1)
    r = _rec("x", 1.1, FEATS)
    r.update(cell=list(a.cell(FEATS)), descriptors=a.normalized(FEATS), audit={"status": "pending"},
             fitness_low=0.9, fitness_high=1.2, interval_method="corners")
    a.insert(r)
    a.records["y"] = _rec("y", 0.0, {}, status="invalid")
    gens = [{"generation": 0, "best": 1.1, "median": 1.1, "qd_score": 1.1, "coverage": 0.01, "evaluations": 2,
             "invalid_count": 1}]
    write_05_archive(tmp_path, cfg, a, gens, {"x": {"heldout_score": 1.05}}, {"git_hash": "abc"})
    meta = json.loads((tmp_path / "archive.json").read_text())
    assert [x["name"] for x in meta["axes"]] == ["machine_balance", "onchip_bytes", "flops_per_tile"]
    t = ipc.open_file(str(tmp_path / "designs.arrow")).read_all()
    rows = {row["design_id"]: row for row in t.to_pylist()}
    assert rows["x"]["status"] == "elite" and rows["y"]["status"] == "invalid"
    assert rows["x"]["heldout_score"] == 1.05 and rows["x"]["audit_status"] == "pending"
    assert t.schema.field("generation").type == pa.uint32()
    g = ipc.open_file(str(tmp_path / "generations.arrow")).read_all().to_pylist()
    assert g[0]["evaluations"] == 2
