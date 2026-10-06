//! Versioned data files (04 §2), embedded so every binary carries the exact tables it was built with; their
//! sha256 enters provenance and the calibration-set hash.

use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub const FILES: &[(&str, &str)] = &[
    ("calib/a100_40gb-2026-10-05.json", include_str!("../data/calib/a100_40gb-2026-10-05.json")),
    ("calib/generic-2026-10-05.json", include_str!("../data/calib/generic-2026-10-05.json")),
    ("datapath.json", include_str!("../data/datapath.json")),
    ("dram/custom.json", include_str!("../data/dram/custom.json")),
    ("dram/ddr5.json", include_str!("../data/dram/ddr5.json")),
    ("dram/gddr6.json", include_str!("../data/dram/gddr6.json")),
    ("dram/gddr7.json", include_str!("../data/dram/gddr7.json")),
    ("dram/hbm2.json", include_str!("../data/dram/hbm2.json")),
    ("dram/hbm2e.json", include_str!("../data/dram/hbm2e.json")),
    ("dram/hbm3.json", include_str!("../data/dram/hbm3.json")),
    ("dram/hbm3e.json", include_str!("../data/dram/hbm3e.json")),
    ("dram/hbm4.json", include_str!("../data/dram/hbm4.json")),
    ("dram/lpddr5x.json", include_str!("../data/dram/lpddr5x.json")),
    ("dram/stacked_dram.json", include_str!("../data/dram/stacked_dram.json")),
    ("package/cowos_l.json", include_str!("../data/package/cowos_l.json")),
    ("package/cowos_s.json", include_str!("../data/package/cowos_s.json")),
    ("package/emib.json", include_str!("../data/package/emib.json")),
    ("package/organic.json", include_str!("../data/package/organic.json")),
    ("package/soic.json", include_str!("../data/package/soic.json")),
    ("params.json", include_str!("../data/params.json")),
    ("phy/hbm2.json", include_str!("../data/phy/hbm2.json")),
    ("phy/hbm2e.json", include_str!("../data/phy/hbm2e.json")),
    ("phy/hbm3.json", include_str!("../data/phy/hbm3.json")),
    ("phy/hbm3e.json", include_str!("../data/phy/hbm3e.json")),
    ("phy/hbm4.json", include_str!("../data/phy/hbm4.json")),
    ("phy/hybrid_bond.json", include_str!("../data/phy/hybrid_bond.json")),
    ("phy/lpddr.json", include_str!("../data/phy/lpddr.json")),
    ("phy/microbump.json", include_str!("../data/phy/microbump.json")),
    ("phy/nvlink_c2c.json", include_str!("../data/phy/nvlink_c2c.json")),
    ("phy/optical.json", include_str!("../data/phy/optical.json")),
    ("phy/pcie.json", include_str!("../data/phy/pcie.json")),
    ("phy/serdes_112g.json", include_str!("../data/phy/serdes_112g.json")),
    ("phy/serdes_224g.json", include_str!("../data/phy/serdes_224g.json")),
    ("phy/serdes_56g.json", include_str!("../data/phy/serdes_56g.json")),
    ("phy/tsv.json", include_str!("../data/phy/tsv.json")),
    ("phy/ucie_adv.json", include_str!("../data/phy/ucie_adv.json")),
    ("phy/ucie_std.json", include_str!("../data/phy/ucie_std.json")),
    ("sources.json", include_str!("../data/sources.json")),
    ("targets/dies.json", include_str!("../data/targets/dies.json")),
    ("targets/fractions.json", include_str!("../data/targets/fractions.json")),
    ("targets/power.json", include_str!("../data/targets/power.json")),
    ("tech/aliases.json", include_str!("../data/tech/aliases.json")),
    ("tech/dram_logic_1y.json", include_str!("../data/tech/dram_logic_1y.json")),
    ("tech/tsmc_n16.json", include_str!("../data/tech/tsmc_n16.json")),
    ("tech/tsmc_n2.json", include_str!("../data/tech/tsmc_n2.json")),
    ("tech/tsmc_n3e.json", include_str!("../data/tech/tsmc_n3e.json")),
    ("tech/tsmc_n4.json", include_str!("../data/tech/tsmc_n4.json")),
    ("tech/tsmc_n5.json", include_str!("../data/tech/tsmc_n5.json")),
    ("tech/tsmc_n7.json", include_str!("../data/tech/tsmc_n7.json")),
];

pub fn file(path: &str) -> Option<&'static str> {
    FILES.iter().find(|(p, _)| *p == path).map(|(_, t)| *t)
}

/// `phys1-` + 32 hex chars of sha256 over every data file (path, NUL, content, NUL) in path order.
pub fn data_hash() -> &'static str {
    static H: OnceLock<String> = OnceLock::new();
    H.get_or_init(|| {
        let mut h = Sha256::new();
        for (p, t) in FILES {
            h.update(p.as_bytes());
            h.update([0]);
            h.update(t.as_bytes());
            h.update([0]);
        }
        format!("phys1-{}", &hex::encode(h.finalize())[..32])
    })
}

/// Like `data_hash` but without the fitted sets (`calib/`): the inputs a calibration set is fitted from, so a set can
/// record it without hashing itself.
pub fn inputs_hash() -> &'static str {
    static H: OnceLock<String> = OnceLock::new();
    H.get_or_init(|| {
        let mut h = Sha256::new();
        for (p, t) in FILES.iter().filter(|(p, _)| !p.starts_with("calib/")) {
            h.update(p.as_bytes());
            h.update([0]);
            h.update(t.as_bytes());
            h.update([0]);
        }
        format!("phys1-{}", &hex::encode(h.finalize())[..32])
    })
}
