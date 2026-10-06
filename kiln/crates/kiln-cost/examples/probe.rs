use kiln_cost::*;
use kiln_ir::hw::{check_file, Profile};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let r = check_file(&args[1], Profile::Reference);
    let hw = r.model.expect("model");
    let unit = hw.units.iter().position(|u| hw.nodes[u.node].path.ends_with(&args[2])).expect("unit");
    let gang: u32 = args.get(3).map_or(1, |g| g.parse().unwrap());
    let t = UnitTemplate::from_hw(&hw, unit, &TemplateOptions { gang, ..Default::default() }).unwrap();
    println!("{}", serde_json::to_string_pretty(&t).unwrap());
}
