//! Graphs (02 §7).

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::op::Node;
use super::tensor::TensorDecl;
use crate::common::Id;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionKind {
    Fusion,
    PipelineStage,
    User,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    pub nodes: Vec<Id>,
    pub kind: RegionKind,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub params: Vec<Id>,
    pub results: Vec<Id>,
    pub tensors: IndexMap<Id, TensorDecl>,
    pub nodes: Vec<Node>,
    #[serde(default)]
    pub regions: IndexMap<Id, Region>,
}

impl Graph {
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id.as_str() == id)
    }

    /// Canonical node order (02 §11.5): Kahn's algorithm with the ready set ordered by id. Nodes on a cycle
    /// are appended in id order (validation reports the cycle).
    pub fn topo_order(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let producer = |t: &Id| self.nodes.iter().position(|nd| nd.outputs.contains(t));
        let deps: Vec<Vec<usize>> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, nd)| {
                let mut d: Vec<usize> = nd
                    .inputs
                    .iter()
                    .filter_map(producer)
                    .filter(|&p| p != i)
                    .collect();
                d.sort_unstable();
                d.dedup();
                d
            })
            .collect();
        let mut indeg: Vec<usize> = deps.iter().map(Vec::len).collect();
        let key = |i: usize| self.nodes[i].id.clone();
        let mut ready: std::collections::BTreeSet<(Id, usize)> = (0..n)
            .filter(|&i| indeg[i] == 0)
            .map(|i| (key(i), i))
            .collect();
        let mut order = Vec::with_capacity(n);
        while let Some(first) = ready.pop_first() {
            let i = first.1;
            order.push(i);
            for (j, d) in deps.iter().enumerate() {
                if d.contains(&i) {
                    indeg[j] -= 1;
                    if indeg[j] == 0 {
                        ready.insert((key(j), j));
                    }
                }
            }
        }
        let mut rest: Vec<usize> = (0..n).filter(|i| !order.contains(i)).collect();
        rest.sort_by_key(|&i| key(i));
        order.extend(rest);
        order
    }
}
