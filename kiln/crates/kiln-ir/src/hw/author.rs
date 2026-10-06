//! Authoring pipeline steps 1-5 (01 §14, §15) on the untyped value tree: JSON5 parsing, schema migration,
//! `extends`/`imports`, params and expressions, template instantiation, shorthands, and `set` patches.

use std::collections::BTreeMap;
use std::path::{Path as FsPath, PathBuf};

use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::expr;
use super::legacy;
use super::quantity::{Bits, Quantity};
use super::select::{Seg, Selector, Start, glob, index_matches};
use super::types::{Layout, Replication, SCHEMA_CURRENT, instance_ids};
use crate::common::Diagnostic;

/// Resolves import/extends paths. `from` is the key of the importing document (None for in-memory roots).
pub trait Loader {
    fn load(&self, from: Option<&str>, rel: &str) -> Result<(String, String), Diagnostic>;
}

/// Loads files relative to the importing file's directory.
pub struct FsLoader;

impl Loader for FsLoader {
    fn load(&self, from: Option<&str>, rel: &str) -> Result<(String, String), Diagnostic> {
        let base = from.and_then(|f| FsPath::new(f).parent()).map(FsPath::to_path_buf).unwrap_or_default();
        let path: PathBuf = base.join(rel);
        let text = std::fs::read_to_string(&path).map_err(|e| {
            Diagnostic::error("E-IR-0209", format!("cannot read {}: {e}", path.display()))
                .hint("import and extends paths are relative to the importing file")
        })?;
        let key = std::fs::canonicalize(&path).unwrap_or(path);
        Ok((key.to_string_lossy().into_owned(), text))
    }
}

/// In-memory documents keyed by name (tests, PyO3 callers, the evolution archive).
#[derive(Default)]
pub struct MemLoader {
    pub files: BTreeMap<String, String>,
}

impl Loader for MemLoader {
    fn load(&self, _from: Option<&str>, rel: &str) -> Result<(String, String), Diagnostic> {
        self.files
            .get(rel)
            .map(|t| (rel.to_owned(), t.clone()))
            .ok_or_else(|| Diagnostic::error("E-IR-0209", format!("import {rel:?} not found")))
    }
}

/// Output of steps 1-5: the document with authoring constructs consumed, ready for typed deserialization.
#[derive(Debug)]
pub struct Authored {
    pub value: Value,
    /// Per-entity `notes` strings (not hashed), keyed by template-level path.
    pub notes: IndexMap<String, String>,
    pub warnings: Vec<Diagnostic>,
    /// Field names written by `set` patches, so typed unknown-field errors on them become E-IR-0212.
    pub patched_fields: Vec<String>,
}

struct Template {
    params: IndexMap<String, Option<Value>>,
    body: Value,
}

struct Ns {
    params: IndexMap<String, Value>,
    templates: IndexMap<String, Template>,
    imports: IndexMap<String, usize>,
}

struct Author<'a> {
    loader: &'a dyn Loader,
    arena: Vec<Ns>,
    loading: Vec<String>,
    instantiating: Vec<(usize, String)>,
    notes: IndexMap<String, String>,
    warnings: Vec<Diagnostic>,
    patched: Vec<String>,
}

const AUTHORING_KEYS: &[&str] = &["extends", "imports", "params", "templates", "set"];
const CHILD_KEYS: &[&str] = &[
    "hosts", "boards", "packages", "dies", "mem_stacks", "clusters", "units", "memories", "networks", "blocks", "ports",
    "switches", "links", "local",
];

/// Steps 1-5 for a root document. `key` identifies it for relative imports (a file path for [`FsLoader`]).
pub fn author(loader: &dyn Loader, key: Option<&str>, text: &str) -> Result<Authored, Vec<Diagnostic>> {
    let mut a = Author {
        loader,
        arena: vec![],
        loading: vec![],
        instantiating: vec![],
        notes: IndexMap::new(),
        warnings: vec![],
        patched: vec![],
    };
    let value = a.run(key, text).map_err(|d| vec![d])?;
    Ok(Authored { value, notes: a.notes, warnings: a.warnings, patched_fields: a.patched })
}

pub fn parse_json5(text: &str) -> Result<Value, Diagnostic> {
    json5::from_str::<Value>(text).map_err(|e| {
        Diagnostic::error("E-IR-0100", format!("syntax error: {e}"))
            .hint("documents are JSON5: unquoted keys, comments and trailing commas are fine; check brackets and quotes")
    })
}

fn obj(v: Value, what: &str) -> Result<Map<String, Value>, Diagnostic> {
    match v {
        Value::Object(m) => Ok(m),
        other => Err(Diagnostic::error("E-IR-0103", format!("{what} must be an object, got {other}"))),
    }
}

/// Import alias -> key of the document that declared it, for imports inherited through `extends`.
type Origins = BTreeMap<String, String>;

impl Author<'_> {
    fn run(&mut self, key: Option<&str>, text: &str) -> Result<Value, Diagnostic> {
        let (doc, origins) = self.parse_doc(key.map(str::to_owned), text)?;
        let ns = self.build_ns(key, &doc, &origins)?;
        let env = self.arena[ns].params.clone();
        let mut out = Map::new();
        let mut set = vec![];
        for (k, v) in doc {
            match k.as_str() {
                "schema" => {
                    out.insert(k, Value::from(SCHEMA_CURRENT));
                }
                "meta" => {
                    out.insert(k, v);
                }
                "set" => set = self.walk(v, ns, &env, "set")?.as_array().cloned().unwrap_or_default(),
                k2 if AUTHORING_KEYS.contains(&k2) => {}
                "system" => {
                    let mut sys = obj(v, "system")?;
                    wrap_system(&mut sys)?;
                    let w = self.walk(Value::Object(sys), ns, &env, "")?;
                    out.insert(k, w);
                }
                _ => {
                    let w = self.walk(v, ns, &env, &k)?;
                    out.insert(k, w);
                }
            }
        }
        let mut out = Value::Object(out);
        shorthands(&mut out)?;
        if let Some(sys) = out.get_mut("system") {
            apply_patches(sys, set, &mut self.patched)?;
        }
        shorthands(&mut out)?;
        Ok(out)
    }

    fn read_ref(&mut self, from: Option<&str>, iref: &Value) -> Result<(String, Map<String, Value>, Origins), Diagnostic> {
        let (path, pin) = match iref {
            Value::String(s) => (s.clone(), None),
            Value::Object(m) => match (m.get("path"), m.get("sha256")) {
                (Some(Value::String(p)), Some(Value::String(h))) => (p.clone(), Some(h.to_lowercase())),
                _ => return Err(Diagnostic::error("E-IR-0103", "import must be a path or {path, sha256}")),
            },
            other => return Err(Diagnostic::error("E-IR-0103", format!("import must be a path, got {other}"))),
        };
        if path.starts_with("kiln:") {
            return Err(Diagnostic::error("E-IR-0209", format!("builtin library {path:?} is not available"))
                .hint("the kiln std library ships after M0; inline the template or import a file"));
        }
        let (key, text) = self.loader.load(from, &path)?;
        if let Some(want) = pin {
            let got = hex::encode(Sha256::digest(text.as_bytes()));
            if got != want {
                return Err(Diagnostic::error("E-IR-0209", format!("{path:?}: sha256 {got} does not match pinned {want}"))
                    .hint("update the pin or restore the pinned file"));
            }
        }
        if let Some(pos) = self.loading.iter().position(|k| *k == key) {
            let cycle: Vec<&str> = self.loading[pos..].iter().map(String::as_str).chain([key.as_str()]).collect();
            return Err(Diagnostic::error("E-IR-0208", format!("import/extends cycle: {}", cycle.join(" -> "))));
        }
        let (doc, origins) = self.parse_doc(Some(key.clone()), &text)?;
        Ok((key, doc, origins))
    }

    /// Steps 1-3 for one document: parse, schema/migration, `extends` merge. Imports inherited through
    /// `extends` keep the key of the document that declared them, to resolve relative to it.
    fn parse_doc(&mut self, key: Option<String>, text: &str) -> Result<(Map<String, Value>, Origins), Diagnostic> {
        let mut doc = obj(parse_json5(text)?, "a hardware document")?;
        match doc.get("schema").and_then(Value::as_str) {
            Some("kiln.hw/1.0" | "kiln.hw/1") => {}
            Some("kiln.hw/0") | None if doc.contains_key("clock_mhz") => {
                doc = obj(legacy::migrate_v0(&Value::Object(doc))?, "migrated document")?;
                self.warnings.push(
                    Diagnostic::warning("W-IR-1902", "document migrated from kiln.hw/0 (harness design JSON)")
                        .hint("run `kiln migrate --write` to store the kiln.hw/1.0 form"),
                );
            }
            None => {
                return Err(Diagnostic::error("E-IR-0102", "missing 'schema'").hint("add schema: \"kiln.hw/1.0\""));
            }
            Some(s) => {
                return Err(Diagnostic::error("E-IR-0107", format!("unknown schema version {s:?}"))
                    .hint("supported: kiln.hw/1.0, kiln.hw/0 (harness JSON, migrated); see `kiln migrate`"));
            }
        }
        let Some(base_ref) = doc.remove("extends") else { return Ok((doc, Origins::new())) };
        self.loading.extend(key.clone());
        let base = self.read_ref(key.as_deref(), &base_ref);
        if key.is_some() {
            self.loading.pop();
        }
        let (bkey, base, mut origins) = base?;
        if let Some(Value::Object(im)) = base.get("imports") {
            for alias in im.keys() {
                origins.entry(alias.clone()).or_insert_with(|| bkey.clone());
            }
        }
        if let Some(Value::Object(im)) = doc.get("imports") {
            origins.retain(|alias, _| !im.contains_key(alias));
        }
        Ok((merge_extends(base, doc), origins))
    }

    fn build_ns(&mut self, key: Option<&str>, doc: &Map<String, Value>, origins: &Origins) -> Result<usize, Diagnostic> {
        let raw = match doc.get("params") {
            Some(v) => obj(v.clone(), "params")?,
            None => Map::new(),
        };
        let params = eval_params(raw)?;
        let mut templates = IndexMap::new();
        if let Some(t) = doc.get("templates") {
            for (name, tv) in obj(t.clone(), "templates")? {
                let mut tm = obj(tv, &format!("template {name:?}"))?;
                let body = tm.remove("body").ok_or_else(|| {
                    Diagnostic::error("E-IR-0102", format!("template {name:?} missing 'body'")).at(format!("templates.{name}"))
                })?;
                let mut tparams = IndexMap::new();
                if let Some(p) = tm.remove("params") {
                    for (pn, decl) in obj(p, "template params")? {
                        let default = match decl {
                            Value::Object(mut d) => d.remove("default"),
                            _ => None,
                        };
                        tparams.insert(pn, default);
                    }
                }
                tm.remove("kind");
                if let Some(k) = tm.keys().next() {
                    return Err(Diagnostic::error("E-IR-0101", format!("template {name:?}: unknown field '{k}'"))
                        .hint("templates have kind, params, body"));
                }
                templates.insert(name, Template { params: tparams, body });
            }
        }
        let mut imports = IndexMap::new();
        if let Some(Value::Object(im)) = doc.get("imports") {
            for (alias, iref) in im {
                let from = origins.get(alias).map(String::as_str).or(key);
                self.loading.extend(from.map(str::to_owned));
                let loaded = self.read_ref(from, iref);
                if from.is_some() {
                    self.loading.pop();
                }
                let (ikey, idoc, iorigins) = loaded.map_err(|d| d.at(format!("imports.{alias}")))?;
                self.loading.push(ikey.clone());
                let ins = self.build_ns(Some(&ikey), &idoc, &iorigins);
                self.loading.pop();
                imports.insert(alias.clone(), ins?);
            }
        }
        self.arena.push(Ns { params, templates, imports });
        Ok(self.arena.len() - 1)
    }

    fn walk(&mut self, v: Value, ns: usize, env: &IndexMap<String, Value>, path: &str) -> Result<Value, Diagnostic> {
        match v {
            Value::String(s) if expr::is_expr(&s) => {
                expr::eval(&s, &|n| env.get(n).cloned()).map_err(|d| d.at(path.trim_start_matches('.')))
            }
            Value::Array(items) => Ok(Value::Array(
                items.into_iter().map(|x| self.walk(x, ns, env, path)).collect::<Result<_, _>>()?,
            )),
            Value::Object(m) if m.contains_key("use") => self.instantiate(m, ns, env, path),
            Value::Object(m) => {
                let path = child_path(path, &m);
                let mut out = Map::new();
                for (k, x) in m {
                    if k == "notes" && x.is_string() {
                        self.notes.insert(path.trim_start_matches('.').to_owned(), x.as_str().unwrap_or("").to_owned());
                        continue;
                    }
                    out.insert(k, self.walk(x, ns, env, &path)?);
                }
                Ok(Value::Object(out))
            }
            other => Ok(other),
        }
    }

    fn instantiate(
        &mut self,
        mut m: Map<String, Value>,
        ns: usize,
        env: &IndexMap<String, Value>,
        path: &str,
    ) -> Result<Value, Diagnostic> {
        let inner = child_path(path, &m);
        let at = inner.trim_start_matches('.').to_owned();
        let name = match m.remove("use") {
            Some(Value::String(s)) => s,
            other => return Err(Diagnostic::error("E-IR-0103", format!("'use' must name a template, got {other:?}")).at(at)),
        };
        let with = match m.remove("with") {
            Some(w) => obj(self.walk(w, ns, env, path)?, "'with'").map_err(|d| d.at(&at))?,
            None => Map::new(),
        };
        let set = m.remove("set");
        let (tns, tname) = match name.split_once('.') {
            Some((alias, rest)) => match self.arena[ns].imports.get(alias) {
                Some(&i) => (i, rest.to_owned()),
                None => return Err(unknown_template(&name, &self.arena[ns]).at(at)),
            },
            None => (ns, name.clone()),
        };
        let Some(t) = self.arena[tns].templates.get(&tname) else {
            return Err(unknown_template(&name, &self.arena[tns]).at(at));
        };
        if self.instantiating.contains(&(tns, tname.clone())) {
            let chain: Vec<&str> = self.instantiating.iter().map(|(_, n)| n.as_str()).collect();
            return Err(Diagnostic::error("E-IR-0208", format!("template cycle: {} -> {tname}", chain.join(" -> "))).at(at));
        }
        let mut tenv = self.arena[tns].params.clone();
        for k in with.keys() {
            if !t.params.contains_key(k) && !tenv.contains_key(k) {
                let declared: Vec<&String> = t.params.keys().collect();
                return Err(Diagnostic::error("E-IR-0203", format!("template {name:?} has no param '{k}'"))
                    .hint(format!("declared params: {declared:?}; document params of the template's file may also be overridden"))
                    .at(at));
            }
        }
        let tparams: Vec<(String, Option<Value>)> = t.params.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let body = t.body.clone();
        for (p, default) in tparams {
            let v = match (with.get(&p), default) {
                (Some(v), _) => v.clone(),
                (None, Some(d)) => self.walk(d, tns, &tenv, &at)?,
                (None, None) => {
                    return Err(Diagnostic::error("E-IR-0202", format!("template {name:?} needs param '{p}'"))
                        .hint(format!("add with: {{ {p}: ... }} at the use site"))
                        .at(at));
                }
            };
            tenv.insert(p, v);
        }
        tenv.extend(with);
        self.instantiating.push((tns, tname));
        let body = self.walk(body, tns, &tenv, &inner);
        self.instantiating.pop();
        let mut body = obj(body?, &format!("template {name:?} body")).map_err(|d| d.at(&at))?;
        for (k, v) in m {
            let w = self.walk(v, ns, env, path)?;
            body.insert(k, w);
        }
        let mut out = Value::Object(body);
        if let Some(set) = set {
            let set = self.walk(set, ns, env, path)?;
            let patches = set.as_array().cloned().unwrap_or_else(|| vec![set]);
            apply_patches(&mut out, patches, &mut self.patched).map_err(|d| d.at(&at))?;
        }
        Ok(out)
    }
}

fn child_path(path: &str, m: &Map<String, Value>) -> String {
    match m.get("id").and_then(Value::as_str) {
        Some(id) if path.is_empty() => id.to_owned(),
        Some(id) => format!("{path}.{id}"),
        None => path.to_owned(),
    }
}

fn unknown_template(name: &str, ns: &Ns) -> Diagnostic {
    let mut known: Vec<String> = ns.templates.keys().cloned().collect();
    known.extend(ns.imports.keys().map(|a| format!("{a}.*")));
    Diagnostic::error("E-IR-0201", format!("unknown template {name:?}")).hint(format!("known templates: {known:?}"))
}

/// Evaluates document params; params may reference each other in any order (cycles are E-IR-0204).
fn eval_params(raw: Map<String, Value>) -> Result<IndexMap<String, Value>, Diagnostic> {
    let mut done: IndexMap<String, Value> = IndexMap::new();
    let mut pending: Vec<(String, Value)> = raw.into_iter().collect();
    while !pending.is_empty() {
        let pending_names: Vec<String> = pending.iter().map(|(k, _)| k.clone()).collect();
        let mut next = vec![];
        let before = pending.len();
        for (k, v) in pending {
            match eval_value(&v, &done) {
                Ok(r) => {
                    done.insert(k, r);
                }
                Err(d) if pending_names.iter().any(|p| d.message.contains(&format!("'{p}'"))) => {
                    next.push((k, v));
                }
                Err(d) => return Err(d.at(format!("params.{k}"))),
            }
        }
        if next.len() == before {
            let names: Vec<&String> = next.iter().map(|(k, _)| k).collect();
            return Err(Diagnostic::error("E-IR-0204", format!("params reference each other cyclically: {names:?}")));
        }
        pending = next;
    }
    let mut ordered = IndexMap::new();
    for (k, v) in done {
        ordered.insert(k, v);
    }
    Ok(ordered)
}

fn eval_value(v: &Value, env: &IndexMap<String, Value>) -> Result<Value, Diagnostic> {
    match v {
        Value::String(s) if expr::is_expr(s) => expr::eval(s, &|n| env.get(n).cloned()),
        Value::Array(a) => a.iter().map(|x| eval_value(x, env)).collect::<Result<_, _>>().map(Value::Array),
        Value::Object(m) => {
            m.iter().map(|(k, x)| Ok((k.clone(), eval_value(x, env)?))).collect::<Result<_, _>>().map(Value::Object)
        }
        other => Ok(other.clone()),
    }
}

/// `extends`: the derived document overrides top-level fields; `params`/`templates`/`imports` merge by key,
/// `set` patches append, `meta` citations/notes merge while claims and description are replaced.
fn merge_extends(mut base: Map<String, Value>, cur: Map<String, Value>) -> Map<String, Value> {
    for (k, v) in cur {
        match (k.as_str(), base.get_mut(&k), v) {
            ("params" | "templates" | "imports", Some(Value::Object(b)), Value::Object(c)) => b.extend(c),
            ("set", Some(Value::Array(b)), Value::Array(c)) => b.extend(c),
            ("meta", Some(Value::Object(b)), Value::Object(c)) => {
                for (mk, mv) in c {
                    match (mk.as_str(), b.get_mut(&mk), mv) {
                        ("citations" | "notes", Some(Value::Object(bm)), Value::Object(cm)) => bm.extend(cm),
                        (_, _, mv) => {
                            b.insert(mk, mv);
                        }
                    }
                }
            }
            (_, _, v) => {
                base.insert(k, v);
            }
        }
    }
    base
}

fn wrap_system(sys: &mut Map<String, Value>) -> Result<(), Diagnostic> {
    if let Some(die) = sys.remove("die") {
        if sys.contains_key("package") {
            return Err(Diagnostic::error("E-IR-0103", "system has both 'die' and 'package' shorthands").at("system"));
        }
        sys.insert("package".into(), json!({"id": "chip", "dies": [die]}));
    }
    if let Some(pkg) = sys.remove("package") {
        let board = json!({"id": "board", "packages": [pkg]});
        match sys.entry("boards").or_insert_with(|| json!([])) {
            Value::Array(b) => b.insert(0, board),
            _ => return Err(Diagnostic::error("E-IR-0103", "system.boards must be a list").at("system")),
        }
    }
    Ok(())
}

/// String and partial-object shorthands of 01 §14.7, rewritten by key.
pub fn shorthands(v: &mut Value) -> Result<(), Diagnostic> {
    match v {
        Value::Array(a) => a.iter_mut().try_for_each(shorthands),
        Value::Object(m) => {
            for (k, x) in m.iter_mut() {
                match (k.as_str(), &mut *x) {
                    ("feeds", Value::Object(f)) => {
                        for fv in f.values_mut() {
                            if let Value::String(s) = fv {
                                *fv = json!({ "from": s.clone() });
                            }
                        }
                    }
                    ("link" | "port", Value::String(s)) => {
                        let bits = Bits::parse_str(s)?;
                        *x = json!({ "width_bits": bits.0 });
                    }
                    ("topology", Value::String(s)) => *x = json!({ "type": s.clone() }),
                    ("endpoints", Value::Array(eps)) => {
                        for e in eps.iter_mut() {
                            if let Value::String(s) = e {
                                *e = json!({ "select": s.clone() });
                            }
                        }
                    }
                    ("attach", Value::String(s)) => *x = json!({ "network": s.clone() }),
                    ("at", Value::Object(o)) if o.len() == 1 => {
                        if let Some(r) = o.remove("router") {
                            *x = json!({ "fixed": { "router": r } });
                        } else if let Some(p) = o.remove("per_router") {
                            *x = json!({ "concentrated": { "per_router": p } });
                        }
                    }
                    _ => {}
                }
                shorthands(x)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Applies `set` patches (`{"sel#field": value}` or `{path, value}`) to a tree whose root is `root`.
pub fn apply_patches(root: &mut Value, patches: Vec<Value>, patched: &mut Vec<String>) -> Result<(), Diagnostic> {
    for p in patches {
        let entries: Vec<(String, Value)> = match p {
            Value::Object(m) if m.len() == 2 && m.contains_key("path") && m.contains_key("value") => {
                let mut m = m;
                match (m.remove("path"), m.remove("value")) {
                    (Some(Value::String(path)), Some(v)) => vec![(path, v)],
                    _ => return Err(Diagnostic::error("E-IR-0103", "set entry {path, value} needs a string path")),
                }
            }
            Value::Object(m) => m.into_iter().collect(),
            other => return Err(Diagnostic::error("E-IR-0103", format!("set entries must be objects, got {other}"))),
        };
        for (path, value) in entries {
            let (sel, field) = path.split_once('#').ok_or_else(|| {
                Diagnostic::error("E-IR-0212", format!("patch path {path:?} has no '#field'"))
                    .hint("write \"<selector>#<field>\", e.g. \"board.chip.die#clusters+\"")
            })?;
            let targets = if sel.is_empty() { vec![String::new()] } else { match_compact(root, sel)? };
            if targets.is_empty() {
                return Err(Diagnostic::error("E-IR-0205", format!("patch selector {sel:?} matches nothing"))
                    .hint("patch selectors are absolute from the system root, e.g. board.chip.die")
                    .at(format!("set.{path}")));
            }
            for t in targets {
                let target = root.pointer_mut(&t).expect("pointer from match_compact");
                apply_field(target, field, value.clone(), patched).map_err(|d| d.at(format!("set.{path}")))?;
            }
        }
    }
    Ok(())
}

pub(crate) fn apply_field(target: &mut Value, field: &str, value: Value, patched: &mut Vec<String>) -> Result<(), Diagnostic> {
    let (field, append) = match field.strip_suffix('+') {
        Some(f) => (f, true),
        None => (field, false),
    };
    let segs: Vec<&str> = field.split('.').collect();
    let (last, parents) = segs.split_last().expect("split yields one segment");
    let missing = |s: &str| {
        Diagnostic::error("E-IR-0212", format!("patch targets non-existent field '{s}'"))
            .hint("check the field name against 01's entity schema")
    };
    let mut cur = target;
    for s in parents {
        cur = cur.get_mut(*s).filter(|c| c.is_object()).ok_or_else(|| missing(s))?;
    }
    let Value::Object(m) = cur else { return Err(missing(last)) };
    if append {
        let Value::Array(arr) = m.entry(*last).or_insert_with(|| json!([])) else {
            return Err(Diagnostic::error("E-IR-0212", format!("'{last}+' appends, but '{last}' is not a list")));
        };
        match value {
            Value::Array(items) => arr.extend(items),
            v => arr.push(v),
        }
    } else {
        m.insert((*last).to_owned(), value);
        patched.push((*last).to_owned());
    }
    Ok(())
}

fn compact_children(node: &Value, ptr: &str) -> Vec<String> {
    let mut out = vec![];
    let Value::Object(m) = node else { return out };
    let mut scan = |m: &Map<String, Value>, base: String| {
        for key in CHILD_KEYS {
            if let Some(Value::Array(a)) = m.get(*key) {
                for (i, el) in a.iter().enumerate() {
                    if el.get("id").is_some_and(Value::is_string) {
                        out.push(format!("{base}/{key}/{i}"));
                    }
                }
            }
        }
    };
    scan(m, ptr.to_owned());
    if let Some(Value::Object(ld)) = m.get("logic_die") {
        scan(ld, format!("{ptr}/logic_die"));
    }
    out
}

fn compact_rep(v: &Value) -> Replication {
    let count = v.get("count").and_then(Value::as_u64).and_then(|c| u32::try_from(c).ok());
    let layout = v.get("layout").and_then(|l| serde_json::from_value::<Layout>(l.clone()).ok()).unwrap_or_default();
    Replication { count, layout, ..Default::default() }
}

/// Matches a selector against template-level entities (before instance expansion). A segment must select every
/// instance of an entity or none; picking individual instances is what `vary` is for.
pub fn match_compact(root: &Value, sel: &str) -> Result<Vec<String>, Diagnostic> {
    let s = Selector::parse(sel)?;
    if matches!(s.start, Start::Up(_)) {
        return Err(Diagnostic::error("E-IR-0212", format!("patch selector {sel:?} cannot start with '^.'")));
    }
    let mut frontier = vec![String::new()];
    for seg in &s.segs {
        let mut next: Vec<String> = vec![];
        match seg {
            Seg::AnyDepth => {
                let mut stack = frontier.clone();
                while let Some(p) = stack.pop() {
                    let node = root.pointer(&p).expect("valid pointer");
                    stack.extend(compact_children(node, &p));
                    next.push(p);
                }
                next.sort();
            }
            Seg::Pat { name, index } => {
                for p in &frontier {
                    for c in compact_children(root.pointer(p).expect("valid pointer"), p) {
                        let node = root.pointer(&c).expect("valid pointer");
                        let id = node["id"].as_str().unwrap_or_default();
                        let insts = instance_ids(id, &compact_rep(node));
                        let hits = insts
                            .iter()
                            .enumerate()
                            .filter(|(i, (inst, coord))| {
                                (name == id || glob(name, inst))
                                    && index.as_ref().is_none_or(|ax| index_matches(ax, *i as u32, coord))
                            })
                            .count();
                        if hits == insts.len() {
                            next.push(c);
                        } else if hits > 0 {
                            return Err(Diagnostic::error(
                                "E-IR-0212",
                                format!("patch selector {sel:?} picks {hits} of {} instances of '{id}'", insts.len()),
                            )
                            .hint("patches apply to whole template-level entities; use `vary` for per-instance values"));
                        }
                    }
                }
            }
        }
        next.dedup();
        frontier = next;
    }
    Ok(frontier.into_iter().filter(|p| !p.is_empty()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(files: &[(&str, &str)], root: &str) -> Result<Authored, Vec<Diagnostic>> {
        let loader = MemLoader { files: files.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect() };
        author(&loader, Some(root), &loader.files[root])
    }

    const BASE: &str = r#"{
      schema: "kiln.hw/1.0", name: "base", tech: "tsmc_n5",
      params: { n: 4, cap: "1MiB", twice: "= cap * 2" },
      templates: { tile: { kind: "cluster", params: { lanes: { default: 8 } }, body: {
        units: [ { id: "v", kind: "vector", lanes: "=lanes", precisions: ["fp32@1"], feeds: { any: "m" } } ],
        memories: [ { id: "m", kind: "scratchpad", capacity: "=twice", ports: [ { dir: "rw", width_bits: 64 } ] } ],
      } } },
      system: { die: { id: "die", clusters: [ { id: "t", count: "=n", use: "tile", with: { lanes: 16 } } ] } },
    }"#;

    #[test]
    fn templates_params_and_wrap() {
        let a = run(&[("base", BASE)], "base").unwrap();
        let t = &a.value["system"]["boards"][0]["packages"][0]["dies"][0]["clusters"][0];
        assert_eq!(a.value["system"]["boards"][0]["packages"][0]["id"], "chip");
        assert_eq!(t["count"], 4);
        assert_eq!(t["units"][0]["lanes"], 16);
        assert_eq!(t["units"][0]["feeds"]["any"], json!({"from": "m"}));
        assert_eq!(t["memories"][0]["capacity"], "2097152B");
        assert!(a.value.get("templates").is_none() && a.value.get("params").is_none());
    }

    #[test]
    fn extends_overrides_params_and_appends_set() {
        let derived = r#"{ schema: "kiln.hw/1.0", name: "derived", extends: "base", params: { n: 2, cap: "4MiB" },
          set: [ { "board.chip.die.t#memories+": [ { id: "x", kind: "fifo", capacity: 64, ports: [ { dir: "rw", width_bits: 8 } ] } ] },
                 { "board.chip.die.t.m#banks": 2 } ] }"#;
        let a = run(&[("base", BASE), ("derived", derived)], "derived").unwrap();
        let t = &a.value["system"]["boards"][0]["packages"][0]["dies"][0]["clusters"][0];
        assert_eq!(a.value["name"], "derived");
        assert_eq!(t["count"], 2);
        assert_eq!(t["memories"][0]["capacity"], "8388608B");
        assert_eq!(t["memories"][0]["banks"], 2);
        assert_eq!(t["memories"][1]["id"], "x");
    }

    #[test]
    fn authoring_errors() {
        let code = |files: &[(&str, &str)], root: &str| run(files, root).unwrap_err()[0].code.clone();
        let cyc_a = r#"{ schema: "kiln.hw/1.0", extends: "b" }"#;
        let cyc_b = r#"{ schema: "kiln.hw/1.0", extends: "a" }"#;
        assert_eq!(code(&[("a", cyc_a), ("b", cyc_b)], "a"), "E-IR-0208");
        let missing = BASE.replace("use: \"tile\"", "use: \"tyle\"");
        assert_eq!(code(&[("x", &missing)], "x"), "E-IR-0201");
        let bad_with = BASE.replace("with: { lanes: 16 }", "with: { lanez: 16 }");
        assert_eq!(code(&[("x", &bad_with)], "x"), "E-IR-0203");
        let no_default = BASE.replace("{ default: 8 }", "{}").replace("with: { lanes: 16 }", "with: {}");
        assert_eq!(code(&[("x", &no_default)], "x"), "E-IR-0202");
        let bad_patch = BASE.replace("system:", r#"set: [ { "board.nope#x": 1 } ], system:"#);
        assert_eq!(code(&[("x", &bad_patch)], "x"), "E-IR-0205");
        assert_eq!(code(&[("x", "{ schema: \"kiln.hw/9.0\" }")], "x"), "E-IR-0107");
        assert_eq!(code(&[("x", "{ schema: ")], "x"), "E-IR-0100");
        let pinned = r#"{ schema: "kiln.hw/1.0", imports: { b: { path: "base", sha256: "00" } } }"#;
        assert_eq!(code(&[("x", pinned), ("base", BASE)], "x"), "E-IR-0209");
    }
}
