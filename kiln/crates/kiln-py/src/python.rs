//! `kiln._kiln`: PyO3 bindings over [`Session`]. Inputs are data only (str, dict, list, numbers); no Python is run.

use std::sync::{Arc, OnceLock};

use kiln_ir::common::{Diagnostic, canonical_json};
use kiln_trace::result::{EvalResult, Status};
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Number, Value};

use crate::explain::{DEFAULT_MAX_CHARS, explain};
use crate::features::DESCRIPTORS;
use crate::inputs::{DesignInput, WorkloadInput};
use crate::options::Options;
use crate::session::{Session, SessionConfig};

fn to_value(o: &Bound<'_, PyAny>, path: &str) -> PyResult<Value> {
    if o.is_none() {
        Ok(Value::Null)
    } else if let Ok(b) = o.cast::<PyBool>() {
        Ok(Value::Bool(b.is_true()))
    } else if o.cast::<PyInt>().is_ok() {
        if let Ok(i) = o.extract::<i64>() {
            Ok(i.into())
        } else if let Ok(u) = o.extract::<u64>() {
            Ok(u.into())
        } else {
            float(o.extract::<f64>()?, path)
        }
    } else if let Ok(f) = o.cast::<PyFloat>() {
        float(f.value(), path)
    } else if let Ok(s) = o.cast::<PyString>() {
        Ok(Value::String(s.to_str()?.to_owned()))
    } else if let Ok(d) = o.cast::<PyDict>() {
        let mut m = Map::new();
        for (k, v) in d.iter() {
            let k: String = k
                .extract()
                .map_err(|_| PyTypeError::new_err(format!("{path}: dict keys must be str")))?;
            let child = format!("{path}.{k}");
            m.insert(k, to_value(&v, &child)?);
        }
        Ok(Value::Object(m))
    } else if o.cast::<PyList>().is_ok() || o.cast::<PyTuple>().is_ok() {
        o.try_iter()?
            .enumerate()
            .map(|(i, v)| to_value(&v?, &format!("{path}[{i}]")))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array)
    } else {
        Err(PyTypeError::new_err(format!(
            "{path}: unsupported type {}; kiln accepts JSON data only (dict, list, str, int, float, bool, None)",
            o.get_type().name()?
        )))
    }
}

fn float(f: f64, path: &str) -> PyResult<Value> {
    Number::from_f64(f)
        .map(Value::Number)
        .ok_or_else(|| PyValueError::new_err(format!("{path}: non-finite number {f}")))
}

fn to_py(py: Python<'_>, v: &Value) -> PyResult<Py<PyAny>> {
    Ok(match v {
        Value::Null => py.None(),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any().unbind(),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.into_pyobject(py)?.into_any().unbind(),
            (_, Some(u)) => u.into_pyobject(py)?.into_any().unbind(),
            _ => n
                .as_f64()
                .unwrap_or(f64::NAN)
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        },
        Value::String(s) => PyString::new(py, s).into_any().unbind(),
        Value::Array(a) => {
            let items = a
                .iter()
                .map(|x| to_py(py, x))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, items)?.into_any().unbind()
        }
        Value::Object(m) => {
            let d = PyDict::new(py);
            for (k, x) in m {
                d.set_item(k, to_py(py, x)?)?;
            }
            d.into_any().unbind()
        }
    })
}

fn diag_err(d: &Diagnostic) -> PyErr {
    PyValueError::new_err(serde_json::to_string(d).expect("diagnostic serializes"))
}

fn design_input(o: &Bound<'_, PyAny>) -> PyResult<DesignInput> {
    if let Ok(s) = o.extract::<String>() {
        return Ok(DesignInput::Str(s));
    }
    if let Ok(p) = o.extract::<std::path::PathBuf>() {
        return Ok(DesignInput::Str(p.to_string_lossy().into_owned()));
    }
    Ok(DesignInput::Value(to_value(o, "design")?))
}

fn workload_input(o: &Bound<'_, PyAny>) -> PyResult<WorkloadInput> {
    if let Ok(s) = o.extract::<String>() {
        return Ok(WorkloadInput::Str(s));
    }
    if let Ok(p) = o.extract::<std::path::PathBuf>() {
        return Ok(WorkloadInput::Str(p.to_string_lossy().into_owned()));
    }
    Ok(WorkloadInput::Value(to_value(o, "workload")?))
}

fn options(o: Option<&Bound<'_, PyAny>>) -> PyResult<Options> {
    match o {
        None => Ok(Options::default()),
        Some(o) if o.is_none() => Ok(Options::default()),
        Some(o) => Options::from_value(&to_value(o, "options")?).map_err(|d| diag_err(&d)),
    }
}

/// `kiln.result/1` with convenience accessors (06 §6.3).
#[pyclass(frozen, name = "Result", module = "kiln")]
pub struct PyEvalResult {
    inner: EvalResult,
    value: Value,
}

impl PyEvalResult {
    fn new(inner: EvalResult) -> Self {
        let value = inner.to_value();
        Self { inner, value }
    }

    fn field(&self, py: Python<'_>, k: &str) -> PyResult<Py<PyAny>> {
        to_py(py, self.value.get(k).unwrap_or(&Value::Null))
    }
}

#[pymethods]
impl PyEvalResult {
    #[getter]
    fn status(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.field(py, "status")
    }

    #[getter]
    fn ok(&self) -> bool {
        self.inner.status == Status::Ok
    }

    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }

    #[getter]
    fn score_interval(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.field(py, "score_interval")
    }

    #[getter]
    fn score_realistic(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.field(py, "score_realistic")
    }

    #[getter]
    fn stage_reached(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.field(py, "stage_reached")
    }

    #[getter]
    fn phases(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.field(py, "phases")
    }

    #[getter]
    fn physical(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.field(py, "physical")
    }

    #[getter]
    fn features(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.field(py, "features")
    }

    #[getter]
    fn violations(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.field(py, "violations")
    }

    #[getter]
    fn errors(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.field(py, "errors")
    }

    #[getter]
    fn warnings(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.field(py, "warnings")
    }

    #[getter]
    fn provenance(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.field(py, "provenance")
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(py, &self.value)
    }

    fn to_json(&self) -> String {
        canonical_json(&self.value)
    }

    fn deterministic_hash(&self) -> String {
        self.inner.deterministic_hash()
    }

    #[pyo3(signature = (max_items = 8, max_chars = DEFAULT_MAX_CHARS))]
    fn explain(&self, max_items: usize, max_chars: usize) -> String {
        explain(&self.inner, max_items, max_chars)
    }

    fn __getitem__(&self, py: Python<'_>, k: &str) -> PyResult<Py<PyAny>> {
        match self.value.get(k) {
            Some(v) => to_py(py, v),
            None => Err(pyo3::exceptions::PyKeyError::new_err(k.to_string())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "<kiln.Result status={} score={} design={}>",
            self.value["status"].as_str().unwrap_or("?"),
            self.inner.score,
            self.inner.provenance.design_hash
        )
    }
}

/// 06 §6.1 `kiln.Session`. Thread-safe; evaluation releases the GIL.
#[pyclass(frozen, name = "Session", module = "kiln")]
pub struct PySession {
    inner: Arc<Session>,
}

#[pymethods]
impl PySession {
    #[new]
    #[pyo3(signature = (calibration = None, cache_dir = None, cache = "disk", threads = None, calib = None, designs_dir = None))]
    fn new(
        calibration: Option<String>,
        cache_dir: Option<std::path::PathBuf>,
        cache: &str,
        threads: Option<usize>,
        calib: Option<String>,
        designs_dir: Option<std::path::PathBuf>,
    ) -> PyResult<Self> {
        let no_cache = match cache {
            "disk" => false,
            "none" | "off" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "cache must be \"disk\" or \"none\", got {other:?}"
                )));
            }
        };
        let cfg = SessionConfig {
            calibration: calibration.or(calib),
            cache_dir,
            no_cache,
            threads,
            designs_dir,
        };
        Ok(Self {
            inner: Arc::new(Session::new(cfg).map_err(|d| diag_err(&d))?),
        })
    }

    #[pyo3(signature = (design, workload, options = None))]
    fn evaluate(
        &self,
        py: Python<'_>,
        design: &Bound<'_, PyAny>,
        workload: &Bound<'_, PyAny>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyEvalResult> {
        let (d, w, o) = (
            design_input(design)?,
            workload_input(workload)?,
            self::options(options)?,
        );
        let s = self.inner.clone();
        Ok(PyEvalResult::new(py.detach(move || s.evaluate(&d, &w, &o))))
    }

    /// Items are `(design, workload)` pairs, or designs with `options["workload"]`; results keep input order.
    #[pyo3(signature = (items, options = None, max_workers = None, ordered = true))]
    fn evaluate_batch(
        &self,
        py: Python<'_>,
        items: &Bound<'_, PyAny>,
        options: Option<&Bound<'_, PyAny>>,
        max_workers: Option<usize>,
        ordered: bool,
    ) -> PyResult<Vec<PyEvalResult>> {
        let o = self::options(options)?;
        if !ordered {
            return Err(PyNotImplementedError::new_err(
                "ordered=False is not supported; results are always in input order",
            ));
        }
        let default_wl = o.workload.clone().map(|v| match v {
            Value::String(s) => WorkloadInput::Str(s),
            v => WorkloadInput::Value(v),
        });
        let mut pairs = Vec::new();
        for (i, it) in items.try_iter()?.enumerate() {
            let it = it?;
            let pair = match it.cast::<PyTuple>() {
                Ok(t) if t.len() == 2 => (
                    design_input(&t.get_item(0)?)?,
                    workload_input(&t.get_item(1)?)?,
                ),
                _ => match &default_wl {
                    Some(w) => (design_input(&it)?, w.clone()),
                    None => {
                        return Err(PyValueError::new_err(format!(
                            "items[{i}] is a bare design but options has no \"workload\"; pass (design, workload) tuples"
                        )));
                    }
                },
            };
            pairs.push(pair);
        }
        let s = self.inner.clone();
        let rs = py.detach(move || s.evaluate_batch(&pairs, &o, max_workers));
        Ok(rs.into_iter().map(PyEvalResult::new).collect())
    }

    #[pyo3(signature = (design, profile = "search"))]
    fn validate(
        &self,
        py: Python<'_>,
        design: &Bound<'_, PyAny>,
        profile: &str,
    ) -> PyResult<Py<PyAny>> {
        let d = design_input(design)?;
        let errs = self.inner.validate(&d, profile);
        to_py(py, &serde_json::to_value(errs).expect("errors serialize"))
    }

    #[pyo3(signature = (name, workload, options = None))]
    fn baseline(
        &self,
        py: Python<'_>,
        name: &str,
        workload: &Bound<'_, PyAny>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyEvalResult> {
        let (w, o) = (workload_input(workload)?, self::options(options)?);
        let wl = crate::inputs::resolve_workload(&w).map_err(|d| diag_err(&d))?;
        let s = self.inner.clone();
        let name = name.to_string();
        Ok(PyEvalResult::new(
            py.detach(move || s.baseline(&name, &wl, &o)),
        ))
    }

    #[getter]
    fn calibration(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let c = self.inner.calibration();
        to_py(py, &serde_json::json!({"id": c.id, "hash": c.hash}))
    }

    #[getter]
    fn cache_dir(&self) -> Option<String> {
        self.inner.cache_dir().map(|p| p.display().to_string())
    }

    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
    }
}

fn default_session() -> PyResult<Arc<Session>> {
    static S: OnceLock<Arc<Session>> = OnceLock::new();
    if let Some(s) = S.get() {
        return Ok(s.clone());
    }
    let s = Arc::new(Session::new(SessionConfig::default()).map_err(|d| diag_err(&d))?);
    Ok(S.get_or_init(|| s).clone())
}

#[pymodule]
mod _kiln {
    use super::*;

    #[pymodule_export]
    use super::{PyEvalResult, PySession};

    #[pymodule_export]
    const KILN_VERSION: &str = kiln_trace::KILN_VERSION;

    #[pymodule_export]
    const GIT_HASH: &str = crate::GIT_HASH;

    /// Module-level convenience using a default Session.
    #[pyfunction]
    #[pyo3(signature = (design, workload, options = None))]
    fn evaluate(
        py: Python<'_>,
        design: &Bound<'_, PyAny>,
        workload: &Bound<'_, PyAny>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyEvalResult> {
        let (d, w, o) = (
            design_input(design)?,
            workload_input(workload)?,
            super::options(options)?,
        );
        let s = default_session()?;
        Ok(PyEvalResult::new(py.detach(move || s.evaluate(&d, &w, &o))))
    }

    /// IR + profile checks only (S0); returns the structured errors and warnings.
    #[pyfunction]
    #[pyo3(signature = (design, profile = "search"))]
    fn validate(py: Python<'_>, design: &Bound<'_, PyAny>, profile: &str) -> PyResult<Py<PyAny>> {
        let d = design_input(design)?;
        let errs = default_session()?.validate(&d, profile);
        to_py(py, &serde_json::to_value(errs).expect("errors serialize"))
    }

    /// Bench manifest (`kiln.bench/1`) for a suite (`legacy`, `standard`, `smoke`) or one `<preset>:<scenario>`.
    #[pyfunction]
    fn bench_export(py: Python<'_>, suite: &str) -> PyResult<Py<PyAny>> {
        match crate::bench_export(suite) {
            Ok(v) => to_py(py, &v),
            Err(ds) => Err(PyValueError::new_err(
                serde_json::to_string(&ds).expect("diagnostics serialize"),
            )),
        }
    }

    /// LLM-readable summary (06 §6.5) of a Result or a `kiln.result/1` dict.
    #[pyfunction]
    #[pyo3(signature = (result, op = None, max_items = 8, max_chars = DEFAULT_MAX_CHARS))]
    fn explain(
        result: &Bound<'_, PyAny>,
        op: Option<&str>,
        max_items: usize,
        max_chars: usize,
    ) -> PyResult<String> {
        if op.is_some() {
            return Err(PyNotImplementedError::new_err(
                "E-NOT-IMPLEMENTED: per-op explanations need kiln_sim::explain_run (03 §10)",
            ));
        }
        if let Ok(r) = result.cast::<PyEvalResult>() {
            return Ok(super::explain(&r.get().inner, max_items, max_chars));
        }
        let r: EvalResult = serde_json::from_value(to_value(result, "result")?)
            .map_err(|e| PyValueError::new_err(format!("not a kiln.result/1 document: {e}")))?;
        Ok(super::explain(&r, max_items, max_chars))
    }

    /// Headless rendering is owned by 05 (`kiln-viz-render`) and not in this build.
    #[pyfunction]
    #[pyo3(signature = (result_or_path, view = "floorplan", fmt = "png", **_opts))]
    fn render(
        result_or_path: &Bound<'_, PyAny>,
        view: &str,
        fmt: &str,
        _opts: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<u8>> {
        let _ = result_or_path;
        Err(PyNotImplementedError::new_err(format!(
            "E-NOT-IMPLEMENTED: kiln.render(view={view:?}, fmt={fmt:?}) needs kiln-viz-render (05 §7, milestone M4-M5)"
        )))
    }

    /// Standard descriptors mapped to `[0, 1]` on their fixed ranges; `energy_split` becomes three keys.
    #[pyfunction]
    fn normalize_features(py: Python<'_>, features: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let f = serde_json::from_value(to_value(features, "features")?)
            .map_err(|e| PyValueError::new_err(format!("features: {e}")))?;
        to_py(
            py,
            &serde_json::to_value(crate::features::normalized(&f)).expect("floats serialize"),
        )
    }

    /// Standard MAP-Elites descriptors with fixed ranges (06 §6.3).
    #[pyfunction]
    fn descriptors(py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(
            py,
            &serde_json::to_value(DESCRIPTORS).expect("descriptors serialize"),
        )
    }
}
