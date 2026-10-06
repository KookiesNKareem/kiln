//! Native visualizer (05 §4): app state, linked selection and view switching over `kiln-viz-render` scenes.
//! The eframe/wgpu shell is behind the `gui` feature; the state machine below is GUI-free and tested
//! headless.

pub mod state;

#[cfg(feature = "gui")]
mod app;

use std::path::PathBuf;

use kiln_trace::archive::Archive;
use kiln_trace::calib_report::CalibRow;
use kiln_trace::trace::Trace;

pub use state::AppState;

/// Everything the viewer opens.
pub struct Data {
    pub runs: Vec<Trace>,
    pub archive: Option<Archive>,
    pub calib: Option<Vec<CalibRow>>,
    /// Archive directory to watch for appended batches (05 §6.7 live mode).
    pub watch: Option<PathBuf>,
}

#[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
pub fn run(data: Data, spec: kiln_viz_render::ViewSpec) -> Result<(), String> {
    app::run(data, spec)
}

#[cfg(all(feature = "gui", target_arch = "wasm32"))]
pub use app::start_web;
