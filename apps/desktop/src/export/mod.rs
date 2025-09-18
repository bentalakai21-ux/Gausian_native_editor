pub mod ffmpeg;
pub mod ui;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ExportCodec {
    H264,
    AV1,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ExportPreset {
    Source,
    P1080,
    P4K,
}

#[derive(Default, Clone)]
pub struct ExportProgress {
    pub progress: f32,
    pub eta: Option<String>,
    pub done: bool,
    pub error: Option<String>,
}

pub use ui::ExportUiState;
