use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TimelineError {
    #[error("invalid operation: {0}")]
    InvalidOp(String),
}

pub type Frame = i64; // 1-based time in frames, supports negatives for offsets

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Fps {
    pub num: u32,
    pub den: u32,
}

impl Fps {
    pub const fn new(num: u32, den: u32) -> Self { Self { num, den } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")] 
pub enum ItemKind {
    #[serde(rename = "solid")]
    Solid { color: String },

    #[serde(rename = "text")]
    Text { text: String, color: String },

    #[serde(rename = "video")]
    Video {
        src: String,
        frame_rate: Option<f32>,
        #[serde(default)]
        in_offset_sec: f64,
        #[serde(default = "default_rate")]
        rate: f32,
    },

    #[serde(rename = "image")]
    Image { src: String },

    #[serde(rename = "audio")]
    Audio {
        src: String,
        #[serde(default)]
        in_offset_sec: f64,
        #[serde(default = "default_rate")]
        rate: f32,
    },
}

fn default_rate() -> f32 { 1.0 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    pub id: String,
    pub from: Frame,
    pub duration_in_frames: Frame,
    #[serde(flatten)]
    pub kind: ItemKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    pub name: String,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequence {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub fps: Fps,
    pub duration_in_frames: Frame,
    pub tracks: Vec<Track>,
}

impl Sequence {
    pub fn new(name: impl Into<String>, width: u32, height: u32, fps: Fps, duration_in_frames: Frame) -> Self {
        Self { name: name.into(), width, height, fps, duration_in_frames, tracks: Vec::new() }
    }

    pub fn add_track(&mut self, track: Track) { self.tracks.push(track); }
}
