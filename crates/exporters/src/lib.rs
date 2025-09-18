use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;
use timeline::{Fps, Item, ItemKind, Sequence, Track};
use uuid::Uuid;

pub mod edl;
pub mod fcp7xml;
pub mod fcpxml;

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("XML serialization error: {0}")]
    XmlSerialization(#[from] quick_xml::Error),
    #[error("JSON serialization error: {0}")]
    JsonSerialization(#[from] serde_json::Error),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("Invalid timecode: {0}")]
    InvalidTimecode(String),
    #[error("Missing asset: {0}")]
    MissingAsset(String),
    #[error("Relinking failed: {0}")]
    RelinkingFailed(String),
}

/// Supported export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    FcpXml1_9,
    FcpXml1_10,
    Fcp7Xml,
    Edl,
    AvidEdl,
    Json,
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    pub format: ExportFormat,
    pub output_path: PathBuf,
    pub project_name: String,
    pub sequence_name: String,
    pub relink_strategy: RelinkStrategy,
    pub timecode_format: TimecodeFormat,
    pub frame_rate: Fps,
    pub audio_sample_rate: u32,
    pub preserve_folder_structure: bool,
    pub include_unused_media: bool,
    pub color_space: ColorSpace,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RelinkStrategy {
    /// Keep absolute paths as-is
    Absolute,
    /// Convert to relative paths from export location
    Relative,
    /// Use path heuristics to find moved files
    Heuristic,
    /// Copy media to export folder
    Copy,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TimecodeFormat {
    /// Drop frame timecode (e.g., 29.97fps)
    DropFrame,
    /// Non-drop frame timecode
    NonDropFrame,
    /// Frames only
    Frames,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ColorSpace {
    Rec709,
    Rec2020,
    DciP3,
    AdobeRgb,
}

/// Main exporter struct
pub struct Exporter {
    config: ExportConfig,
}

impl Exporter {
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Export sequence to specified format
    pub fn export_sequence(&self, sequence: &Sequence, assets: &[AssetInfo]) -> Result<()> {
        match self.config.format {
            ExportFormat::FcpXml1_9 | ExportFormat::FcpXml1_10 => {
                fcpxml::export_fcpxml(sequence, assets, &self.config)
            }
            ExportFormat::Fcp7Xml => fcp7xml::export_fcp7xml(sequence, assets, &self.config),
            ExportFormat::Edl | ExportFormat::AvidEdl => {
                edl::export_edl(sequence, assets, &self.config)
            }
            ExportFormat::Json => self.export_json(sequence, assets),
        }
    }

    /// Import sequence from file
    pub fn import_sequence(&self, path: &Path) -> Result<(Sequence, Vec<AssetInfo>)> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| ExportError::UnsupportedFormat("Unknown file extension".to_string()))?;

        match extension.to_lowercase().as_str() {
            "fcpxml" => fcpxml::import_fcpxml(path, &self.config),
            "xml" => {
                // Try to detect if it's FCP7 XML or FCPXML
                let content = std::fs::read_to_string(path)?;
                if content.contains("xmeml") {
                    fcp7xml::import_fcp7xml(path, &self.config)
                } else if content.contains("fcpxml") {
                    fcpxml::import_fcpxml(path, &self.config)
                } else {
                    Err(ExportError::UnsupportedFormat("Unknown XML format".to_string()).into())
                }
            }
            "edl" => edl::import_edl(path, &self.config),
            "json" => self.import_json(path),
            _ => Err(
                ExportError::UnsupportedFormat(format!("Unsupported format: {}", extension)).into(),
            ),
        }
    }

    fn export_json(&self, sequence: &Sequence, assets: &[AssetInfo]) -> Result<()> {
        let export_data = ExportData {
            sequence: sequence.clone(),
            assets: assets.to_vec(),
            metadata: ExportMetadata {
                exported_at: Utc::now(),
                exporter_version: env!("CARGO_PKG_VERSION").to_string(),
                config: self.config.clone(),
            },
        };

        let json = serde_json::to_string_pretty(&export_data)?;
        std::fs::write(&self.config.output_path, json)?;
        Ok(())
    }

    fn import_json(&self, path: &Path) -> Result<(Sequence, Vec<AssetInfo>)> {
        let content = std::fs::read_to_string(path)?;
        let export_data: ExportData = serde_json::from_str(&content)?;
        Ok((export_data.sequence, export_data.assets))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetInfo {
    pub id: String,
    pub path: PathBuf,
    pub relative_path: Option<PathBuf>,
    pub kind: AssetKind,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration_frames: Option<i64>,
    pub fps: Option<Fps>,
    pub audio_channels: Option<u32>,
    pub sample_rate: Option<u32>,
    pub timecode: Option<String>,
    pub color_space: Option<ColorSpace>,
    pub file_size: Option<u64>,
    pub hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetKind {
    Video,
    Audio,
    Image,
    Sequence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExportData {
    sequence: Sequence,
    assets: Vec<AssetInfo>,
    metadata: ExportMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExportMetadata {
    exported_at: DateTime<Utc>,
    exporter_version: String,
    config: ExportConfig,
}

/// Utility functions for timecode conversion
pub mod timecode {
    use super::*;

    pub fn frames_to_timecode(frames: i64, fps: Fps, format: TimecodeFormat) -> String {
        let fps_float = fps.num as f64 / fps.den as f64;

        match format {
            TimecodeFormat::Frames => frames.to_string(),
            TimecodeFormat::NonDropFrame => {
                let total_seconds = frames as f64 / fps_float;
                let hours = (total_seconds / 3600.0) as u32;
                let minutes = ((total_seconds % 3600.0) / 60.0) as u32;
                let seconds = (total_seconds % 60.0) as u32;
                let frame = (frames % fps.num as i64) as u32;
                format!("{:02}:{:02}:{:02}:{:02}", hours, minutes, seconds, frame)
            }
            TimecodeFormat::DropFrame => {
                // Simplified drop frame calculation for 29.97fps
                if (fps.num == 30000 && fps.den == 1001) || (fps.num == 30 && fps.den == 1) {
                    let total_seconds = frames as f64 / fps_float;
                    let hours = (total_seconds / 3600.0) as u32;
                    let minutes = ((total_seconds % 3600.0) / 60.0) as u32;
                    let seconds = (total_seconds % 60.0) as u32;
                    let frame = (frames % fps.num as i64) as u32;
                    format!("{:02}:{:02}:{:02};{:02}", hours, minutes, seconds, frame)
                } else {
                    // Fall back to non-drop frame for other rates
                    frames_to_timecode(frames, fps, TimecodeFormat::NonDropFrame)
                }
            }
        }
    }

    pub fn timecode_to_frames(timecode: &str, fps: Fps, format: TimecodeFormat) -> Result<i64> {
        match format {
            TimecodeFormat::Frames => timecode
                .parse::<i64>()
                .map_err(|_| ExportError::InvalidTimecode(timecode.to_string()).into()),
            TimecodeFormat::NonDropFrame | TimecodeFormat::DropFrame => {
                let parts: Vec<&str> = timecode.split(&[':', ';'][..]).collect();
                if parts.len() != 4 {
                    return Err(ExportError::InvalidTimecode(timecode.to_string()).into());
                }

                let hours: u32 = parts[0]
                    .parse()
                    .map_err(|_| ExportError::InvalidTimecode(timecode.to_string()))?;
                let minutes: u32 = parts[1]
                    .parse()
                    .map_err(|_| ExportError::InvalidTimecode(timecode.to_string()))?;
                let seconds: u32 = parts[2]
                    .parse()
                    .map_err(|_| ExportError::InvalidTimecode(timecode.to_string()))?;
                let frame: u32 = parts[3]
                    .parse()
                    .map_err(|_| ExportError::InvalidTimecode(timecode.to_string()))?;

                let fps_float = fps.num as f64 / fps.den as f64;
                let total_seconds = hours as f64 * 3600.0 + minutes as f64 * 60.0 + seconds as f64;
                let total_frames = (total_seconds * fps_float).round() as i64 + frame as i64;

                Ok(total_frames)
            }
        }
    }
}

/// Asset relinking utilities
pub mod relinking {
    use super::*;
    use std::path::{Path, PathBuf};
    use walkdir::WalkDir;

    pub fn relink_assets(
        assets: &mut [AssetInfo],
        search_paths: &[PathBuf],
        strategy: RelinkStrategy,
    ) -> Result<Vec<RelinkResult>> {
        let mut results = Vec::new();

        for asset in assets.iter_mut() {
            let result = match strategy {
                RelinkStrategy::Absolute => RelinkResult::Unchanged,
                RelinkStrategy::Relative => relink_relative(asset)?,
                RelinkStrategy::Heuristic => relink_heuristic(asset, search_paths)?,
                RelinkStrategy::Copy => RelinkResult::Unchanged, // Handled separately
            };
            results.push(result);
        }

        Ok(results)
    }

    fn relink_relative(asset: &mut AssetInfo) -> Result<RelinkResult> {
        if let Some(rel_path) = &asset.relative_path {
            asset.path = rel_path.clone();
            Ok(RelinkResult::Relinked(rel_path.clone()))
        } else {
            Ok(RelinkResult::Failed(
                "No relative path available".to_string(),
            ))
        }
    }

    fn relink_heuristic(asset: &mut AssetInfo, search_paths: &[PathBuf]) -> Result<RelinkResult> {
        let filename = asset
            .path
            .file_name()
            .ok_or_else(|| ExportError::RelinkingFailed("Invalid asset path".to_string()))?;

        // First try exact filename match
        for search_path in search_paths {
            for entry in WalkDir::new(search_path).into_iter().filter_map(|e| e.ok()) {
                if entry.file_name() == filename {
                    let new_path = entry.path().to_path_buf();

                    // Verify it's the same file if we have a hash
                    if let Some(expected_hash) = &asset.hash {
                        if let Ok(actual_hash) = calculate_file_hash(&new_path) {
                            if &actual_hash == expected_hash {
                                asset.path = new_path.clone();
                                return Ok(RelinkResult::Relinked(new_path));
                            }
                        }
                    } else {
                        // No hash available, use first match
                        asset.path = new_path.clone();
                        return Ok(RelinkResult::Relinked(new_path));
                    }
                }
            }
        }

        Ok(RelinkResult::Failed("File not found".to_string()))
    }

    fn calculate_file_hash(path: &Path) -> Result<String> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let mut buffer = [0; 8192];

        loop {
            match file.read(&mut buffer)? {
                0 => break,
                n => {
                    use std::hash::{Hash, Hasher};
                    buffer[..n].hash(&mut hasher);
                }
            }
        }

        use std::hash::Hasher;
        Ok(format!("{:x}", hasher.finish()))
    }

    #[derive(Debug, Clone)]
    pub enum RelinkResult {
        Unchanged,
        Relinked(PathBuf),
        Failed(String),
    }
}

// Custom serialization for ExportFormat
impl Serialize for ExportFormat {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let s = match self {
            ExportFormat::FcpXml1_9 => "fcpxml_1_9",
            ExportFormat::FcpXml1_10 => "fcpxml_1_10",
            ExportFormat::Fcp7Xml => "fcp7xml",
            ExportFormat::Edl => "edl",
            ExportFormat::AvidEdl => "avid_edl",
            ExportFormat::Json => "json",
        };
        serializer.serialize_str(s)
    }
}

impl<'de> Deserialize<'de> for ExportFormat {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "fcpxml_1_9" => Ok(ExportFormat::FcpXml1_9),
            "fcpxml_1_10" => Ok(ExportFormat::FcpXml1_10),
            "fcp7xml" => Ok(ExportFormat::Fcp7Xml),
            "edl" => Ok(ExportFormat::Edl),
            "avid_edl" => Ok(ExportFormat::AvidEdl),
            "json" => Ok(ExportFormat::Json),
            _ => Err(serde::de::Error::unknown_variant(
                &s,
                &[
                    "fcpxml_1_9",
                    "fcpxml_1_10",
                    "fcp7xml",
                    "edl",
                    "avid_edl",
                    "json",
                ],
            )),
        }
    }
}
