use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use thiserror::Error;
mod yuv_decode;
pub use yuv_decode::{best_decoder, decode_yuv_at, VideoDecoder, YuvFrame, YuvPixFmt};

#[derive(Debug, Error)]
pub enum ProbeError {
    #[error("ffprobe not found on PATH; please install FFmpeg (ffprobe)")]
    FfprobeMissing,
    #[error("ffprobe failed: {0}")]
    FfprobeFailed(String),
    #[error("parse error: {0}")]
    Parse(String),
}

#[derive(Debug, Clone, Deserialize)]
struct FfprobeStream {
    codec_type: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    r_frame_rate: Option<String>,
    avg_frame_rate: Option<String>,
    sample_rate: Option<String>,
    channels: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
struct FfprobeFormat {
    duration: Option<String>,
    format_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct FfprobeJson {
    streams: Option<Vec<FfprobeStream>>,
    format: Option<FfprobeFormat>,
}

#[derive(Debug, Clone)]
pub struct MediaInfo {
    pub path: PathBuf,
    pub kind: MediaKind,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps_num: Option<u32>,
    pub fps_den: Option<u32>,
    pub duration_seconds: Option<f64>,
    pub audio_channels: Option<u32>,
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Video,
    Image,
    Audio,
}

fn parse_rate(s: &str) -> Option<(u32, u32)> {
    let s = s.trim();
    if s == "0/0" || s == "0" || s.is_empty() {
        return None;
    }
    if let Some((a, b)) = s.split_once('/') {
        let num = a.parse().ok()?;
        let den = b.parse().ok()?;
        if den == 0 {
            return None;
        }
        return Some((num, den));
    }
    // integer fallback
    let v: u32 = s.parse().ok()?;
    Some((v, 1))
}

pub fn probe_media(path: &Path) -> Result<MediaInfo, ProbeError> {
    let ffprobe = which::which("ffprobe").map_err(|_| ProbeError::FfprobeMissing)?;
    let path_str = path.to_string_lossy().to_string();
    let out = Command::new(ffprobe)
        .arg("-v")
        .arg("error")
        .arg("-show_format")
        .arg("-show_streams")
        .arg("-print_format")
        .arg("json")
        .arg(path_str)
        .output()
        .map_err(|e| ProbeError::FfprobeFailed(e.to_string()))?;
    if !out.status.success() {
        return Err(ProbeError::FfprobeFailed(
            String::from_utf8_lossy(&out.stderr).into(),
        ));
    }
    let parsed: FfprobeJson =
        serde_json::from_slice(&out.stdout).map_err(|e| ProbeError::Parse(e.to_string()))?;

    let mut kind = MediaKind::Video;
    let mut width = None;
    let mut height = None;
    let mut fps = None;
    let mut audio_channels = None;
    let mut sample_rate = None;

    if let Some(streams) = &parsed.streams {
        for s in streams {
            match s.codec_type.as_deref() {
                Some("video") => {
                    kind = MediaKind::Video;
                    width = width.or(s.width);
                    height = height.or(s.height);
                    fps = fps
                        .or_else(|| s.avg_frame_rate.as_deref().and_then(parse_rate))
                        .or_else(|| s.r_frame_rate.as_deref().and_then(parse_rate));
                }
                Some("audio") => {
                    if kind != MediaKind::Video {
                        kind = MediaKind::Audio;
                    }
                    audio_channels = audio_channels.or(s.channels);
                    sample_rate =
                        sample_rate.or(s.sample_rate.as_deref().and_then(|x| x.parse().ok()));
                }
                Some("image") => {
                    if kind != MediaKind::Video {
                        kind = MediaKind::Image;
                    }
                    width = width.or(s.width);
                    height = height.or(s.height);
                }
                _ => {}
            }
        }
    }

    let duration_seconds = parsed
        .format
        .as_ref()
        .and_then(|f| f.duration.as_deref())
        .and_then(|d| d.parse().ok());

    let (fps_num, fps_den) = fps.map(|(n, d)| (Some(n), Some(d))).unwrap_or((None, None));

    Ok(MediaInfo {
        path: path.to_path_buf(),
        kind,
        width,
        height,
        fps_num,
        fps_den,
        duration_seconds,
        audio_channels,
        sample_rate,
    })
}

/// Generate proxy/transcode for media file
pub fn generate_proxy(
    input_path: &Path,
    output_path: &Path,
    width: u32,
    height: u32,
    bitrate_kbps: u32,
) -> Result<(), ProbeError> {
    let ffmpeg = which::which("ffmpeg").map_err(|_| ProbeError::FfprobeMissing)?;

    let output = Command::new(ffmpeg)
        .arg("-i")
        .arg(input_path)
        .arg("-vf")
        .arg(format!("scale={}:{}", width, height))
        .arg("-c:v")
        .arg("libx264")
        .arg("-b:v")
        .arg(format!("{}k", bitrate_kbps))
        .arg("-c:a")
        .arg("aac")
        .arg("-b:a")
        .arg("128k")
        .arg("-preset")
        .arg("fast")
        .arg("-y") // Overwrite output
        .arg(output_path)
        .output()
        .map_err(|e| ProbeError::FfprobeFailed(e.to_string()))?;

    if !output.status.success() {
        return Err(ProbeError::FfprobeFailed(
            String::from_utf8_lossy(&output.stderr).into(),
        ));
    }

    Ok(())
}

/// Generate thumbnail from video at specific time
pub fn generate_thumbnail(
    input_path: &Path,
    output_path: &Path,
    time_seconds: f64,
    width: u32,
    height: u32,
) -> Result<(), ProbeError> {
    let ffmpeg = which::which("ffmpeg").map_err(|_| ProbeError::FfprobeMissing)?;

    let output = Command::new(ffmpeg)
        .arg("-ss")
        .arg(format!("{:.3}", time_seconds))
        .arg("-i")
        .arg(input_path)
        .arg("-vframes")
        .arg("1")
        .arg("-vf")
        .arg(format!("scale={}:{}", width, height))
        .arg("-y") // Overwrite output
        .arg(output_path)
        .output()
        .map_err(|e| ProbeError::FfprobeFailed(e.to_string()))?;

    if !output.status.success() {
        return Err(ProbeError::FfprobeFailed(
            String::from_utf8_lossy(&output.stderr).into(),
        ));
    }

    Ok(())
}

/// Generate audio waveform data
pub fn generate_waveform(input_path: &Path, samples: u32) -> Result<Vec<f32>, ProbeError> {
    let ffmpeg = which::which("ffmpeg").map_err(|_| ProbeError::FfprobeMissing)?;

    let output = Command::new(ffmpeg)
        .arg("-i")
        .arg(input_path)
        .arg("-ac")
        .arg("1") // Mono
        .arg("-ar")
        .arg("8000") // 8kHz sample rate
        .arg("-f")
        .arg("f32le") // 32-bit float little-endian
        .arg("-")
        .output()
        .map_err(|e| ProbeError::FfprobeFailed(e.to_string()))?;

    if !output.status.success() {
        return Err(ProbeError::FfprobeFailed(
            String::from_utf8_lossy(&output.stderr).into(),
        ));
    }

    // Convert raw audio data to waveform samples
    let audio_data = output.stdout;
    let mut waveform = Vec::new();
    let chunk_size = (audio_data.len() / 4) / samples as usize; // 4 bytes per f32

    for chunk_start in (0..audio_data.len()).step_by(chunk_size * 4) {
        let chunk_end = (chunk_start + chunk_size * 4).min(audio_data.len());
        let chunk = &audio_data[chunk_start..chunk_end];

        let mut max_amplitude = 0.0f32;
        for sample_bytes in chunk.chunks_exact(4) {
            if sample_bytes.len() == 4 {
                let sample = f32::from_le_bytes([
                    sample_bytes[0],
                    sample_bytes[1],
                    sample_bytes[2],
                    sample_bytes[3],
                ]);
                max_amplitude = max_amplitude.max(sample.abs());
            }
        }

        waveform.push(max_amplitude);
    }

    Ok(waveform)
}

/// Export presets for different codecs
#[derive(Debug, Clone)]
pub struct ExportPreset {
    pub name: String,
    pub codec: String,
    pub container: String,
    pub video_bitrate: Option<u32>,
    pub audio_bitrate: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<(u32, u32)>,
    pub additional_args: Vec<String>,
}

impl ExportPreset {
    pub fn h264_1080p() -> Self {
        Self {
            name: "H.264 1080p".to_string(),
            codec: "libx264".to_string(),
            container: "mp4".to_string(),
            video_bitrate: Some(8000),
            audio_bitrate: Some(320),
            width: Some(1920),
            height: Some(1080),
            fps: Some((30, 1)),
            additional_args: vec!["-preset".to_string(), "medium".to_string()],
        }
    }

    pub fn h264_720p() -> Self {
        Self {
            name: "H.264 720p".to_string(),
            codec: "libx264".to_string(),
            container: "mp4".to_string(),
            video_bitrate: Some(4000),
            audio_bitrate: Some(320),
            width: Some(1280),
            height: Some(720),
            fps: Some((30, 1)),
            additional_args: vec!["-preset".to_string(), "medium".to_string()],
        }
    }

    pub fn av1_1080p() -> Self {
        Self {
            name: "AV1 1080p".to_string(),
            codec: "libsvtav1".to_string(),
            container: "mp4".to_string(),
            video_bitrate: Some(5000),
            audio_bitrate: Some(320),
            width: Some(1920),
            height: Some(1080),
            fps: Some((30, 1)),
            additional_args: vec!["-crf".to_string(), "28".to_string()],
        }
    }
}

/// Export video with given preset
pub fn export_video(
    input_path: &Path,
    output_path: &Path,
    preset: &ExportPreset,
) -> Result<(), ProbeError> {
    let ffmpeg = which::which("ffmpeg").map_err(|_| ProbeError::FfprobeMissing)?;

    let mut cmd = Command::new(ffmpeg);
    cmd.arg("-i").arg(input_path);

    // Video codec
    cmd.arg("-c:v").arg(&preset.codec);

    // Video bitrate
    if let Some(bitrate) = preset.video_bitrate {
        cmd.arg("-b:v").arg(format!("{}k", bitrate));
    }

    // Resolution
    if let (Some(width), Some(height)) = (preset.width, preset.height) {
        cmd.arg("-vf").arg(format!("scale={}:{}", width, height));
    }

    // Frame rate
    if let Some((num, den)) = preset.fps {
        cmd.arg("-r").arg(format!("{}/{}", num, den));
    }

    // Audio codec and bitrate
    cmd.arg("-c:a").arg("aac");
    if let Some(audio_bitrate) = preset.audio_bitrate {
        cmd.arg("-b:a").arg(format!("{}k", audio_bitrate));
    }

    // Additional arguments
    for arg in &preset.additional_args {
        cmd.arg(arg);
    }

    // Output
    cmd.arg("-y").arg(output_path);

    let output = cmd
        .output()
        .map_err(|e| ProbeError::FfprobeFailed(e.to_string()))?;

    if !output.status.success() {
        return Err(ProbeError::FfprobeFailed(
            String::from_utf8_lossy(&output.stderr).into(),
        ));
    }

    Ok(())
}

/// Get available hardware encoders on the system
pub fn get_hardware_encoders() -> HashMap<String, Vec<String>> {
    let mut encoders = HashMap::new();

    if let Ok(ffmpeg) = which::which("ffmpeg") {
        if let Ok(output) = Command::new(ffmpeg).arg("-encoders").output() {
            let output_str = String::from_utf8_lossy(&output.stdout);

            // Parse available hardware encoders
            let mut h264_encoders = Vec::new();
            let mut hevc_encoders = Vec::new();
            let mut av1_encoders = Vec::new();

            for line in output_str.lines() {
                if line.contains("h264") {
                    if line.contains("videotoolbox") {
                        h264_encoders.push("h264_videotoolbox".to_string());
                    }
                    if line.contains("nvenc") {
                        h264_encoders.push("h264_nvenc".to_string());
                    }
                    if line.contains("qsv") {
                        h264_encoders.push("h264_qsv".to_string());
                    }
                    if line.contains("vaapi") {
                        h264_encoders.push("h264_vaapi".to_string());
                    }
                }

                if line.contains("hevc") || line.contains("h265") {
                    if line.contains("videotoolbox") {
                        hevc_encoders.push("hevc_videotoolbox".to_string());
                    }
                    if line.contains("nvenc") {
                        hevc_encoders.push("hevc_nvenc".to_string());
                    }
                    if line.contains("qsv") {
                        hevc_encoders.push("hevc_qsv".to_string());
                    }
                    if line.contains("vaapi") {
                        hevc_encoders.push("hevc_vaapi".to_string());
                    }
                }

                if line.contains("av1") {
                    if line.contains("nvenc") {
                        av1_encoders.push("av1_nvenc".to_string());
                    }
                    if line.contains("qsv") {
                        av1_encoders.push("av1_qsv".to_string());
                    }
                    if line.contains("vaapi") {
                        av1_encoders.push("av1_vaapi".to_string());
                    }
                }
            }

            if !h264_encoders.is_empty() {
                encoders.insert("H.264".to_string(), h264_encoders);
            }
            if !hevc_encoders.is_empty() {
                encoders.insert("HEVC".to_string(), hevc_encoders);
            }
            if !av1_encoders.is_empty() {
                encoders.insert("AV1".to_string(), av1_encoders);
            }
        }
    }

    encoders
}
