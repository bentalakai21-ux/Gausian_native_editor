use std::path::Path;
use thiserror::Error;
use tracing::debug;

#[cfg(target_os = "macos")]
use native_decoder::{create_decoder, DecoderConfig, YuvPixFmt as NativeYuvPixFmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum YuvPixFmt {
    Nv12,
    P010,
}

#[derive(Debug, Clone)]
pub struct YuvFrame {
    pub fmt: YuvPixFmt,
    pub y: Vec<u8>,
    pub uv: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("backend not available")]
    NotAvailable,
    #[error("decode failed: {0}")]
    Failed(String),
}

pub trait VideoDecoder {
    fn decode_yuv_at(&mut self, path: &Path, t_sec: f64) -> Result<YuvFrame, DecodeError>;
}

pub fn best_decoder() -> Box<dyn VideoDecoder + Send> {
    // Try native decoder first, fallback to FFmpeg
    #[cfg(target_os = "macos")]
    {
        if native_decoder::is_native_decoding_available() {
            debug!("Using native decoder");
            return Box::new(NativeDecoderWrapper::new());
        } else {
            debug!("Native decoder not available, using FFmpeg");
        }
    }
    // Fallback to FFmpeg
    debug!("Using FFmpeg decoder");
    Box::new(FfmpegDecoder {})
}

pub fn decode_yuv_at(path: &Path, t_sec: f64) -> Result<YuvFrame, DecodeError> {
    let mut dec = best_decoder();
    dec.decode_yuv_at(path, t_sec)
}

#[cfg(target_os = "macos")]
struct NativeDecoderWrapper {
    decoder: Option<Box<dyn native_decoder::NativeVideoDecoder + Send>>,
}

#[cfg(target_os = "macos")]
impl NativeDecoderWrapper {
    fn new() -> Self {
        Self { decoder: None }
    }

    fn get_or_create_decoder(
        &mut self,
        path: &Path,
    ) -> Result<&mut Box<dyn native_decoder::NativeVideoDecoder + Send>, DecodeError> {
        if self.decoder.is_none() {
            let config = DecoderConfig {
                hardware_acceleration: true,
                preferred_format: Some(NativeYuvPixFmt::Nv12),
                zero_copy: false,
            };
            self.decoder = Some(create_decoder(path, config).map_err(|e| {
                DecodeError::Failed(format!("Failed to create native decoder: {}", e))
            })?);
        }
        Ok(self.decoder.as_mut().unwrap())
    }
}

#[cfg(target_os = "macos")]
impl VideoDecoder for NativeDecoderWrapper {
    fn decode_yuv_at(&mut self, path: &Path, t_sec: f64) -> Result<YuvFrame, DecodeError> {
        debug!(
            "NativeDecoderWrapper::decode_yuv_at called for path: {:?}, t_sec: {}",
            path, t_sec
        );
        let decoder = self.get_or_create_decoder(path)?;
        let frame = decoder
            .decode_frame(t_sec)
            .map_err(|e| DecodeError::Failed(format!("Native decode failed: {}", e)))?;

        if let Some(frame) = frame {
            // Convert native format to our format
            let fmt = match frame.format {
                NativeYuvPixFmt::Nv12 => YuvPixFmt::Nv12,
                NativeYuvPixFmt::P010 => YuvPixFmt::P010,
            };

            debug!(
                "Successfully decoded frame: {}x{} format: {:?}",
                frame.width, frame.height, fmt
            );
            Ok(YuvFrame {
                fmt,
                y: frame.y_plane,
                uv: frame.uv_plane,
                width: frame.width,
                height: frame.height,
            })
        } else {
            debug!("No frame available from native decoder");
            Err(DecodeError::Failed("No frame available".into()))
        }
    }
}

struct FfmpegDecoder {}

impl VideoDecoder for FfmpegDecoder {
    fn decode_yuv_at(&mut self, path: &Path, t_sec: f64) -> Result<YuvFrame, DecodeError> {
        // Try P010 first
        if let Some(frame) = ffmpeg_decode(path, t_sec, true) {
            return Ok(frame);
        }
        // Fallback to NV12
        if let Some(frame) = ffmpeg_decode(path, t_sec, false) {
            return Ok(frame);
        }
        Err(DecodeError::Failed("ffmpeg decode failed".into()))
    }
}

fn ffmpeg_decode(path: &Path, t_sec: f64, p010: bool) -> Option<YuvFrame> {
    let pixfmt = if p010 { "p010le" } else { "nv12" };
    let info = crate::probe_media(path).ok()?;
    let w = info.width?;
    let h = info.height?;
    let out = std::process::Command::new("ffmpeg")
        .arg("-ss")
        .arg(format!("{:.3}", t_sec.max(0.0)))
        .arg("-i")
        .arg(path)
        .arg("-frames:v")
        .arg("1")
        .arg("-f")
        .arg("rawvideo")
        .arg("-pix_fmt")
        .arg(pixfmt)
        .arg("-threads")
        .arg("1")
        .arg("-")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    if p010 {
        // Expected size: Y plane w*h*2, UV plane w*h (2x16-bit at half res)
        let y_bytes = (w as usize) * (h as usize) * 2;
        let uv_bytes = (w as usize) * (h as usize);
        if out.stdout.len() < y_bytes + uv_bytes {
            return None;
        }
        let y = out.stdout[..y_bytes].to_vec();
        let uv = out.stdout[y_bytes..y_bytes + uv_bytes].to_vec();
        Some(YuvFrame {
            fmt: YuvPixFmt::P010,
            y,
            uv,
            width: w,
            height: h,
        })
    } else {
        // NV12: Y w*h, UV w*h/2
        let expected = (w as usize) * (h as usize) + (w as usize) * (h as usize) / 2;
        if out.stdout.len() < expected {
            return None;
        }
        let y_size = (w as usize) * (h as usize);
        let y = out.stdout[..y_size].to_vec();
        let uv = out.stdout[y_size..y_size + (expected - y_size)].to_vec();
        Some(YuvFrame {
            fmt: YuvPixFmt::Nv12,
            y,
            uv,
            width: w,
            height: h,
        })
    }
}
