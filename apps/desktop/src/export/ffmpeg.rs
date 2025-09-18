use std::{
    io::{BufRead, BufReader},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
};

use crate::timeline_crate::{Item, ItemKind, Sequence};

use super::{ExportCodec, ExportProgress};

#[derive(Clone)]
struct VideoSegment {
    kind: VideoSegKind,
    start_sec: f32,
    duration: f32,
}

#[derive(Clone)]
enum VideoSegKind {
    Video { path: String, start_sec: f32 },
    Image { path: String },
    Black,
}

#[derive(Clone)]
struct AudioClip {
    path: String,
    offset_sec: f32,
    duration: f32,
}

struct ExportTimeline {
    video_segments: Vec<VideoSegment>,
    audio_clips: Vec<AudioClip>,
}

pub(crate) fn run_ffmpeg_timeline(
    out_path: String,
    size: (u32, u32),
    fps: f32,
    codec: ExportCodec,
    selected_encoder: Option<String>,
    crf: i32,
    total_ms: u64,
    seq: Sequence,
    progress: Arc<Mutex<ExportProgress>>,
) {
    let (w, h) = size;
    let timeline = build_export_timeline(&seq);
    let mut args: Vec<String> = Vec::new();
    args.push("-y".into());

    let mut input_index = 0usize;
    let mut video_labels: Vec<String> = Vec::new();
    for seg in &timeline.video_segments {
        match &seg.kind {
            VideoSegKind::Video { path, start_sec } => {
                args.push("-ss".into());
                args.push(format!("{:.3}", start_sec));
                args.push("-t".into());
                args.push(format!("{:.3}", seg.duration));
                args.push("-i".into());
                args.push(path.clone());
            }
            VideoSegKind::Image { path } => {
                args.push("-loop".into());
                args.push("1".into());
                args.push("-t".into());
                args.push(format!("{:.3}", seg.duration));
                args.push("-i".into());
                args.push(path.clone());
            }
            VideoSegKind::Black => {
                args.push("-f".into());
                args.push("lavfi".into());
                args.push("-t".into());
                args.push(format!("{:.3}", seg.duration));
                args.push("-r".into());
                args.push(format!("{}", fps.max(1.0) as i32));
                args.push("-i".into());
                args.push(format!("color=black:s={}x{}", w, h));
            }
        }
        video_labels.push(format!("v{}", input_index));
        input_index += 1;
    }

    let audio_input_start = input_index;
    for clip in &timeline.audio_clips {
        args.push("-i".into());
        args.push(clip.path.clone());
        input_index += 1;
    }

    let mut filters: Vec<String> = Vec::new();
    let mut vouts: Vec<String> = Vec::new();
    for (i, _seg) in timeline.video_segments.iter().enumerate() {
        let label_in = format!("{}:v", i);
        let label_out = format!("v{}o", i);
        filters.push(format!(
            "[{}]scale={}x{}:flags=lanczos,fps={},format=yuv420p[{}]",
            label_in,
            w,
            h,
            fps.max(1.0) as i32,
            label_out
        ));
        vouts.push(format!("[{}]", label_out));
    }
    if !vouts.is_empty() {
        filters.push(format!(
            "{}concat=n={}:v=1:a=0[vout]",
            vouts.join(""),
            vouts.len()
        ));
    }

    let mut aouts: Vec<String> = Vec::new();
    for (j, clip) in timeline.audio_clips.iter().enumerate() {
        let in_idx = audio_input_start + j;
        let label_in = format!("{}:a", in_idx);
        let label_out = format!("a{}o", j);
        let delay_ms = (clip.offset_sec * 1000.0).round() as u64;
        let total_s = total_ms as f32 / 1000.0;
        filters.push(format!(
            "[{}]adelay={}|{},atrim=0:{:.3},aresample=async=1[{}]",
            label_in, delay_ms, delay_ms, total_s, label_out
        ));
        aouts.push(format!("[{}]", label_out));
    }
    let has_audio = !aouts.is_empty();
    if has_audio {
        filters.push(format!(
            "{}amix=inputs={}:normalize=0:duration=longest[aout]",
            aouts.join(""),
            aouts.len()
        ));
    }

    if !filters.is_empty() {
        args.push("-filter_complex".into());
        args.push(filters.join(";"));
    }

    args.push("-map".into());
    args.push("[vout]".into());
    if has_audio {
        args.push("-map".into());
        args.push("[aout]".into());
    } else {
        args.push("-an".into());
    }

    args.push("-pix_fmt".into());
    args.push("yuv420p".into());
    match codec {
        ExportCodec::H264 => {
            let encoder = selected_encoder.unwrap_or_else(|| "libx264".into());
            args.push("-c:v".into());
            args.push(encoder);
            args.push("-crf".into());
            args.push(crf.to_string());
            args.push("-preset".into());
            args.push("medium".into());
            args.push("-movflags".into());
            args.push("+faststart".into());
        }
        ExportCodec::AV1 => {
            let encoder = selected_encoder.unwrap_or_else(|| "libaom-av1".into());
            args.push("-c:v".into());
            args.push(encoder.clone());
            if encoder.starts_with("libaom") {
                args.push("-b:v".into());
                args.push("0".into());
                args.push("-crf".into());
                args.push(crf.to_string());
                args.push("-row-mt".into());
                args.push("1".into());
            } else {
                args.push("-cq".into());
                args.push(crf.to_string());
            }
        }
    }

    args.push("-progress".into());
    args.push("pipe:2".into());
    args.push(out_path.clone());

    let mut cmd = Command::new("ffmpeg");
    cmd.args(args.iter().map(|s| s.as_str()));
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            if let Ok(mut p) = progress.lock() {
                p.done = true;
                p.error = Some(format!("ffmpeg spawn failed: {}", e));
            }
            return;
        }
    };

    if let Some(stderr) = child.stderr.take() {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while let Ok(n) = reader.read_line(&mut line) {
            if n == 0 {
                break;
            }
            if let Some((k, v)) = line.trim().split_once('=') {
                if k == "out_time_ms" {
                    if let Ok(ms) = v.parse::<u64>() {
                        let prog = if total_ms > 0 {
                            (ms as f32 / total_ms as f32).min(1.0)
                        } else {
                            0.0
                        };
                        if let Ok(mut p) = progress.lock() {
                            p.progress = prog;
                        }
                    }
                }
            }
            line.clear();
        }
    }

    let status = child.wait().ok();
    if let Ok(mut p) = progress.lock() {
        p.done = true;
        if let Some(st) = status {
            if !st.success() {
                p.error = Some(format!("ffmpeg failed: {:?}", st.code()));
            }
        }
    }
}

fn build_export_timeline(seq: &Sequence) -> ExportTimeline {
    let mut points: Vec<i64> = vec![0, seq.duration_in_frames];
    for track in seq.tracks.iter() {
        for it in &track.items {
            if !matches!(it.kind, ItemKind::Audio { .. }) {
                points.push(it.from);
                points.push(it.from + it.duration_in_frames);
            }
        }
    }
    points.sort_unstable();
    points.dedup();

    let fps = seq.fps.num.max(1) as f32 / seq.fps.den.max(1) as f32;
    let mut video_segments: Vec<VideoSegment> = Vec::new();
    for w in points.windows(2) {
        let a = w[0];
        let b = w[1];
        if b <= a {
            continue;
        }
        let (item_opt, _ti) = topmost_item_covering(seq, a);
        let kind = if let Some(item) = item_opt {
            match &item.kind {
                ItemKind::Video { src, .. } => {
                    let start_into = (a - item.from).max(0) as f32 / fps;
                    VideoSegKind::Video {
                        path: src.clone(),
                        start_sec: start_into,
                    }
                }
                ItemKind::Image { src } => VideoSegKind::Image { path: src.clone() },
                _ => VideoSegKind::Black,
            }
        } else {
            VideoSegKind::Black
        };
        let seg = VideoSegment {
            kind,
            start_sec: a as f32 / fps,
            duration: (b - a) as f32 / fps,
        };
        video_segments.push(seg);
    }

    let mut audio_clips: Vec<AudioClip> = Vec::new();
    for track in &seq.tracks {
        for it in &track.items {
            if let ItemKind::Audio { src, .. } = &it.kind {
                audio_clips.push(AudioClip {
                    path: src.clone(),
                    offset_sec: it.from as f32 / fps,
                    duration: it.duration_in_frames as f32 / fps,
                });
            }
        }
    }

    ExportTimeline {
        video_segments,
        audio_clips,
    }
}

fn topmost_item_covering<'a>(seq: &'a Sequence, frame: i64) -> (Option<&'a Item>, Option<usize>) {
    for (ti, track) in seq.tracks.iter().enumerate().rev() {
        for it in &track.items {
            if frame >= it.from && frame < it.from + it.duration_in_frames {
                if !matches!(it.kind, ItemKind::Audio { .. }) {
                    return (Some(it), Some(ti));
                }
            }
        }
    }
    (None, None)
}

#[allow(dead_code)]
fn detect_hw_encoder<const N: usize>(candidates: [&str; N]) -> Option<String> {
    let out = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-encoders")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    for cand in candidates {
        if s.contains(cand) {
            return Some(cand.to_string());
        }
    }
    None
}
