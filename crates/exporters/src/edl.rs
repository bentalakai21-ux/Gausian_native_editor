use crate::{timecode, AssetInfo, ExportConfig, ExportError};
use anyhow::Result;
use std::path::Path;
use timeline::{ItemKind, Sequence};

/// Export sequence to EDL format
pub fn export_edl(sequence: &Sequence, assets: &[AssetInfo], config: &ExportConfig) -> Result<()> {
    let edl_content = generate_edl(sequence, assets, config)?;
    std::fs::write(&config.output_path, edl_content)?;
    Ok(())
}

/// Import sequence from EDL file
pub fn import_edl(path: &Path, config: &ExportConfig) -> Result<(Sequence, Vec<AssetInfo>)> {
    let content = std::fs::read_to_string(path)?;
    parse_edl(&content, config)
}

fn generate_edl(
    sequence: &Sequence,
    assets: &[AssetInfo],
    config: &ExportConfig,
) -> Result<String> {
    let mut edl = String::new();

    // EDL header
    edl.push_str(&format!("TITLE: {}\n", sequence.name));
    edl.push_str(&format!("FCM: NON-DROP FRAME\n\n"));

    let mut edit_number = 1;
    let mut timeline_position = 0i64;

    // Process each track
    for (track_index, track) in sequence.tracks.iter().enumerate() {
        for item in &track.items {
            // Skip non-video items for basic EDL
            if !matches!(item.kind, ItemKind::Video { .. }) {
                continue;
            }

            let source_name = match &item.kind {
                ItemKind::Video { src, .. } => Path::new(src)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("UNKNOWN")
                    .to_uppercase(),
                _ => continue,
            };

            // Calculate timecodes
            let source_in = timecode::frames_to_timecode(0, sequence.fps, config.timecode_format);
            let source_out = timecode::frames_to_timecode(
                item.duration_in_frames,
                sequence.fps,
                config.timecode_format,
            );
            let record_in =
                timecode::frames_to_timecode(item.from, sequence.fps, config.timecode_format);
            let record_out = timecode::frames_to_timecode(
                item.from + item.duration_in_frames,
                sequence.fps,
                config.timecode_format,
            );

            // EDL line format: EDIT# SOURCE TRACK TRANSITION SOURCE_IN SOURCE_OUT RECORD_IN RECORD_OUT
            let track_type = if track_index == 0 { "V" } else { "A" };
            edl.push_str(&format!(
                "{:03} {} {} C {} {} {} {}\n",
                edit_number, source_name, track_type, source_in, source_out, record_in, record_out
            ));

            edit_number += 1;
        }
    }

    Ok(edl)
}

fn parse_edl(content: &str, config: &ExportConfig) -> Result<(Sequence, Vec<AssetInfo>)> {
    let mut sequence = Sequence::new("Imported EDL", 1920, 1080, timeline::Fps::new(30, 1), 0);
    let mut assets = Vec::new();

    let mut video_track = timeline::Track {
        name: "V1".to_string(),
        items: Vec::new(),
    };
    let mut audio_track = timeline::Track {
        name: "A1".to_string(),
        items: Vec::new(),
    };

    for line in content.lines() {
        let line = line.trim();

        // Skip comments and empty lines
        if line.is_empty() || line.starts_with("TITLE:") || line.starts_with("FCM:") {
            continue;
        }

        // Parse EDL line
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 8 {
            let source_name = parts[1];
            let track_type = parts[2];
            let source_in = parts[4];
            let source_out = parts[5];
            let record_in = parts[6];
            let record_out = parts[7];

            // Convert timecodes to frames
            let from =
                timecode::timecode_to_frames(record_in, sequence.fps, config.timecode_format)?;
            let to =
                timecode::timecode_to_frames(record_out, sequence.fps, config.timecode_format)?;
            let duration = to - from;

            // Create item
            let item_id = uuid::Uuid::new_v4().to_string();
            let src_path = format!("{}.mov", source_name.to_lowercase()); // Assume .mov extension

            let item = timeline::Item {
                id: item_id,
                from,
                duration_in_frames: duration,
                kind: if track_type.starts_with('V') {
                    ItemKind::Video {
                        src: src_path.clone(),
                        frame_rate: Some(sequence.fps.num as f32 / sequence.fps.den as f32),
                        in_offset_sec: 0.0,
                        rate: 1.0,
                    }
                } else {
                    ItemKind::Audio {
                        src: src_path.clone(),
                        in_offset_sec: 0.0,
                        rate: 1.0,
                    }
                },
            };

            // Add to appropriate track
            if track_type.starts_with('V') {
                video_track.items.push(item);
            } else {
                audio_track.items.push(item);
            }

            // Create asset info
            let asset_info = AssetInfo {
                id: uuid::Uuid::new_v4().to_string(),
                path: std::path::PathBuf::from(&src_path),
                relative_path: None,
                kind: if track_type.starts_with('V') {
                    crate::AssetKind::Video
                } else {
                    crate::AssetKind::Audio
                },
                width: Some(1920),
                height: Some(1080),
                duration_frames: Some(duration),
                fps: Some(sequence.fps),
                audio_channels: if track_type.starts_with('A') {
                    Some(2)
                } else {
                    None
                },
                sample_rate: if track_type.starts_with('A') {
                    Some(48000)
                } else {
                    None
                },
                timecode: None,
                color_space: Some(config.color_space),
                file_size: None,
                hash: None,
            };
            assets.push(asset_info);
        }
    }

    if !video_track.items.is_empty() {
        sequence.add_track(video_track);
    }
    if !audio_track.items.is_empty() {
        sequence.add_track(audio_track);
    }

    // Update sequence duration
    let max_end = sequence
        .tracks
        .iter()
        .flat_map(|t| t.items.iter())
        .map(|i| i.from + i.duration_in_frames)
        .max()
        .unwrap_or(0);
    sequence.duration_in_frames = max_end;

    Ok((sequence, assets))
}
