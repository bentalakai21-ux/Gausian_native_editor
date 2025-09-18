use crate::{AssetInfo, ColorSpace, ExportConfig, ExportError, TimecodeFormat};
use anyhow::Result;
use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::path::Path;
use timeline::{Fps, Item, ItemKind, Sequence, Track};
use uuid::Uuid;

/// Export sequence to FCPXML format
pub fn export_fcpxml(
    sequence: &Sequence,
    assets: &[AssetInfo],
    config: &ExportConfig,
) -> Result<()> {
    let fcpxml = FcpXml::from_sequence(sequence, assets, config)?;
    let xml_content = fcpxml.to_xml()?;
    std::fs::write(&config.output_path, xml_content)?;
    Ok(())
}

/// Import sequence from FCPXML file
pub fn import_fcpxml(path: &Path, config: &ExportConfig) -> Result<(Sequence, Vec<AssetInfo>)> {
    let content = std::fs::read_to_string(path)?;
    let fcpxml = FcpXml::from_xml(&content)?;
    fcpxml.to_sequence(config)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FcpXml {
    version: String,
    project: FcpProject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FcpProject {
    name: String,
    uid: String,
    sequence: FcpSequence,
    events: Vec<FcpEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FcpSequence {
    name: String,
    uid: String,
    duration: String,
    format: FcpFormat,
    spine: FcpSpine,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FcpFormat {
    id: String,
    name: String,
    frameDuration: String,
    width: u32,
    height: u32,
    colorSpace: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FcpSpine {
    clips: Vec<FcpClip>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FcpClip {
    name: String,
    uid: String,
    duration: String,
    start: String,
    offset: String,
    ref_id: String,
    format: Option<String>,
    audio_subitems: Vec<FcpAudioSubitem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FcpAudioSubitem {
    lane: i32,
    offset: String,
    duration: String,
    ref_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FcpEvent {
    name: String,
    uid: String,
    resources: FcpResources,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FcpResources {
    assets: Vec<FcpAsset>,
    media: Vec<FcpMedia>,
    formats: Vec<FcpFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FcpAsset {
    id: String,
    name: String,
    uid: String,
    src: String,
    start: String,
    duration: String,
    hasVideo: bool,
    hasAudio: bool,
    format: Option<String>,
    audioSources: Option<u32>,
    audioChannels: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FcpMedia {
    id: String,
    name: String,
    uid: String,
    projectRef: String,
}

impl FcpXml {
    fn from_sequence(
        sequence: &Sequence,
        assets: &[AssetInfo],
        config: &ExportConfig,
    ) -> Result<Self> {
        let project_uid = Uuid::new_v4().to_string();
        let sequence_uid = Uuid::new_v4().to_string();
        let event_uid = Uuid::new_v4().to_string();

        // Convert frame duration to rational string
        let frame_duration = format!("{}s", sequence.fps.den as f64 / sequence.fps.num as f64);
        let total_duration = format!(
            "{}s",
            (sequence.duration_in_frames * sequence.fps.den as i64) as f64
                / sequence.fps.num as f64
        );

        // Create format
        let format = FcpFormat {
            id: "r1".to_string(),
            name: format!("{}p{}", sequence.height, sequence.fps.num),
            frameDuration: frame_duration.clone(),
            width: sequence.width,
            height: sequence.height,
            colorSpace: match config.color_space {
                ColorSpace::Rec709 => "1-1-1 (Rec. 709)".to_string(),
                ColorSpace::Rec2020 => "9-18-9 (Rec. 2020 HLG)".to_string(),
                ColorSpace::DciP3 => "5-4-6 (DCI-P3 D65)".to_string(),
                ColorSpace::AdobeRgb => "1-1-1 (Rec. 709)".to_string(), // Fallback
            },
        };

        // Convert timeline items to FCPXML clips
        let mut clips = Vec::new();
        let mut fcp_assets = Vec::new();

        for track in &sequence.tracks {
            for item in &track.items {
                let clip_uid = Uuid::new_v4().to_string();
                let asset_uid = Uuid::new_v4().to_string();

                // Find matching asset info
                let asset_info = assets.iter().find(|a| match &item.kind {
                    ItemKind::Video { src, .. }
                    | ItemKind::Audio { src, .. }
                    | ItemKind::Image { src } => a.path.to_string_lossy() == *src,
                    _ => false,
                });

                let clip_duration = format!(
                    "{}s",
                    (item.duration_in_frames * sequence.fps.den as i64) as f64
                        / sequence.fps.num as f64
                );
                let clip_start = format!(
                    "{}s",
                    (item.from * sequence.fps.den as i64) as f64 / sequence.fps.num as f64
                );

                match &item.kind {
                    ItemKind::Video {
                        src, frame_rate: _, ..
                    } => {
                        let asset_name = Path::new(src)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("Unknown")
                            .to_string();

                        let asset = FcpAsset {
                            id: asset_uid.clone(),
                            name: asset_name.clone(),
                            uid: asset_uid.clone(),
                            src: src.clone(),
                            start: "0s".to_string(),
                            duration: clip_duration.clone(),
                            hasVideo: true,
                            hasAudio: matches!(item.kind, ItemKind::Video { .. }),
                            format: Some("r1".to_string()),
                            audioSources: asset_info.and_then(|a| a.audio_channels),
                            audioChannels: asset_info.and_then(|a| a.audio_channels),
                        };
                        fcp_assets.push(asset);

                        let clip = FcpClip {
                            name: asset_name,
                            uid: clip_uid,
                            duration: clip_duration,
                            start: clip_start,
                            offset: "0s".to_string(),
                            ref_id: asset_uid,
                            format: Some("r1".to_string()),
                            audio_subitems: Vec::new(),
                        };
                        clips.push(clip);
                    }
                    ItemKind::Image { src } => {
                        let asset_name = Path::new(src)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("Unknown")
                            .to_string();

                        let asset = FcpAsset {
                            id: asset_uid.clone(),
                            name: asset_name.clone(),
                            uid: asset_uid.clone(),
                            src: src.clone(),
                            start: "0s".to_string(),
                            duration: clip_duration.clone(),
                            hasVideo: true,
                            hasAudio: false,
                            format: Some("r1".to_string()),
                            audioSources: asset_info.and_then(|a| a.audio_channels),
                            audioChannels: asset_info.and_then(|a| a.audio_channels),
                        };
                        fcp_assets.push(asset);

                        let clip = FcpClip {
                            name: asset_name,
                            uid: clip_uid,
                            duration: clip_duration,
                            start: clip_start,
                            offset: "0s".to_string(),
                            ref_id: asset_uid,
                            format: Some("r1".to_string()),
                            audio_subitems: Vec::new(),
                        };
                        clips.push(clip);
                    }
                    ItemKind::Audio { src, .. } => {
                        let asset_name = Path::new(src)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("Unknown")
                            .to_string();

                        let asset = FcpAsset {
                            id: asset_uid.clone(),
                            name: asset_name.clone(),
                            uid: asset_uid.clone(),
                            src: src.clone(),
                            start: "0s".to_string(),
                            duration: clip_duration.clone(),
                            hasVideo: false,
                            hasAudio: true,
                            format: None,
                            audioSources: asset_info.and_then(|a| a.audio_channels),
                            audioChannels: asset_info.and_then(|a| a.audio_channels),
                        };
                        fcp_assets.push(asset);

                        let clip = FcpClip {
                            name: asset_name,
                            uid: clip_uid,
                            duration: clip_duration,
                            start: clip_start,
                            offset: "0s".to_string(),
                            ref_id: asset_uid,
                            format: None,
                            audio_subitems: Vec::new(),
                        };
                        clips.push(clip);
                    }
                    _ => {
                        // Handle text, solid color, etc.
                        continue;
                    }
                }
            }
        }

        let spine = FcpSpine { clips };

        let fcp_sequence = FcpSequence {
            name: sequence.name.clone(),
            uid: sequence_uid,
            duration: total_duration,
            format: format.clone(),
            spine,
        };

        let resources = FcpResources {
            assets: fcp_assets,
            media: Vec::new(),
            formats: vec![format],
        };

        let event = FcpEvent {
            name: config.project_name.clone(),
            uid: event_uid,
            resources,
        };

        let project = FcpProject {
            name: config.project_name.clone(),
            uid: project_uid,
            sequence: fcp_sequence,
            events: vec![event],
        };

        Ok(FcpXml {
            version: match config.format {
                crate::ExportFormat::FcpXml1_9 => "1.9".to_string(),
                crate::ExportFormat::FcpXml1_10 => "1.10".to_string(),
                _ => "1.10".to_string(),
            },
            project,
        })
    }

    fn to_xml(&self) -> Result<String> {
        let mut buffer = Vec::new();
        let mut writer = Writer::new(Cursor::new(&mut buffer));

        // Write XML declaration
        writer.write_event(Event::Decl(quick_xml::events::BytesDecl::new(
            "1.0",
            Some("UTF-8"),
            None,
        )))?;

        // Write root fcpxml element
        let mut fcpxml_elem = BytesStart::new("fcpxml");
        fcpxml_elem.push_attribute(("version", self.version.as_str()));
        writer.write_event(Event::Start(fcpxml_elem))?;

        // Write project
        let mut project_elem = BytesStart::new("project");
        project_elem.push_attribute(("name", self.project.name.as_str()));
        project_elem.push_attribute(("uid", self.project.uid.as_str()));
        writer.write_event(Event::Start(project_elem))?;

        // Write sequence
        self.write_sequence(&mut writer)?;

        // Write events
        for event in &self.project.events {
            self.write_event(&mut writer, event)?;
        }

        writer.write_event(Event::End(BytesEnd::new("project")))?;
        writer.write_event(Event::End(BytesEnd::new("fcpxml")))?;

        Ok(String::from_utf8(buffer)?)
    }

    fn write_sequence(&self, writer: &mut Writer<Cursor<&mut Vec<u8>>>) -> Result<()> {
        let seq = &self.project.sequence;

        let mut seq_elem = BytesStart::new("sequence");
        seq_elem.push_attribute(("name", seq.name.as_str()));
        seq_elem.push_attribute(("uid", seq.uid.as_str()));
        seq_elem.push_attribute(("duration", seq.duration.as_str()));
        seq_elem.push_attribute(("format", seq.format.id.as_str()));
        writer.write_event(Event::Start(seq_elem))?;

        // Write spine
        writer.write_event(Event::Start(BytesStart::new("spine")))?;

        for clip in &seq.spine.clips {
            self.write_clip(writer, clip)?;
        }

        writer.write_event(Event::End(BytesEnd::new("spine")))?;
        writer.write_event(Event::End(BytesEnd::new("sequence")))?;

        Ok(())
    }

    fn write_clip(&self, writer: &mut Writer<Cursor<&mut Vec<u8>>>, clip: &FcpClip) -> Result<()> {
        let mut clip_elem = BytesStart::new("clip");
        clip_elem.push_attribute(("name", clip.name.as_str()));
        clip_elem.push_attribute(("uid", clip.uid.as_str()));
        clip_elem.push_attribute(("duration", clip.duration.as_str()));
        clip_elem.push_attribute(("start", clip.start.as_str()));
        clip_elem.push_attribute(("offset", clip.offset.as_str()));
        clip_elem.push_attribute(("ref", clip.ref_id.as_str()));

        if let Some(format) = &clip.format {
            clip_elem.push_attribute(("format", format.as_str()));
        }

        writer.write_event(Event::Start(clip_elem))?;
        writer.write_event(Event::End(BytesEnd::new("clip")))?;

        Ok(())
    }

    fn write_event(
        &self,
        writer: &mut Writer<Cursor<&mut Vec<u8>>>,
        event: &FcpEvent,
    ) -> Result<()> {
        let mut event_elem = BytesStart::new("event");
        event_elem.push_attribute(("name", event.name.as_str()));
        event_elem.push_attribute(("uid", event.uid.as_str()));
        writer.write_event(Event::Start(event_elem))?;

        // Write resources
        writer.write_event(Event::Start(BytesStart::new("resources")))?;

        // Write formats
        for format in &event.resources.formats {
            let mut format_elem = BytesStart::new("format");
            format_elem.push_attribute(("id", format.id.as_str()));
            format_elem.push_attribute(("name", format.name.as_str()));
            format_elem.push_attribute(("frameDuration", format.frameDuration.as_str()));
            format_elem.push_attribute(("width", format.width.to_string().as_str()));
            format_elem.push_attribute(("height", format.height.to_string().as_str()));
            format_elem.push_attribute(("colorSpace", format.colorSpace.as_str()));
            writer.write_event(Event::Empty(format_elem))?;
        }

        // Write assets
        for asset in &event.resources.assets {
            let mut asset_elem = BytesStart::new("asset");
            asset_elem.push_attribute(("id", asset.id.as_str()));
            asset_elem.push_attribute(("name", asset.name.as_str()));
            asset_elem.push_attribute(("uid", asset.uid.as_str()));
            asset_elem.push_attribute(("src", asset.src.as_str()));
            asset_elem.push_attribute(("start", asset.start.as_str()));
            asset_elem.push_attribute(("duration", asset.duration.as_str()));
            asset_elem.push_attribute(("hasVideo", if asset.hasVideo { "1" } else { "0" }));
            asset_elem.push_attribute(("hasAudio", if asset.hasAudio { "1" } else { "0" }));

            if let Some(format) = &asset.format {
                asset_elem.push_attribute(("format", format.as_str()));
            }

            if let Some(channels) = asset.audioChannels {
                asset_elem.push_attribute(("audioChannels", channels.to_string().as_str()));
            }

            writer.write_event(Event::Empty(asset_elem))?;
        }

        writer.write_event(Event::End(BytesEnd::new("resources")))?;
        writer.write_event(Event::End(BytesEnd::new("event")))?;

        Ok(())
    }

    fn from_xml(xml_content: &str) -> Result<Self> {
        // This is a simplified XML parser - in a real implementation,
        // you would use a proper XML parsing library with full FCPXML support
        let mut reader = Reader::from_str(xml_content);
        reader.config_mut().trim_text(true);

        // For now, return a minimal structure
        // A full implementation would parse the entire FCPXML structure
        Ok(FcpXml {
            version: "1.10".to_string(),
            project: FcpProject {
                name: "Imported Project".to_string(),
                uid: Uuid::new_v4().to_string(),
                sequence: FcpSequence {
                    name: "Imported Sequence".to_string(),
                    uid: Uuid::new_v4().to_string(),
                    duration: "0s".to_string(),
                    format: FcpFormat {
                        id: "r1".to_string(),
                        name: "1080p30".to_string(),
                        frameDuration: "1/30s".to_string(),
                        width: 1920,
                        height: 1080,
                        colorSpace: "1-1-1 (Rec. 709)".to_string(),
                    },
                    spine: FcpSpine { clips: Vec::new() },
                },
                events: Vec::new(),
            },
        })
    }

    fn to_sequence(&self, config: &ExportConfig) -> Result<(Sequence, Vec<AssetInfo>)> {
        let fps = Fps::new(30, 1); // Parse from format.frameDuration
        let mut sequence = Sequence::new(
            &self.project.sequence.name,
            self.project.sequence.format.width,
            self.project.sequence.format.height,
            fps,
            0, // Will be calculated from clips
        );

        let mut assets = Vec::new();
        let mut _tracks: Vec<Track> = Vec::new();
        let mut video_track = Track {
            name: "V1".to_string(),
            items: Vec::new(),
        };
        let mut audio_track = Track {
            name: "A1".to_string(),
            items: Vec::new(),
        };

        // Convert FCPXML clips back to timeline items
        for clip in &self.project.sequence.spine.clips {
            // Find the corresponding asset
            let asset = self
                .project
                .events
                .iter()
                .flat_map(|e| &e.resources.assets)
                .find(|a| a.id == clip.ref_id);

            if let Some(asset) = asset {
                let asset_info = AssetInfo {
                    id: asset.id.clone(),
                    path: std::path::PathBuf::from(&asset.src),
                    relative_path: None,
                    kind: if asset.hasVideo {
                        crate::AssetKind::Video
                    } else if asset.hasAudio {
                        crate::AssetKind::Audio
                    } else {
                        crate::AssetKind::Image
                    },
                    width: Some(self.project.sequence.format.width),
                    height: Some(self.project.sequence.format.height),
                    duration_frames: None, // Parse from duration
                    fps: Some(fps),
                    audio_channels: asset.audioChannels,
                    sample_rate: None,
                    timecode: None,
                    color_space: Some(config.color_space),
                    file_size: None,
                    hash: None,
                };
                assets.push(asset_info);

                let item_kind = if asset.hasVideo {
                    ItemKind::Video {
                        src: asset.src.clone(),
                        frame_rate: Some(fps.num as f32 / fps.den as f32),
                        in_offset_sec: 0.0,
                        rate: 1.0,
                    }
                } else if asset.hasAudio {
                    ItemKind::Audio {
                        src: asset.src.clone(),
                        in_offset_sec: 0.0,
                        rate: 1.0,
                    }
                } else {
                    ItemKind::Image {
                        src: asset.src.clone(),
                    }
                };

                let item = Item {
                    id: clip.uid.clone(),
                    from: 0,                 // Parse from clip.start
                    duration_in_frames: 150, // Parse from clip.duration
                    kind: item_kind,
                };

                if asset.hasVideo {
                    video_track.items.push(item);
                } else {
                    audio_track.items.push(item);
                }
            }
        }

        if !video_track.items.is_empty() {
            sequence.add_track(video_track);
        }
        if !audio_track.items.is_empty() {
            sequence.add_track(audio_track);
        }

        Ok((sequence, assets))
    }
}
