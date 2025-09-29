use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
};

use eframe::egui::{self, Widget};
use project::ProjectDb;

use crate::timeline_crate::Sequence;

use super::{ffmpeg, ExportCodec, ExportPreset, ExportProgress};

pub struct ExportUiState {
    pub(crate) open: bool,
    codec: ExportCodec,
    preset: ExportPreset,
    crf: i32,
    output_path: String,
    running: bool,
    progress: f32,
    status: String,
    progress_shared: Option<Arc<Mutex<ExportProgress>>>,
    worker: Option<JoinHandle<()>>,
    encoders_h264: Vec<String>,
    encoders_av1: Vec<String>,
    selected_encoder: Option<String>,
}

impl Default for ExportUiState {
    fn default() -> Self {
        Self {
            open: false,
            codec: ExportCodec::H264,
            preset: ExportPreset::Source,
            crf: 23,
            output_path: String::new(),
            running: false,
            progress: 0.0,
            status: String::new(),
            progress_shared: None,
            worker: None,
            encoders_h264: Vec::new(),
            encoders_av1: Vec::new(),
            selected_encoder: None,
        }
    }
}

impl ExportUiState {
    pub(crate) fn ui(
        &mut self,
        ctx: &egui::Context,
        frame: &eframe::Frame,
        seq: &Sequence,
        db: &ProjectDb,
        project_id: &str,
    ) {
        if !self.open {
            return;
        }
        let mut keep_open = true;
        egui::Window::new("Export")
            .open(&mut keep_open)
            .resizable(true)
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    if self.encoders_h264.is_empty() && self.encoders_av1.is_empty() {
                        let map = media_io::get_hardware_encoders();
                        if let Some(v) = map.get("h264") {
                            self.encoders_h264 = v.clone();
                        }
                        if let Some(v) = map.get("av1") {
                            self.encoders_av1 = v.clone();
                        }
                        if !self.encoders_h264.iter().any(|e| e == "libx264") {
                            self.encoders_h264.insert(0, "libx264".into());
                        }
                        if !self.encoders_av1.iter().any(|e| e == "libaom-av1") {
                            self.encoders_av1.insert(0, "libaom-av1".into());
                        }
                    }

                    ui.horizontal(|ui| {
                        ui.label("Output:");
                        ui.text_edit_singleline(&mut self.output_path);
                        if ui.button("Browse").clicked() {
                            let default_name = match self.codec {
                                ExportCodec::H264 => "export.mp4",
                                ExportCodec::AV1 => "export.mkv",
                            };
                            let dialog = rfd::FileDialog::new()
                                .set_parent(frame)
                                .set_file_name(default_name);
                            if let Some(path) = dialog.save_file() {
                                self.output_path = path.display().to_string();
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Codec:");
                        let mut codec_idx = match self.codec {
                            ExportCodec::H264 => 0,
                            ExportCodec::AV1 => 1,
                        };
                        egui::ComboBox::from_id_salt("codec_combo")
                            .selected_text(match self.codec {
                                ExportCodec::H264 => "H.264",
                                ExportCodec::AV1 => "AV1",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut codec_idx, 0, "H.264");
                                ui.selectable_value(&mut codec_idx, 1, "AV1");
                            });
                        let prev_codec = self.codec;
                        self.codec = if codec_idx == 0 {
                            ExportCodec::H264
                        } else {
                            ExportCodec::AV1
                        };
                        if self.codec != prev_codec && !self.output_path.is_empty() {
                            self.output_path = adjust_extension(
                                &self.output_path,
                                match self.codec {
                                    ExportCodec::H264 => "mp4",
                                    ExportCodec::AV1 => "mkv",
                                },
                            );
                        }

                        ui.label("Encoder:");
                        let list = match self.codec {
                            ExportCodec::H264 => &mut self.encoders_h264,
                            ExportCodec::AV1 => &mut self.encoders_av1,
                        };
                        if list.is_empty() {
                            list.push(match self.codec {
                                ExportCodec::H264 => "libx264".into(),
                                ExportCodec::AV1 => "libaom-av1".into(),
                            });
                        }
                        let mut selection = self
                            .selected_encoder
                            .clone()
                            .unwrap_or_else(|| list[0].clone());
                        egui::ComboBox::from_id_salt("encoder_combo")
                            .selected_text(selection.clone())
                            .show_ui(ui, |ui| {
                                for enc in list.iter() {
                                    ui.selectable_value(&mut selection, enc.clone(), enc);
                                }
                            });
                        self.selected_encoder = Some(selection);
                    });

                    ui.horizontal(|ui| {
                        ui.label("Preset:");
                        let mut preset_idx = match self.preset {
                            ExportPreset::Source => 0,
                            ExportPreset::P1080 => 1,
                            ExportPreset::P4K => 2,
                        };
                        egui::ComboBox::from_id_salt("preset_combo")
                            .selected_text(match self.preset {
                                ExportPreset::Source => "Source",
                                ExportPreset::P1080 => "1080p",
                                ExportPreset::P4K => "4K",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut preset_idx, 0, "Source");
                                ui.selectable_value(&mut preset_idx, 1, "1080p");
                                ui.selectable_value(&mut preset_idx, 2, "4K");
                            });
                        self.preset = match preset_idx {
                            1 => ExportPreset::P1080,
                            2 => ExportPreset::P4K,
                            _ => ExportPreset::Source,
                        };

                        ui.label("CRF:");
                        let crf_range = if matches!(self.codec, ExportCodec::H264) {
                            12..=32
                        } else {
                            20..=50
                        };
                        ui.add(egui::Slider::new(&mut self.crf, crf_range));
                    });

                    let (src_path, total_ms) =
                        default_export_source_and_duration(db, project_id, seq);
                    ui.label(format!(
                        "Input: {}",
                        src_path.as_deref().unwrap_or("<none>")
                    ));
                    ui.label(format!("Duration: {:.2}s", total_ms as f32 / 1000.0));

                    ui.separator();
                    if !self.running {
                        let can_start = src_path.is_some() && !self.output_path.trim().is_empty();
                        if ui
                            .add_enabled(can_start, egui::Button::new("Start Export"))
                            .clicked()
                        {
                            if src_path.is_some() {
                                let fps = seq.fps.num.max(1) as f32 / seq.fps.den.max(1) as f32;
                                let (w, h) = match self.preset {
                                    ExportPreset::Source => (seq.width, seq.height),
                                    ExportPreset::P1080 => (1920, 1080),
                                    ExportPreset::P4K => (3840, 2160),
                                };
                                let codec = self.codec;
                                if !self.output_path.is_empty() {
                                    self.output_path = adjust_extension(
                                        &self.output_path,
                                        match codec {
                                            ExportCodec::H264 => "mp4",
                                            ExportCodec::AV1 => "mkv",
                                        },
                                    );
                                }
                                let crf = self.crf;
                                let out_path = self.output_path.clone();
                                let progress = Arc::new(Mutex::new(ExportProgress::default()));
                                self.progress_shared = Some(progress.clone());
                                self.running = true;
                                self.status.clear();
                                let selected_encoder = self.selected_encoder.clone();
                                let seq_owned = seq.clone();

                                self.worker = Some(thread::spawn(move || {
                                    ffmpeg::run_ffmpeg_timeline(
                                        out_path,
                                        (w, h),
                                        fps,
                                        codec,
                                        selected_encoder,
                                        crf,
                                        total_ms as u64,
                                        seq_owned,
                                        progress,
                                    );
                                }));
                            }
                        }
                    } else {
                        if let Some(p) = &self.progress_shared {
                            if let Ok(p) = p.lock() {
                                self.progress = p.progress;
                                if let Some(eta) = &p.eta {
                                    self.status = format!("ETA: {}", eta);
                                }
                                if p.done {
                                    self.running = false;
                                    self.status =
                                        p.error.clone().unwrap_or_else(|| "Done".to_string());
                                }
                            }
                        }
                        ui.add(egui::ProgressBar::new(self.progress).show_percentage());
                        ui.label(&self.status);
                    }
                });
            });
        if !keep_open {
            self.open = false;
        }
    }
}

fn default_export_source_and_duration(
    db: &ProjectDb,
    project_id: &str,
    seq: &Sequence,
) -> (Option<String>, u64) {
    let assets = db.list_assets(project_id).unwrap_or_default();
    let src = assets
        .into_iter()
        .find(|a| a.kind.eq_ignore_ascii_case("video"))
        .map(|a| a.src_abs);
    let fps = seq.fps.num.max(1) as f32 / seq.fps.den.max(1) as f32;
    let total_ms = ((seq.duration_in_frames as f32 / fps) * 1000.0) as u64;
    (src, total_ms)
}

fn adjust_extension(path: &str, ext: &str) -> String {
    let mut p = PathBuf::from(path);
    p.set_extension(ext);
    p.display().to_string()
}
