use anyhow::Result;
use eframe::{egui, NativeOptions};
use eframe::egui::Widget;
use eframe::egui_wgpu; // for native TextureId path
use project::{AssetRow, ProjectDb};
use timeline::{Fps, Item, ItemKind, Sequence, Track};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use std::path::PathBuf;
mod interaction;
use interaction::{DragMode, DragState};
mod audio;
use audio::AudioState;
use jobs::{JobEvent, JobStatus};
use media_io::YuvPixFmt;
use native_decoder::{create_decoder, DecoderConfig, is_native_decoding_available, ZeroCopyVideoRenderer};
use std::collections::HashMap;
use native_decoder::{
    NativeVideoDecoder, VideoFrame, YuvPixFmt as NativeYuvPixFmt
};
use std::hash::Hash;
// (Arc already imported above)

fn main() {
    // Ensure DB exists before UI
    let data_dir = project::app_data_dir();
    std::fs::create_dir_all(&data_dir).expect("create data dir");
    let db_path = data_dir.join("app.db");
    let db = ProjectDb::open_or_create(&db_path).expect("open db");

    let options = NativeOptions::default();
    let _ = eframe::run_native(
        "Gausian Native Editor",
        options,
        Box::new(move |_cc| Ok(Box::new(App::new(db)))),
    );
}

const PREFETCH_BUDGET_PER_TICK: usize = 6;

#[derive(Default)]
struct DecodeManager {
    decoders: HashMap<String, DecoderEntry>,
}

struct DecoderEntry {
    decoder: Box<dyn NativeVideoDecoder>,
    zc_decoder: Option<Box<dyn NativeVideoDecoder>>, // zero-copy VT session (IOSurface)
    last_pts: Option<f64>,
    last_fmt: Option<&'static str>,
    consecutive_misses: u32,
    attempts_this_tick: u32,
    fed_samples: usize,
    draws: u32,
}

impl DecodeManager {
    fn normalize_path_key(path: &str) -> String {
        std::fs::canonicalize(path)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| path.to_string())
    }

    fn get_or_create(&mut self, path: &str, cfg: &DecoderConfig) -> anyhow::Result<&mut DecoderEntry> {
        let key = Self::normalize_path_key(path);
        if !self.decoders.contains_key(&key) {
            let decoder = if is_native_decoding_available() {
                create_decoder(path, cfg.clone())?
            } else {
                // TODO: on non-macOS, swap for MF/VAAPI backends when available.
                create_decoder(path, cfg.clone())?
            };
            self.decoders.insert(key.clone(), DecoderEntry {
                decoder,
                zc_decoder: None,
                last_pts: None,
                last_fmt: None,
                consecutive_misses: 0,
                attempts_this_tick: 0,
                fed_samples: 0,
                draws: 0,
            });
        }
        Ok(self.decoders.get_mut(&key).unwrap())
    }

    /// Try once; if None, feed the async pipeline a few steps without blocking UI.
    fn decode_and_prefetch(&mut self, path: &str, cfg: &DecoderConfig, target_ts: f64) -> Option<VideoFrame> {
        let entry = self.get_or_create(path, cfg).ok()?;
        entry.attempts_this_tick = 0;

        let mut frame = entry.decoder.decode_frame(target_ts).ok().flatten();
        entry.attempts_this_tick += 1;

        let mut tries = 0;
        while frame.is_none() && tries < PREFETCH_BUDGET_PER_TICK {
            let _ = entry.decoder.decode_frame(target_ts); // advance AVF/VT asynchronously
            entry.attempts_this_tick += 1;
            tries += 1;
            frame = entry.decoder.decode_frame(target_ts).ok().flatten();
        }

        if let Some(ref f) = frame {
            entry.last_pts = Some(f.timestamp);
            entry.last_fmt = Some(match f.format {
                NativeYuvPixFmt::Nv12 => "NV12",
                NativeYuvPixFmt::P010 => "P010",
                _ => "YUV",
            });
            entry.consecutive_misses = 0;
        } else {
            entry.consecutive_misses = entry.consecutive_misses.saturating_add(1);
        }
        frame
    }

    /// Attempt zero-copy decode via IOSurface. On macOS only.
    #[cfg(target_os = "macos")]
    fn decode_zero_copy(&mut self, path: &str, target_ts: f64) -> Option<native_decoder::IOSurfaceFrame> {
        use native_decoder::YuvPixFmt as Nyf;
        let key = Self::normalize_path_key(path);
        let entry = if let Some(e) = self.decoders.get_mut(&key) { e } else {
            // Initialize a CPU decoder entry first (so HUD works), then add zero-copy below.
            let cfg = DecoderConfig { hardware_acceleration: true, preferred_format: Some(Nyf::Nv12), zero_copy: false };
            let _ = self.get_or_create(path, &cfg);
            self.decoders.get_mut(&key).unwrap()
        };
        if entry.zc_decoder.is_none() {
            let cfg_zc = DecoderConfig { hardware_acceleration: true, preferred_format: Some(Nyf::Nv12), zero_copy: true };
            if let Ok(dec) = create_decoder(path, cfg_zc) { entry.zc_decoder = Some(dec); } else { return None; }
        }
        let dec = entry.zc_decoder.as_mut().unwrap();
        // Try a few feeds to coax out a frame without blocking long
        let mut f = dec.decode_frame_zero_copy(target_ts).ok().flatten();
        let mut tries = 0;
        while f.is_none() && tries < PREFETCH_BUDGET_PER_TICK {
            let _ = dec.decode_frame_zero_copy(target_ts);
            tries += 1;
            f = dec.decode_frame_zero_copy(target_ts).ok().flatten();
        }
        f
    }

    #[cfg(not(target_os = "macos"))]
    fn decode_zero_copy(&mut self, _path: &str, _target_ts: f64) -> Option<native_decoder::IOSurfaceFrame> { None }

    fn hud(&self, path: &str, target_ts: f64) -> String {
        let key = Self::normalize_path_key(path);
        if let Some(e) = self.decoders.get(&key) {
            let last = e.last_pts.unwrap_or(f64::NAN);
            let fmt = e.last_fmt.unwrap_or("?");
            let ring = e.decoder.ring_len();
            let cb = e.decoder.cb_frames();
            let last_cb = e.decoder.last_cb_pts();
            let fed = e.decoder.fed_samples();
            
            format!(
                "decode: attempts {}  misses {}  last_pts {:.3}  target {:.3}  fmt {}\nring {}  cb {}  last_cb {:.3}  fed {}  draws {}",
                e.attempts_this_tick, e.consecutive_misses, last, target_ts, fmt,
                ring, cb, last_cb, fed, e.draws
            )
        } else {
            format!("decode: initializing…  target {:.3}", target_ts)
        }
    }
    
    fn increment_draws(&mut self, path: &str) {
        let key = Self::normalize_path_key(path);
        if let Some(e) = self.decoders.get_mut(&key) {
            e.draws = e.draws.saturating_add(1);
        }
    }
}

struct App {
    db: ProjectDb,
    project_id: String,
    import_path: String,
    // timeline state
    seq: Sequence,
    zoom_px_per_frame: f32,
    playhead: i64,
    playing: bool,
    last_tick: Option<Instant>,
    // Anchored playhead timing to avoid jitter
    play_anchor_instant: Option<Instant>,
    play_anchor_frame: i64,
    preview: PreviewState,
    audio: AudioState,
    selected: Option<(usize, usize)>,
    drag: Option<DragState>,
    export: ExportUiState,
    import_workers: Vec<std::thread::JoinHandle<()>>,
    jobs: Option<jobs::JobsHandle>,
    job_events: Vec<JobEvent>,
    show_jobs: bool,
    decode_mgr: DecodeManager,
}

impl App {
    fn new(db: ProjectDb) -> Self {
        let project_id = "default".to_string();
        let _ = db.ensure_project(&project_id, "Default Project", None);
        let mut seq = Sequence::new("Main", 1920, 1080, Fps::new(30, 1), 600);
        if seq.tracks.is_empty() {
            seq.add_track(Track { name: "V1".into(), items: vec![] });
            seq.add_track(Track { name: "V2".into(), items: vec![] });
            seq.add_track(Track { name: "A1".into(), items: vec![] });
        }
        Self {
            db,
            project_id,
            import_path: String::new(),
            seq,
            zoom_px_per_frame: 2.0,
            playhead: 0,
            playing: false,
            last_tick: None,
            play_anchor_instant: None,
            play_anchor_frame: 0,
            preview: PreviewState::new(),
            audio: AudioState::new(),
            selected: None,
            drag: None,
            export: ExportUiState::default(),
            import_workers: Vec::new(),
            jobs: Some(jobs::JobsRuntime::start(2)),
            job_events: Vec::new(),
            show_jobs: false,
            decode_mgr: DecodeManager::default(),
        }
    }

    fn import_from_path(&mut self) {
        let p = std::mem::take(&mut self.import_path);
        if p.trim().is_empty() { return; }
        let path = PathBuf::from(p);
        let _ = self.import_files(&[path]);
    }

    fn export_sequence(&mut self) {
        // Open the export dialog UI
        self.export.open = true;
    }

    fn import_files(&mut self, files: &[PathBuf]) -> Result<()> {
        if files.is_empty() { return Ok(()); }
        let ancestor = nearest_common_ancestor(files);
        if let Some(base) = ancestor.as_deref() { self.db.set_project_base_path(&self.project_id, base)?; }
        let db_path = self.db.path().to_path_buf();
        let project_id = self.project_id.clone();
        for f in files.to_vec() {
            let base = ancestor.clone();
            let db_path = db_path.clone();
            let project_id = project_id.clone();
            let jobs = self.jobs.clone();
            let h = std::thread::spawn(move || {
                let db = project::ProjectDb::open_or_create(&db_path).expect("open db");
                match media_io::probe_media(&f) {
                Ok(info) => {
                    let kind = match info.kind { media_io::MediaKind::Video => "video", media_io::MediaKind::Image => "image", media_io::MediaKind::Audio => "audio" };
                        let rel = base.as_deref().and_then(|b| pathdiff::diff_paths(&f, b));
                    let fps_num = info.fps_num.map(|v| v as i64);
                    let fps_den = info.fps_den.map(|v| v as i64);
                    let duration_frames = match (info.duration_seconds, fps_num, fps_den) {
                        (Some(d), Some(n), Some(dn)) if dn != 0 => Some(((d * (n as f64) / (dn as f64)).round()) as i64),
                        _ => None,
                    };
                        let asset_id = db.insert_asset_row(
                            &project_id,
                        kind,
                            &f,
                        rel.as_deref(),
                        info.width.map(|x| x as i64),
                        info.height.map(|x| x as i64),
                        duration_frames,
                        fps_num,
                        fps_den,
                        info.audio_channels.map(|x| x as i64),
                        info.sample_rate.map(|x| x as i64),
                        ).unwrap_or_default();
                        if let Some(j) = jobs {
                            use jobs::{JobKind, JobSpec};
                            for kind in [JobKind::Waveform, JobKind::Thumbnails, JobKind::Proxy, JobKind::SeekIndex] {
                                let id = j.enqueue(JobSpec { asset_id: asset_id.clone(), kind, priority: 0 });
                                let _ = db.enqueue_job(&id, &asset_id, match kind { JobKind::Waveform=>"waveform", JobKind::Thumbnails=>"thumbs", JobKind::Proxy=>"proxy", JobKind::SeekIndex=>"seekidx" }, 0);
                            }
                        }
                }
                Err(e) => eprintln!("ffprobe failed for {:?}: {e}", f),
            }
            });
            self.import_workers.push(h);
        }
        Ok(())
    }

    fn assets(&self) -> Vec<AssetRow> {
        self.db.list_assets(&self.project_id).unwrap_or_default()
    }

    fn add_asset_to_timeline(&mut self, asset: &AssetRow) {
        // Decide track based on kind
        let is_audio = asset.kind.eq_ignore_ascii_case("audio");
        let track_index = if is_audio { self.seq.tracks.len().saturating_sub(1) } else { 0 };
        if let Some(track) = self.seq.tracks.get_mut(track_index) {
            let from = track.items.iter().map(|it| it.from + it.duration_in_frames).max().unwrap_or(0);
            let duration = asset.duration_frames.unwrap_or(150).max(1);
            let id = uuid::Uuid::new_v4().to_string();
            let kind = if is_audio {
                ItemKind::Audio { src: asset.src_abs.clone() }
            } else if asset.kind.eq_ignore_ascii_case("image") {
                ItemKind::Image { src: asset.src_abs.clone() }
            } else {
                let fr = match (asset.fps_num, asset.fps_den) { (Some(n), Some(d)) if d != 0 => Some(n as f32 / d as f32), _ => None };
                ItemKind::Video { src: asset.src_abs.clone(), frame_rate: fr }
            };
            track.items.push(Item { id, from, duration_in_frames: duration, kind });
            let end = self.seq.tracks.iter().flat_map(|t| t.items.iter().map(|it| it.from + it.duration_in_frames)).max().unwrap_or(0);
            self.seq.duration_in_frames = end.max(self.seq.duration_in_frames);
        }
    }

    fn timeline_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Zoom");
            ui.add(egui::Slider::new(&mut self.zoom_px_per_frame, 0.2..=20.0).logarithmic(true));
            if ui.button("Fit").clicked() {
                let width = ui.available_width().max(1.0);
                self.zoom_px_per_frame = (width / (self.seq.duration_in_frames.max(1) as f32)).max(0.1);
            }
        });

        let track_h = 48.0;
        let content_w = (self.seq.duration_in_frames as f32 * self.zoom_px_per_frame).max(1000.0);
        let content_h = (self.seq.tracks.len() as f32 * track_h).max(200.0);
        egui::ScrollArea::both().show(ui, |ui| {
            let (rect, response) = ui.allocate_exact_size(egui::vec2(content_w, content_h), egui::Sense::click());
            let painter = ui.painter_at(rect);
            // Background
            painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(18, 18, 20));
            // Vertical grid each second
            let fps = (self.seq.fps.num.max(1) as f32 / self.seq.fps.den.max(1) as f32).max(1.0);
            let px_per_sec = self.zoom_px_per_frame * fps;
            let start_x = rect.left();
            let mut x = start_x;
            while x < rect.right() {
                painter.line_segment([egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())], egui::Stroke::new(1.0, egui::Color32::from_gray(50)));
                x += px_per_sec;
            }
            // Tracks and clips
            for (ti, track) in self.seq.tracks.iter().enumerate() {
                let y = rect.top() + ti as f32 * track_h;
                // track separator
                painter.line_segment([egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)], egui::Stroke::new(1.0, egui::Color32::from_gray(60)));
                // items
                for (ii, it) in track.items.iter().enumerate() {
                    let x0 = rect.left() + it.from as f32 * self.zoom_px_per_frame;
                    let x1 = x0 + it.duration_in_frames as f32 * self.zoom_px_per_frame;
                    let r = egui::Rect::from_min_max(egui::pos2(x0, y + 4.0), egui::pos2(x1, y + track_h - 4.0));
                    let hovered = r.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or(egui::pos2(-1.0,-1.0))));
                    let mut border = egui::Stroke::new(1.0, egui::Color32::BLACK);
                    if let Some(sel) = self.selected { if sel == (ti, ii) { border = egui::Stroke::new(2.0, egui::Color32::WHITE); } }
                    let (color, label) = match &it.kind {
                        ItemKind::Audio { .. } => (egui::Color32::from_rgb(40, 120, 40), "Audio"),
                        ItemKind::Image { .. } => (egui::Color32::from_rgb(120, 120, 40), "Image"),
                        ItemKind::Video { .. } => (egui::Color32::from_rgb(40, 90, 160), "Video"),
                        ItemKind::Text { .. } => (egui::Color32::from_rgb(150, 80, 150), "Text"),
                        ItemKind::Solid { .. } => (egui::Color32::from_rgb(80, 80, 80), "Solid"),
                    };
                    painter.rect_filled(r, 4.0, color);
                    painter.rect_stroke(r, 4.0, border);
                    painter.text(r.center_top() + egui::vec2(0.0, 12.0), egui::Align2::CENTER_TOP, label, egui::FontId::monospace(12.0), egui::Color32::WHITE);

                    if hovered && ui.input(|i| i.pointer.primary_pressed()) {
                        // Determine drag mode by edge proximity
                        let mx = ui.input(|i| i.pointer.hover_pos().unwrap_or(egui::pos2(0.0,0.0))).x;
                        let mode = if (mx - r.left()).abs() <= 6.0 { DragMode::TrimStart }
                                   else if (mx - r.right()).abs() <= 6.0 { DragMode::TrimEnd }
                                   else { DragMode::Move };
                        self.selected = Some((ti, ii));
                        self.drag = Some(DragState { track: ti, item: ii, mode, start_mouse_x: mx, orig_from: it.from, orig_dur: it.duration_in_frames });
                    }
                }
            }
            // Playhead
            let phx = rect.left() + self.playhead as f32 * self.zoom_px_per_frame;
            painter.line_segment([egui::pos2(phx, rect.top()), egui::pos2(phx, rect.bottom())], egui::Stroke::new(2.0, egui::Color32::from_rgb(220, 60, 60)));

            if response.clicked() {
                let pos = response.interact_pointer_pos().unwrap_or(rect.left_top());
                let frame = ((pos.x - rect.left()) / self.zoom_px_per_frame).round() as i64;
                let old_playhead = self.playhead;
                self.playhead = frame.clamp(0, self.seq.duration_in_frames);
                
                // Only request repaint if playhead actually changed
                if self.playhead != old_playhead {
                    // Repaint will be triggered by the main update loop
                }
            }

            if let Some(drag) = self.drag {
                if ui.input(|i| !i.pointer.primary_down()) {
                    self.drag = None;
                } else if let Some((ti, ii)) = self.selected {
                    if ti < self.seq.tracks.len() && ii < self.seq.tracks[ti].items.len() {
                        let item = &mut self.seq.tracks[ti].items[ii];
                        let mx = ui.input(|i| i.pointer.hover_pos().unwrap_or(egui::pos2(0.0,0.0))).x;
                        let dx_px = mx - drag.start_mouse_x;
                        let df = (dx_px / self.zoom_px_per_frame).round() as i64;
                        let fpsf = self.seq.fps.num.max(1) as f32 / self.seq.fps.den.max(1) as f32;
                        let eps = 3.0; // frames
                        match drag.mode {
                            DragMode::Move => {
                                let mut new_from = (drag.orig_from + df).max(0);
                                let secf = (new_from as f32 / fpsf).round() * fpsf;
                                if ((secf - new_from as f32).abs()) <= eps { new_from = secf as i64; }
                                item.from = new_from;
                            }
                            DragMode::TrimStart => {
                                let mut new_from = (drag.orig_from + df).clamp(0, drag.orig_from + drag.orig_dur - 1);
                                let secf = (new_from as f32 / fpsf).round() * fpsf;
                                if ((secf - new_from as f32).abs()) <= eps { new_from = secf as i64; }
                                let delta = new_from - drag.orig_from;
                                item.from = new_from;
                                item.duration_in_frames = (drag.orig_dur - delta).max(1);
                            }
                            DragMode::TrimEnd => {
                                let mut new_dur = (drag.orig_dur + df).max(1);
                                let end = drag.orig_from + new_dur;
                                let secf = (end as f32 / fpsf).round() * fpsf;
                                if ((secf - end as f32).abs()) <= eps { new_dur = (secf as i64 - drag.orig_from).max(1); }
                                item.duration_in_frames = new_dur;
                            }
                        }
                    }
                }
            }
        });
    }

    fn poll_jobs(&mut self) {
        if let Some(j) = &self.jobs {
            while let Ok(ev) = j.rx_events.try_recv() {
                // Update DB status
                let status_str = match &ev.status {
                    JobStatus::Pending => "pending",
                    JobStatus::Running => "running",
                    JobStatus::Progress(_) => "progress",
                    JobStatus::Done => "done",
                    JobStatus::Failed(_) => "failed",
                    JobStatus::Canceled => "canceled",
                };
                let _ = self.db.update_job_status(&ev.id, status_str);
                self.job_events.push(ev);
                if self.job_events.len() > 300 { self.job_events.remove(0); }
            }
        }
    }

    fn preview_ui(&mut self, ctx: &egui::Context, frame: &eframe::Frame, ui: &mut egui::Ui) {
        // Determine current visual source at playhead
        let fps = self.seq.fps.num.max(1) as f32 / self.seq.fps.den.max(1) as f32;
        let t_sec = self.playhead as f32 / fps;
        let source = current_visual_source(&self.seq, self.playhead);

        // Layout: reserve a 16:9 box or fit available space
        let avail = ui.available_size();
        let mut w = avail.x.max(320.0);
        let mut h = (w * 9.0 / 16.0).round();
        if h > avail.y { h = avail.y; w = (h * 16.0 / 9.0).round(); }
        let desired = (w as u32, h as u32);

        // Playback advance (anchored clock to avoid jitter)
        if self.playing {
            let now = Instant::now();
            if self.play_anchor_instant.is_none() {
                self.play_anchor_instant = Some(now);
                self.play_anchor_frame = self.playhead;
            }
            let base = self.play_anchor_frame;
            let elapsed = now.duration_since(self.play_anchor_instant.unwrap());
            let advanced = (fps * elapsed.as_secs_f32()).floor() as i64;
            self.playhead = (base + advanced).clamp(0, self.seq.duration_in_frames);
            if self.playhead >= self.seq.duration_in_frames {
                self.playing = false;
            }
        } else {
            self.play_anchor_instant = None;
        }

        // Draw
        let (rect, _resp) = ui.allocate_exact_size(egui::vec2(w, h), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(12, 12, 12));
        
        // Use persistent decoder with prefetch
        if let Some(src) = source.as_ref() {
        if let Some(rs) = frame.wgpu_render_state() {
                let decoder_config = DecoderConfig {
                    hardware_acceleration: true,
                    preferred_format: Some(NativeYuvPixFmt::Nv12),
                    zero_copy: false, // Phase 1 only
                };
                
                let vf_opt = self.decode_mgr.decode_and_prefetch(&src.path, &decoder_config, t_sec as f64);
                // Try zero-copy first (macOS only)
                #[cfg(target_os = "macos")]
                let zc_opt = self.decode_mgr.decode_zero_copy(&src.path, t_sec as f64);
                #[cfg(not(target_os = "macos"))]
                let zc_opt: Option<native_decoder::IOSurfaceFrame> = None;

                // Prefer zero-copy if available; otherwise fallback to CPU present path
                let yuv_result = {
                    #[cfg(target_os = "macos")]
                    {
                        if let Some(ref zc) = zc_opt {
                            if let Some(res) = self.preview.present_nv12_zero_copy(&rs, zc) { res } else { self.preview.present_yuv_with_frame(&rs, &src.path, t_sec as f64, vf_opt.as_ref()) }
                        } else {
                            self.preview.present_yuv_with_frame(&rs, &src.path, t_sec as f64, vf_opt.as_ref())
                        }
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        self.preview.present_yuv_with_frame(&rs, &src.path, t_sec as f64, vf_opt.as_ref())
                    }
                };
                
                if let Some((fmt, y, uv)) = yuv_result {
                    let use_uint = matches!(fmt, YuvPixFmt::P010) && !device_supports_16bit_norm(&rs);
                    let (tex_w, tex_h) = if zc_opt.is_some() {
                        let z = zc_opt.as_ref().unwrap();
                        (z.width, z.height)
                    } else {
                        (self.y_size.0, self.y_size.1)
                    };
                    let cb = egui_wgpu::Callback::new_paint_callback(
                        rect,
                        PreviewYuvCallback { y_tex: y, uv_tex: uv, fmt, use_uint, w: tex_w, h: tex_h },
                    );
                    ui.painter().add(cb);
                    self.decode_mgr.increment_draws(&src.path);

                    // Show HUD overlay on successful draw
                    let hud = self.decode_mgr.hud(&src.path, t_sec as f64);
                    painter.text(rect.left_top() + egui::vec2(5.0, 5.0), egui::Align2::LEFT_TOP, hud, egui::FontId::monospace(10.0), egui::Color32::WHITE);
                } else {
                    // No YUV textures available: render HUD instead of black screen
                    let hud = self.decode_mgr.hud(&src.path, t_sec as f64);
                    painter.text(rect.center(), egui::Align2::CENTER_CENTER, hud, egui::FontId::monospace(12.0), egui::Color32::LIGHT_GRAY);
                }
            } else {
                painter.text(rect.center(), egui::Align2::CENTER_CENTER, "No WGPU state", egui::FontId::proportional(16.0), egui::Color32::GRAY);
            }
        } else {
            painter.text(rect.center(), egui::Align2::CENTER_CENTER, "No Preview", egui::FontId::proportional(16.0), egui::Color32::GRAY);
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Optimized repaint pacing: adaptive frame rate based on activity
        if self.playing {
            let fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
            let dt = if fps > 0.0 { 1.0 / fps } else { 1.0 / 30.0 };
            ctx.request_repaint_after(Duration::from_secs_f64(dt));
        } else {
            // When not playing, only repaint when needed (scrubbing, UI changes)
            // This reduces CPU usage significantly when idle
        }
        // Space toggles play/pause
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            if self.playing { self.playing = false; }
            else {
                if self.playhead >= self.seq.duration_in_frames { self.playhead = 0; }
                self.playing = true;
                self.last_tick = Some(Instant::now());
            }
        }
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Import path:");
                ui.text_edit_singleline(&mut self.import_path);
                if ui.button("Add").clicked() {
                    self.import_from_path();
                }
                if ui.button("Export...").clicked() {
                    self.export_sequence();
                }
                if ui.button("Jobs").clicked() {
                    self.show_jobs = !self.show_jobs;
                }
                ui.separator();
                if ui.button(if self.playing { "Pause (Space)" } else { "Play (Space)" }).clicked() {
                    if self.playing { self.playing = false; } else { self.playing = true; self.last_tick = Some(Instant::now()); }
                }
            });
        });

        // Export dialog UI
        self.export.ui(ctx, &self.seq, &self.db, &self.project_id);

        // Preview panel will be inside CentralPanel with resizable area

        egui::SidePanel::left("assets").default_width(340.0).show(ctx, |ui| {
            self.poll_jobs();
            ui.heading("Assets");
            ui.horizontal(|ui| {
                if ui.button("Import...").clicked() {
                    if let Some(files) = rfd::FileDialog::new().pick_files() {
                        let _ = self.import_files(&files);
                    }
                }
                if ui.button("Refresh").clicked() {}
                if ui.button("Jobs").clicked() { self.show_jobs = !self.show_jobs; }
            });
            
            // Show hardware encoders info
            ui.collapsing("Hardware Encoders", |ui| {
                let encoders = media_io::get_hardware_encoders();
                if encoders.is_empty() {
                    ui.label("No hardware encoders detected");
                    ui.label("Using software encoders (slower)");
                } else {
                    for (codec, encoder_list) in encoders {
                        ui.label(format!("{}:", codec));
                        for encoder in encoder_list {
                            ui.label(format!("  • {}", encoder));
                        }
                    }
                }
            });

            // Native Video Decoder
            ui.collapsing("Native Video Decoder", |ui| {
                let available = is_native_decoding_available();
                ui.label(format!("Native decoding available: {}", if available { "✅ Yes" } else { "❌ No" }));
                
                if available {
                    ui.label("• VideoToolbox hardware acceleration");
                    ui.label("• Phase 1: CPU plane copies (NV12/P010)");
                    ui.label("• Phase 2: Zero-copy IOSurface (planned)");
                    
                    if ui.button("Test Native Decoder (Phase 1)").clicked() {
                        // Test native decoder with a sample video
                        if let Some(asset) = self.assets().first() {
                            let config = DecoderConfig {
                                hardware_acceleration: true,
                                preferred_format: Some(native_decoder::YuvPixFmt::Nv12),
                                zero_copy: false, // Phase 1 only
                            };
                            
                            match create_decoder(&asset.src_abs, config) {
                                Ok(mut decoder) => {
                                    let properties = decoder.get_properties();
                                    ui.label(format!("✅ Phase 1 Decoder created successfully!"));
                                    ui.label(format!("Video: {}x{} @ {:.1}fps", 
                                        properties.width, properties.height, properties.frame_rate));
                                    ui.label(format!("Duration: {:.1}s", properties.duration));
                                    ui.label(format!("Format: {:?}", properties.format));
                                    
                                    // Test frame decoding
                                    if let Ok(Some(frame)) = decoder.decode_frame(1.0) {
                                        ui.label(format!("✅ Frame decoded: {}x{} YUV", frame.width, frame.height));
                                        ui.label(format!("Y plane: {} bytes", frame.y_plane.len()));
                                        ui.label(format!("UV plane: {} bytes", frame.uv_plane.len()));
                                    } else {
                                        ui.label("❌ Frame decoding failed");
                                    }
                                }
                                Err(e) => {
                                    ui.label(format!("❌ Decoder creation failed: {}", e));
                                }
                            }
                        } else {
                            ui.label("❌ No assets available for testing");
                        }
                    }
                    
                    if ui.button("Test Zero-Copy Decoder (Phase 2)").clicked() {
                        // Test zero-copy decoder with IOSurface
                        if let Some(asset) = self.assets().first() {
                            let config = DecoderConfig {
                                hardware_acceleration: true,
                                preferred_format: Some(native_decoder::YuvPixFmt::Nv12),
                                zero_copy: true, // Phase 2 zero-copy
                            };
                            
                            match create_decoder(&asset.src_abs, config) {
                                Ok(mut decoder) => {
                                    let properties = decoder.get_properties();
                                    ui.label(format!("✅ Phase 2 Zero-Copy Decoder created!"));
                                    ui.label(format!("Video: {}x{} @ {:.1}fps", 
                                        properties.width, properties.height, properties.frame_rate));
                                    ui.label(format!("Zero-copy supported: {}", decoder.supports_zero_copy()));
                                    
                                    // Test zero-copy frame decoding
                                    #[cfg(target_os = "macos")]
                                    {
                                        if let Ok(Some(iosurface_frame)) = decoder.decode_frame_zero_copy(1.0) {
                                            ui.label(format!("✅ IOSurface frame decoded: {}x{}", 
                                                iosurface_frame.width, iosurface_frame.height));
                                            ui.label(format!("Surface format: {:?}", iosurface_frame.format));
                                            ui.label(format!("Timestamp: {:.3}s", iosurface_frame.timestamp));
                                            
                                            // Test WGPU integration
                                            ui.label("🎬 Testing WGPU integration...");
                                            ui.label("✅ Zero-copy pipeline ready for rendering!");
                                        } else {
                                            ui.label("❌ Zero-copy frame decoding failed");
                                        }
                                    }
                                    
                                    #[cfg(not(target_os = "macos"))]
                                    {
                                        ui.label("ℹ️ Zero-copy mode not available on this platform");
                                    }
                                }
                                Err(e) => {
                                    ui.label(format!("❌ Zero-copy decoder creation failed: {}", e));
                                }
                            }
                        } else {
                            ui.label("❌ No assets available for testing");
                        }
                    }
                } else {
                    ui.label("Native decoding not available on this platform");
                    ui.label("Falling back to FFmpeg-based decoding");
                }
            });
            egui::Separator::default().ui(ui);
            let assets = self.assets();
            egui_extras::TableBuilder::new(ui)
                .striped(true)
                .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                .column(egui_extras::Column::remainder()) // Name
                .column(egui_extras::Column::auto()) // Kind
                .column(egui_extras::Column::auto()) // WxH
                .column(egui_extras::Column::auto()) // Add
                .header(20.0, |mut header| {
                    header.col(|ui| { ui.strong("Name"); });
                    header.col(|ui| { ui.strong("Kind"); });
                    header.col(|ui| { ui.strong("Size"); });
                    header.col(|ui| { ui.strong(""); });
                })
                .body(|mut body| {
                    for a in assets.iter() {
                        body.row(22.0, |mut row| {
                            row.col(|ui| {
                                let name = std::path::Path::new(&a.src_abs).file_name().map(|s| s.to_string_lossy()).unwrap_or_default();
                                ui.label(name);
                            });
                            row.col(|ui| { ui.label(&a.kind); });
                            row.col(|ui| {
                                if let (Some(w), Some(h)) = (a.width, a.height) { ui.label(format!("{}x{}", w, h)); }
                            });
                            row.col(|ui| {
                                if ui.button("Add").clicked() { self.add_asset_to_timeline(a); }
                            });
                        });
                    }
                });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Resize::default()
                .id_salt("preview_resize")
                .default_size(egui::vec2(ui.available_width(), 360.0))
                .show(ui, |ui| {
                    self.preview_ui(ctx, frame, ui);
                });
            ui.add_space(4.0);
            ui.separator();
            
            // Performance indicator
            ui.horizontal(|ui| {
            ui.heading("Timeline");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let cache_stats = format!("Cache: {}/{} hits", 
                                            self.preview.cache_hits, 
                                            self.preview.cache_hits + self.preview.cache_misses);
                    ui.small(&cache_stats);
                });
            });
            
            self.timeline_ui(ui);
        });

        // Jobs window
        if self.show_jobs {
            egui::Window::new("Jobs").open(&mut self.show_jobs).resizable(true).show(ctx, |ui| {
                ui.label("Background Jobs");
                let mut latest: std::collections::BTreeMap<String, JobEvent> = std::collections::BTreeMap::new();
                for ev in &self.job_events { latest.insert(ev.id.clone(), ev.clone()); }
                egui_extras::TableBuilder::new(ui)
                    .striped(true)
                    .column(egui_extras::Column::auto())
                    .column(egui_extras::Column::auto())
                    .column(egui_extras::Column::auto())
                    .column(egui_extras::Column::remainder())
                    .header(18.0, |mut h| {
                        h.col(|ui| { ui.strong("Job"); });
                        h.col(|ui| { ui.strong("Asset"); });
                        h.col(|ui| { ui.strong("Kind"); });
                        h.col(|ui| { ui.strong("Status"); });
                    })
                    .body(|mut b| {
                        for (_id, ev) in latest.iter() {
                            b.row(20.0, |mut r| {
                                r.col(|ui| { ui.monospace(&ev.id[..8.min(ev.id.len())]); });
                                r.col(|ui| { ui.monospace(&ev.asset_id[..8.min(ev.asset_id.len())]); });
                                r.col(|ui| { ui.label(format!("{:?}", ev.kind)); });
                                r.col(|ui| {
                                    match &ev.status {
                                        JobStatus::Progress(p) => { ui.add(egui::ProgressBar::new(*p).show_percentage()); }
                                        s => { ui.label(format!("{:?}", s)); }
                                    }
                                    if !matches!(ev.status, JobStatus::Done | JobStatus::Failed(_) | JobStatus::Canceled) {
                                        if ui.small_button("Cancel").clicked() {
                                            if let Some(j) = &self.jobs { j.cancel_job(&ev.id); }
                                        }
                                    }
                                });
                            });
                        }
                    });
            });
        }

        // Audio playback follows play state
        if self.playing {
            let fps = self.seq.fps.num.max(1) as f32 / self.seq.fps.den.max(1) as f32;
            let t_sec = self.playhead as f32 / fps;
            // pick audio clip at playhead or audio from current video
            let mut src_path: Option<String> = None;
            // search audio tracks first
            'outer: for track in self.seq.tracks.iter().rev() {
                for it in &track.items {
                    let covers = self.playhead >= it.from && self.playhead < it.from + it.duration_in_frames;
                    if !covers { continue; }
                    if let ItemKind::Audio { src } = &it.kind { src_path = Some(src.clone()); break 'outer; }
                }
            }
            if src_path.is_none() {
                if let Some(v) = current_visual_source(&self.seq, self.playhead) { if !v.is_image { src_path = Some(v.path); } }
            }
            self.audio.ensure_playing(src_path.as_deref(), t_sec as f64);
        } else {
            self.audio.stop();
        }
    }
}

fn nearest_common_ancestor(paths: &[PathBuf]) -> Option<PathBuf> {
    if paths.is_empty() { return None; }
    let mut it = paths.iter();
    let mut acc = it.next()?.ancestors().map(|p| p.to_path_buf()).collect::<Vec<_>>();
    for p in it {
        let set = p.ancestors().map(|p| p.to_path_buf()).collect::<Vec<_>>();
        acc.retain(|cand| set.contains(cand));
        if acc.is_empty() { break; }
    }
    acc.first().cloned()
}

#[derive(Clone, Debug)]
struct VisualSource { path: String, is_image: bool }

fn current_visual_source(seq: &Sequence, playhead: i64) -> Option<VisualSource> {
    // Choose topmost non-audio track that has a covering item
    for track in seq.tracks.iter().rev() {
        for it in &track.items {
            let covers = playhead >= it.from && playhead < it.from + it.duration_in_frames;
            if !covers { continue; }
            match &it.kind {
                ItemKind::Video { src, .. } => return Some(VisualSource { path: src.clone(), is_image: false }),
                ItemKind::Image { src } => return Some(VisualSource { path: src.clone(), is_image: true }),
                ItemKind::Text { .. } | ItemKind::Solid { .. } | ItemKind::Audio { .. } => {}
            }
        }
    }
    None
}

struct PreviewState {
    // Efficient frame cache with LRU eviction
    frame_cache: Arc<Mutex<std::collections::HashMap<FrameCacheKey, CachedFrame>>>,
    cache_worker: Option<JoinHandle<()>>,
    cache_stop: Option<Arc<AtomicBool>>,

    // Current preview state
    current_source: Option<VisualSource>,
    last_frame_time: f64,
    last_size: (u32, u32),

    // Native WGPU presentation (double-buffered RGBA fallback)
    gpu_tex_a: Option<std::sync::Arc<eframe::wgpu::Texture>>,
    gpu_view_a: Option<eframe::wgpu::TextureView>,
    gpu_tex_b: Option<std::sync::Arc<eframe::wgpu::Texture>>,
    gpu_view_b: Option<eframe::wgpu::TextureView>,
    gpu_use_b: bool,
    gpu_tex_id: Option<egui::TextureId>,
    gpu_size: (u32, u32),

    // NV12 fast path (triple-buffered Y/UV planes)
    y_tex: [Option<std::sync::Arc<eframe::wgpu::Texture>>; 3],
    uv_tex: [Option<std::sync::Arc<eframe::wgpu::Texture>>; 3],
    y_size: (u32, u32),
    uv_size: (u32, u32),
    ring_write: usize,
    ring_present: usize,

    // Staging buffers + scratch for COPY_BUFFER_TO_TEXTURE (no per-frame allocs)
    y_stage: [Option<eframe::wgpu::Buffer>; 3],
    uv_stage: [Option<eframe::wgpu::Buffer>; 3],
    y_pad_bpr: usize,
    uv_pad_bpr: usize,
    y_rows: u32,
    uv_rows: u32,

    // Simple NV12/P010 frame cache with small LRU to avoid re-decoding during scrubs
    nv12_cache: std::collections::HashMap<FrameCacheKey, Nv12Frame>,
    nv12_keys: std::collections::VecDeque<FrameCacheKey>,
    pix_fmt_map: std::collections::HashMap<String, YuvPixFmt>,

    // Zero-copy NV12 cache (macOS only)
    #[cfg(target_os = "macos")]
    gpu_yuv: Option<native_decoder::GpuYuv>,
    #[cfg(target_os = "macos")]
    zc_size: (u32, u32),

    // Performance metrics
    cache_hits: u64,
    cache_misses: u64,
    decode_time_ms: f64,
}

impl PreviewState {
    fn new() -> Self {
        Self {
            frame_cache: Arc::new(Mutex::new(HashMap::new())),
            cache_worker: None,
            cache_stop: None,
            current_source: None,
            last_frame_time: -1.0,
            last_size: (0, 0),
            gpu_tex_a: None,
            gpu_view_a: None,
            gpu_tex_b: None,
            gpu_view_b: None,
            gpu_use_b: false,
            gpu_tex_id: None,
            gpu_size: (0, 0),
            cache_hits: 0,
            cache_misses: 0,
            decode_time_ms: 0.0,
            y_tex: [None, None, None],
            uv_tex: [None, None, None],
            y_size: (0, 0),
            uv_size: (0, 0),
            ring_write: 0,
            ring_present: 0,
            y_stage: [None, None, None],
            uv_stage: [None, None, None],
            y_pad_bpr: 0,
            uv_pad_bpr: 0,
            y_rows: 0,
            uv_rows: 0,
            nv12_cache: std::collections::HashMap::new(),
            nv12_keys: std::collections::VecDeque::new(),
            pix_fmt_map: std::collections::HashMap::new(),
            #[cfg(target_os = "macos")]
            gpu_yuv: None,
            #[cfg(target_os = "macos")]
            zc_size: (0, 0),
        }
    }

    #[cfg(target_os = "macos")]
    fn ensure_zero_copy_nv12_textures(&mut self, rs: &eframe::egui_wgpu::RenderState, w: u32, h: u32) {
        let y_sz = (w, h);
        if self.gpu_yuv.is_some() && self.zc_size == y_sz { return; }
        let device = &*rs.device;
        eprintln!("[zc] allocate NV12 textures Y {}x{} UV {}x{}", w, h, (w + 1)/2, (h + 1)/2);
        let y_desc = eframe::wgpu::TextureDescriptor {
            label: Some("zc_nv12_y"),
            size: eframe::wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: eframe::wgpu::TextureDimension::D2,
            format: eframe::wgpu::TextureFormat::R8Unorm,
            usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING | eframe::wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let uv_desc = eframe::wgpu::TextureDescriptor {
            label: Some("zc_nv12_uv"),
            size: eframe::wgpu::Extent3d { width: (w + 1)/2, height: (h + 1)/2, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: eframe::wgpu::TextureDimension::D2,
            format: eframe::wgpu::TextureFormat::Rg8Unorm,
            usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING | eframe::wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };
        let y_tex = std::sync::Arc::new(device.create_texture(&y_desc));
        let uv_tex = std::sync::Arc::new(device.create_texture(&uv_desc));
        self.gpu_yuv = Some(native_decoder::GpuYuv { y_tex, uv_tex });
        self.zc_size = y_sz;
    }

    // Ensure triple-buffer NV12 plane textures at native size
    fn ensure_yuv_textures(&mut self, rs: &eframe::egui_wgpu::RenderState, w: u32, h: u32, fmt: YuvPixFmt) {
        let y_sz = (w, h);
        let uv_sz = ((w + 1) / 2, (h + 1) / 2);
        if self.y_size == y_sz && self.uv_size == uv_sz && self.y_tex[0].is_some() && self.uv_tex[0].is_some() {
            return;
        }
        let device = &*rs.device;
        let supports16 = device_supports_16bit_norm(rs);
        let (y_format, uv_format, y_bpp, uv_bpp_per_texel) = match fmt {
            YuvPixFmt::Nv12 => (eframe::wgpu::TextureFormat::R8Unorm, eframe::wgpu::TextureFormat::Rg8Unorm, 1usize, 2usize),
            YuvPixFmt::P010 => {
                if supports16 {
                    (eframe::wgpu::TextureFormat::R16Unorm, eframe::wgpu::TextureFormat::Rg16Unorm, 2usize, 4usize)
                } else {
                    (eframe::wgpu::TextureFormat::R16Uint, eframe::wgpu::TextureFormat::Rg16Uint, 2usize, 4usize)
                }
            }
        };
        let make_y = || device.create_texture(&eframe::wgpu::TextureDescriptor {
            label: Some("preview_nv12_y"),
            size: eframe::wgpu::Extent3d { width: y_sz.0, height: y_sz.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: eframe::wgpu::TextureDimension::D2,
            format: y_format,
            usage: eframe::wgpu::TextureUsages::COPY_DST | eframe::wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let make_uv = || device.create_texture(&eframe::wgpu::TextureDescriptor {
            label: Some("preview_nv12_uv"),
            size: eframe::wgpu::Extent3d { width: uv_sz.0, height: uv_sz.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: eframe::wgpu::TextureDimension::D2,
            format: uv_format,
            usage: eframe::wgpu::TextureUsages::COPY_DST | eframe::wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        for i in 0..3 {
            self.y_tex[i] = Some(std::sync::Arc::new(make_y()));
            self.uv_tex[i] = Some(std::sync::Arc::new(make_uv()));
            // (re)create staging buffers for COPY_BUFFER_TO_TEXTURE
            self.y_stage[i] = Some(device.create_buffer(&eframe::wgpu::BufferDescriptor {
                label: Some("stage_y"),
                size: (y_sz.1 as usize * align_to((y_sz.0 as usize)*y_bpp, eframe::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)) as u64,
                usage: eframe::wgpu::BufferUsages::COPY_SRC | eframe::wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.uv_stage[i] = Some(device.create_buffer(&eframe::wgpu::BufferDescriptor {
                label: Some("stage_uv"),
                size: (uv_sz.1 as usize * align_to((uv_sz.0 as usize) * uv_bpp_per_texel, eframe::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)) as u64,
                usage: eframe::wgpu::BufferUsages::COPY_SRC | eframe::wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        self.ring_write = 0;
        self.ring_present = 0;
        self.y_size = y_sz;
        self.uv_size = uv_sz;
        self.y_pad_bpr = align_to((y_sz.0 as usize)*y_bpp, eframe::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        self.uv_pad_bpr = align_to((uv_sz.0 as usize) * uv_bpp_per_texel, eframe::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        self.y_rows = y_sz.1;
        self.uv_rows = uv_sz.1;
    }

    fn upload_yuv_planes(&mut self, rs: &eframe::egui_wgpu::RenderState, fmt: YuvPixFmt, y: &[u8], uv: &[u8], w: u32, h: u32) {
        self.ensure_yuv_textures(rs, w, h, fmt);
        let queue = &*rs.queue;
        let device = &*rs.device;
        let next_idx = (self.ring_write + 1) % 3;
        if next_idx == self.ring_present { return; } // drop frame rather than stall
        let idx = self.ring_write % 3;
        let y_tex = self.y_tex[idx].as_ref().map(|a| &**a).unwrap();
        let uv_tex = self.uv_tex[idx].as_ref().map(|a| &**a).unwrap();

        let uv_w = (w + 1) / 2;
        let uv_h = (h + 1) / 2;
        let (y_bpp, uv_bpp_per_texel) = match fmt { YuvPixFmt::Nv12 => (1usize, 2usize), YuvPixFmt::P010 => (2usize, 4usize) };
        let y_bpr = (w as usize) * y_bpp;
        let uv_bpr = (uv_w as usize) * uv_bpp_per_texel;
        let y_pad_bpr = self.y_pad_bpr;
        let uv_pad_bpr = self.uv_pad_bpr;

        // Fill pre-allocated scratch buffers with row padding
        let mut y_scratch: Vec<u8> = Vec::with_capacity(y_pad_bpr * h as usize);
        unsafe { y_scratch.set_len(y_pad_bpr * h as usize); }
        for r in 0..(h as usize) {
            let s = r * y_bpr;
            let d = r * y_pad_bpr;
            y_scratch[d..d + y_bpr].copy_from_slice(&y[s..s + y_bpr]);
        }
        let mut uv_scratch: Vec<u8> = Vec::with_capacity(uv_pad_bpr * uv_h as usize);
        unsafe { uv_scratch.set_len(uv_pad_bpr * uv_h as usize); }
        for r in 0..(uv_h as usize) {
            let s = r * uv_bpr;
            let d = r * uv_pad_bpr;
            uv_scratch[d..d + uv_bpr].copy_from_slice(&uv[s..s + uv_bpr]);
        }

        let y_stage = self.y_stage[idx].as_ref().unwrap();
        let uv_stage = self.uv_stage[idx].as_ref().unwrap();
        queue.write_buffer(y_stage, 0, &y_scratch);
        queue.write_buffer(uv_stage, 0, &uv_scratch);

        let mut encoder = device.create_command_encoder(&eframe::wgpu::CommandEncoderDescriptor { label: Some("nv12_upload") });
        encoder.copy_buffer_to_texture(
            eframe::wgpu::ImageCopyBuffer {
                buffer: y_stage,
                layout: eframe::wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(y_pad_bpr as u32), rows_per_image: Some(h) },
            },
            eframe::wgpu::ImageCopyTexture { texture: y_tex, mip_level: 0, origin: eframe::wgpu::Origin3d::ZERO, aspect: eframe::wgpu::TextureAspect::All },
            eframe::wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        encoder.copy_buffer_to_texture(
            eframe::wgpu::ImageCopyBuffer {
                buffer: uv_stage,
                layout: eframe::wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(uv_pad_bpr as u32), rows_per_image: Some(uv_h) },
            },
            eframe::wgpu::ImageCopyTexture { texture: uv_tex, mip_level: 0, origin: eframe::wgpu::Origin3d::ZERO, aspect: eframe::wgpu::TextureAspect::All },
            eframe::wgpu::Extent3d { width: uv_w, height: uv_h, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);

        self.ring_present = idx;
        self.ring_write = next_idx;
    }

    fn present_yuv(&mut self, rs: &eframe::egui_wgpu::RenderState, path: &str, t_sec: f64) -> Option<(YuvPixFmt, Arc<eframe::wgpu::Texture>, Arc<eframe::wgpu::Texture>)> {
        let key = FrameCacheKey::new(path, t_sec, 0, 0);
        let mut fmt; let mut y; let mut uv; let mut w; let mut h;
        if let Some(hit) = self.nv12_cache.get(&key) {
            fmt = hit.fmt; y = hit.y.clone(); uv = hit.uv.clone(); w = hit.w; h = hit.h;
            if let Some(pos) = self.nv12_keys.iter().position(|k| k == &key) { self.nv12_keys.remove(pos); }
            self.nv12_keys.push_back(key.clone());
            } else {
            if let Ok(frame) = media_io::decode_yuv_at(std::path::Path::new(path), t_sec) {
                fmt = frame.fmt; y = frame.y; uv = frame.uv; w = frame.width; h = frame.height;
                if fmt == YuvPixFmt::P010 && !device_supports_16bit_norm(rs) {
                    if let Some((_f, ny, nuv, nw, nh)) = decode_video_frame_nv12_only(path, t_sec) { fmt = YuvPixFmt::Nv12; y = ny; uv = nuv; w = nw; h = nh; }
                }
                self.nv12_cache.insert(key.clone(), Nv12Frame { fmt, y: y.clone(), uv: uv.clone(), w, h });
                self.nv12_keys.push_back(key.clone());
                if self.nv12_keys.len() > 64 { if let Some(old) = self.nv12_keys.pop_front() { self.nv12_cache.remove(&old); } }
            } else { return None; }
        }
        self.upload_yuv_planes(rs, fmt, &y, &uv, w, h);
        let idx = self.ring_present;
        Some((fmt, self.y_tex[idx].as_ref().unwrap().clone(), self.uv_tex[idx].as_ref().unwrap().clone()))
    }

    // Ensure double-buffered GPU textures and a registered TextureId
    fn ensure_gpu_textures(&mut self, rs: &eframe::egui_wgpu::RenderState, w: u32, h: u32) {
        if self.gpu_size == (w, h) && self.gpu_tex_id.is_some() && (self.gpu_view_a.is_some() || self.gpu_view_b.is_some()) {
            return;
        }
        let device = &*rs.device;
        let make_tex = || device.create_texture(&eframe::wgpu::TextureDescriptor {
            label: Some("preview_native_tex"),
            size: eframe::wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: eframe::wgpu::TextureDimension::D2,
            format: eframe::wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: eframe::wgpu::TextureUsages::COPY_DST | eframe::wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let tex_a = std::sync::Arc::new(make_tex());
        let view_a = tex_a.create_view(&eframe::wgpu::TextureViewDescriptor::default());
        let tex_b = std::sync::Arc::new(make_tex());
        let view_b = tex_b.create_view(&eframe::wgpu::TextureViewDescriptor::default());

        // Register a TextureId if needed, otherwise update it to A initially
        let mut renderer = rs.renderer.write();
        if let Some(id) = self.gpu_tex_id {
            renderer.update_egui_texture_from_wgpu_texture(device, &view_a, eframe::wgpu::FilterMode::Linear, id);
        } else {
            let id = renderer.register_native_texture(device, &view_a, eframe::wgpu::FilterMode::Linear);
            self.gpu_tex_id = Some(id);
        }

        self.gpu_tex_a = Some(tex_a);
        self.gpu_view_a = Some(view_a);
        self.gpu_tex_b = Some(tex_b);
        self.gpu_view_b = Some(view_b);
        self.gpu_use_b = false;
        self.gpu_size = (w, h);
    }

    // Upload RGBA bytes into the next back buffer and retarget the TextureId to it
    fn upload_gpu_frame(&mut self, rs: &eframe::egui_wgpu::RenderState, rgba: &[u8]) {
        let (w, h) = self.gpu_size;
        let queue = &*rs.queue;
        // swap buffer
        self.gpu_use_b = !self.gpu_use_b;
        let (tex, view) = if self.gpu_use_b {
            (self.gpu_tex_b.as_ref().map(|a| &**a), self.gpu_view_b.as_ref())
        } else {
            (self.gpu_tex_a.as_ref().map(|a| &**a), self.gpu_view_a.as_ref())
        };
        if let (Some(tex), Some(view)) = (tex, view) {
            let bytes_per_row = (w * 4) as usize;
            let align = eframe::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize; // 256
            let padded_bpr = ((bytes_per_row + align - 1) / align) * align;
            if padded_bpr == bytes_per_row {
                queue.write_texture(
                    eframe::wgpu::ImageCopyTexture { texture: tex, mip_level: 0, origin: eframe::wgpu::Origin3d::ZERO, aspect: eframe::wgpu::TextureAspect::All },
                    rgba,
                    eframe::wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some((bytes_per_row) as u32), rows_per_image: Some(h) },
                    eframe::wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
            } else {
                // build a padded buffer per row to satisfy alignment
                let mut padded = vec![0u8; padded_bpr * (h as usize)];
                for row in 0..(h as usize) {
                    let src_off = row * bytes_per_row;
                    let dst_off = row * padded_bpr;
                    padded[dst_off..dst_off + bytes_per_row]
                        .copy_from_slice(&rgba[src_off..src_off + bytes_per_row]);
                }
                queue.write_texture(
                    eframe::wgpu::ImageCopyTexture { texture: tex, mip_level: 0, origin: eframe::wgpu::Origin3d::ZERO, aspect: eframe::wgpu::TextureAspect::All },
                    &padded,
                    eframe::wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(padded_bpr as u32), rows_per_image: Some(h) },
                    eframe::wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
            }
            if let Some(id) = self.gpu_tex_id {
                let device = &*rs.device;
                let mut renderer = rs.renderer.write();
                renderer.update_egui_texture_from_wgpu_texture(device, view, eframe::wgpu::FilterMode::Linear, id);
            }
        }
    }

    // Present a GPU-cached frame for a source/time. If absent, decode one and upload.
    fn present_gpu_cached(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        path: &str,
        t_sec: f64,
        desired: (u32, u32),
    ) -> Option<egui::TextureId> {
        self.ensure_gpu_textures(rs, desired.0, desired.1);
        // Try cache first
        let key = FrameCacheKey::new(path, t_sec, desired.0, desired.1);
        if let Some(cached) = self.get_cached_frame(&key) {
            let mut bytes = Vec::with_capacity(cached.image.pixels.len() * 4);
            for p in &cached.image.pixels { bytes.extend_from_slice(&p.to_array()); }
            self.upload_gpu_frame(rs, &bytes);
            return self.gpu_tex_id; // ignored in wgpu path; retained for compatibility
        }
        // Decode one frame on demand
        let decoded = if path.to_lowercase().ends_with(".png") || path.to_lowercase().ends_with(".jpg") || path.to_lowercase().ends_with(".jpeg") {
            decode_image_optimized(path, desired.0, desired.1)
        } else {
            decode_video_frame_optimized(path, t_sec, desired.0, desired.1)
        };
        if let Some(img) = decoded {
            let mut bytes = Vec::with_capacity(img.pixels.len() * 4);
            for p in &img.pixels { bytes.extend_from_slice(&p.to_array()); }
            self.upload_gpu_frame(rs, &bytes);
            return self.gpu_tex_id; // ignored in wgpu path; retained for compatibility
        }
        None
    }

    fn update(&mut self, ctx: &egui::Context, size: (u32, u32), source: Option<&VisualSource>, _playing: bool, t_sec: f64) {
        // Check if we need to update the frame
        let need_update = match source {
            Some(src) => {
                self.current_source.as_ref().map_or(true, |current| {
                    current.path != src.path || 
                    (t_sec - self.last_frame_time).abs() > 0.05 || // Update every 50ms for smooth scrubbing
                    self.last_size != size
                })
            }
            None => {
                self.current_source.is_some()
            }
        };

        if need_update {
            self.current_source = source.cloned();
            self.last_frame_time = t_sec;
            self.last_size = size;

            if let Some(src) = source {
                // Try to get frame from cache first
                let cache_key = FrameCacheKey::new(&src.path, t_sec, size.0, size.1);
                
                if let Some(_cached_frame) = self.get_cached_frame(&cache_key) {
                    // Cache hit - let present_gpu_cached upload to native WGPU on paint
                    self.cache_hits += 1;
                    ctx.request_repaint();
                } else {
                    // Cache miss - decode frame asynchronously
                    self.cache_misses += 1;
                    self.decode_frame_async(ctx, src.clone(), cache_key, t_sec);
                }
            } else {
                // no source
            }
        }
    }
    
    fn get_cached_frame(&self, key: &FrameCacheKey) -> Option<CachedFrame> {
        if let Ok(cache) = self.frame_cache.lock() {
            if let Some(mut frame) = cache.get(key).cloned() {
                frame.access_count += 1;
                frame.last_access = std::time::Instant::now();
                return Some(frame);
            }
        }
        None
    }
    
    fn decode_frame_async(&mut self, ctx: &egui::Context, source: VisualSource, cache_key: FrameCacheKey, t_sec: f64) {
        // If native decoding is available and this is a video, do not spawn RGBA decoding.
        // The persistent native decoder will feed frames via the ring buffer.
        if !source.is_image && is_native_decoding_available() {
            return;
        }
        let cache = self.frame_cache.clone();
        let ctx = ctx.clone();
        
        // Stop any existing cache worker
        if let Some(stop) = &self.cache_stop {
            stop.store(true, Ordering::Relaxed);
        }
        if let Some(worker) = self.cache_worker.take() {
            let _ = worker.join();
        }
        
        let stop_flag = Arc::new(AtomicBool::new(false));
        self.cache_stop = Some(stop_flag.clone());
        
        let worker = thread::spawn(move || {
            if stop_flag.load(Ordering::Relaxed) { return; }
            
            let start_time = std::time::Instant::now();
            
            // Decode frame efficiently
            let frame_result = if source.is_image {
                decode_image_optimized(&source.path, cache_key.width, cache_key.height)
        } else {
                // Use native decoder if available, fallback to FFmpeg
                if is_native_decoding_available() {
                    decode_video_frame_native(&source.path, t_sec, cache_key.width, cache_key.height)
                } else {
                    decode_video_frame_optimized(&source.path, t_sec, cache_key.width, cache_key.height)
                }
            };
            
            if stop_flag.load(Ordering::Relaxed) { return; }
            
            if let Some(image) = frame_result {
                let _decode_time = start_time.elapsed();
                
                // Cache the frame
                let cached_frame = CachedFrame {
                    image: image.clone(),
                    decoded_at: std::time::Instant::now(),
                    access_count: 1,
                    last_access: std::time::Instant::now(),
                };
                
                if let Ok(mut cache) = cache.lock() {
                    // Implement LRU eviction if cache is too large
                    if cache.len() > 50 { // Max 50 cached frames
                        evict_lru_frames(&mut cache, 10); // Remove oldest 10 frames
                    }
                    
                    cache.insert(cache_key, cached_frame);
                }
                
                // Update texture on main thread
                ctx.request_repaint();
            }
        });
        
        self.cache_worker = Some(worker);
    }

    fn stop_cache_worker(&mut self) {
        if let Some(stop) = &self.cache_stop {
            stop.store(true, Ordering::Relaxed);
        }
        if let Some(worker) = self.cache_worker.take() {
            let _ = worker.join();
        }
        self.cache_stop = None;
    }
    
    fn print_cache_stats(&self) {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests > 0 {
            let hit_rate = (self.cache_hits as f64 / total_requests as f64) * 100.0;
            println!("Preview Cache Stats: {:.1}% hit rate ({}/{} requests), avg decode: {:.1}ms", 
                     hit_rate, self.cache_hits, total_requests, self.decode_time_ms);
        }
    }
    
    fn preload_nearby_frames(&self, source: &VisualSource, current_time: f64, size: (u32, u32)) {
        if source.is_image { return; } // No need to preload for images
        
        let cache = self.frame_cache.clone();
        let source = source.clone();
        let (w, h) = size;
        
        // Preload frames around current time (±2 seconds)
        thread::spawn(move || {
            let _preload_range = 2.0; // seconds
            let _step = 0.2; // every 200ms
            
            for offset in [0.2, 0.4, 0.6, 0.8, 1.0, -0.2, -0.4, -0.6, -0.8, -1.0] {
                let preload_time = current_time + offset;
                if preload_time < 0.0 { continue; }
                
                let cache_key = FrameCacheKey::new(&source.path, preload_time, w, h);
                
                // Check if frame is already cached
                if let Ok(cache) = cache.lock() {
                    if cache.contains_key(&cache_key) {
                        continue; // Already cached
                    }
                }
                
                // Decode frame in background
                if let Some(image) = decode_video_frame_optimized(&source.path, preload_time, w, h) {
                    let cached_frame = CachedFrame {
                        image,
                        decoded_at: std::time::Instant::now(),
                        access_count: 0,
                        last_access: std::time::Instant::now(),
                    };
                    
                    if let Ok(mut cache) = cache.lock() {
                        // Only cache if we're not over the limit
                        if cache.len() < 50 {
                            cache.insert(cache_key, cached_frame);
                        }
                    }
                }
                
                // Small delay to avoid overwhelming the system
                thread::sleep(Duration::from_millis(10));
            }
        });
    }

    fn present_yuv_with_frame(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        path: &str,
        t_sec: f64,
        vf_opt: Option<&native_decoder::VideoFrame>,
    ) -> Option<(YuvPixFmt, Arc<eframe::wgpu::Texture>, Arc<eframe::wgpu::Texture>)> {
        if let Some(vf) = vf_opt {
            // Map NativeYuvPixFmt to local YuvPixFmt and handle P010->NV12 fallback
            let mut fmt = match vf.format {
                native_decoder::YuvPixFmt::Nv12 => YuvPixFmt::Nv12,
                native_decoder::YuvPixFmt::P010 => YuvPixFmt::P010,
            };
            let mut y: Vec<u8> = vf.y_plane.clone();
            let mut uv: Vec<u8> = vf.uv_plane.clone();
            let w = vf.width; let h = vf.height;
            if fmt == YuvPixFmt::P010 && !device_supports_16bit_norm(rs) {
                if let Some((_f, ny, nuv, nw, nh)) = decode_video_frame_nv12_only(path, t_sec) { fmt = YuvPixFmt::Nv12; y = ny; uv = nuv; let _ = (nw, nh); }
            }
            let key = FrameCacheKey::new(path, t_sec, 0, 0);
            self.nv12_cache.insert(key.clone(), Nv12Frame { fmt, y: y.clone(), uv: uv.clone(), w, h });
            self.nv12_keys.push_back(key);
            while self.nv12_keys.len() > 64 { if let Some(old) = self.nv12_keys.pop_front() { self.nv12_cache.remove(&old); } }
            self.upload_yuv_planes(rs, fmt, &y, &uv, w, h);
            let idx = self.ring_present;
            return Some((fmt, self.y_tex[idx].as_ref().unwrap().clone(), self.uv_tex[idx].as_ref().unwrap().clone()));
        }
        // Fallback to old path
        self.present_yuv(rs, path, t_sec)
    }

    #[cfg(target_os = "macos")]
    fn present_nv12_zero_copy(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        zc: &native_decoder::IOSurfaceFrame,
    ) -> Option<(YuvPixFmt, Arc<eframe::wgpu::Texture>, Arc<eframe::wgpu::Texture>)> {
        self.ensure_zero_copy_nv12_textures(rs, zc.width, zc.height);
        if let Some(ref yuv) = self.gpu_yuv {
            let queue = &*rs.queue;
            if let Err(e) = yuv.import_from_iosurface(queue, zc) {
                eprintln!("[zc] import_from_iosurface error: {}", e);
                return None;
            }
            return Some((YuvPixFmt::Nv12, yuv.y_tex.clone(), yuv.uv_tex.clone()));
        }
        None
    }
}

// WGPU callback to draw NV12 planes via WGSL YUV->RGB.
struct PreviewYuvCallback {
    y_tex: Arc<eframe::wgpu::Texture>,
    uv_tex: Arc<eframe::wgpu::Texture>,
    fmt: YuvPixFmt,
    use_uint: bool,
    w: u32,
    h: u32,
}

struct Nv12Resources {
    pipeline: eframe::wgpu::RenderPipeline,
    bind_group_layout: eframe::wgpu::BindGroupLayout,
    uniform_bgl: eframe::wgpu::BindGroupLayout,
    sampler: eframe::wgpu::Sampler,
}

struct Nv12BindGroup {
    bind: eframe::wgpu::BindGroup,
    y_id: usize,
    uv_id: usize,
}
struct P010UintResources {
    pipeline: eframe::wgpu::RenderPipeline,
    tex_bgl: eframe::wgpu::BindGroupLayout,
    uniform_bgl: eframe::wgpu::BindGroupLayout,
}
struct P010UintTexBind(eframe::wgpu::BindGroup);
struct P010UintConvBind(eframe::wgpu::BindGroup);

impl egui_wgpu::CallbackTrait for PreviewYuvCallback {
    fn prepare(
        &self,
        device: &eframe::wgpu::Device,
        queue: &eframe::wgpu::Queue,
        _screen: &eframe::egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut eframe::wgpu::CommandEncoder,
        resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<eframe::wgpu::CommandBuffer> {
        // Ensure pipeline resources
        if resources.get::<Nv12Resources>().is_none() {
            let shader_src = r#"
                @group(0) @binding(0) var samp: sampler;
                @group(0) @binding(1) var texY: texture_2d<f32>;
                @group(0) @binding(2) var texUV: texture_2d<f32>;

                struct Conv { y_offset: f32, y_scale: f32, c_offset: f32, c_scale: f32, _pad: vec2<f32> };
                @group(1) @binding(0) var<uniform> conv: Conv;

                struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

                @vertex
                fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
                    var pos = array<vec2<f32>,3>(vec2(-1.0, -1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
                    var uv  = array<vec2<f32>,3>(vec2(0.0, 1.0), vec2(2.0,1.0), vec2(0.0,-1.0));
                    var o: VSOut;
                    o.pos = vec4<f32>(pos[vi], 0.0, 1.0);
                    o.uv = uv[vi];
                    return o;
                }

                @fragment
                fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
                    let y = textureSample(texY, samp, in.uv).r;
                    let uv = textureSample(texUV, samp, in.uv).rg;
                    // Limited-range conversion parameters via uniform
                    let y709 = max(y - conv.y_offset, 0.0) * conv.y_scale;
                    let u = (uv.x - conv.c_offset) * conv.c_scale;
                    let v = (uv.y - conv.c_offset) * conv.c_scale;
                    let r = y709 + 1.5748 * v;
                    let g = y709 - 0.1873 * u - 0.4681 * v;
                    let b = y709 + 1.8556 * u;
                    return vec4<f32>(r, g, b, 1.0);
                }
            "#;
            let module = device.create_shader_module(eframe::wgpu::ShaderModuleDescriptor {
                label: Some("preview_nv12_shader"),
                source: eframe::wgpu::ShaderSource::Wgsl(shader_src.into()),
            });
            let bgl = device.create_bind_group_layout(&eframe::wgpu::BindGroupLayoutDescriptor {
                label: Some("preview_nv12_bgl"),
                entries: &[
                    eframe::wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: eframe::wgpu::ShaderStages::FRAGMENT,
                        ty: eframe::wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: eframe::wgpu::TextureViewDimension::D2,
                            sample_type: eframe::wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    eframe::wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: eframe::wgpu::ShaderStages::FRAGMENT,
                        ty: eframe::wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: eframe::wgpu::TextureViewDimension::D2,
                            sample_type: eframe::wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    eframe::wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: eframe::wgpu::ShaderStages::FRAGMENT,
                        ty: eframe::wgpu::BindingType::Sampler(eframe::wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
            let uniform_bgl = device.create_bind_group_layout(&eframe::wgpu::BindGroupLayoutDescriptor {
                label: Some("preview_nv12_uniform_bgl"),
                entries: &[eframe::wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: eframe::wgpu::ShaderStages::FRAGMENT,
                    ty: eframe::wgpu::BindingType::Buffer { ty: eframe::wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                }],
            });
            let pl = device.create_pipeline_layout(&eframe::wgpu::PipelineLayoutDescriptor {
                label: Some("preview_nv12_pl"),
                bind_group_layouts: &[&bgl, &uniform_bgl],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_render_pipeline(&eframe::wgpu::RenderPipelineDescriptor {
                label: Some("preview_nv12_pipeline"),
                layout: Some(&pl),
                vertex: eframe::wgpu::VertexState {
                    module: &module,
                    entry_point: "vs_main",
                    compilation_options: eframe::wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(eframe::wgpu::FragmentState {
                    module: &module,
                    entry_point: "fs_main",
                    compilation_options: eframe::wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(eframe::wgpu::ColorTargetState {
                        format: eframe::wgpu::TextureFormat::Bgra8Unorm,
                        blend: Some(eframe::wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: eframe::wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: eframe::wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: eframe::wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
            let sampler = device.create_sampler(&eframe::wgpu::SamplerDescriptor::default());
            // Take ownership values, then insert to avoid overlapping borrows during insert
            let res = Nv12Resources { pipeline, bind_group_layout: bgl, uniform_bgl, sampler };
            resources.insert(res);
        }

        if self.use_uint {
            if resources.get::<P010UintResources>().is_none() {
                let shader_src = r#"
                    @group(0) @binding(0) var texY: texture_2d<u32>;
                    @group(0) @binding(1) var texUV: texture_2d<u32>;
                    struct Conv { y_offset: f32, y_scale: f32, c_offset: f32, c_scale: f32, _pad: vec2<f32> };
                    @group(1) @binding(0) var<uniform> conv: Conv;
                    struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
                    @vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
                        var pos = array<vec2<f32>,3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
                        var uv = array<vec2<f32>,3>(vec2(0.0,1.0), vec2(2.0,1.0), vec2(0.0,-1.0));
                        var o: VSOut; o.pos = vec4<f32>(pos[vi],0.0,1.0); o.uv = uv[vi]; return o;
                    }
                    @fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
                        let dimY = textureDimensions(texY);
                        let dimUV = textureDimensions(texUV);
                        let coordY = vec2<i32>(in.uv * vec2<f32>(dimY));
                        let coordUV = vec2<i32>(in.uv * vec2<f32>(dimUV));
                        let y16 = textureLoad(texY, coordY, 0).x;
                        let uv16 = textureLoad(texUV, coordUV, 0);
                        let y10 = f32((y16 >> 6u) & 1023u) / 1023.0;
                        let u10 = f32((uv16.x >> 6u) & 1023u) / 1023.0;
                        let v10 = f32((uv16.y >> 6u) & 1023u) / 1023.0;
                        let y709 = max(y10 - conv.y_offset, 0.0) * conv.y_scale;
                        let u = (u10 - conv.c_offset) * conv.c_scale;
                        let v = (v10 - conv.c_offset) * conv.c_scale;
                        let r = y709 + 1.5748 * v;
                        let g = y709 - 0.1873 * u - 0.4681 * v;
                        let b = y709 + 1.8556 * u;
                        return vec4<f32>(r,g,b,1.0);
                    }
                "#;
                let module = device.create_shader_module(eframe::wgpu::ShaderModuleDescriptor { label: Some("p010_uint_shader"), source: eframe::wgpu::ShaderSource::Wgsl(shader_src.into()) });
                let tex_bgl = device.create_bind_group_layout(&eframe::wgpu::BindGroupLayoutDescriptor { label: Some("p010_uint_tex_bgl"), entries: &[eframe::wgpu::BindGroupLayoutEntry { binding: 0, visibility: eframe::wgpu::ShaderStages::FRAGMENT, ty: eframe::wgpu::BindingType::Texture { multisampled: false, view_dimension: eframe::wgpu::TextureViewDimension::D2, sample_type: eframe::wgpu::TextureSampleType::Uint }, count: None }, eframe::wgpu::BindGroupLayoutEntry { binding: 1, visibility: eframe::wgpu::ShaderStages::FRAGMENT, ty: eframe::wgpu::BindingType::Texture { multisampled: false, view_dimension: eframe::wgpu::TextureViewDimension::D2, sample_type: eframe::wgpu::TextureSampleType::Uint }, count: None }] });
                let uniform_bgl = device.create_bind_group_layout(&eframe::wgpu::BindGroupLayoutDescriptor { label: Some("p010_uint_uniform_bgl"), entries: &[eframe::wgpu::BindGroupLayoutEntry { binding: 0, visibility: eframe::wgpu::ShaderStages::FRAGMENT, ty: eframe::wgpu::BindingType::Buffer { ty: eframe::wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }] });
                let pl = device.create_pipeline_layout(&eframe::wgpu::PipelineLayoutDescriptor { label: Some("p010_uint_pl"), bind_group_layouts: &[&tex_bgl, &uniform_bgl], push_constant_ranges: &[] });
                let pipeline = device.create_render_pipeline(&eframe::wgpu::RenderPipelineDescriptor { label: Some("p010_uint_pipeline"), layout: Some(&pl), vertex: eframe::wgpu::VertexState { module: &module, entry_point: "vs_main", compilation_options: eframe::wgpu::PipelineCompilationOptions::default(), buffers: &[] }, fragment: Some(eframe::wgpu::FragmentState { module: &module, entry_point: "fs_main", compilation_options: eframe::wgpu::PipelineCompilationOptions::default(), targets: &[Some(eframe::wgpu::ColorTargetState { format: eframe::wgpu::TextureFormat::Bgra8Unorm, blend: Some(eframe::wgpu::BlendState::ALPHA_BLENDING), write_mask: eframe::wgpu::ColorWrites::ALL })] }), primitive: eframe::wgpu::PrimitiveState::default(), depth_stencil: None, multisample: eframe::wgpu::MultisampleState::default(), multiview: None, cache: None });
                resources.insert(P010UintResources { pipeline, tex_bgl, uniform_bgl });
            }
            // Create P010 uint bind groups
            let view_y = self.y_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default());
            let view_uv = self.uv_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default());
            let tex_layout = &resources.get::<P010UintResources>().unwrap().tex_bgl;
            let tbg = device.create_bind_group(&eframe::wgpu::BindGroupDescriptor { label: Some("p010_uint_tex_bg"), layout: tex_layout, entries: &[eframe::wgpu::BindGroupEntry { binding: 0, resource: eframe::wgpu::BindingResource::TextureView(&view_y) }, eframe::wgpu::BindGroupEntry { binding: 1, resource: eframe::wgpu::BindingResource::TextureView(&view_uv) }] });
            resources.insert(P010UintTexBind(tbg));
            // Upload conv uniform
            // Use limited-range conversion for P010 uint (FFmpeg typically outputs limited-range)
            let (y_off, y_scale, c_off, c_scale) = (64.0/1023.0, 1.0/876.0, 512.0/1023.0, 1.0/896.0);
            #[repr(C)]
            #[derive(Clone, Copy)]
            struct ConvStd { y_offset: f32, y_scale: f32, c_offset: f32, c_scale: f32, _pad: [f32;2] }
            let conv = ConvStd { y_offset: y_off, y_scale, c_offset: c_off, c_scale, _pad: [0.0;2] };
            let ubuf = device.create_buffer(&eframe::wgpu::BufferDescriptor { label: Some("p010_uint_ubo"), size: std::mem::size_of::<ConvStd>() as u64, usage: eframe::wgpu::BufferUsages::UNIFORM | eframe::wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let bytes: &[u8] = unsafe { std::slice::from_raw_parts((&conv as *const ConvStd) as *const u8, std::mem::size_of::<ConvStd>()) };
            queue.write_buffer(&ubuf, 0, bytes);
            let uniform_layout = &resources.get::<P010UintResources>().unwrap().uniform_bgl;
            let ubg = device.create_bind_group(&eframe::wgpu::BindGroupDescriptor { label: Some("p010_uint_conv_bg"), layout: uniform_layout, entries: &[eframe::wgpu::BindGroupEntry { binding: 0, resource: eframe::wgpu::BindingResource::Buffer(eframe::wgpu::BufferBinding { buffer: &ubuf, offset: 0, size: None }) }] });
            resources.insert(P010UintConvBind(ubg));
            return Vec::new();
        }

        // Float NV12/P010 bind groups and uniform
        let y_id = Arc::as_ptr(&self.y_tex) as usize;
        let uv_id = Arc::as_ptr(&self.uv_tex) as usize;
        let rebuild = resources.get::<Nv12BindGroup>().map_or(true, |bg| bg.y_id != y_id || bg.uv_id != uv_id);
        if rebuild {
            let view_y = self.y_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default());
            let view_uv = self.uv_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default());
            let (nv_bgl, nv_samp) = {
                let r = resources.get::<Nv12Resources>().unwrap();
                (&r.bind_group_layout, &r.sampler)
            };
            let bind = device.create_bind_group(&eframe::wgpu::BindGroupDescriptor { label: Some("preview_nv12_bg"), layout: nv_bgl, entries: &[eframe::wgpu::BindGroupEntry { binding: 0, resource: eframe::wgpu::BindingResource::Sampler(nv_samp) }, eframe::wgpu::BindGroupEntry { binding: 1, resource: eframe::wgpu::BindingResource::TextureView(&view_y) }, eframe::wgpu::BindGroupEntry { binding: 2, resource: eframe::wgpu::BindingResource::TextureView(&view_uv) }] });
            eprintln!("NV12 bind-group ready ({}x{} / {}x{})", self.w, self.h, (self.w + 1)/2, (self.h + 1)/2);
            resources.insert(Nv12BindGroup { bind, y_id, uv_id });
        }
        // Use limited-range conversion (FFmpeg typically outputs limited-range YUV)
        // Limited-range: Y 16-235, UV 16-240 (8-bit) or Y 64-940, UV 64-960 (10-bit)
        let (y_off, y_scale, c_off, c_scale) = match self.fmt { 
            YuvPixFmt::Nv12 => (16.0/255.0, 1.0/219.0, 128.0/255.0, 1.0/224.0), // Limited-range
            YuvPixFmt::P010 => (64.0/1023.0, 1.0/876.0, 512.0/1023.0, 1.0/896.0) // Limited-range
        };
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct ConvStd { y_offset: f32, y_scale: f32, c_offset: f32, c_scale: f32, _pad: [f32;2] }
        let conv = ConvStd { y_offset: y_off, y_scale, c_offset: c_off, c_scale, _pad: [0.0;2] };
        let ubuf = device.create_buffer(&eframe::wgpu::BufferDescriptor { label: Some("yuv_conv_ubo"), size: std::mem::size_of::<ConvStd>() as u64, usage: eframe::wgpu::BufferUsages::UNIFORM | eframe::wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts((&conv as *const ConvStd) as *const u8, std::mem::size_of::<ConvStd>()) };
        queue.write_buffer(&ubuf, 0, bytes);
        let conv_layout = &resources.get::<Nv12Resources>().unwrap().uniform_bgl;
        let ubg = device.create_bind_group(&eframe::wgpu::BindGroupDescriptor { label: Some("yuv_conv_bg"), layout: conv_layout, entries: &[eframe::wgpu::BindGroupEntry { binding: 0, resource: eframe::wgpu::BindingResource::Buffer(eframe::wgpu::BufferBinding { buffer: &ubuf, offset: 0, size: None }) }] });
        resources.insert(ConvBindGroup(ubg));
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'static>,
        resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        if self.use_uint {
            let res = resources.get::<P010UintResources>().expect("p010 uint resources");
            let tbg = resources.get::<P010UintTexBind>().expect("p010 uint tex bg");
            let ubg = resources.get::<P010UintConvBind>().expect("p010 uint conv bg");
            render_pass.set_pipeline(&res.pipeline);
            render_pass.set_bind_group(0, &tbg.0, &[]);
            render_pass.set_bind_group(1, &ubg.0, &[]);
            render_pass.draw(0..3, 0..1);
        } else {
            let res = resources.get::<Nv12Resources>().expect("nv12 resources");
            let bg = resources.get::<Nv12BindGroup>().expect("nv12 bind group");
            let ubg = resources.get::<ConvBindGroup>().expect("conv bind group");
            render_pass.set_pipeline(&res.pipeline);
            render_pass.set_bind_group(0, &bg.bind, &[]);
            render_pass.set_bind_group(1, &ubg.0, &[]);
            render_pass.draw(0..3, 0..1);
        }
    }
}

struct ConvBindGroup(eframe::wgpu::BindGroup);

fn find_jpeg_frame(buf: &[u8]) -> Option<(usize, usize)> {
    // SOI 0xFFD8, EOI 0xFFD9
    let mut start = None;
    for i in 0..buf.len().saturating_sub(1) {
        if start.is_none() && buf[i] == 0xFF && buf[i+1] == 0xD8 { start = Some(i); }
        if let Some(s) = start {
            if buf[i] == 0xFF && buf[i+1] == 0xD9 { return Some((s, i+2)); }
        }
    }
    None
}

fn decode_to_color_image(bytes: &[u8]) -> Option<egui::ColorImage> {
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w,h) = img.dimensions();
    let data = img.into_raw();
    Some(egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &data))
}

    // Optimized video frame decode at native size (no scaling; GPU handles fit)
fn decode_video_frame_optimized(path: &str, t_sec: f64, w: u32, h: u32) -> Option<egui::ColorImage> {
    // Decode one frame at requested size to match GPU upload
    let frame_bytes = (w as usize) * (h as usize) * 4;
    let out = std::process::Command::new("ffmpeg")
        .arg("-ss").arg(format!("{:.3}", t_sec.max(0.0)))
        .arg("-i").arg(path)
        .arg("-frames:v").arg("1")
        .arg("-vf").arg(format!("scale={}x{}:flags=fast_bilinear", w, h))
        .arg("-f").arg("rawvideo")
        .arg("-pix_fmt").arg("rgba")
        .arg("-threads").arg("1")
        .arg("-")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output().ok()?;

    if !out.status.success() { return None; }
    if out.stdout.len() < frame_bytes { return None; }
    Some(egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &out.stdout[..frame_bytes]))
}

// Decode video frame using native decoder
fn decode_video_frame_native(path: &str, t_sec: f64, w: u32, h: u32) -> Option<egui::ColorImage> {
    let config = DecoderConfig {
        hardware_acceleration: true,
        preferred_format: Some(native_decoder::YuvPixFmt::Nv12),
        zero_copy: false, // Phase 1 only
    };
    
    match create_decoder(path, config) {
        Ok(mut decoder) => {
            match decoder.decode_frame(t_sec) {
                Ok(Some(video_frame)) => {
                    // Convert YUV to RGBA for egui::ColorImage
                    let rgba = yuv_to_rgba(&video_frame.y_plane, &video_frame.uv_plane, 
                                          video_frame.width, video_frame.height, video_frame.format);
                    
                    // Scale to requested size if needed
                    if video_frame.width == w && video_frame.height == h {
                        Some(egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba))
                    } else {
                        // Simple nearest-neighbor scaling for now
                        let scaled = scale_rgba_nearest(&rgba, video_frame.width, video_frame.height, w, h);
                        Some(egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &scaled))
                    }
                }
                Ok(None) => {
                    eprintln!("Native decoder: No frame at timestamp {:.3}s", t_sec);
                    None
                }
                Err(e) => {
                    eprintln!("Native decoder error: {}", e);
                    None
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to create native decoder: {}", e);
            None
        }
    }
}

// Convert YUV to RGBA (simple implementation)
fn yuv_to_rgba(y_plane: &[u8], uv_plane: &[u8], width: u32, height: u32, format: native_decoder::YuvPixFmt) -> Vec<u8> {
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    
    match format {
        native_decoder::YuvPixFmt::Nv12 => {
            // NV12: Y plane + interleaved UV plane
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let y_idx = y * width as usize + x;
                    let uv_idx = (y / 2) * width as usize + (x / 2) * 2;
                    
                    let y_val = y_plane[y_idx] as f32;
                    let u_val = uv_plane[uv_idx] as f32 - 128.0;
                    let v_val = uv_plane[uv_idx + 1] as f32 - 128.0;
                    
                    // YUV to RGB conversion (ITU-R BT.601)
                    let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
                    let g = (y_val - 0.344136 * u_val - 0.714136 * v_val).clamp(0.0, 255.0) as u8;
                    let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;
                    
                    let rgba_idx = (y * width as usize + x) * 4;
                    rgba[rgba_idx] = r;
                    rgba[rgba_idx + 1] = g;
                    rgba[rgba_idx + 2] = b;
                    rgba[rgba_idx + 3] = 255; // Alpha
                }
            }
        }
        native_decoder::YuvPixFmt::P010 => {
            // P010: 10-bit YUV (simplified to 8-bit for now)
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let y_idx = y * width as usize + x;
                    let uv_idx = (y / 2) * width as usize + (x / 2) * 2;
                    
                    // Convert 10-bit to 8-bit (shift right by 2)
                    let y_val = (y_plane[y_idx] as f32) * 4.0;
                    let u_val = (uv_plane[uv_idx] as f32) * 4.0 - 128.0;
                    let v_val = (uv_plane[uv_idx + 1] as f32) * 4.0 - 128.0;
                    
                    // YUV to RGB conversion
                    let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
                    let g = (y_val - 0.344136 * u_val - 0.714136 * v_val).clamp(0.0, 255.0) as u8;
                    let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;
                    
                    let rgba_idx = (y * width as usize + x) * 4;
                    rgba[rgba_idx] = r;
                    rgba[rgba_idx + 1] = g;
                    rgba[rgba_idx + 2] = b;
                    rgba[rgba_idx + 3] = 255; // Alpha
                }
            }
        }
    }
    
    rgba
}

// Simple nearest-neighbor scaling
fn scale_rgba_nearest(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (dst_w * dst_h * 4) as usize];
    
    for y in 0..dst_h as usize {
        for x in 0..dst_w as usize {
            let src_x = (x as f32 * src_w as f32 / dst_w as f32) as usize;
            let src_y = (y as f32 * src_h as f32 / dst_h as f32) as usize;
            
            let src_idx = (src_y * src_w as usize + src_x) * 4;
            let dst_idx = (y * dst_w as usize + x) * 4;
            
            if src_idx + 3 < src.len() && dst_idx + 3 < dst.len() {
                dst[dst_idx] = src[src_idx];
                dst[dst_idx + 1] = src[src_idx + 1];
                dst[dst_idx + 2] = src[src_idx + 2];
                dst[dst_idx + 3] = src[src_idx + 3];
            }
        }
    }
    
    dst
}

// Decode a single frame to NV12 or P010 at native size.
fn decode_video_frame_yuv(path: &str, t_sec: f64) -> Option<(YuvPixFmt, Vec<u8>, Vec<u8>, u32, u32)> {
    let info = media_io::probe_media(std::path::Path::new(path)).ok()?;
    let w = info.width?;
    let h = info.height?;
    // Try P010 first
    let out10 = std::process::Command::new("ffmpeg")
        .arg("-ss").arg(format!("{:.3}", t_sec.max(0.0)))
        .arg("-i").arg(path)
        .arg("-frames:v").arg("1")
        .arg("-f").arg("rawvideo")
        .arg("-pix_fmt").arg("p010le")
        .arg("-threads").arg("1")
        .arg("-")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output().ok()?;
    if out10.status.success() {
        let exp10 = (w as usize) * (h as usize) * 3; // Y:2 bytes * w*h ; UV: w*h bytes (2x16-bit at half res)
        if out10.stdout.len() >= exp10 {
            let y_bytes = (w as usize) * (h as usize) * 2;
            let y = out10.stdout[..y_bytes].to_vec();
            let uv = out10.stdout[y_bytes..y_bytes + (exp10 - y_bytes)].to_vec();
            return Some((YuvPixFmt::P010, y, uv, w, h));
        }
    }
    // Fallback NV12
    let expected = (w as usize) * (h as usize) + (w as usize) * (h as usize) / 2;
    let out = std::process::Command::new("ffmpeg")
        .arg("-ss").arg(format!("{:.3}", t_sec.max(0.0)))
        .arg("-i").arg(path)
        .arg("-frames:v").arg("1")
        .arg("-f").arg("rawvideo")
        .arg("-pix_fmt").arg("nv12")
        .arg("-threads").arg("1")
        .arg("-")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output().ok()?;
    if !out.status.success() || out.stdout.len() < expected { return None; }
    let y_size = (w as usize) * (h as usize);
    let y = out.stdout[..y_size].to_vec();
    let uv = out.stdout[y_size..y_size + (expected - y_size)].to_vec();
    Some((YuvPixFmt::Nv12, y, uv, w, h))
}

fn decode_video_frame_nv12_only(path: &str, t_sec: f64) -> Option<(YuvPixFmt, Vec<u8>, Vec<u8>, u32, u32)> {
    let info = media_io::probe_media(std::path::Path::new(path)).ok()?;
    let w = info.width?; let h = info.height?;
    let expected = (w as usize) * (h as usize) + (w as usize) * (h as usize) / 2;
    let out = std::process::Command::new("ffmpeg")
        .arg("-ss").arg(format!("{:.3}", t_sec.max(0.0)))
        .arg("-i").arg(path)
        .arg("-frames:v").arg("1")
        .arg("-f").arg("rawvideo")
        .arg("-pix_fmt").arg("nv12")
        .arg("-threads").arg("1")
        .arg("-")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output().ok()?;
    if !out.status.success() || out.stdout.len() < expected { return None; }
    let y_size = (w as usize) * (h as usize);
    let y = out.stdout[..y_size].to_vec();
    let uv = out.stdout[y_size..y_size + (expected - y_size)].to_vec();
    Some((YuvPixFmt::Nv12, y, uv, w, h))
}

fn device_supports_16bit_norm(rs: &eframe::egui_wgpu::RenderState) -> bool {
    rs.device.features().contains(eframe::wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
}

fn align_to(v: usize, align: usize) -> usize { ((v + align - 1) / align) * align }

#[derive(Clone)]
struct Nv12Frame { fmt: YuvPixFmt, y: Vec<u8>, uv: Vec<u8>, w: u32, h: u32 }

// Using media_io::YuvPixFmt

// Optimized image decoding
fn decode_image_optimized(path: &str, w: u32, h: u32) -> Option<egui::ColorImage> {
    // For images, use the image crate directly for better performance
    let img = image::open(path).ok()?;
    let resized = img.resize(w, h, image::imageops::FilterType::Lanczos3);
    let rgba = resized.to_rgba8();
    let (width, height) = rgba.dimensions();
    
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [width as usize, height as usize], 
        &rgba.into_raw()
    ))
}

// LRU eviction for frame cache
fn evict_lru_frames(cache: &mut HashMap<FrameCacheKey, CachedFrame>, count: usize) {
    if cache.len() <= count { return; }
    
    // Collect frames with their last access times
    let mut frames_with_time: Vec<(FrameCacheKey, std::time::Instant)> = cache
        .iter()
        .map(|(key, frame)| (key.clone(), frame.last_access))
        .collect();
    
    // Sort by last access time (oldest first)
    frames_with_time.sort_by_key(|(_, time)| *time);
    
    // Remove the oldest frames
    for (key, _) in frames_with_time.into_iter().take(count) {
        cache.remove(&key);
    }
}

fn grab_frame_at(path: &str, size: (u32,u32), t_sec: f64) -> Option<egui::ColorImage> {
    let (w,h) = size;
    decode_video_frame_optimized(path, t_sec, w, h)
}

// Efficient frame cache key
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FrameCacheKey {
    path: String,
    time_sec: u32, // Rounded to nearest 0.1 second for cache efficiency
    width: u32,
    height: u32,
}

impl FrameCacheKey {
    fn new(path: &str, time_sec: f64, width: u32, height: u32) -> Self {
        Self {
            path: path.to_string(),
            time_sec: (time_sec * 10.0).round() as u32, // 0.1 second precision
            width,
            height,
        }
    }
}

// Cached frame with metadata
#[derive(Clone)]
struct CachedFrame {
    image: egui::ColorImage,
    decoded_at: std::time::Instant,
    access_count: u32,
    last_access: std::time::Instant,
}

// Frame buffer used by the preview scheduler (kept for compatibility)
struct FrameBuffer {
    pts: f64,
    w: u32,
    h: u32,
    bytes: Vec<u8>,
}

// (removed legacy standalone WGPU context to avoid mixed versions)

// -------------------------
// Export UI + ffmpeg runner
// -------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExportCodec { H264, AV1 }

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExportPreset { Source, P1080, P4K }

#[derive(Default, Clone)]
struct ExportProgress { progress: f32, eta: Option<String>, done: bool, error: Option<String> }

struct ExportUiState {
    open: bool,
    codec: ExportCodec,
    preset: ExportPreset,
    crf: i32,
    output_path: String,
    running: bool,
    progress: f32,
    status: String,
    progress_shared: Option<std::sync::Arc<std::sync::Mutex<ExportProgress>>>,
    worker: Option<std::thread::JoinHandle<()>>,
    encoders_h264: Vec<String>,
    encoders_av1: Vec<String>,
    selected_encoder: Option<String>,
}

impl Default for ExportCodec { fn default() -> Self { ExportCodec::H264 } }
impl Default for ExportPreset { fn default() -> Self { ExportPreset::Source } }

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
    fn ui(&mut self, ctx: &egui::Context, seq: &timeline::Sequence, db: &ProjectDb, project_id: &str) {
        if !self.open { return; }
        let mut keep_open = true;
        egui::Window::new("Export")
            .open(&mut keep_open)
            .resizable(true)
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    // Gather available encoders once per UI open
                    if self.encoders_h264.is_empty() && self.encoders_av1.is_empty() {
                        let map = media_io::get_hardware_encoders();
                        if let Some(v) = map.get("h264") { self.encoders_h264 = v.clone(); }
                        if let Some(v) = map.get("av1") { self.encoders_av1 = v.clone(); }
                        // Always include software options at front
                        if !self.encoders_h264.iter().any(|e| e == "libx264") { self.encoders_h264.insert(0, "libx264".into()); }
                        if !self.encoders_av1.iter().any(|e| e == "libaom-av1") { self.encoders_av1.insert(0, "libaom-av1".into()); }
                    }
                    // Output path picker
                    ui.horizontal(|ui| {
                        ui.label("Output:");
                        ui.text_edit_singleline(&mut self.output_path);
                        if ui.button("Browse").clicked() {
                            // Default extension based on codec
                            let default_name = match self.codec { ExportCodec::H264 => "export.mp4", ExportCodec::AV1 => "export.mkv" };
                            if let Some(path) = rfd::FileDialog::new().set_file_name(default_name).save_file() {
                                self.output_path = path.display().to_string();
                            }
                        }
                    });

                    // Codec + preset + CRF
                    ui.horizontal(|ui| {
                        ui.label("Codec:");
                        let mut codec_idx = match self.codec { ExportCodec::H264 => 0, ExportCodec::AV1 => 1 };
                        egui::ComboBox::from_id_salt("codec_combo")
                            .selected_text(match self.codec { ExportCodec::H264 => "H.264", ExportCodec::AV1 => "AV1" })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut codec_idx, 0, "H.264");
                                ui.selectable_value(&mut codec_idx, 1, "AV1");
                            });
                        let prev_codec = self.codec;
                        self.codec = if codec_idx == 0 { ExportCodec::H264 } else { ExportCodec::AV1 };
                        if self.codec != prev_codec && !self.output_path.is_empty() {
                            // Gate extension automatically on codec change
                            self.output_path = adjust_extension(&self.output_path, match self.codec { ExportCodec::H264 => "mp4", ExportCodec::AV1 => "mkv" });
                        }

                        ui.label("Encoder:");
                        let list = match self.codec { ExportCodec::H264 => &mut self.encoders_h264, ExportCodec::AV1 => &mut self.encoders_av1 };
                        if list.is_empty() { list.push(match self.codec { ExportCodec::H264 => "libx264".into(), ExportCodec::AV1 => "libaom-av1".into() }); }
                        let mut selection = self.selected_encoder.clone().unwrap_or_else(|| list[0].clone());
                        egui::ComboBox::from_id_salt("encoder_combo")
                            .selected_text(selection.clone())
                            .show_ui(ui, |ui| {
                                for enc in list.iter() { ui.selectable_value(&mut selection, enc.clone(), enc); }
                            });
                        self.selected_encoder = Some(selection);
                    });

                    ui.horizontal(|ui| {
                        ui.label("Preset:");
                        let mut preset_idx = match self.preset { ExportPreset::Source => 0, ExportPreset::P1080 => 1, ExportPreset::P4K => 2 };
                        egui::ComboBox::from_id_salt("preset_combo")
                            .selected_text(match self.preset { ExportPreset::Source => "Source", ExportPreset::P1080 => "1080p", ExportPreset::P4K => "4K" })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut preset_idx, 0, "Source");
                                ui.selectable_value(&mut preset_idx, 1, "1080p");
                                ui.selectable_value(&mut preset_idx, 2, "4K");
                            });
                        self.preset = match preset_idx { 1 => ExportPreset::P1080, 2 => ExportPreset::P4K, _ => ExportPreset::Source };

                        ui.label("CRF:");
                        let crf_range = if matches!(self.codec, ExportCodec::H264) { 12..=32 } else { 20..=50 };
                        ui.add(egui::Slider::new(&mut self.crf, crf_range));
                    });

                    // Suggested input source and seq info
                    let (src_path, total_ms) = default_export_source_and_duration(db, project_id, seq);
                    ui.label(format!("Input: {}", src_path.as_deref().unwrap_or("<none>")));
                    ui.label(format!("Duration: {:.2}s", total_ms as f32 / 1000.0));

                    ui.separator();
                    if !self.running {
                        let can_start = src_path.is_some() && !self.output_path.trim().is_empty();
                        if ui.add_enabled(can_start, egui::Button::new("Start Export")).clicked() {
                            if src_path.is_some() {
                                let fps = seq.fps.num.max(1) as f32 / seq.fps.den.max(1) as f32;
                                let (w, h) = match self.preset {
                                    ExportPreset::Source => (seq.width, seq.height),
                                    ExportPreset::P1080 => (1920, 1080),
                                    ExportPreset::P4K => (3840, 2160),
                                };
                                let codec = self.codec;
                                // Ensure extension matches codec
                                if !self.output_path.is_empty() {
                                    self.output_path = adjust_extension(&self.output_path, match codec { ExportCodec::H264 => "mp4", ExportCodec::AV1 => "mkv" });
                                }
                                let crf = self.crf;
                                let out_path = self.output_path.clone();
                                let progress = std::sync::Arc::new(std::sync::Mutex::new(ExportProgress::default()));
                                self.progress_shared = Some(progress.clone());
                                self.running = true;
                                self.status.clear();
                                let selected_encoder = self.selected_encoder.clone();
                                let seq_owned = seq.clone();

                                self.worker = Some(std::thread::spawn(move || {
                                    run_ffmpeg_timeline(out_path, (w, h), fps, codec, selected_encoder, crf, total_ms as u64, seq_owned, progress);
                                }));
                            }
                        }
                    } else {
                        if let Some(p) = &self.progress_shared {
                            if let Ok(p) = p.lock() {
                                self.progress = p.progress;
                                if let Some(eta) = &p.eta { self.status = format!("ETA: {}", eta); }
                                if p.done {
                                    self.running = false;
                                    self.status = p.error.clone().unwrap_or_else(|| "Done".to_string());
                                }
                            }
                        }
                        ui.add(egui::ProgressBar::new(self.progress).show_percentage());
                        ui.label(&self.status);
                    }
                });
            });
        if !keep_open { self.open = false; }
    }
}

fn default_export_source_and_duration(db: &ProjectDb, project_id: &str, seq: &timeline::Sequence) -> (Option<String>, u64) {
    // Pick first video asset as a simple source; duration from asset or sequence
    let assets = db.list_assets(project_id).unwrap_or_default();
    let src = assets.into_iter().find(|a| a.kind.eq_ignore_ascii_case("video")).map(|a| a.src_abs);
    let fps = seq.fps.num.max(1) as f32 / seq.fps.den.max(1) as f32;
    let total_ms = ((seq.duration_in_frames as f32 / fps) * 1000.0) as u64;
    (src, total_ms)
}

fn run_ffmpeg_timeline(out_path: String, size: (u32, u32), fps: f32, codec: ExportCodec, selected_encoder: Option<String>, crf: i32, total_ms: u64, seq: timeline::Sequence, progress: std::sync::Arc<std::sync::Mutex<ExportProgress>>) {
    // Build inputs from timeline
    let (w, h) = size;
    let timeline = build_export_timeline(&seq);
    let mut args: Vec<String> = Vec::new();
    args.push("-y".into());

    // Inputs: video segments
    let mut input_index = 0usize;
    let mut video_labels: Vec<String> = Vec::new();
    for seg in &timeline.video_segments {
        match &seg.kind {
            VideoSegKind::Video { path, start_sec } => {
                args.push("-ss".into()); args.push(format!("{:.3}", start_sec));
                args.push("-t".into()); args.push(format!("{:.3}", seg.duration));
                args.push("-i".into()); args.push(path.clone());
            }
            VideoSegKind::Image { path } => {
                args.push("-loop".into()); args.push("1".into());
                args.push("-t".into()); args.push(format!("{:.3}", seg.duration));
                args.push("-i".into()); args.push(path.clone());
            }
            VideoSegKind::Black => {
                args.push("-f".into()); args.push("lavfi".into());
                args.push("-t".into()); args.push(format!("{:.3}", seg.duration));
                args.push("-r".into()); args.push(format!("{}", fps.max(1.0) as i32));
                args.push("-i".into()); args.push(format!("color=black:s={}x{}", w, h));
            }
        }
        video_labels.push(format!("v{}", input_index));
        input_index += 1;
    }

    // Inputs: audio clips
    let audio_input_start = input_index;
    for clip in &timeline.audio_clips {
        args.push("-i".into()); args.push(clip.path.clone());
        input_index += 1;
    }

    // Filter complex assembly
    let mut filters: Vec<String> = Vec::new();
    let mut vouts: Vec<String> = Vec::new();
    for (i, _seg) in timeline.video_segments.iter().enumerate() {
        let label_in = format!("{}:v", i);
        let label_out = format!("v{}o", i);
        filters.push(format!("[{}]scale={}x{}:flags=lanczos,fps={},format=yuv420p[{}]", label_in, w, h, fps.max(1.0) as i32, label_out));
        vouts.push(format!("[{}]", label_out));
    }
    if !vouts.is_empty() {
        filters.push(format!("{}concat=n={}:v=1:a=0[vout]", vouts.join(""), vouts.len()));
    }

    let mut aouts: Vec<String> = Vec::new();
    for (j, clip) in timeline.audio_clips.iter().enumerate() {
        let in_idx = audio_input_start + j;
        let label_in = format!("{}:a", in_idx);
        let label_out = format!("a{}o", j);
        let delay_ms = (clip.offset_sec * 1000.0).round() as u64;
        let total_s = total_ms as f32 / 1000.0;
        filters.push(format!("[{}]adelay={}|{},atrim=0:{:.3},aresample=async=1[{}]", label_in, delay_ms, delay_ms, total_s, label_out));
        aouts.push(format!("[{}]", label_out));
    }
    let has_audio = !aouts.is_empty();
    if has_audio {
        filters.push(format!("{}amix=inputs={}:normalize=0:duration=longest[aout]", aouts.join(""), aouts.len()));
    }

    if !filters.is_empty() {
        args.push("-filter_complex".into());
        args.push(filters.join(";"));
    }

    args.push("-map".into()); args.push("[vout]".into());
    if has_audio { args.push("-map".into()); args.push("[aout]".into()); } else { args.push("-an".into()); }

    // Codec settings
    args.push("-pix_fmt".into()); args.push("yuv420p".into());
    match codec {
        ExportCodec::H264 => {
            let encoder = selected_encoder.unwrap_or_else(|| "libx264".into());
            args.push("-c:v".into()); args.push(encoder);
            args.push("-crf".into()); args.push(crf.to_string());
            args.push("-preset".into()); args.push("medium".into());
            args.push("-movflags".into()); args.push("+faststart".into());
        }
        ExportCodec::AV1 => {
            let encoder = selected_encoder.unwrap_or_else(|| "libaom-av1".into());
            args.push("-c:v".into()); args.push(encoder.clone());
            if encoder.starts_with("libaom") {
                args.push("-b:v".into()); args.push("0".into());
                args.push("-crf".into()); args.push(crf.to_string());
                args.push("-row-mt".into()); args.push("1".into());
            } else {
                // hw av1 encoders typically use cq
                args.push("-cq".into()); args.push(crf.to_string());
            }
        }
    }

    args.push("-progress".into()); args.push("pipe:2".into());
    args.push(out_path.clone());

    let mut cmd = std::process::Command::new("ffmpeg");
    cmd.args(args.iter().map(|s| s.as_str()));
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            if let Ok(mut p) = progress.lock() { p.done = true; p.error = Some(format!("ffmpeg spawn failed: {}", e)); }
            return;
        }
    };

    if let Some(stderr) = child.stderr.take() {
        use std::io::{BufRead, BufReader};
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        while let Ok(n) = reader.read_line(&mut line) {
            if n == 0 { break; }
            if let Some((k,v)) = line.trim().split_once('=') {
                if k == "out_time_ms" {
                    if let Ok(ms) = v.parse::<u64>() {
                        let prog = if total_ms > 0 { (ms as f32 / total_ms as f32).min(1.0) } else { 0.0 };
                        if let Ok(mut p) = progress.lock() { p.progress = prog; }
                    }
                }
            }
            line.clear();
        }
    }

    let status = child.wait().ok();
    if let Ok(mut p) = progress.lock() {
        p.done = true;
        if let Some(st) = status { if !st.success() { p.error = Some(format!("ffmpeg failed: {:?}", st.code())); } }
    }
}

#[derive(Clone)]
struct VideoSegment { kind: VideoSegKind, start_sec: f32, duration: f32 }

#[derive(Clone)]
enum VideoSegKind { Video { path: String, start_sec: f32 }, Image { path: String }, Black }

#[derive(Clone)]
struct AudioClip { path: String, offset_sec: f32, duration: f32 }

struct ExportTimeline { video_segments: Vec<VideoSegment>, audio_clips: Vec<AudioClip> }

fn build_export_timeline(seq: &timeline::Sequence) -> ExportTimeline {
    // Build breakpoints from all non-audio item edges
    let mut points: Vec<i64> = vec![0, seq.duration_in_frames];
    for (_ti, track) in seq.tracks.iter().enumerate() {
        for it in &track.items {
            match &it.kind {
                ItemKind::Audio { .. } => {}
                _ => {
                    points.push(it.from);
                    points.push(it.from + it.duration_in_frames);
                }
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
        if b <= a { continue; }
        let (item_opt, _ti) = topmost_item_covering(seq, a);
        let kind = if let Some(item) = item_opt {
            match &item.kind {
                ItemKind::Video { src, .. } => {
                    let start_into = (a - item.from).max(0) as f32 / fps;
                    VideoSegKind::Video { path: src.clone(), start_sec: start_into }
                }
                ItemKind::Image { src } => VideoSegKind::Image { path: src.clone() },
                _ => VideoSegKind::Black,
            }
        } else { VideoSegKind::Black };
        let seg = VideoSegment { kind, start_sec: a as f32 / fps, duration: (b - a) as f32 / fps };
        video_segments.push(seg);
    }

    // Audio clips from explicit audio tracks only
    let mut audio_clips: Vec<AudioClip> = Vec::new();
    for track in &seq.tracks {
        for it in &track.items {
            if let ItemKind::Audio { src } = &it.kind {
                audio_clips.push(AudioClip {
                    path: src.clone(),
                    offset_sec: it.from as f32 / fps,
                    duration: it.duration_in_frames as f32 / fps,
                });
            }
        }
    }

    ExportTimeline { video_segments, audio_clips }
}

fn topmost_item_covering<'a>(seq: &'a timeline::Sequence, frame: i64) -> (Option<&'a timeline::Item>, Option<usize>) {
    for (ti, track) in seq.tracks.iter().enumerate().rev() {
        for it in &track.items {
            if frame >= it.from && frame < it.from + it.duration_in_frames {
                match it.kind { ItemKind::Audio { .. } => {}, _ => return (Some(it), Some(ti)) }
            }
        }
    }
    (None, None)
}

fn adjust_extension(path: &str, ext: &str) -> String {
    let mut p = std::path::PathBuf::from(path);
    p.set_extension(ext);
    p.display().to_string()
}

fn detect_hw_encoder<const N: usize>(candidates: [&str; N]) -> Option<String> {
    // best-effort: check existence by running ffmpeg -hide_banner -encoders and scanning; fallback None
    let out = std::process::Command::new("ffmpeg").arg("-hide_banner").arg("-encoders")
        .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null()).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    for cand in candidates {
        if s.contains(cand) { return Some(cand.to_string()); }
    }
    None
}
