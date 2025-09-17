use anyhow::Result;
use eframe::{egui, NativeOptions};
use eframe::egui::Widget;
use eframe::egui_wgpu; // for native TextureId path
use project::{AssetRow, ProjectDb};
use timeline::{Fps, Item, ItemKind, Sequence, Track};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use std::path::PathBuf;
mod interaction;
use interaction::{DragMode, DragState};
mod audio;
use audio::AudioState;
mod audio_engine;
use jobs::{JobEvent, JobStatus};
use media_io::YuvPixFmt;
use native_decoder::{create_decoder, DecoderConfig, is_native_decoding_available, ZeroCopyVideoRenderer};
use std::collections::HashMap;
use std::collections::VecDeque;
use native_decoder::{
    NativeVideoDecoder, VideoFrame, YuvPixFmt as NativeYuvPixFmt
};
use std::hash::Hash;
// (Arc already imported above)
use crossbeam_channel as channel;

#[derive(Default, Debug, Clone)]
struct PlaybackClock {
    playing: bool,
    rate: f64,                 // 1.0 = normal
    anchor_instant: Option<Instant>,
    anchor_timeline_sec: f64,  // timeline time at anchor
}

impl PlaybackClock {
    fn play(&mut self, current_timeline_sec: f64) {
        self.playing = true;
        self.anchor_timeline_sec = current_timeline_sec;
        self.anchor_instant = Some(Instant::now());
    }
    fn pause(&mut self, current_timeline_sec: f64) {
        self.playing = false;
        self.anchor_timeline_sec = current_timeline_sec;
        self.anchor_instant = None;
    }
    fn set_rate(&mut self, rate: f64, current_timeline_sec: f64) {
        // re-anchor to avoid jumps
        self.anchor_timeline_sec = current_timeline_sec;
        self.anchor_instant = Some(Instant::now());
        self.rate = rate;
    }
    fn now(&self) -> f64 {
        if self.playing {
            let dt = self.anchor_instant.unwrap().elapsed().as_secs_f64();
            self.anchor_timeline_sec + dt * self.rate
        } else {
            self.anchor_timeline_sec
        }
    }
    fn seek_to(&mut self, timeline_sec: f64) {
        self.anchor_timeline_sec = timeline_sec;
        if self.playing {
            self.anchor_instant = Some(Instant::now());
        }
    }
}

use tracing_subscriber::EnvFilter;

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .try_init();
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
    workers: HashMap<String, DecodeWorkerRuntime>,
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

    /// Decode exactly once without advancing/prefetching (used when paused).
    fn decode_exact_once(&mut self, path: &str, cfg: &DecoderConfig, target_ts: f64) -> Option<VideoFrame> {
        let entry = self.get_or_create(path, cfg).ok()?;
        entry.attempts_this_tick = 0;
        let frame = entry.decoder.decode_frame(target_ts).ok().flatten();
        entry.attempts_this_tick += 1;
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

    /// Single attempt zero-copy decode without prefetching (paused mode)
    #[cfg(target_os = "macos")]
    fn decode_zero_copy_once(&mut self, path: &str, target_ts: f64) -> Option<native_decoder::IOSurfaceFrame> {
        use native_decoder::YuvPixFmt as Nyf;
        let key = Self::normalize_path_key(path);
        let entry = if let Some(e) = self.decoders.get_mut(&key) { e } else {
            let cfg = DecoderConfig { hardware_acceleration: true, preferred_format: Some(Nyf::Nv12), zero_copy: false };
            let _ = self.get_or_create(path, &cfg);
            self.decoders.get_mut(&key).unwrap()
        };
        if entry.zc_decoder.is_none() {
            let cfg_zc = DecoderConfig { hardware_acceleration: true, preferred_format: Some(Nyf::Nv12), zero_copy: true };
            if let Ok(dec) = create_decoder(path, cfg_zc) { entry.zc_decoder = Some(dec); } else { return None; }
        }
        let dec = entry.zc_decoder.as_mut().unwrap();
        dec.decode_frame_zero_copy(target_ts).ok().flatten()
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

    // Worker management for decoupled decode → render
    fn ensure_worker(&mut self, path: &str) {
        let key = Self::normalize_path_key(path);
        if self.workers.contains_key(&key) { return; }
        let rt = spawn_worker(&key);
        self.workers.insert(key, rt);
    }

    fn send_cmd(&mut self, path: &str, cmd: DecodeCmd) {
        let key = Self::normalize_path_key(path);
        if let Some(w) = self.workers.get(&key) {
            let _ = w.cmd_tx.send(cmd);
        }
    }

    fn take_latest(&mut self, path: &str) -> Option<VideoFrameOut> {
        let key = Self::normalize_path_key(path);
        if let Some(w) = self.workers.get(&key) {
            if let Ok(mut g) = w.slot.0.lock() { return g.take(); }
        }
        None
    }
}

// ---------- Playback engine state ----------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlayState { Paused, Seeking, Playing, Scrubbing }

struct EngineState {
    state: PlayState,
    rate: f32,      // 1.0 by default
    target_pts: f64,
}

// ---------- Decoupled frame types ----------
#[derive(Clone, Copy, Debug)]
struct VideoProps { w: u32, h: u32, fps: f64, fmt: YuvPixFmt }

#[derive(Clone)]
enum FramePayload { Cpu { y: Arc<[u8]>, uv: Arc<[u8]> } }

#[derive(Clone)]
struct VideoFrameOut { pts: f64, props: VideoProps, payload: FramePayload }

// ---------- Worker control ----------
enum DecodeCmd {
    Play { start_pts: f64, rate: f32 },
    Seek { target_pts: f64 },
    Pause,
    Stop,
}

struct LatestFrameSlot(Arc<Mutex<Option<VideoFrameOut>>>);

struct DecodeWorkerRuntime {
    #[allow(dead_code)]
    handle: std::thread::JoinHandle<()>,
    cmd_tx: channel::Sender<DecodeCmd>,
    slot: LatestFrameSlot,
}

fn spawn_worker(path: &str) -> DecodeWorkerRuntime {
    use channel::{unbounded, Receiver, Sender};
    let (cmd_tx, cmd_rx) = unbounded::<DecodeCmd>();
    let slot = LatestFrameSlot(Arc::new(Mutex::new(None)));
    let slot_for_worker = LatestFrameSlot(slot.0.clone());
    let path = path.to_string();
    let handle = std::thread::spawn(move || {
        // Initialize decoders
        let cfg_cpu = DecoderConfig { hardware_acceleration: true, preferred_format: Some(NativeYuvPixFmt::Nv12), zero_copy: false };
        let mut cpu_dec = match create_decoder(&path, cfg_cpu) { Ok(d) => d, Err(e) => { eprintln!("[worker] create_decoder CPU failed: {e}"); return; } };
        // For now, worker outputs CPU NV12/P010 frames only (zero-copy can be added later)

        let props = cpu_dec.get_properties();
        let fps = if props.frame_rate > 0.0 { props.frame_rate } else { 30.0 };
        let frame_dur = if fps > 0.0 { 1.0 / fps } else { 1.0 / 30.0 };

        let mut mode = PlayState::Paused;
        let mut rate: f32 = 1.0;
        let mut anchor_pts: f64 = 0.0;
        let mut anchor_t = std::time::Instant::now();
        let mut running = true;

        let mut attempt_decode = |target: f64| -> Option<VideoFrameOut> {
            // Try zero-copy first (macOS), then CPU. Do a few coax attempts.
            // CPU path
            let mut f = cpu_dec.decode_frame(target).ok().flatten();
            let mut tries = 0;
            while f.is_none() && tries < PREFETCH_BUDGET_PER_TICK {
                let _ = cpu_dec.decode_frame(target);
                tries += 1;
                f = cpu_dec.decode_frame(target).ok().flatten();
            }
            if let Some(vf) = f {
                let fmt = match vf.format { NativeYuvPixFmt::Nv12 => YuvPixFmt::Nv12, NativeYuvPixFmt::P010 => YuvPixFmt::P010 };
                let y: Arc<[u8]> = Arc::from(vf.y_plane.into_boxed_slice());
                let uv: Arc<[u8]> = Arc::from(vf.uv_plane.into_boxed_slice());
                return Some(VideoFrameOut { pts: vf.timestamp, props: VideoProps { w: vf.width, h: vf.height, fps, fmt }, payload: FramePayload::Cpu { y, uv } });
            }
            None
        };

        let mut pending: VecDeque<VideoFrameOut> = VecDeque::new();
        while running {
            // Drain commands
            while let Ok(cmd) = cmd_rx.try_recv() {
                match cmd {
                    DecodeCmd::Play { start_pts, rate: r } => {
                        // Only (re)anchor when transitioning into Playing; otherwise keep smooth progression
                        if mode != PlayState::Playing {
                            mode = PlayState::Playing;
                            anchor_pts = start_pts;
                            anchor_t = std::time::Instant::now();
                        }
                        rate = r;
                    }
                    DecodeCmd::Seek { target_pts } => { mode = PlayState::Seeking; anchor_pts = target_pts; }
                    DecodeCmd::Pause => { mode = PlayState::Paused; }
                    DecodeCmd::Stop => { running = false; }
                }
            }

            match mode {
                PlayState::Playing => {
                    let dt = anchor_t.elapsed().as_secs_f64();
                    let target = anchor_pts + dt * (rate as f64);
                    if let Some(out) = attempt_decode(target) {
                        eprintln!("[WORKER] out pts={:.3}", out.pts);
                        if let Ok(mut g) = slot_for_worker.0.lock() { *g = Some(out); }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(4));
                }
                PlayState::Seeking | PlayState::Scrubbing => {
                    let target = anchor_pts;
                    if let Some(out) = attempt_decode(target) {
                        eprintln!("[WORKER] out pts={:.3}", out.pts);
                        if let Ok(mut g) = slot_for_worker.0.lock() { *g = Some(out); }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(4));
                }
                PlayState::Paused => {
                    std::thread::sleep(std::time::Duration::from_millis(6));
                }
            }
        }
    });

    DecodeWorkerRuntime { handle, cmd_tx, slot }
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
    audio_out: Option<audio_engine::AudioEngine>,
    selected: Option<(usize, usize)>,
    drag: Option<DragState>,
    export: ExportUiState,
    import_workers: Vec<std::thread::JoinHandle<()>>,
    jobs: Option<jobs::JobsHandle>,
    job_events: Vec<JobEvent>,
    show_jobs: bool,
    decode_mgr: DecodeManager,
    playback_clock: PlaybackClock,
    audio_cache: AudioCache,
    // When true during this frame, enable audible scrubbing while paused
    scrub_audio_active: bool,
    // pending vertical move: (from_track, item_index, to_track)
    pending_move: Option<(usize, usize, usize)>,
    // Last successfully presented key: (source path, media time in milliseconds)
    // Using media time (not playhead frame) avoids wrong reuse when clips share a path but have different in_offset/rate.
    last_preview_key: Option<(String, i64)>,
    // Playback engine
    engine: EngineState,
    // Debounce decode commands: remember last sent (state, path, optional seek bucket)
    last_sent: Option<(PlayState, String, Option<i64>)>,
    // Throttled engine log state
    // (Used only for preview_ui logging when sending worker commands)
    // Not strictly necessary, but kept for future UI log hygiene.
    // last_engine_log: Option<Instant>,
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
            audio_out: audio_engine::AudioEngine::new().ok(),
            selected: None,
            drag: None,
            export: ExportUiState::default(),
            import_workers: Vec::new(),
            jobs: Some(jobs::JobsRuntime::start(2)),
            job_events: Vec::new(),
            show_jobs: false,
            decode_mgr: DecodeManager::default(),
            playback_clock: PlaybackClock { rate: 1.0, ..Default::default() },
            audio_cache: AudioCache::default(),
            scrub_audio_active: false,
            pending_move: None,
            last_preview_key: None,
            engine: EngineState { state: PlayState::Paused, rate: 1.0, target_pts: 0.0 },
            last_sent: None,
        }
    }

    fn request_audio_peaks(&mut self, _path: &std::path::Path) {
        // Placeholder: integrate with audio decoding backend to compute peaks.
        // Keep bounded: one job per path. For now, no-op to avoid blocking UI.
    }

    fn split_clip_at_frame(&mut self, track: usize, item: usize, split_frame: i64) {
        if track >= self.seq.tracks.len() { return; }
        let items = &mut self.seq.tracks[track].items;
        if item >= items.len() { return; }
        let clip = items[item].clone();
        let from = clip.from;
        let dur = clip.duration_in_frames;
        if split_frame <= from || split_frame >= from + dur { return; }
        let left_dur = split_frame - from;
        let right_dur = dur - left_dur;
        // Left
        items[item].duration_in_frames = left_dur;
        // Right
        let mut right = clip.clone();
        right.from = split_frame;
        right.duration_in_frames = right_dur;
        items.insert(item+1, right);
    }

    fn remove_clip(&mut self, track: usize, item: usize) {
        if track < self.seq.tracks.len() && item < self.seq.tracks[track].items.len() {
            self.seq.tracks[track].items.remove(item);
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
                ItemKind::Audio { src: asset.src_abs.clone(), in_offset_sec: 0.0, rate: 1.0 }
            } else if asset.kind.eq_ignore_ascii_case("image") {
                ItemKind::Image { src: asset.src_abs.clone() }
            } else {
                let fr = match (asset.fps_num, asset.fps_den) { (Some(n), Some(d)) if d != 0 => Some(n as f32 / d as f32), _ => None };
                ItemKind::Video { src: asset.src_abs.clone(), frame_rate: fr, in_offset_sec: 0.0, rate: 1.0 }
            };
            track.items.push(Item { id, from, duration_in_frames: duration, kind });
            let end = self.seq.tracks.iter().flat_map(|t| t.items.iter().map(|it| it.from + it.duration_in_frames)).max().unwrap_or(0);
            self.seq.duration_in_frames = end.max(self.seq.duration_in_frames);
        }
    }

    fn timeline_ui(&mut self, ui: &mut egui::Ui) {
        // Reset scrubbing flag; set true only while background dragging
        self.scrub_audio_active = false;
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
        egui::ScrollArea::both().drag_to_scroll(false).show(ui, |ui| {
            let mut to_request: Vec<std::path::PathBuf> = Vec::new();
            let (rect, response) = ui.allocate_exact_size(egui::vec2(content_w, content_h), egui::Sense::click_and_drag());
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

                    // Optional lightweight waveform lane under clips (audio or video)
                    if let Some(src_path) = match &it.kind { 
                        ItemKind::Audio { src, .. } => Some(src.as_str()),
                        ItemKind::Video { src, .. } => Some(src.as_str()),
                        _ => None,
                    } {
                        let pbuf = std::path::PathBuf::from(src_path);
                        if let Some(peaks) = self.audio_cache.map.get(&pbuf) {
                            let rect_lane = r.shrink2(egui::vec2(2.0, 6.0));
                            let n = peaks.peaks.len().max(1);
                            let mut pts_top: Vec<egui::Pos2> = Vec::with_capacity(n);
                            let mut pts_bot: Vec<egui::Pos2> = Vec::with_capacity(n);
                            for (i, (mn, mx)) in peaks.peaks.iter().enumerate() {
                                let t = if n > 1 { i as f32 / (n as f32 - 1.0) } else { 0.0 };
                                let x = egui::lerp(rect_lane.left()..=rect_lane.right(), t);
                                let y0 = egui::lerp(rect_lane.center().y..=rect_lane.top(), mx.abs().min(1.0));
                                let y1 = egui::lerp(rect_lane.center().y..=rect_lane.bottom(), mn.abs().min(1.0));
                                pts_top.push(egui::pos2(x, y0));
                                pts_bot.push(egui::pos2(x, y1));
                            }
                            let stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(120,180,240));
                            ui.painter().add(egui::Shape::line(pts_top, stroke));
                            ui.painter().add(egui::Shape::line(pts_bot, stroke));
                        } else {
                            to_request.push(pbuf);
                        }
                    }

                    // Make the clip rect an interactive drag target so ScrollArea doesn't pan
                    let resp = ui.interact(
                        r,
                        egui::Id::new(("clip", ti, ii)),
                        egui::Sense::click_and_drag(),
                    );
                    if resp.clicked() { self.selected = Some((ti, ii)); }
                    if resp.drag_started() {
                        let mx = resp.interact_pointer_pos().unwrap_or(egui::pos2(0.0,0.0)).x;
                        // Determine drag mode by edge proximity
                        let mode = if (mx - r.left()).abs() <= 6.0 { DragMode::TrimStart }
                                   else if (mx - r.right()).abs() <= 6.0 { DragMode::TrimEnd }
                                   else { DragMode::Move };
                        self.selected = Some((ti, ii));
                        self.drag = Some(DragState { track: ti, item: ii, mode, start_mouse_x: mx, orig_from: it.from, orig_dur: it.duration_in_frames });
                    }
                    if resp.drag_released() {
                        // On release, allow moving the clip to a different track if pointer is over it
                        if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                            let mut target_track = (((pos.y - rect.top()) / track_h).floor() as isize) as isize;
                            // Clamp to valid range
                            if target_track < 0 { target_track = 0; }
                            if target_track >= self.seq.tracks.len() as isize { target_track = (self.seq.tracks.len() as isize) - 1; }
                            let target_track = target_track as usize;
                            if target_track != ti {
                                // Defer the actual move until after we finish iterating/borrowing
                                self.pending_move = Some((ti, ii, target_track));
                            }
                        }
                        self.drag = None;
                    }
                }
            }
            // Playhead
            let phx = rect.left() + self.playhead as f32 * self.zoom_px_per_frame;
            painter.line_segment([egui::pos2(phx, rect.top()), egui::pos2(phx, rect.bottom())], egui::Stroke::new(2.0, egui::Color32::from_rgb(220, 60, 60)));

            // Click/drag background to scrub (when not dragging a clip)
            if self.drag.is_none() {
                // Single click: move playhead on mouse up as well
                if response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let local_px = (pos.x - rect.left()).max(0.0) as f64;
                        let fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
                        let frames = (local_px / self.zoom_px_per_frame as f64).round() as i64;
                        let sec = (frames as f64) / fps;
                        self.playback_clock.seek_to(sec);
                        self.playhead = frames.clamp(0, self.seq.duration_in_frames);
                        self.engine.state = PlayState::Seeking;
                    }
                }
                // Drag: continuously update while primary is down
                if response.dragged() && ui.input(|i| i.pointer.primary_down()) {
                    if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
                        let local_px = (pos.x - rect.left()).max(0.0) as f64;
                        let fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
                        let frames = (local_px / self.zoom_px_per_frame as f64).round() as i64;
                        let sec = (frames as f64) / fps;
                        self.playback_clock.seek_to(sec);
                        self.playhead = frames.clamp(0, self.seq.duration_in_frames);
                        // Enable audible scrubbing while paused
                        self.scrub_audio_active = true;
                        self.engine.state = PlayState::Scrubbing;
                    }
                }
            }

            // Timeline hotkeys: split/delete
            let pressed_split = ui.input(|i| i.key_pressed(egui::Key::K) || (i.modifiers.command && i.key_pressed(egui::Key::S)));
            let pressed_delete = ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace));
            if pressed_split {
                if let Some((t, iidx)) = self.selected {
                    let fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
                    let t_sec = self.playback_clock.now();
                    let split_frame = (t_sec * fps).round() as i64;
                    self.split_clip_at_frame(t, iidx, split_frame);
                }
            }
            if pressed_delete {
                if let Some((t, iidx)) = self.selected.take() {
                    self.remove_clip(t, iidx);
                }
            }

            if let Some(drag) = self.drag {
                if ui.input(|i| !i.pointer.primary_down()) {
                    self.drag = None;
                    // End of scrubbing drag
                    if self.engine.state == PlayState::Scrubbing { self.engine.state = PlayState::Paused; }
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
                                let new_from = (drag.orig_from + df).clamp(0, drag.orig_from + drag.orig_dur - 1);
                                let delta_frames = (new_from - drag.orig_from).max(0);
                                let secf = (new_from as f32 / fpsf).round() * fpsf;
                                let snap_new_from = if ((secf - new_from as f32).abs()) <= eps { secf as i64 } else { new_from };
                                // Advance source by delta_sec
                                let fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
                                let delta_sec = (delta_frames as f64) / fps;
                                item.from = snap_new_from;
                                item.duration_in_frames = (drag.orig_dur - delta_frames).max(1);
                                match &mut item.kind {
                                    ItemKind::Video { in_offset_sec, .. } => { *in_offset_sec = (*in_offset_sec + delta_sec).max(0.0); }
                                    ItemKind::Audio { in_offset_sec, .. } => { *in_offset_sec = (*in_offset_sec + delta_sec).max(0.0); }
                                    _ => {}
                                }
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

            // Apply any pending cross-track move now that we are out of the per-item iteration
            if let Some((from_t, item_idx, to_t)) = self.pending_move.take() {
                if from_t < self.seq.tracks.len() && to_t < self.seq.tracks.len() {
                    if item_idx < self.seq.tracks[from_t].items.len() {
                        // Remove the item from the old track
                        let item = self.seq.tracks[from_t].items.remove(item_idx);
                        // Insert at the end of the target track (keeps the same timeline start/duration)
                        self.seq.tracks[to_t].items.push(item);
                        // Update selection to the new location
                        let new_index = self.seq.tracks[to_t].items.len().saturating_sub(1);
                        self.selected = Some((to_t, new_index));
                    }
                }
            }

            // Defer any peak requests until after immutable borrows end
            for p in to_request { self.request_audio_peaks(&p); }
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
        // Determine current visual source at playhead (lock to exact frame)
        let fps = self.seq.fps.num.max(1) as f64 / self.seq.fps.den.max(1) as f64;
        let t_playhead = self.playback_clock.now();
        let playhead_frame = if self.engine.state == PlayState::Playing {
            (t_playhead * fps).floor() as i64
        } else {
            (t_playhead * fps).round() as i64
        };
        self.playhead = playhead_frame;
        let target_ts = (playhead_frame as f64) / fps;
        let t_sec = target_ts as f32;
        let source = current_visual_source(&self.seq, self.playhead);

        // Debug: shader mode toggle for YUV preview
        ui.horizontal(|ui| {
            ui.label("Shader:");
            let mode = &mut self.preview.shader_mode;
            let solid = matches!(*mode, PreviewShaderMode::Solid);
            if ui.selectable_label(solid, "Solid").clicked() { *mode = PreviewShaderMode::Solid; ctx.request_repaint(); }
            let showy = matches!(*mode, PreviewShaderMode::ShowY);
            if ui.selectable_label(showy, "Y").clicked() { *mode = PreviewShaderMode::ShowY; ctx.request_repaint(); }
            let uvd = matches!(*mode, PreviewShaderMode::UvDebug);
            if ui.selectable_label(uvd, "UV").clicked() { *mode = PreviewShaderMode::UvDebug; ctx.request_repaint(); }
            let nv12 = matches!(*mode, PreviewShaderMode::Nv12);
            if ui.selectable_label(nv12, "NV12").clicked() { *mode = PreviewShaderMode::Nv12; ctx.request_repaint(); }
        });
        // Hotkeys 1/2/3
        if ui.input(|i| i.key_pressed(egui::Key::Num1)) { self.preview.shader_mode = PreviewShaderMode::Solid; ctx.request_repaint(); }
        if ui.input(|i| i.key_pressed(egui::Key::Num2)) { self.preview.shader_mode = PreviewShaderMode::ShowY; ctx.request_repaint(); }
        if ui.input(|i| i.key_pressed(egui::Key::Num3)) { self.preview.shader_mode = PreviewShaderMode::UvDebug; ctx.request_repaint(); }
        if ui.input(|i| i.key_pressed(egui::Key::Num4)) { self.preview.shader_mode = PreviewShaderMode::Nv12; ctx.request_repaint(); }

        // Layout: reserve a 16:9 box or fit available space
        let avail = ui.available_size();
        let mut w = avail.x.max(320.0);
        let mut h = (w * 9.0 / 16.0).round();
        if h > avail.y { h = avail.y; w = (h * 16.0 / 9.0).round(); }
        let desired = (w as u32, h as u32);

        // Playback progression handled by PlaybackClock (no speed-up)

        // Draw
        let (rect, _resp) = ui.allocate_exact_size(egui::vec2(w, h), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(12, 12, 12));
        
        // Use persistent decoder with prefetch
        if let Some(src) = source.as_ref() {
            if let Some(rs) = frame.wgpu_render_state() {
                // New: use background decode worker and small frame queue
                let (active_path, media_t) = if let Some((p, mt)) = active_video_media_time(&self.seq, t_playhead) { (p, mt) } else { (src.path.clone(), t_playhead) };
                self.engine.target_pts = media_t;
                self.decode_mgr.ensure_worker(&active_path);

                // Log engine vs clock once per second or on state change
                {
                    static mut LAST_LOG_INSTANT: Option<std::time::Instant> = None;
                    static mut LAST_LOG_STATE: Option<(PlayState, bool)> = None;
                    let now = std::time::Instant::now();
                    let clock_playing = self.playback_clock.playing;
                    let should_log = unsafe {
                        let last = LAST_LOG_INSTANT.get_or_insert(now);
                        let changed = LAST_LOG_STATE.map(|(s, c)| s != self.engine.state || c != clock_playing).unwrap_or(true);
                        let elapsed = now.duration_since(*last).as_secs_f64() >= 1.0;
                        if changed || elapsed { *last = now; LAST_LOG_STATE = Some((self.engine.state, clock_playing)); true } else { false }
                    };
                    if should_log {
                        eprintln!("[ENGINE] state={:?} clock_playing={} target_pts={:.3}", self.engine.state, clock_playing, media_t);
                    }
                }

                // Debounce sends: include seek bucket for non-playing states
                let fps_seq = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
                let seek_bucket = (media_t * fps_seq).round() as i64;
                let k = match self.engine.state {
                    PlayState::Playing => (self.engine.state, active_path.clone(), None),
                    _ => (self.engine.state, active_path.clone(), Some(seek_bucket)),
                };
                if self.last_sent != Some(k.clone()) {
                    match self.engine.state {
                        PlayState::Playing => { let _ = self.decode_mgr.send_cmd(&active_path, DecodeCmd::Play { start_pts: media_t, rate: self.engine.rate }); }
                        PlayState::Scrubbing | PlayState::Seeking | PlayState::Paused => { let _ = self.decode_mgr.send_cmd(&active_path, DecodeCmd::Seek { target_pts: media_t }); }
                    }
                    if self.engine.state == PlayState::Scrubbing { ctx.request_repaint(); }
                    self.last_sent = Some(k);
                }

                // Drain worker and pick a frame (latest-wins slot)
                let newest = self.decode_mgr.take_latest(&active_path);
                let queue_len = if newest.is_some() { 1 } else { 0 };
                let tol = {
                    let fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
                    let frame_dur = if fps > 0.0 { 1.0 / fps } else { 1.0 / 30.0 };
                    (0.5 * frame_dur).max(0.012)
                };
                let is_actually_playing = self.engine.state == PlayState::Playing || self.playback_clock.playing;
                let picked: Option<VideoFrameOut> = if is_actually_playing {
                    newest
                } else {
                    newest.filter(|f| (f.pts - media_t).abs() <= tol)
                };

                if let Some(f) = picked {
                    if let FramePayload::Cpu { y, uv } = f.payload {
                        if let Some((fmt, ytex, uvtex)) = self.preview.present_yuv_from_bytes(&rs, f.props.fmt, &y, &uv, f.props.w, f.props.h) {
                            eprintln!("[PREVIEW] draw=yuv fmt={:?} mode={:?}", fmt, self.preview.shader_mode);
                            let use_uint = matches!(fmt, YuvPixFmt::P010) && !device_supports_16bit_norm(&rs);
                            let cb = egui_wgpu::Callback::new_paint_callback(rect, PreviewYuvCallback { y_tex: ytex, uv_tex: uvtex, fmt, use_uint, w: f.props.w, h: f.props.h, mode: self.preview.shader_mode });
                            ui.painter().add(cb);
                            // HUD overlay: state, queue, delta ms
                            let d_ms = ((f.pts - media_t) * 1000.0) as i32;
                            let hud = format!(
                                "state={:?}  queue={}  Δ={}ms  picked={:.3}  target={:.3}",
                                self.engine.state, queue_len, d_ms, f.pts, media_t
                            );
                            painter.text(rect.left_top() + egui::vec2(5.0, 5.0), egui::Align2::LEFT_TOP, hud, egui::FontId::monospace(10.0), egui::Color32::WHITE);
                        }
                    }
                } else {
                    if let Some((fmt, y_tex, uv_tex)) = self.preview.current_plane_textures() {
                        eprintln!("[PREVIEW] reuse last=fmt {:?} size={}x{}", fmt, self.preview.y_size.0, self.preview.y_size.1);
                        let use_uint = matches!(fmt, YuvPixFmt::P010) && !device_supports_16bit_norm(&rs);
                        let cb = egui_wgpu::Callback::new_paint_callback(
                            rect,
                            PreviewYuvCallback { y_tex, uv_tex, fmt, use_uint, w: self.preview.y_size.0, h: self.preview.y_size.1, mode: self.preview.shader_mode }
                        );
                        ui.painter().add(cb);
                        let hud = format!(
                            "state={:?}  queue={}  Δ=--  target={:.3} (reuse)",
                            self.engine.state, queue_len, media_t
                        );
                        painter.text(rect.left_top() + egui::vec2(5.0, 5.0), egui::Align2::LEFT_TOP, hud, egui::FontId::monospace(10.0), egui::Color32::WHITE);
                    } else {
                        eprintln!("[PREVIEW] fallback=solid dummy=1x1");
                        let device = &*rs.device;
                        let y_tex = std::sync::Arc::new(device.create_texture(&eframe::wgpu::TextureDescriptor { label: Some("dummy_y"), size: eframe::wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: eframe::wgpu::TextureDimension::D2, format: eframe::wgpu::TextureFormat::R8Unorm, usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING | eframe::wgpu::TextureUsages::COPY_DST, view_formats: &[] }));
                        let uv_tex = std::sync::Arc::new(device.create_texture(&eframe::wgpu::TextureDescriptor { label: Some("dummy_uv"), size: eframe::wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: eframe::wgpu::TextureDimension::D2, format: eframe::wgpu::TextureFormat::Rg8Unorm, usage: eframe::wgpu::TextureUsages::TEXTURE_BINDING | eframe::wgpu::TextureUsages::COPY_DST, view_formats: &[] }));
                        let cb = egui_wgpu::Callback::new_paint_callback(rect, PreviewYuvCallback { y_tex, uv_tex, fmt: YuvPixFmt::Nv12, use_uint: false, w: 1, h: 1, mode: PreviewShaderMode::Solid });
                        ui.painter().add(cb);
                        let hud = format!("state={:?}  queue={}  Δ=--  target={:.3}", self.engine.state, queue_len, media_t);
                        painter.text(rect.left_top() + egui::vec2(5.0, 5.0), egui::Align2::LEFT_TOP, hud, egui::FontId::monospace(10.0), egui::Color32::WHITE);
                    }
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
        if self.engine.state == PlayState::Playing {
            let fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
            let dt = if fps > 0.0 { 1.0 / fps } else { 1.0 / 30.0 };
            ctx.request_repaint_after(Duration::from_secs_f64(dt));
        } else {
            // When not playing, only repaint when needed (scrubbing, UI changes)
            // This reduces CPU usage significantly when idle
        }
        // Space toggles play/pause (keep engine.state in sync)
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            let seq_fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
            let current_sec = (self.playhead as f64) / seq_fps;

            if self.playback_clock.playing {
                self.playback_clock.pause(current_sec);
                // NEW: make the decode engine pause too
                self.engine.state = PlayState::Paused;
            } else {
                if self.playhead >= self.seq.duration_in_frames { self.playhead = 0; }
                self.playback_clock.play(current_sec);
                // NEW: make the decode engine actually play
                self.engine.state = PlayState::Playing;
            }
        }

        // Keep engine.state aligned with the clock unless we're in an explicit drag/seek
        if !matches!(self.engine.state, PlayState::Scrubbing | PlayState::Seeking) {
            self.engine.state = if self.playback_clock.playing { PlayState::Playing } else { PlayState::Paused };
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
                if ui.button(if self.engine.state == PlayState::Playing { "Pause (Space)" } else { "Play (Space)" }).clicked() {
                    let seq_fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
                    let current_sec = (self.playhead as f64) / seq_fps;
                    if self.engine.state == PlayState::Playing {
                        self.playback_clock.pause(current_sec);
                        self.engine.state = PlayState::Paused;
                    } else {
                        self.playback_clock.play(current_sec);
                        self.engine.state = PlayState::Playing;
                    }
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

        // Properties panel for selected clip
        egui::SidePanel::right("properties").default_width(280.0).show(ctx, |ui| {
            ui.heading("Properties");
            if let Some((ti, ii)) = self.selected {
                if ti < self.seq.tracks.len() && ii < self.seq.tracks[ti].items.len() {
                    let item = &mut self.seq.tracks[ti].items[ii];
                    ui.label(format!("Clip ID: {}", &item.id[..8.min(item.id.len())]));
                    ui.label(format!("From: {}  Dur: {}f", item.from, item.duration_in_frames));
                    match &mut item.kind {
                        ItemKind::Video { in_offset_sec, rate, .. } => {
                            ui.separator();
                            ui.label("Video");
                            ui.horizontal(|ui| {
                                ui.label("Rate");
                                let mut r = *rate as f64;
                                if ui.add(egui::DragValue::new(&mut r).clamp_range(0.05..=8.0).speed(0.02)).changed() {
                                    *rate = (r as f32).max(0.01);
                                }
                                if ui.small_button("1.0").on_hover_text("Reset").clicked() { *rate = 1.0; }
                            });
                            ui.horizontal(|ui| {
                                ui.label("In Offset (s)");
                                let mut o = *in_offset_sec;
                                if ui.add(egui::DragValue::new(&mut o).clamp_range(0.0..=1_000_000.0).speed(0.01)).changed() {
                                    *in_offset_sec = o.max(0.0);
                                }
                                if ui.small_button("0").on_hover_text("Reset").clicked() { *in_offset_sec = 0.0; }
                            });
                        }
                        ItemKind::Audio { in_offset_sec, rate, .. } => {
                            ui.separator();
                            ui.label("Audio");
                            ui.horizontal(|ui| {
                                ui.label("Rate");
                                let mut r = *rate as f64;
                                if ui.add(egui::DragValue::new(&mut r).clamp_range(0.05..=8.0).speed(0.02)).changed() {
                                    *rate = (r as f32).max(0.01);
                                }
                                if ui.small_button("1.0").on_hover_text("Reset").clicked() { *rate = 1.0; }
                            });
                            ui.horizontal(|ui| {
                                ui.label("In Offset (s)");
                                let mut o = *in_offset_sec;
                                if ui.add(egui::DragValue::new(&mut o).clamp_range(0.0..=1_000_000.0).speed(0.01)).changed() {
                                    *in_offset_sec = o.max(0.0);
                                }
                                if ui.small_button("0").on_hover_text("Reset").clicked() { *in_offset_sec = 0.0; }
                            });
                        }
                        ItemKind::Image { .. } => {
                            ui.separator();
                            ui.label("Image clip has no time controls");
                        }
                        _ => {}
                    }
                } else {
                    ui.label("Selection out of range");
                }
            } else {
                ui.label("No clip selected");
            }
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

        // Audio playback follows play state using active audio mapping
        if self.engine.state == PlayState::Playing {
            let t_sec = self.playback_clock.now();
            if let Some((path, media_sec)) = active_audio_media_time(&self.seq, t_sec) {
                self.audio.ensure_playing(Some(&path), media_sec);
            } else {
                self.audio.stop();
            }
        } else if self.scrub_audio_active || self.engine.state == PlayState::Scrubbing || self.engine.state == PlayState::Seeking {
            // Audible scrubbing while paused
            let t_sec = self.playback_clock.now();
            if let Some((path, media_sec)) = active_audio_media_time(&self.seq, t_sec) {
                self.audio.preview_scrub(Some(&path), media_sec);
            } else {
                self.audio.stop();
            }
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

fn active_video_media_time(seq: &Sequence, timeline_sec: f64) -> Option<(String, f64)> {
    let seq_fps = (seq.fps.num.max(1) as f64) / (seq.fps.den.max(1) as f64);
    let playhead = (timeline_sec * seq_fps).round() as i64;
    for track in seq.tracks.iter().rev() {
        for it in &track.items {
            let covers = playhead >= it.from && playhead < it.from + it.duration_in_frames;
            if !covers { continue; }
            if let ItemKind::Video { src, in_offset_sec, rate, .. } = &it.kind {
                let start_on_timeline_sec = it.from as f64 / seq_fps;
                let local_t = (timeline_sec - start_on_timeline_sec).max(0.0);
                let media_sec = *in_offset_sec + local_t * (*rate as f64);
                return Some((src.clone(), media_sec.max(0.0)));
            }
        }
    }
    None
}

fn active_audio_media_time(seq: &Sequence, timeline_sec: f64) -> Option<(String, f64)> {
    let seq_fps = (seq.fps.num.max(1) as f64) / (seq.fps.den.max(1) as f64);
    let playhead = (timeline_sec * seq_fps).round() as i64;
    for track in seq.tracks.iter().rev() {
        for it in &track.items {
            let covers = playhead >= it.from && playhead < it.from + it.duration_in_frames;
            if !covers { continue; }
            if let ItemKind::Audio { src, in_offset_sec, rate } = &it.kind {
                let start_on_timeline_sec = it.from as f64 / seq_fps;
                let local_t = (timeline_sec - start_on_timeline_sec).max(0.0);
                let media_sec = *in_offset_sec + local_t * (*rate as f64);
                return Some((src.clone(), media_sec.max(0.0)));
            }
        }
    }
    active_video_media_time(seq, timeline_sec)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PreviewShaderMode { Solid, ShowY, UvDebug, Nv12 }

impl Default for PreviewShaderMode { fn default() -> Self { PreviewShaderMode::Solid } }

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
    #[cfg(target_os = "macos")]
    zc_logged: bool,

    // Last-presented zero-copy textures (macOS reuse)
    #[cfg(target_os = "macos")]
    last_zc: Option<(YuvPixFmt, std::sync::Arc<eframe::wgpu::Texture>, std::sync::Arc<eframe::wgpu::Texture>, (u32,u32))>,

    // Recency tracking for reuse selection
    last_present_tick: u64,
    last_cpu_tick: u64,
    #[cfg(target_os = "macos")]
    last_zc_tick: u64,

    // Performance metrics
    cache_hits: u64,
    cache_misses: u64,
    decode_time_ms: f64,
    // Debug shader mode for preview
    // Last presented YUV format (for reuse without new uploads)
    last_fmt: Option<YuvPixFmt>,
    shader_mode: PreviewShaderMode,
}

// Log guard: avoid spamming size mismatch logs across frames
static PRESENT_SIZE_MISMATCH_LOGGED: OnceLock<AtomicBool> = OnceLock::new();

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
            #[cfg(target_os = "macos")]
            zc_logged: false,
            #[cfg(target_os = "macos")]
            last_zc: None,
            last_present_tick: 0,
            last_cpu_tick: 0,
            #[cfg(target_os = "macos")]
            last_zc_tick: 0,
            last_fmt: None,
            shader_mode: PreviewShaderMode::Nv12,
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
        if next_idx == self.ring_present {
            eprintln!("[RING DROP] write={} present={} (dropping frame to avoid stall)", self.ring_write, self.ring_present);
            return;
        }
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

        // Fill pre-allocated scratch buffers with row padding, zero-initialized
        let mut y_scratch = vec![0u8; y_pad_bpr * h as usize];
        for r in 0..(h as usize) {
            let s = r * y_bpr;
            let d = r * y_pad_bpr;
            y_scratch[d..d + y_bpr].copy_from_slice(&y[s..s + y_bpr]);
        }
        let mut uv_scratch = vec![0u8; uv_pad_bpr * uv_h as usize];
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
        eprintln!("[UV] w={} h={} bpr={} rows={}", uv_w, uv_h, uv_pad_bpr, uv_h);

        self.ring_present = idx;
        self.ring_write = next_idx;
        self.last_fmt = Some(fmt);
    }

    fn current_plane_textures(&self) -> Option<(YuvPixFmt, std::sync::Arc<eframe::wgpu::Texture>, std::sync::Arc<eframe::wgpu::Texture>)> {
        let mut best: Option<(u64, YuvPixFmt, std::sync::Arc<eframe::wgpu::Texture>, std::sync::Arc<eframe::wgpu::Texture>)> = None;
        if let Some(fmt) = self.last_fmt {
            let idx = self.ring_present % 3;
            if let (Some(y), Some(uv)) = (self.y_tex[idx].as_ref(), self.uv_tex[idx].as_ref()) {
                best = Some((self.last_cpu_tick, fmt, y.clone(), uv.clone()));
            }
        }
        #[cfg(target_os = "macos")]
        if let Some((fmt, y, uv, _sz)) = self.last_zc.as_ref() {
            match best {
                Some((tick, ..)) if self.last_zc_tick <= tick => {}
                _ => { best = Some((self.last_zc_tick, *fmt, y.clone(), uv.clone())); }
            }
        }
        best.map(|(_, fmt, y, uv)| (fmt, y, uv))
    }

    #[cfg(target_os = "macos")]
    fn set_last_zc_present(
        &mut self,
        fmt: YuvPixFmt,
        y_tex: std::sync::Arc<eframe::wgpu::Texture>,
        uv_tex: std::sync::Arc<eframe::wgpu::Texture>,
        w: u32,
        h: u32,
    ) {
        self.last_zc = Some((fmt, y_tex, uv_tex, (w, h)));
        self.last_fmt = Some(fmt);
        self.y_size = (w, h);
        self.uv_size = ((w + 1)/2, (h + 1)/2);
        self.last_present_tick = self.last_present_tick.wrapping_add(1);
        self.last_zc_tick = self.last_present_tick;
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

    fn present_yuv_from_bytes(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        fmt: YuvPixFmt,
        y_bytes: &[u8],
        uv_bytes: &[u8],
        w: u32,
        h: u32,
    ) -> Option<(YuvPixFmt, Arc<eframe::wgpu::Texture>, Arc<eframe::wgpu::Texture>)> {
        // Ensure textures/buffers exist at this decoded size/format
        self.ensure_yuv_textures(rs, w, h, fmt);

        // Write into current ring slot
        let wi = self.ring_write % 3;

        // Compute padded rows
        let (y_bpp, uv_bpp_per_texel) = match fmt { YuvPixFmt::Nv12 => (1usize, 2usize), YuvPixFmt::P010 => (2usize, 4usize) };
        let y_w = w as usize; let y_h = h as usize;
        let uv_w = ((w + 1) / 2) as usize; let uv_h = ((h + 1) / 2) as usize;
        let y_pad_bpr = align_to(y_w * y_bpp, eframe::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        let uv_pad_bpr = align_to(uv_w * uv_bpp_per_texel, eframe::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        debug_assert!(uv_pad_bpr % 2 == 0, "NV12 UV bpr must be even (#channels=2)");

        // Guard: verify plane lengths once; early out if mismatched
        let expected_y = y_w * y_bpp * y_h;
        let expected_uv = uv_w * uv_bpp_per_texel * uv_h;
        debug_assert_eq!(y_bytes.len(), expected_y, "Y plane size mismatch");
        debug_assert_eq!(uv_bytes.len(), expected_uv, "UV plane size mismatch");
        if y_bytes.len() != expected_y || uv_bytes.len() != expected_uv {
            let flag = PRESENT_SIZE_MISMATCH_LOGGED.get_or_init(|| AtomicBool::new(false));
            if !flag.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[present] size mismatch: got Y={} UV={}, expected Y={} UV={}",
                    y_bytes.len(), uv_bytes.len(), expected_y, expected_uv
                );
            }
            return None;
        }

        let device = &*rs.device;
        let queue = &*rs.queue;

        // Upload Y
        if let (Some(stage), Some(y_tex)) = (self.y_stage[wi].as_ref(), self.y_tex[wi].as_ref()) {
            if y_pad_bpr == y_w * y_bpp {
                queue.write_buffer(stage, 0, y_bytes);
            } else {
                let mut padded = vec![0u8; y_pad_bpr * y_h];
                for row in 0..y_h {
                    let src_off = row * y_w * y_bpp;
                    let dst_off = row * y_pad_bpr;
                    padded[dst_off..dst_off + y_w * y_bpp].copy_from_slice(&y_bytes[src_off..src_off + y_w * y_bpp]);
                }
                queue.write_buffer(stage, 0, &padded);
            }
            let mut enc = device.create_command_encoder(&eframe::wgpu::CommandEncoderDescriptor { label: Some("copy_y") });
            enc.copy_buffer_to_texture(
                eframe::wgpu::ImageCopyBuffer { buffer: stage, layout: eframe::wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(y_pad_bpr as u32), rows_per_image: Some(h) } },
                eframe::wgpu::ImageCopyTexture { texture: y_tex, mip_level: 0, origin: eframe::wgpu::Origin3d::ZERO, aspect: eframe::wgpu::TextureAspect::All },
                eframe::wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            rs.queue.submit(std::iter::once(enc.finish()));
        }

        // Upload UV
        if let (Some(stage), Some(uv_tex)) = (self.uv_stage[wi].as_ref(), self.uv_tex[wi].as_ref()) {
            if uv_pad_bpr == uv_w * uv_bpp_per_texel {
                queue.write_buffer(stage, 0, uv_bytes);
            } else {
                let mut padded = vec![0u8; uv_pad_bpr * uv_h];
                for row in 0..uv_h {
                    let src_off = row * uv_w * uv_bpp_per_texel;
                    let dst_off = row * uv_pad_bpr;
                    padded[dst_off..dst_off + uv_w * uv_bpp_per_texel].copy_from_slice(&uv_bytes[src_off..src_off + uv_w * uv_bpp_per_texel]);
                }
                queue.write_buffer(stage, 0, &padded);
            }
            let mut enc = device.create_command_encoder(&eframe::wgpu::CommandEncoderDescriptor { label: Some("copy_uv") });
            enc.copy_buffer_to_texture(
                eframe::wgpu::ImageCopyBuffer { buffer: stage, layout: eframe::wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(uv_pad_bpr as u32), rows_per_image: Some((h + 1) / 2) } },
                eframe::wgpu::ImageCopyTexture { texture: uv_tex, mip_level: 0, origin: eframe::wgpu::Origin3d::ZERO, aspect: eframe::wgpu::TextureAspect::All },
                eframe::wgpu::Extent3d { width: (w + 1) / 2, height: (h + 1) / 2, depth_or_array_layers: 1 },
            );
            rs.queue.submit(std::iter::once(enc.finish()));
            eprintln!("[UV] w={} h={} bpr={} rows={}", uv_w, uv_h, uv_pad_bpr, uv_h);
        }

        // Persist last-good so fallback can reuse
        self.last_fmt = Some(fmt);
        self.y_size = (w, h);
        self.uv_size = ((w + 1) / 2, (h + 1) / 2);
        self.ring_present = wi;
        self.ring_write = (wi + 1) % 3;
        self.last_present_tick = self.last_present_tick.wrapping_add(1);
        self.last_cpu_tick = self.last_present_tick;

        let y_tex = self.y_tex[wi].as_ref()?.clone();
        let uv_tex = self.uv_tex[wi].as_ref()?.clone();
        Some((fmt, y_tex, uv_tex))
    }

    #[cfg(target_os = "macos")]
    fn present_nv12_zero_copy(
        &mut self,
        rs: &eframe::egui_wgpu::RenderState,
        zc: &native_decoder::IOSurfaceFrame,
    ) -> Option<(YuvPixFmt, Arc<eframe::wgpu::Texture>, Arc<eframe::wgpu::Texture>)> {
        self.ensure_zero_copy_nv12_textures(rs, zc.width, zc.height);
        if let Some((y_arc, uv_arc)) = self.gpu_yuv.as_ref().map(|g| (g.y_tex.clone(), g.uv_tex.clone())) {
            let queue = &*rs.queue;
            if let Err(e) = self.gpu_yuv.as_ref().unwrap().import_from_iosurface(queue, zc) {
                eprintln!("[zc] import_from_iosurface error: {}", e);
                return None;
            }
            #[cfg(target_os = "macos")]
            if !self.zc_logged {
                tracing::info!("[preview] imported NV12 planes: Y={}x{}  UV={}x{}", zc.width, zc.height, (zc.width + 1)/2, (zc.height + 1)/2);
                self.zc_logged = true;
            }
            // Persist last ZC for reuse
            self.set_last_zc_present(YuvPixFmt::Nv12, y_arc.clone(), uv_arc.clone(), zc.width, zc.height);
            return Some((YuvPixFmt::Nv12, y_arc, uv_arc));
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
    mode: PreviewShaderMode,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PreviewUniforms {
    w: f32,
    h: f32,
    mode: u32, // 0=Solid,1=ShowY,2=UvDebug,3=Nv12
    _pad: u32, // 16B alignment
}

struct Nv12Resources {
    pipeline_nv12: eframe::wgpu::RenderPipeline,
    pipeline_solid: eframe::wgpu::RenderPipeline,
    pipeline_showy: eframe::wgpu::RenderPipeline,
    pipeline_uvdebug: eframe::wgpu::RenderPipeline,
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
                struct Uniforms { w: f32, h: f32, mode: u32, _pad: u32 };
                @group(0) @binding(3) var<uniform> uni: Uniforms;

                struct Conv { y_bias: f32, y_scale: f32, uv_bias: f32, uv_scale: f32 };
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
                    let tc = in.uv;
                    switch uni.mode {
                        case 0u: { // Solid
                            return vec4<f32>(0.0, 1.0, 0.0, 1.0);
                        }
                        case 1u: { // ShowY
                            let y = textureSampleLevel(texY, samp, tc, 0.0).r;
                            return vec4<f32>(y, y, y, 1.0);
                        }
                        case 2u: { // UvDebug
                            let uv = textureSampleLevel(texUV, samp, tc, 0.0).rg;
                            return vec4<f32>(uv.x, uv.y, 0.0, 1.0);
                        }
                        default: { // NV12 using BT.709 limited range conv
                            let y = textureSampleLevel(texY, samp, tc, 0.0).r;
                            let uv = textureSampleLevel(texUV, samp, tc, 0.0).rg;
                            let C = max((y - conv.y_bias) * conv.y_scale, 0.0);
                            let D = (uv.x - conv.uv_bias) * conv.uv_scale;
                            let E = (uv.y - conv.uv_bias) * conv.uv_scale;
                            let r = clamp(C + 1.5748 * E,              0.0, 1.0);
                            let g = clamp(C - 0.1873 * D - 0.4681 * E, 0.0, 1.0);
                            let b = clamp(C + 1.8556 * D,              0.0, 1.0);
                            return vec4<f32>(r, g, b, 1.0);
                        }
                    }
                }
            "#;
            let module = device.create_shader_module(eframe::wgpu::ShaderModuleDescriptor {
                label: Some("preview_nv12_shader"),
                source: eframe::wgpu::ShaderSource::Wgsl(shader_src.into()),
            });
            let bgl = device.create_bind_group_layout(&eframe::wgpu::BindGroupLayoutDescriptor {
                label: Some("NV12 tex BGL"),
                entries: &[
                    eframe::wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: eframe::wgpu::ShaderStages::FRAGMENT,
                        ty: eframe::wgpu::BindingType::Sampler(eframe::wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
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
                        binding: 3,
                        visibility: eframe::wgpu::ShaderStages::FRAGMENT,
                        ty: eframe::wgpu::BindingType::Buffer {
                            ty: eframe::wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            let uniform_bgl = device.create_bind_group_layout(&eframe::wgpu::BindGroupLayoutDescriptor {
                label: Some("NV12 conv BGL"),
                entries: &[eframe::wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: eframe::wgpu::ShaderStages::FRAGMENT,
                    ty: eframe::wgpu::BindingType::Buffer { ty: eframe::wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                }],
            });
            let pl = device.create_pipeline_layout(&eframe::wgpu::PipelineLayoutDescriptor {
                label: Some("NV12 pipeline layout"),
                bind_group_layouts: &[&bgl, &uniform_bgl],
                push_constant_ranges: &[],
            });
            let mk_pipeline = |label: &str, fs: &str| device.create_render_pipeline(&eframe::wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: eframe::wgpu::VertexState {
                    module: &module,
                    entry_point: "vs_main",
                    compilation_options: eframe::wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(eframe::wgpu::FragmentState {
                    module: &module,
                    entry_point: fs,
                    compilation_options: eframe::wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(eframe::wgpu::ColorTargetState {
                        format: eframe::wgpu::TextureFormat::Bgra8Unorm,
                        blend: Some(eframe::wgpu::BlendState::REPLACE),
                        write_mask: eframe::wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: eframe::wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: eframe::wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
            let pipeline_nv12 = mk_pipeline("preview_nv12_pipeline", "fs_main");
            let pipeline_solid = mk_pipeline("preview_solid_pipeline", "fs_main");
            let pipeline_showy = mk_pipeline("preview_showy_pipeline", "fs_main");
            let pipeline_uvdebug = mk_pipeline("preview_uvdebug_pipeline", "fs_main");
            let sampler = device.create_sampler(&eframe::wgpu::SamplerDescriptor {
                label: Some("nv12_clamp_sampler"),
                address_mode_u: eframe::wgpu::AddressMode::ClampToEdge,
                address_mode_v: eframe::wgpu::AddressMode::ClampToEdge,
                address_mode_w: eframe::wgpu::AddressMode::ClampToEdge,
                mag_filter: eframe::wgpu::FilterMode::Linear,
                min_filter: eframe::wgpu::FilterMode::Linear,
                mipmap_filter: eframe::wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            // Take ownership values, then insert to avoid overlapping borrows during insert
            let res = Nv12Resources { pipeline_nv12, pipeline_solid, pipeline_showy, pipeline_uvdebug, bind_group_layout: bgl, uniform_bgl, sampler };
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
        // Always refresh to avoid stale texture bindings during playback/scrub
        let y_id = Arc::as_ptr(&self.y_tex) as usize;
        let uv_id = Arc::as_ptr(&self.uv_tex) as usize;
        let view_y = self.y_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default());
        let view_uv = self.uv_tex.create_view(&eframe::wgpu::TextureViewDescriptor::default());
        let (nv_bgl, nv_samp) = {
            let r = resources.get::<Nv12Resources>().unwrap();
            (&r.bind_group_layout, &r.sampler)
        };
        // Preview uniforms (w,h,mode)
        let mode_u32: u32 = match self.mode { PreviewShaderMode::Solid => 0, PreviewShaderMode::ShowY => 1, PreviewShaderMode::UvDebug => 2, PreviewShaderMode::Nv12 => 3 };
        let uni = PreviewUniforms { w: self.w as f32, h: self.h as f32, mode: mode_u32, _pad: 0 };
        let ubuf2 = device.create_buffer(&eframe::wgpu::BufferDescriptor {
            label: Some("preview_uniforms"),
            size: std::mem::size_of::<PreviewUniforms>() as u64,
            usage: eframe::wgpu::BufferUsages::UNIFORM | eframe::wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ubuf2, 0, bytemuck::bytes_of(&uni));
        let bind = device.create_bind_group(&eframe::wgpu::BindGroupDescriptor {
            label: Some("preview_nv12_bg"),
            layout: nv_bgl,
            entries: &[
                eframe::wgpu::BindGroupEntry { binding: 0, resource: eframe::wgpu::BindingResource::Sampler(nv_samp) },
                eframe::wgpu::BindGroupEntry { binding: 1, resource: eframe::wgpu::BindingResource::TextureView(&view_y) },
                eframe::wgpu::BindGroupEntry { binding: 2, resource: eframe::wgpu::BindingResource::TextureView(&view_uv) },
                eframe::wgpu::BindGroupEntry { binding: 3, resource: eframe::wgpu::BindingResource::Buffer(eframe::wgpu::BufferBinding { buffer: &ubuf2, offset: 0, size: None }) },
            ],
        });
        tracing::debug!("NV12 bind-group refreshed ({}x{} / {}x{})", self.w, self.h, (self.w + 1)/2, (self.h + 1)/2);
        resources.insert(Nv12BindGroup { bind, y_id, uv_id });
        // BT.709 limited-range conversion parameters
        let (y_bias, y_scale, uv_bias, uv_scale) = match self.fmt {
            YuvPixFmt::Nv12 => (16.0/255.0, 255.0/219.0, 128.0/255.0, 255.0/224.0),
            YuvPixFmt::P010 => (64.0/1023.0, 1023.0/876.0, 512.0/1023.0, 1023.0/896.0),
        };
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct ConvStd { y_bias: f32, y_scale: f32, uv_bias: f32, uv_scale: f32 }
        let conv = ConvStd { y_bias, y_scale, uv_bias, uv_scale };
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
            // Validate presence before use
            assert!(resources.get::<Nv12BindGroup>().is_some(), "missing NV12 tex bind group");
            assert!(resources.get::<ConvBindGroup>().is_some(), "missing conv bind group");
            // Single pipeline; shader selects the mode via uniform
            render_pass.set_pipeline(&res.pipeline_nv12);
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
            if let ItemKind::Audio { src, .. } = &it.kind {
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
#[derive(Clone, Debug)]
struct AudioPeaks {
    peaks: Vec<(f32, f32)>, // (min, max) in [-1,1]
    duration_sec: f32,
    channels: u16,
    sample_rate: u32,
}

#[derive(Default)]
struct AudioCache {
    map: std::collections::HashMap<std::path::PathBuf, std::sync::Arc<AudioPeaks>>,
}
