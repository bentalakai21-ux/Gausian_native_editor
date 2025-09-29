mod app_cloud;
mod app_assets;
mod app_modal;
mod app_project;
mod app_timeline;
mod app_ui;

pub(crate) use app_modal::{PhaseAgg, PhasePlan};

// Preview behavior settings (frame-based thresholds)
#[derive(Clone, Copy)]
struct PreviewSettings {
    // Accept frames within this many frames when strict-paused
    strict_tolerance_frames: f32,
    // Accept frames within this many frames when non-strict paused
    paused_tolerance_frames: f32,
    // Only clear the last frame on seek if the target moved beyond this many frames
    clear_threshold_frames: f32,
}

impl Default for PreviewSettings {
    fn default() -> Self {
        Self { strict_tolerance_frames: 2.5, paused_tolerance_frames: 2.0, clear_threshold_frames: 2.0 }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AppMode { ProjectPicker, Editor }

struct App {
    db: ProjectDb,
    project_id: String,
    import_path: String,
    // timeline state
    seq: Sequence,
    timeline_history: CommandHistory,
    zoom_px_per_frame: f32,
    playhead: i64,
    playing: bool,
    last_tick: Option<Instant>,
    // Anchored playhead timing to avoid jitter
    play_anchor_instant: Option<Instant>,
    play_anchor_frame: i64,
    preview: PreviewState,
    audio_out: Option<audio_engine::AudioEngine>,
    selected: Option<(usize, usize)>,
    drag: Option<DragState>,
    export: ExportUiState,
    import_workers: Vec<std::thread::JoinHandle<()>>,
    jobs: Option<jobs_crate::JobsHandle>,
    job_events: Vec<JobEvent>,
    show_jobs: bool,
    decode_mgr: DecodeManager,
    playback_clock: PlaybackClock,
    audio_cache: AudioCache,
    audio_buffers: AudioBufferCache,
    // When true during this frame, enable audible scrubbing while paused
    // Last successfully presented key: (source path, media time in milliseconds)
    // Using media time (not playhead frame) avoids wrong reuse when clips share a path but have different in_offset/rate.
    last_preview_key: Option<(String, i64)>,
    // Playback engine
    engine: EngineState,
    // Debounce decode commands: remember last sent (state, path, optional seek bucket)
    last_sent: Option<(PlayState, String, Option<i64>)>,
    // Epsilon-based dispatch tracking
    last_seek_sent_pts: Option<f64>,
    last_play_reanchor_time: Option<Instant>,
    // Throttled engine log state
    // (Used only for preview_ui logging when sending worker commands)
    // Not strictly necessary, but kept for future UI log hygiene.
    // last_engine_log: Option<Instant>,
    // Strict paused behavior toggle (UI)
    strict_pause: bool,
    // Track when a paused seek was requested (for overlay timing)
    last_seek_request_at: Option<Instant>,
    // Last presented frame PTS for current source (path, pts seconds)
    last_present_pts: Option<(String, f64)>,
    // User settings
    settings: PreviewSettings,
    show_settings: bool,
    // ComfyUI integration (Phase 1)
    comfy: crate::comfyui::ComfyUiManager,
    show_comfy_panel: bool,
    // Editable input for ComfyUI repo path (separate from committed config)
    comfy_repo_input: String,
    // Installer UI state
    comfy_install_dir_input: String,
    comfy_torch_backend: crate::comfyui::TorchBackend,
    comfy_venv_python_input: String,
    comfy_recreate_venv: bool,
    comfy_install_ffmpeg: bool,
    // Remote/Local ComfyUI job monitor
    comfy_ws_monitor: bool,
    comfy_ws_thread: Option<std::thread::JoinHandle<()>>,
    comfy_ws_stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    // Modal cloud job submission
    modal_enabled: bool,
    modal_base_url: String,
    modal_api_key: String,
    modal_payload: String,
    modal_logs: std::collections::VecDeque<String>,
    modal_rx: Receiver<ModalEvent>,
    modal_tx: Sender<ModalEvent>,
    // Cached recent jobs/artifacts from /healthz
    modal_recent: Vec<(String, Vec<(String, String)>)>,
    // Cloud (Modal) live monitor
    cloud_target: CloudTarget,
    // Optional cloud relay (WS/SSE over WS) endpoint for progress + artifacts
    modal_relay_ws_url: String,
    // Live cloud monitor state
    modal_queue_pending: usize,
    modal_queue_running: usize,
    modal_job_progress: std::collections::HashMap<String, (f32, u32, u32, std::time::Instant)>,
    modal_job_source: std::collections::HashMap<String, crate::CloudUpdateSrc>,
    modal_phase_plans: std::collections::HashMap<String, PhasePlan>,
    modal_phase_agg: std::collections::HashMap<String, PhaseAgg>,
    // Active job id; only this job's progress is shown/imported
    modal_active_job: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    // Track expected unique filename prefixes per job (for artifact filtering)
    modal_job_prefixes: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, String>>>,
    // Cloud monitor lifecycle
    modal_monitor_requested: bool,
    modal_last_progress_at: Option<Instant>,
    // Known jobs queued locally this session
    modal_known_jobs: std::collections::HashSet<String>,
    pip_index_url_input: String,
    pip_extra_index_url_input: String,
    pip_trusted_hosts_input: String,
    pip_proxy_input: String,
    pip_no_cache: bool,
    // Embedded ComfyUI webview
    comfy_embed_inside: bool,
    #[allow(dead_code)]
    comfy_webview: Option<Box<dyn crate::embed_webview::WebViewHost>>,
    comfy_devtools: bool,
    comfy_embed_logs: std::collections::VecDeque<String>,
    // Placement and sizing for embedded view
    comfy_embed_in_assets: bool,
    comfy_assets_height: f32,
    // Floating ComfyUI panel window visibility
    show_comfy_view_window: bool,
    // Auto-import from ComfyUI outputs
    comfy_auto_import: bool,
    comfy_import_logs: std::collections::VecDeque<String>,
    comfy_ingest_thread: Option<std::thread::JoinHandle<()>>,
    comfy_ingest_stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    comfy_ingest_rx: Receiver<(String, std::path::PathBuf)>,
    comfy_ingest_tx: Sender<(String, std::path::PathBuf)>,
    // Project id that the ingest thread is currently bound to (for routing)
    comfy_ingest_project_id: Option<String>,
    // Projects page
    show_projects: bool,
    new_project_name: String,
    new_project_base: String,
    // App mode: show project picker before opening editor
    mode: AppMode,
    // Autosave indicator
    last_save_at: Option<Instant>,
    // Assets UI: cached thumbnail textures
    asset_thumb_textures: std::collections::HashMap<String, egui::TextureHandle>,
    // Dragging asset from assets panel into timeline
    dragging_asset: Option<project::AssetRow>,
    // Assets UI: adjustable thumbnail width
    asset_thumb_w: f32,
    assets_drop_rect: Option<egui::Rect>,
    timeline_drop_rect: Option<egui::Rect>,
    pending_timeline_drops: Vec<PendingTimelineDrop>,
}

struct PendingTimelineDrop {
    path: std::path::PathBuf,
    track_hint: usize,
    frame: i64,
}

impl App {
    fn handle_external_file_drops(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if dropped.is_empty() {
            return;
        }

        let drop_pos = ctx
            .input(|i| i.pointer.interact_pos().or_else(|| i.pointer.hover_pos()));

        for file in dropped {
            let Some(path_buf) = file.path else { continue };
            let path = path_buf
                .canonicalize()
                .unwrap_or(path_buf.clone());

            let mut handled = false;

            if let Some(pos) = drop_pos {
                if !handled {
                    if let Some(rect) = self.timeline_drop_rect {
                        if rect.contains(pos) {
                            if let Some((track, frame)) = self.timeline_drop_target(pos) {
                                let _ = self.import_files(&[path.clone()]);
                                self.pending_timeline_drops.push(PendingTimelineDrop {
                                    path: path.clone(),
                                    track_hint: track,
                                    frame,
                                });
                                handled = true;
                            }
                        }
                    }
                }
                if !handled {
                    if let Some(rect) = self.assets_drop_rect {
                        if rect.contains(pos) {
                            let _ = self.import_files(&[path.clone()]);
                            handled = true;
                        }
                    }
                }
            }

            if !handled {
                let _ = self.import_files(&[path.clone()]);
            }
        }
    }

    fn process_pending_timeline_drops(&mut self) {
        let mut remaining = Vec::new();
        let pending_items = std::mem::take(&mut self.pending_timeline_drops);
        for pending in pending_items {
            let path_str = pending.path.to_string_lossy().to_string();
            match self.db.find_asset_by_path(&self.project_id, &path_str) {
                Ok(Some(asset)) => {
                    self.insert_asset_at(&asset, pending.track_hint, pending.frame);
                }
                Ok(None) | Err(_) => {
                    remaining.push(pending);
                }
            }
        }
        self.pending_timeline_drops = remaining;
    }

    fn timeline_drop_target(&self, pos: egui::Pos2) -> Option<(usize, i64)> {
        let rect = self.timeline_drop_rect?;
        let track_count = self.seq.graph.tracks.len();
        if track_count == 0 {
            return None;
        }
        let track_h = 48.0;
        let mut track = ((pos.y - rect.top()) / track_h).floor() as isize;
        track = track.clamp(0, track_count as isize - 1);
        let local_x = (pos.x - rect.left()).max(0.0) as f64;
        let zoom = self.zoom_px_per_frame.max(0.001) as f64;
        let frame = (local_x / zoom).round() as i64;
        Some((track as usize, frame.max(0)))
    }

    fn collect_modal_artifacts(
        candidates: &mut Vec<(String, String)>,
        arr_opt: Option<Vec<serde_json::Value>>,
    ) {
        if let Some(items) = arr_opt {
            for it in items {
                let name = it
                    .get("filename")
                    .and_then(|s| s.as_str())
                    .unwrap_or("");
                let url = it
                    .get("url")
                    .and_then(|s| s.as_str())
                    .unwrap_or("");
                if url.is_empty() {
                    continue;
                }
                if !name.to_ascii_lowercase().ends_with(".mp4") {
                    continue;
                }
                candidates.push((name.to_string(), url.to_string()));
            }
        }
    }

    // cloud/project methods moved to their modules
    fn new(db: ProjectDb) -> Self {
        let project_id = "default".to_string();
        let _ = db.ensure_project(&project_id, "Default Project", None);
        let mut seq = Sequence::new("Main", 1920, 1080, Fps::new(30, 1), 600);
        if seq.tracks.is_empty() {
            // Default to three video and three audio tracks
            for i in 1..=3 { seq.add_track(Track { name: format!("V{}", i), items: vec![] }); }
            for i in 1..=3 { seq.add_track(Track { name: format!("A{}", i), items: vec![] }); }
        }
        seq.graph = timeline_crate::migrate_sequence_tracks(&seq);
        let db_path = db.path().to_path_buf();
        let mut app = Self {
            db,
            project_id,
            import_path: String::new(),
            seq,
            timeline_history: CommandHistory::default(),
            zoom_px_per_frame: 2.0,
            playhead: 0,
            playing: false,
            last_tick: None,
            play_anchor_instant: None,
            play_anchor_frame: 0,
            preview: PreviewState::new(),
            audio_out: audio_engine::AudioEngine::new().ok(),
            selected: None,
            drag: None,
            export: ExportUiState::default(),
            import_workers: Vec::new(),
            jobs: Some(jobs_crate::JobsRuntime::start(db_path, 2)),
            job_events: Vec::new(),
            show_jobs: false,
            decode_mgr: DecodeManager::default(),
            playback_clock: PlaybackClock { rate: 1.0, ..Default::default() },
            audio_cache: AudioCache::default(),
            audio_buffers: AudioBufferCache::default(),
            last_preview_key: None,
            engine: EngineState { state: PlayState::Paused, rate: 1.0, target_pts: 0.0 },
            last_sent: None,
            last_seek_sent_pts: None,
            last_play_reanchor_time: None,
            strict_pause: true,
            last_seek_request_at: None,
            last_present_pts: None,
            settings: PreviewSettings::default(),
            show_settings: false,
            comfy: crate::comfyui::ComfyUiManager::default(),
            show_comfy_panel: false,
            comfy_repo_input: String::new(),
            comfy_install_dir_input: crate::comfyui::ComfyUiManager::default_install_dir()
                .to_string_lossy()
                .to_string(),
            comfy_torch_backend: crate::comfyui::TorchBackend::Auto,
            comfy_venv_python_input: String::new(),
            comfy_recreate_venv: false,
            comfy_install_ffmpeg: true,
            comfy_ws_monitor: false,
            comfy_ws_thread: None,
            comfy_ws_stop: None,
            modal_enabled: true,
            modal_base_url: String::new(),
            modal_api_key: String::new(),
            modal_payload: String::from("{\n  \"workflow\": \"basic-video\",\n  \"params\": { \"width\": 1920, \"height\": 1080, \"fps\": 30, \"seconds\": 5 }\n}"),
            modal_logs: std::collections::VecDeque::with_capacity(256),
            modal_rx: {
                let (_tx, rx) = unbounded();
                rx
            },
            modal_tx: {
                let (tx, _rx) = unbounded();
                tx
            },
            cloud_target: CloudTarget::Prompt,
            modal_relay_ws_url: String::new(),
            modal_queue_pending: 0,
            modal_queue_running: 0,
            modal_job_progress: std::collections::HashMap::new(),
            modal_job_source: std::collections::HashMap::new(),
            modal_phase_plans: std::collections::HashMap::new(),
            modal_phase_agg: std::collections::HashMap::new(),
            modal_active_job: std::sync::Arc::new(std::sync::Mutex::new(None)),
            modal_job_prefixes: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            modal_recent: Vec::new(),
            modal_monitor_requested: false,
            modal_last_progress_at: None,
            modal_known_jobs: std::collections::HashSet::new(),
            pip_index_url_input: String::new(),
            pip_extra_index_url_input: String::new(),
            pip_trusted_hosts_input: String::new(),
            pip_proxy_input: String::new(),
            pip_no_cache: false,
            // Default to not opening ComfyUI inside the editor
            comfy_embed_inside: false,
            comfy_webview: None,
            comfy_devtools: false,
            comfy_embed_logs: std::collections::VecDeque::with_capacity(128),
            comfy_embed_in_assets: true,
            comfy_assets_height: 320.0,
            show_comfy_view_window: true,
            comfy_auto_import: true,
            comfy_import_logs: std::collections::VecDeque::with_capacity(256),
            comfy_ingest_thread: None,
            comfy_ingest_stop: None,
            // channel will be set below
            comfy_ingest_rx: {
                let (_tx, rx) = unbounded::<(String, std::path::PathBuf)>();
                rx
            },
            comfy_ingest_tx: {
                let (tx, _rx) = unbounded::<(String, std::path::PathBuf)>();
                tx
            },
            comfy_ingest_project_id: None,
            show_projects: false,
            new_project_name: String::new(),
            new_project_base: String::new(),
            mode: AppMode::ProjectPicker,
            last_save_at: None,
            asset_thumb_textures: std::collections::HashMap::new(),
            dragging_asset: None,
            asset_thumb_w: 148.0,
            assets_drop_rect: None,
            timeline_drop_rect: None,
            pending_timeline_drops: Vec::new(),
        };
        // Replace placeholder channels with a real pair
        let (tx, rx) = unbounded();
        app.comfy_ingest_tx = tx;
        app.comfy_ingest_rx = rx;
        // Modal events channel
        let (mtx, mrx) = unbounded();
        app.modal_tx = mtx;
        app.modal_rx = mrx;
        // Initialize ComfyUI repo input from current config (if any)
        if let Some(p) = app.comfy.config().repo_path.as_ref() {
            app.comfy_repo_input = p.to_string_lossy().to_string();
        }
        app.sync_tracks_from_graph();
        app
    }

    // asset/timeline helpers moved to their modules

}

// Best-effort converter from a generic "workflow" JSON into a ComfyUI /prompt payload.
// This is intentionally conservative: it tries to recognize a "nodes" array and
// build a minimal prompt map with class_type and any provided literal inputs.
// Complex graph links are not guaranteed to convert; if conversion isn't possible,
// returns an Err with a helpful message.
fn convert_workflow_to_prompt(workflow_json: &str) -> Result<String, String> { app_cloud::convert_workflow_to_prompt(workflow_json) }

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Drain modal events and append to logs
        while let Ok(ev) = self.modal_rx.try_recv() {
            match ev {
                ModalEvent::Log(s) => {
                    self.modal_logs.push_back(s);
                    while self.modal_logs.len() > 256 { self.modal_logs.pop_front(); }
                }
                ModalEvent::JobQueued(id) => {
                    self.modal_logs.push_back(format!("Queued job: {}", id));
                    // Build a phase plan from current payload for this job id
                    let plan = Self::compute_phase_plan_from_payload(&self.modal_payload);
                    self.modal_phase_plans.insert(id.clone(), plan);
                    self.modal_phase_agg.entry(id.clone()).or_insert_with(PhaseAgg::default);
                    // Ensure monitor is requested when a job is queued
                    self.modal_monitor_requested = true;
                    // Track as known job this session
                    self.modal_known_jobs.insert(id.clone());
                }
                ModalEvent::JobQueuedWithPrefix(id, prefix) => {
                    // Mark this as the active job and clear previous progress bars
                    if let Ok(mut a) = self.modal_active_job.lock() { *a = Some(id.clone()); }
                    self.modal_phase_agg.clear();
                    self.modal_job_progress.clear();
                    self.modal_phase_plans.clear();
                    // Also prune known jobs and prefix map to only this id
                    self.modal_known_jobs.clear();
                    self.modal_known_jobs.insert(id.clone());
                    if let Ok(mut m) = self.modal_job_prefixes.lock() { m.retain(|k, _| k == &id); }
                    // Seed phase plan and an empty aggregate entry so the UI shows a placeholder bar immediately
                    let plan = Self::compute_phase_plan_from_payload(&self.modal_payload);
                    self.modal_phase_plans.insert(id.clone(), plan);
                    self.modal_phase_agg.insert(id.clone(), PhaseAgg::default());
                    // Remember expected output filename prefix for this job
                    if let Ok(mut m) = self.modal_job_prefixes.lock() { m.insert(id.clone(), prefix.clone()); }
                    // Lightweight per-job poller for /jobs/{id} to drive progress/import without /progress-status
                    let http_base = {
                        let mut base = self.modal_base_url.trim().to_string();
                        if !base.starts_with("http://") && !base.starts_with("https://") {
                            base = format!("https://{}", base);
                        }
                        if base.ends_with("/health") { base = base[..base.len()-"/health".len()].trim_end_matches('/').to_string(); }
                        if base.ends_with("/healthz") { base = base[..base.len()-"/healthz".len()].trim_end_matches('/').to_string(); }
                        base
                    };
                    // Re-introduce a lightweight /jobs/{id} poller while the job is running
                    // to keep progress fresh when WS events are sparse. It stops as soon as
                    // the job is completed to avoid any cold starts.
                    let token = self.modal_api_key.clone();
                    let jid = id.clone();
                    let tx_log = self.modal_tx.clone();
                    let tx_import = self.comfy_ingest_tx.clone();
                    let proj_id = self.project_id.clone();
                    let app_tmp = project::app_data_dir().join("tmp").join("cloud");
                    let _ = std::fs::create_dir_all(&app_tmp);
                    let active_job = self.modal_active_job.clone();
                    let job_prefixes = self.modal_job_prefixes.clone();
                    std::thread::spawn(move || {
                        use std::time::Duration;
                        loop {
                            // Exit if a different job became active or job cleared
                            let still_active = active_job
                                .lock()
                                .ok()
                                .and_then(|a| a.clone())
                                .map(|cur| cur == jid)
                                .unwrap_or(false);
                            if !still_active { break; }
                            // Poll job state
                            let job_url = format!("{}/jobs/{}", http_base.trim_end_matches('/'), jid);
                            let mut req = ureq::get(&job_url);
                            if !token.trim().is_empty() {
                                req = req.set("Authorization", &format!("Bearer {}", token));
                            }
                            if let Ok(resp) = req.call() {
                                if let Ok(body) = resp.into_string() {
                                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                                        let pr = v
                                            .get("progress_percent")
                                            .and_then(|x| x.as_f64())
                                            .unwrap_or(0.0) as f32;
                                        let cur = v
                                            .get("current_step")
                                            .and_then(|x| x.as_u64())
                                            .unwrap_or(0) as u32;
                                        let tot = v
                                            .get("total_steps")
                                            .and_then(|x| x.as_u64())
                                            .unwrap_or(0) as u32;
                                        let _ = tx_log.send(ModalEvent::CloudProgress {
                                            job_id: jid.clone(),
                                            progress: pr,
                                            current: cur,
                                            total: tot,
                                            node_id: None,
                                        });
                                        let _ = tx_log.send(ModalEvent::CloudSource {
                                            job_id: jid.clone(),
                                            source: crate::CloudUpdateSrc::Jobs,
                                        });
                                        // If no totals yet, try the headless /progress-status for richer details
                                        if (tot == 0) || (pr == 0.0 && cur == 0) {
                                            let status_url = format!("{}/progress-status", http_base.trim_end_matches('/'));
                                            let mut sreq = ureq::get(&status_url);
                                            if !token.trim().is_empty() { sreq = sreq.set("Authorization", &format!("Bearer {}", token)); }
                                            if let Ok(sresp) = sreq.call() {
                                                if let Ok(sbody) = sresp.into_string() {
                                                    if let Ok(sv) = serde_json::from_str::<serde_json::Value>(&sbody) {
                                                        if let Some(arr) = sv.get("job_details").and_then(|a| a.as_array()) {
                                                            for it in arr {
                                                                let sid = it.get("job_id").and_then(|s| s.as_str()).unwrap_or("");
                                                                if sid == jid {
                                                                    let spr = it.get("progress_percent").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                                                                    let scur = it.get("current_step").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                                                    let stot = it.get("total_steps").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                                                    let _ = tx_log.send(ModalEvent::CloudProgress { job_id: jid.clone(), progress: spr, current: scur, total: stot, node_id: None });
                                                                    let _ = tx_log.send(ModalEvent::CloudSource { job_id: jid.clone(), source: crate::CloudUpdateSrc::Status });
                                                                    break;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        // Stop polling promptly when completed; optionally import artifacts
                                        let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("");
                                        if status == "completed" {
                                            // Only import if still the active job (avoid double-import if WS already handled it)
                                            let still_active2 = active_job
                                                .lock()
                                                .ok()
                                                .and_then(|a| a.clone())
                                                .map(|cur| cur == jid)
                                                .unwrap_or(false);
                                            if still_active2 {
                                                let prefix_opt = job_prefixes
                                                    .lock()
                                                    .ok()
                                                    .and_then(|m| m.get(&jid).cloned());
                                                let _ = tx_log.send(ModalEvent::JobImporting(jid.clone()));

                                                let mut candidates: Vec<(String, String)> = Vec::new();
                                                Self::collect_modal_artifacts(
                                                    &mut candidates,
                                                    v.get("artifacts").and_then(|a| a.as_array()).cloned(),
                                                );

                                                if candidates.is_empty() {
                                                    let art_url = format!(
                                                        "{}/artifacts/{}",
                                                        http_base.trim_end_matches('/'),
                                                        jid
                                                    );
                                                    let mut areq = ureq::get(&art_url);
                                                    if !token.trim().is_empty() {
                                                        areq = areq.set("Authorization", &format!("Bearer {}", token));
                                                    }
                                                    if let Ok(aresp) = areq.call() {
                                                        if let Ok(abody) = aresp.into_string() {
                                                            if let Ok(av) = serde_json::from_str::<serde_json::Value>(&abody) {
                                                                Self::collect_modal_artifacts(
                                                                    &mut candidates,
                                                                    av.get("artifacts")
                                                                        .and_then(|a| a.as_array())
                                                                        .cloned(),
                                                                );
                                                            }
                                                        }
                                                    }
                                                }

                                                if candidates.is_empty() {
                                                    let mut tries = 0u8;
                                                    while tries < 5 && candidates.is_empty() {
                                                        std::thread::sleep(std::time::Duration::from_millis(300));
                                                        let job_url = format!(
                                                            "{}/jobs/{}",
                                                            http_base.trim_end_matches('/'),
                                                            jid
                                                        );
                                                        let mut req2 = ureq::get(&job_url);
                                                        if !token.trim().is_empty() {
                                                            req2 = req2.set("Authorization", &format!("Bearer {}", token));
                                                        }
                                                        if let Ok(resp2) = req2.call() {
                                                            if let Ok(body2) = resp2.into_string() {
                                                                if let Ok(v2) = serde_json::from_str::<serde_json::Value>(&body2) {
                                                                    Self::collect_modal_artifacts(
                                                                        &mut candidates,
                                                                        v2.get("artifacts")
                                                                            .and_then(|a| a.as_array())
                                                                            .cloned(),
                                                                    );
                                                                }
                                                            }
                                                        }
                                                        tries += 1;
                                                    }
                                                }

                                                if candidates.is_empty() {
                                                    let hz_url = format!(
                                                        "{}/healthz",
                                                        http_base.trim_end_matches('/')
                                                    );
                                                    let mut hreq = ureq::get(&hz_url);
                                                    if !token.trim().is_empty() {
                                                        hreq = hreq.set("Authorization", &format!("Bearer {}", token));
                                                    }
                                                    if let Ok(hresp) = hreq.call() {
                                                        if let Ok(hbody) = hresp.into_string() {
                                                            if let Ok(hv) = serde_json::from_str::<serde_json::Value>(&hbody) {
                                                                if let Some(recent) = hv.get("recent").and_then(|a| a.as_array()) {
                                                                    for job in recent {
                                                                        if job.get("id").and_then(|s| s.as_str()) == Some("outputs") {
                                                                            if let Some(arts) = job.get("artifacts").and_then(|a| a.as_array()) {
                                                                                Self::collect_modal_artifacts(
                                                                                    &mut candidates,
                                                                                    Some(arts.to_vec()),
                                                                                );
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }

                                                let mut seen_names: std::collections::HashSet<String> =
                                                    std::collections::HashSet::new();
                                                let mut downloaded: Vec<String> = Vec::new();
                                                let mut ack_names: Vec<String> = Vec::new();

                                                let mut attempt_download = |orig_name: &str, url: &str| -> bool {
                                                    let base_name = std::path::Path::new(orig_name)
                                                        .file_name()
                                                        .and_then(|s| s.to_str())
                                                        .unwrap_or(orig_name);
                                                    if !seen_names.insert(base_name.to_string()) {
                                                        return false;
                                                    }
                                                    let mut req = ureq::get(url);
                                                    if !token.trim().is_empty() {
                                                        req = req.set("Authorization", &format!("Bearer {}", token));
                                                    }
                                                    match req.call() {
                                                        Ok(resp) => {
                                                            let mut reader = resp.into_reader();
                                                            let dest = app_tmp.join(base_name);
                                                            if let Some(parent) = dest.parent() {
                                                                let _ = std::fs::create_dir_all(parent);
                                                            }
                                                            match std::fs::File::create(&dest) {
                                                                Ok(mut f) => {
                                                                    if std::io::copy(&mut reader, &mut f).is_ok() {
                                                                        let _ = tx_import.send((proj_id.clone(), dest));
                                                                        downloaded.push(base_name.to_string());
                                                                        ack_names.push(orig_name.to_string());
                                                                        let _ = tx_log.send(ModalEvent::Log(format!(
                                                                            "Downloaded {}",
                                                                            base_name
                                                                        )));
                                                                        true
                                                                    } else {
                                                                        let _ = tx_log.send(ModalEvent::Log(format!(
                                                                            "Download failed (write) {}",
                                                                            base_name
                                                                        )));
                                                                        false
                                                                    }
                                                                }
                                                                Err(e) => {
                                                                    let _ = tx_log.send(ModalEvent::Log(format!(
                                                                        "Download failed (create) {}: {}",
                                                                        base_name, e
                                                                    )));
                                                                    false
                                                                }
                                                            }
                                                        }
                                                        Err(e) => {
                                                            let _ = tx_log.send(ModalEvent::Log(format!(
                                                                "Download failed {}: {}",
                                                                base_name, e
                                                            )));
                                                            false
                                                        }
                                                    }
                                                };

                                                // Prefer strict prefix matches when available; otherwise fall back to any mp4 candidate
                                                let strict_only = if let Some(pref) = prefix_opt.as_ref() {
                                                    let mut c = 0usize;
                                                    for (name, _url) in candidates.iter() {
                                                        let base_name = std::path::Path::new(name)
                                                            .file_name()
                                                            .and_then(|s| s.to_str())
                                                            .unwrap_or(name);
                                                        if base_name.starts_with(pref) { c += 1; }
                                                    }
                                                    c > 0
                                                } else { false };

                                                for (name, url) in candidates.iter() {
                                                    if strict_only {
                                                        if let Some(pref) = prefix_opt.as_ref() {
                                                            let base_name = std::path::Path::new(name)
                                                                .file_name()
                                                                .and_then(|s| s.to_str())
                                                                .unwrap_or(name);
                                                            if !base_name.starts_with(pref) { continue; }
                                                        }
                                                    }
                                                    let _ = attempt_download(name, url);
                                                }

                                                if downloaded.is_empty() {
                                                    // Fallback: attempt canonical /view/{job_id}
                                                    let view_url = format!(
                                                        "{}/view/{}",
                                                        http_base.trim_end_matches('/'),
                                                        jid
                                                    );
                                                    let mut vreq = ureq::get(&view_url);
                                                    if !token.trim().is_empty() { vreq = vreq.set("Authorization", &format!("Bearer {}", token)); }
                                                    match vreq.call() {
                                                        Ok(vresp) => {
                                                            let mut reader = vresp.into_reader();
                                                            let fallback_name = if let Some(pref) = prefix_opt.as_ref() {
                                                                format!("{}-view.mp4", pref)
                                                            } else { format!("{}.mp4", jid) };
                                                            let tmp = app_tmp.join(&fallback_name);
                                                            if let Some(parent) = tmp.parent() { let _ = std::fs::create_dir_all(parent); }
                                                            if let Ok(mut f) = std::fs::File::create(&tmp) {
                                                                let _ = std::io::copy(&mut reader, &mut f);
                                                                let _ = tx_import.send((proj_id.clone(), tmp.clone()));
                                                                let _ = tx_log.send(ModalEvent::Log(format!(
                                                                    "Downloaded via /view/{{job_id}} → queued import: {}",
                                                                    tmp.to_string_lossy()
                                                                )));
                                                            }
                                                        }
                                                        Err(_) => {
                                                            let _ = tx_log.send(ModalEvent::Log(format!(
                                                                "No downloadable artifacts found for {}",
                                                                jid
                                                            )));
                                                        }
                                                    }
                                                } else {
                                                    ack_names.sort();
                                                    ack_names.dedup();
                                                    let ack_url = format!(
                                                        "{}/jobs/{}/imported",
                                                        http_base.trim_end_matches('/'),
                                                        jid
                                                    );
                                                    let mut preq =
                                                        ureq::post(&ack_url).set("Content-Type", "application/json");
                                                    if !token.trim().is_empty() {
                                                        preq = preq.set("Authorization", &format!("Bearer {}", token));
                                                    }
                                                    let body = serde_json::json!({ "filenames": ack_names });
                                                    let body_json = body.to_string();
                                                    let _ = preq.send_string(&body_json);
                                                }
                                            }
                                            // Clear active job and notify imported
                                            if let Ok(mut a) = active_job.lock() { *a = None; }
                                            let _ = tx_log.send(ModalEvent::JobImported(jid.clone()));
                                            break;
                                        }
                                    }
                                }
                            }
                            std::thread::sleep(Duration::from_millis(2000));
                        }
                    });
                }
                ModalEvent::CloudStatus { pending, running } => {
                    self.modal_queue_pending = pending;
                    self.modal_queue_running = running;
                    // Treat any queue activity as recent progress
                    if pending + running > 0 { self.modal_last_progress_at = Some(Instant::now()); }
                }
                ModalEvent::CloudProgress { job_id, progress, current, total, node_id } => {
                    // Only track progress for the active job
                    let is_active = self.modal_active_job.lock().ok().and_then(|a| a.clone()).map(|id| id == job_id).unwrap_or(true);
                    if !is_active { continue; }
                    self.modal_job_progress.insert(job_id.clone(), (progress, current, total, std::time::Instant::now()));
                    // Ensure an aggregate entry exists even if the queued id differed (e.g., prompt_id vs job_id)
                    let _ = self.modal_phase_agg.entry(job_id.clone()).or_insert_with(PhaseAgg::default);
                    // Track as known job this session
                    self.modal_known_jobs.insert(job_id.clone());
                    if let Some(agg) = self.modal_phase_agg.get_mut(&job_id) {
                        // Map node to phase using plan
                        let phase = if let Some(nid) = node_id.as_ref() {
                            if self.modal_phase_plans.get(&job_id).map(|p| p.sampling.contains(nid)).unwrap_or(false) { Some("s") }
                            else if self.modal_phase_plans.get(&job_id).map(|p| p.encoding.contains(nid)).unwrap_or(false) { Some("e") }
                            else { None }
                        } else { None };
                        match phase {
                            Some("s") => { agg.s_cur = agg.s_cur.max(current); agg.s_tot = agg.s_tot.max(total); }
                            Some("e") => { agg.e_cur = agg.e_cur.max(current); agg.e_tot = agg.e_tot.max(total); }
                            _ => {
                                // Heuristic if node unknown
                                if total >= 32 { agg.e_cur = agg.e_cur.max(current); agg.e_tot = agg.e_tot.max(total); }
                                else { agg.s_cur = agg.s_cur.max(current); agg.s_tot = agg.s_tot.max(total); }
                            }
                        }
                    }
                    self.modal_last_progress_at = Some(Instant::now());
                }
                ModalEvent::CloudSource { job_id, source } => {
                    // Only track source for the active job
                    let is_active = self.modal_active_job.lock().ok().and_then(|a| a.clone()).map(|id| id == job_id).unwrap_or(true);
                    if is_active { self.modal_job_source.insert(job_id, source); }
                }
                ModalEvent::JobImporting(jid) => { if let Some(a) = self.modal_phase_agg.get_mut(&jid) { a.importing = true; } }
                ModalEvent::JobImported(jid) => {
                    self.modal_phase_agg.remove(&jid);
                    self.modal_phase_plans.remove(&jid);
                    self.modal_job_progress.remove(&jid);
                    if let Ok(mut a) = self.modal_active_job.lock() {
                        if a.as_ref().map(|cur| cur == &jid).unwrap_or(false) {
                            *a = None;
                        }
                    }
                    self.modal_known_jobs.remove(&jid);
                    self.modal_monitor_requested = false;
                }
                ModalEvent::Recent(list) => {
                    self.modal_recent = list;
                }
            }
        }
        // Drain any completed files from ComfyUI ingest and import them
        while let Ok((proj_id, path)) = self.comfy_ingest_rx.try_recv() {
            // Determine project base path
            let mut base = self
                .db
                .get_project_base_path(&proj_id)
                .ok()
                .flatten()
                .unwrap_or_else(|| {
                    // Default base under app data dir if not set
                    let p = project::app_data_dir().join("projects").join(&proj_id);
                    let _ = std::fs::create_dir_all(&p);
                    let _ = self.db.set_project_base_path(&proj_id, &p);
                    p
                });
            // If base was incorrectly set to a file (e.g., from single-file import), use its parent dir.
            if base.is_file() {
                if let Some(parent) = base.parent() {
                    let parent = parent.to_path_buf();
                    let _ = self.db.set_project_base_path(&proj_id, &parent);
                    base = parent;
                }
            }
            let media_dir = base.join("media").join("comfy");
            let date = chrono::Local::now().format("%Y-%m-%d").to_string();
            let dest_dir = media_dir.join(date);
            let _ = std::fs::create_dir_all(&dest_dir);
            let file_name = std::path::Path::new(&path)
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "output.mp4".to_string());
            let mut dest = dest_dir.join(&file_name);
            // Ensure unique name
            if dest.exists() {
                let stem = dest
                    .file_stem()
                    .and_then(|s| Some(s.to_string_lossy().to_string()))
                    .unwrap_or_else(|| "output".to_string());
                let ext = dest.extension().and_then(|e| Some(e.to_string_lossy().to_string()));
                let mut i = 1;
                loop {
                    let candidate = dest_dir.join(format!(
                        "{}-{}.{}",
                        stem,
                        i,
                        ext.as_deref().unwrap_or("mp4")
                    ));
                    if !candidate.exists() {
                        dest = candidate;
                        break;
                    }
                    i += 1;
                }
            }
            // True move semantics: try rename; on cross-device or other failures, copy then delete.
            let mut did_move = false;
            match std::fs::rename(&path, &dest) {
                Ok(_) => { did_move = true; }
                Err(rename_err) => {
                    match std::fs::copy(&path, &dest) {
                        Ok(_) => {
                            // Best-effort remove original after successful copy
                            if let Err(rem_err) = std::fs::remove_file(&path) {
                                self.comfy_import_logs.push_back(format!(
                                    "Warning: copied (fallback) but failed to remove original {}: {}",
                                    path.to_string_lossy(), rem_err
                                ));
                            }
                        }
                        Err(copy_err) => {
                            self.comfy_import_logs.push_back(format!(
                                "Import move failed (rename: {}, copy: {}) {} -> {}",
                                rename_err,
                                copy_err,
                                path.to_string_lossy(),
                                dest.to_string_lossy(),
                            ));
                            continue; // Skip import on failure
                        }
                    }
                }
            }
            let _ = self.import_files_for(&proj_id, &[dest.clone()]);
            self.comfy_import_logs.push_back(if did_move {
                format!("Moved into {}: {}", proj_id, dest.to_string_lossy())
            } else {
                format!("Copied into {}: {}", proj_id, dest.to_string_lossy())
            });
        }
        // Start/stop ingest thread depending on state
        // Auto-import does not strictly require the server to be running;
        // as long as the ComfyUI repo/output folder is known, watch it.
        // If the open project changes, restart the watcher so events are
        // attributed to the project that was active when detection started.
        if let Some(pid) = &self.comfy_ingest_project_id {
            if Some(pid) != Some(&self.project_id) {
                if let Some(flag) = &self.comfy_ingest_stop { flag.store(true, Ordering::Relaxed); }
                if let Some(h) = self.comfy_ingest_thread.take() { let _ = h.join(); }
                self.comfy_ingest_stop = None;
                self.comfy_ingest_project_id = None;
            }
        }
        let out_dir_cfg = self
            .comfy
            .config()
            .repo_path
            .as_ref()
            .map(|p| p.join("output"));
        let can_watch = out_dir_cfg.as_ref().map(|d| d.exists()).unwrap_or(false);
        if self.comfy_auto_import && can_watch {
            if self.comfy_ingest_thread.is_none() {
                if let Some(dir) = out_dir_cfg {
                    let dir_s = dir.to_string_lossy().to_string();
                    let stop = Arc::new(AtomicBool::new(false));
                    let tx = self.comfy_ingest_tx.clone();
                    let dir_clone = dir.clone();
                    let start_pid = self.project_id.clone();
                    let pid_for_thread = start_pid.clone();
                    let handle = std::thread::spawn({
                        let stop = Arc::clone(&stop);
                        move || {
                            use std::collections::{HashMap, HashSet};
                            use std::thread::sleep;
                            let mut seen: HashSet<String> = HashSet::new();
                            let mut stable: HashMap<String, (u64, u8)> = HashMap::new();
                            let allowed_exts = [
                                // videos
                                "mp4", "mov", "webm", "mkv", "avi", "gif",
                                // images
                                "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "exr",
                            ];
                            while !stop.load(Ordering::Relaxed) {
                                for entry in WalkDir::new(&dir_clone).into_iter().filter_map(|e| e.ok()) {
                                    if !entry.file_type().is_file() { continue; }
                                    let p = entry.path();
                                    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
                                    if !allowed_exts.contains(&ext.as_str()) { continue; }
                                    let key = p.to_string_lossy().to_string();
                                    if seen.contains(&key) { continue; }
                                    if let Ok(md) = entry.metadata() {
                                        let size = md.len();
                                        match stable.get_mut(&key) {
                                            Some((last, hits)) => {
                                                if *last == size {
                                                    *hits += 1;
                                                    if *hits >= 3 {
                                                        let _ = tx.send((pid_for_thread.clone(), p.to_path_buf()));
                                                        stable.remove(&key);
                                                        seen.insert(key.clone());
                                                    }
                                                } else {
                                                    *last = size; *hits = 0;
                                                }
                                            }
                                            None => { stable.insert(key.clone(), (size, 0)); }
                                        }
                                    }
                                }
                                sleep(std::time::Duration::from_millis(700));
                            }
                        }
                    });
                    self.comfy_ingest_stop = Some(stop);
                    self.comfy_ingest_thread = Some(handle);
                    self.comfy_ingest_project_id = Some(start_pid);
                    self.comfy_import_logs.push_back(format!(
                        "Watching Comfy outputs: {}",
                        dir_s
                    ));
                }
            }
        } else {
            if let Some(flag) = &self.comfy_ingest_stop { flag.store(true, Ordering::Relaxed); }
            if let Some(h) = self.comfy_ingest_thread.take() { let _ = h.join(); }
            self.comfy_ingest_stop = None;
            self.comfy_ingest_project_id = None;
        }

        // Cloud (Modal) websocket monitor removed; rely on HTTP polling.
        self.modal_monitor_requested = false;

// Start/stop ComfyUI WebSocket job monitor
        // Runs regardless of embed; needs host/port and monitor toggle
        let ws_needed = self.comfy_ws_monitor;
        let ws_running = self.comfy_ws_thread.is_some();
        if ws_needed && !ws_running {
            let host = self.comfy.config().host.clone();
            let port = self.comfy.config().port;
            if !host.trim().is_empty() && port > 0 {
                let scheme_ws = if self.comfy.config().https { "wss" } else { "ws" };
                let scheme_http = if self.comfy.config().https { "https" } else { "http" };
                let ws_url = format!("{}://{}:{}/ws", scheme_ws, host, port);
                let http_base = format!("{}://{}:{}", scheme_http, host, port);
                let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let tx = self.comfy_ingest_tx.clone();
                let proj_id_at_start = self.project_id.clone();
                let app_data = project::app_data_dir().join("tmp").join("comfy");
                let handle = std::thread::spawn({
                    let stop = std::sync::Arc::clone(&stop);
                    move || {
                        use tungstenite::Message;
                        use serde_json::Value;
                        let _ = std::fs::create_dir_all(&app_data);
                        let mut backoff_ms: u64 = 500;
                        loop {
                            if stop.load(std::sync::atomic::Ordering::Relaxed) { break; }
                            // Connect
                            let url = match url::Url::parse(&ws_url) { Ok(u) => u, Err(_) => break };
                            match tungstenite::client::connect(url) {
                                Ok((mut socket, _)) => {
                                    backoff_ms = 500;
                                    loop {
                                        if stop.load(std::sync::atomic::Ordering::Relaxed) { break; }
                                        match socket.read_message() {
                                            Ok(Message::Text(txt)) => {
                                                if let Ok(v) = serde_json::from_str::<Value>(&txt) {
                                                    let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                                    if typ == "execution_end" {
                                                        // Do not fetch via ComfyUI /view here.
                                                        // Cloud jobs upload artifacts to object storage
                                                        // and are imported by the cloud monitor (/events + /jobs).
                                                    }
                                                }
                                            }
                                            Ok(Message::Binary(_)) => {}
                                            Ok(_) => {}
                                            Err(_) => { break; }
                                        }
                                    }
                                }
                                Err(_) => {
                                    // backoff
                                    std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                                    backoff_ms = (backoff_ms * 2).min(8_000);
                                }
                            }
                        }
                    }
                });
                self.comfy_ws_stop = Some(stop);
                self.comfy_ws_thread = Some(handle);
            }
        } else if !ws_needed && ws_running {
            if let Some(flag) = &self.comfy_ws_stop { flag.store(true, Ordering::Relaxed); }
            // Do not join here to avoid blocking UI if the socket is waiting; let it exit on its own.
            let _ = self.comfy_ws_thread.take();
            self.comfy_ws_stop = None;
        }
        // Push-driven repaint is primary (worker triggers request_repaint on new frames).
        // Safety net: ensure periodic UI updates even if no frames arrive.
        if self.engine.state == PlayState::Playing {
            // Try to pace by the active clip fps, bounded by the sequence fps.
            let seq_fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
            let t_playhead = self.playback_clock.now();
            let active_fps = if let Some((path, _)) = self.active_video_media_time_graph(t_playhead) {
                if let Some(latest) = self.decode_mgr.take_latest(&path) { latest.props.fps as f64 } else { f64::NAN }
            } else {
                f64::NAN
            };
            let clip_fps = if active_fps.is_finite() && active_fps > 0.0 { active_fps } else { seq_fps };
            let target_fps = clip_fps.min(seq_fps).clamp(10.0, 120.0);
            let dt = 1.0f64 / target_fps;
            ctx.request_repaint_after(Duration::from_secs_f64(dt));
        } else {
            ctx.request_repaint_after(Duration::from_millis(150));
        }
        // Space toggles play/pause (keep engine.state in sync)
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            let seq_fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
            let current_sec = (self.playhead as f64) / seq_fps;

            if self.playback_clock.playing {
                self.playback_clock.pause(current_sec);
                // NEW: make the decode engine pause too
                self.engine.state = PlayState::Paused;
                if let Some(engine) = &self.audio_out { engine.pause(current_sec); }
            } else {
                if self.playhead >= self.seq.duration_in_frames { self.playhead = 0; }
                self.playback_clock.play(current_sec);
                // NEW: make the decode engine actually play
                self.engine.state = PlayState::Playing;
                if let Ok(clips) = self.build_audio_clips() {
                    if let Some(engine) = &self.audio_out { engine.start(current_sec, clips); }
                }
            }
        }

        app_ui::top_toolbar(self, ctx, frame);

        app_ui::preview_settings_window(self, ctx);

        if app_ui::show_project_picker_if_needed(self, ctx) { return; }

        // Export dialog UI (editor mode only)
        self.export.ui(ctx, frame, &self.seq, &self.db, &self.project_id);

        // Preview panel will be inside CentralPanel with resizable area

        let assets_panel = egui::SidePanel::left("assets")
            .default_width(200.0)
            .resizable(true)
            .min_width(110.0)
            .max_width(1600.0)
            .show(ctx, |ui| {
                // Top area (not scrolling): toolbar + optional embedded ComfyUI
                self.poll_jobs();
                ui.heading("Assets");
                ui.horizontal(|ui| {
                    if ui.button("Import...").clicked() {
                        let dialog = rfd::FileDialog::new().set_parent(frame);
                        if let Some(files) = dialog.pick_files() {
                            let _ = self.import_files(&files);
                        }
                    }
                    if ui.button("Refresh").clicked() {}
                    if ui.button("Jobs").clicked() { self.show_jobs = !self.show_jobs; }
                    if ui.button("ComfyUI").clicked() { self.show_comfy_panel = !self.show_comfy_panel; }
                });
                if self.show_comfy_panel {
                    app_ui::comfy_settings_panel(self, ui);
                }
                // Thumbnails are fixed-size squares to keep layout consistent
                // Delegate full assets panel content to app_ui helpers
                app_ui::cloud_modal_section(self, ui);
                app_ui::comfy_embed_in_assets(self, ui);
                app_ui::assets_scroll_section(self, ui);
            });

        self.assets_drop_rect = Some(assets_panel.response.rect);

        
        
        
        
        
        
        
        // Properties panel for selected clip
        app_ui::properties_panel(self, ctx);

        app_ui::center_editor(self, ctx, frame);

        self.jobs_window(ctx);

        app_ui::drag_overlay(self, ctx);

        self.handle_external_file_drops(ctx);
        self.process_pending_timeline_drops();
    }
}
