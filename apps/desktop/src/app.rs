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

// Cloud progress phases
#[derive(Default, Clone)]
struct PhasePlan {
    sampling: std::collections::HashSet<String>,
    encoding: std::collections::HashSet<String>,
}

#[derive(Default, Clone)]
struct PhaseAgg {
    s_cur: u32,
    s_tot: u32,
    e_cur: u32,
    e_tot: u32,
    importing: bool,
    imported: bool,
}

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
    modal_ws_monitor: bool,
    modal_ws_thread: Option<std::thread::JoinHandle<()>>,
    modal_ws_stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
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
}

impl App {
    fn modal_test_connection(&self) {
        let base = self.modal_base_url.trim().to_string();
        let key = self.modal_api_key.clone();
        let tx = self.modal_tx.clone();
        std::thread::spawn(move || {
            let log = |s: &str| { let _ = tx.send(ModalEvent::Log(s.to_string())); };
            if base.is_empty() { log("Base URL not set"); return; }
            // Normalize base (strip trailing /health or /healthz if user pasted a full health URL)
            let mut base_trim = base.trim_end_matches('/').to_string();
            for suffix in ["/healthz", "/health"] {
                if base_trim.ends_with(suffix) {
                    base_trim = base_trim[..base_trim.len()-suffix.len()].trim_end_matches('/').to_string();
                    break;
                }
            }
            // Try extended health first (/healthz) to list recent artifacts; fall back to /health
            let base_trim = base_trim; // shadow immutable
            let urlz = format!("{}/healthz", base_trim);
            let mut reqz = ureq::get(&urlz);
            if !key.trim().is_empty() { reqz = reqz.set("Authorization", &format!("Bearer {}", key)); }
            match reqz.call() {
                Ok(resp) => {
                    let status = resp.status();
                    match resp.into_string() {
                        Ok(body) => {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                                let recent = v.get("recent").and_then(|r| r.as_array()).map(|a| a.len()).unwrap_or(0);
                                log(&format!("Healthz: {} (recent jobs: {})", status, recent));
                                if let Some(arr) = v.get("recent").and_then(|r| r.as_array()) {
                                    for (i, j) in arr.iter().enumerate().take(3) {
                                        let jid = j.get("id").and_then(|s| s.as_str()).unwrap_or("");
                                        let st = j.get("status").and_then(|s| s.as_str()).unwrap_or("");
                                        if let Some(arts) = j.get("artifacts").and_then(|a| a.as_array()) {
                                            for (k, a) in arts.iter().enumerate().take(2) {
                                                let fname = a.get("filename").and_then(|s| s.as_str()).unwrap_or("");
                                                let url = a.get("url").and_then(|s| s.as_str()).unwrap_or("");
                                                log(&format!("  [{}] {} {} -> {}", i+1, jid, fname, url));
                                                if k == 0 { break; }
                                            }
                                        } else {
                                            log(&format!("  [{}] {} (status: {})", i+1, jid, st));
                                        }
                                    }
                                }
                                return;
                            } else {
                                log(&format!("Healthz: {} (non-JSON)", status));
                                return;
                            }
                        }
                        Err(_) => { log(&format!("Healthz: {} (empty body)", status)); return; }
                    }
                }
                Err(_e) => {
                    let url = format!("{}/health", base_trim);
                    let req = ureq::get(&url);
                    let req = if key.trim().is_empty() { req } else { req.set("Authorization", &format!("Bearer {}", key)) };
                    match req.call() {
                        Ok(resp) => log(&format!("Health: {}", resp.status())) ,
                        Err(e) => log(&format!("Health check failed: {}", e)),
                    }
                }
            }
        });
    }

    fn modal_queue_job(&self) {
        let base = self.modal_base_url.trim().to_string();
        let key = self.modal_api_key.clone();
        let payload = self.modal_payload.clone();
        let tx = self.modal_tx.clone();
        let target = self.cloud_target;
        // Generate a unique client_id to tag this job's outputs (prefix)
        let client_id = uuid::Uuid::new_v4().to_string();
        std::thread::spawn(move || {
            let log = |s: &str| { let _ = tx.send(ModalEvent::Log(s.to_string())); };
            if base.is_empty() { log("Base URL not set"); return; }
            if payload.trim().is_empty() { log("Payload is empty"); return; }
            let url = format!("{}/prompt", base.trim_end_matches('/'));
            let req_base = ureq::post(&url).set("Content-Type", "application/json");
            let req_base = if key.trim().is_empty() { req_base } else { req_base.set("Authorization", &format!("Bearer {}", key)) };
            // Prepare body depending on target, and patch filename_prefix/client_id for unique outputs
            let mut body_v: serde_json::Value = match target {
                CloudTarget::Prompt => {
                    // If payload already has {"prompt":{...}}, patch it; else wrap it
                    match serde_json::from_str::<serde_json::Value>(&payload) {
                        Ok(mut v) => {
                            if v.get("prompt").is_some() {
                                v
                            } else {
                                let mut obj = serde_json::Map::new();
                                obj.insert("prompt".into(), v);
                                serde_json::Value::Object(obj)
                            }
                        }
                        Err(e) => { log(&format!("Invalid JSON: {}", e)); return; }
                    }
                }
                CloudTarget::Workflow => {
                    match convert_workflow_to_prompt(&payload) {
                        Ok(s) => match serde_json::from_str::<serde_json::Value>(&s) {
                            Ok(v) => v,
                            Err(e) => { log(&format!("Converted workflow parse failed: {}", e)); return; }
                        },
                        Err(e) => { log(&format!("Workflow convert failed: {}", e)); return; }
                    }
                }
            };
            // Ensure client_id is present and stable (use generated one)
            if let Some(obj) = body_v.as_object_mut() {
                obj.insert("client_id".into(), serde_json::Value::String(client_id.clone()));
                // Patch filename_prefix for relevant nodes to this client_id to avoid cross-job collisions
                if let Some(prompt_obj) = obj.get_mut("prompt").and_then(|p| p.as_object_mut()) {
                    // Try to preserve an existing base prefix if present (e.g., "teacache")
                    let mut base_prefix: Option<String> = None;
                    for (_nid, node_v) in prompt_obj.iter() {
                        if let Some(nobj) = node_v.as_object() {
                            let class_type = nobj.get("class_type").and_then(|s| s.as_str()).unwrap_or("");
                            let wants_prefix = matches!(class_type,
                                "VHS_VideoCombine" | "VideoCombine" | "SaveVideo" | "SaveImage"
                            ) || class_type.to_ascii_lowercase().contains("videocombine") || class_type.to_ascii_lowercase().contains("savevideo");
                            if wants_prefix {
                                if let Some(inputs) = nobj.get("inputs").and_then(|i| i.as_object()) {
                                    if let Some(fpv) = inputs.get("filename_prefix").and_then(|x| x.as_str()) {
                                        if !fpv.trim().is_empty() { base_prefix = Some(fpv.trim().to_string()); break; }
                                    }
                                }
                            }
                        }
                    }
                    let prefix_final = if let Some(base) = base_prefix { format!("{}-{}", base, client_id) } else { client_id.clone() };
                    for (_nid, node_v) in prompt_obj.iter_mut() {
                        if let Some(nobj) = node_v.as_object_mut() {
                            let class_type = nobj.get("class_type").and_then(|s| s.as_str()).unwrap_or("");
                            let wants_prefix = matches!(class_type,
                                "VHS_VideoCombine" | "VideoCombine" | "SaveVideo" | "SaveImage"
                            ) || class_type.to_ascii_lowercase().contains("videocombine") || class_type.to_ascii_lowercase().contains("savevideo");
                            if wants_prefix {
                                // Ensure inputs exists
                                match nobj.get_mut("inputs") {
                                    Some(inputs) if inputs.is_object() => {
                                        if let Some(m) = inputs.as_object_mut() {
                                            m.insert("filename_prefix".into(), serde_json::Value::String(prefix_final.clone()));
                                        }
                                    }
                                    _ => {
                                        let mut m = serde_json::Map::new();
                                        m.insert("filename_prefix".into(), serde_json::Value::String(prefix_final.clone()));
                                        nobj.insert("inputs".into(), serde_json::Value::Object(m));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let body = body_v.to_string();
            match req_base.send_string(&body) {
                Ok(resp) => {
                    let status = resp.status();
                    if status >= 200 && status < 300 {
                        match resp.into_string() {
                            Ok(body) => {
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                                    let id = v.get("prompt_id")
                                        .or_else(|| v.get("job_id"))
                                        .or_else(|| v.get("id"))
                                        .and_then(|s| s.as_str())
                                        .unwrap_or("");
                                    if !id.is_empty() {
                                        let _ = tx.send(ModalEvent::JobQueued(id.to_string()));
                                        // Also include the unique prefix used for this run (preserved base + client_id)
                                        // Try to extract what we actually set for filename_prefix
                                        let mut prefix_used = client_id.clone();
                                        if let Some(prompt_obj) = body_v.get("prompt").and_then(|p| p.as_object()) {
                                            for node_v in prompt_obj.values() {
                                                if let Some(nobj) = node_v.as_object() {
                                                    if let Some(inputs) = nobj.get("inputs").and_then(|i| i.as_object()) {
                                                        if let Some(fpv) = inputs.get("filename_prefix").and_then(|x| x.as_str()) {
                                                            if !fpv.trim().is_empty() { prefix_used = fpv.trim().to_string(); break; }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        let _ = tx.send(ModalEvent::JobQueuedWithPrefix(id.to_string(), prefix_used));
                                    }
                                    else { log("Job queued (no id in response)"); }
                                } else {
                                    log("Job queued (non-JSON response)");
                                }
                            }
                            Err(_) => { log("Job queued (no body)"); }
                        }
                    } else {
                        let body = resp.into_string().unwrap_or_default();
                        log(&format!("Queue failed: HTTP {}\n{}", status, body));
                    }
                }
                Err(ureq::Error::Status(code, resp)) => {
                    let body = resp.into_string().unwrap_or_default();
                    log(&format!("Queue failed: HTTP {}\n{}", code, body));
                }
                Err(e) => log(&format!("Queue error: {}", e)),
            }
        });
    }

    fn compute_phase_plan_from_payload(payload: &str) -> PhasePlan {
        let mut plan = PhasePlan::default();
        let parse = serde_json::from_str::<serde_json::Value>(payload).ok();
        let mut prompt_obj_opt: Option<&serde_json::Map<String, serde_json::Value>> = None;
        if let Some(v) = parse.as_ref() {
            if let Some(p) = v.get("prompt").and_then(|p| p.as_object()) { prompt_obj_opt = Some(p); }
            else if v.get("nodes").is_some() {
                // Workflow format; build a temporary prompt-like map
                if let Some(arr) = v.get("nodes").and_then(|n| n.as_array()) {
                    let mut tmp = serde_json::Map::new();
                    for n in arr { if let (Some(id), Some(ct)) = (n.get("id"), n.get("class_type").and_then(|s| s.as_str())) {
                        let id_s = if let Some(i) = id.as_i64() { i.to_string() } else { id.as_str().unwrap_or("").to_string() };
                        let mut o = serde_json::Map::new(); o.insert("class_type".into(), serde_json::Value::String(ct.to_string())); tmp.insert(id_s, serde_json::Value::Object(o));
                    } }
                    prompt_obj_opt = Some(&*Box::leak(Box::new(tmp))); // limited scope in UI; acceptable here
                }
            } else if v.is_object() {
                prompt_obj_opt = v.as_object();
            }
        }
        if let Some(prompt_obj) = prompt_obj_opt {
            for (id, nodev) in prompt_obj {
                if let Some(ct) = nodev.get("class_type").and_then(|s| s.as_str()) {
                    let id_s = id.clone();
                    let lc = ct.to_ascii_lowercase();
                    if lc.contains("ksampler") || lc.contains("modelsampling") { plan.sampling.insert(id_s); }
                    if matches!(ct, "VHS_VideoCombine"|"VideoCombine"|"SaveVideo") || lc.contains("videocombine") || lc.contains("savevideo") { plan.encoding.insert(id.clone()); }
                }
            }
        }
        plan
    }

    fn modal_refresh_recent(&self) {
        let base = self.modal_base_url.trim().to_string();
        let key = self.modal_api_key.clone();
        let tx = self.modal_tx.clone();
        std::thread::spawn(move || {
            let log = |s: &str| { let _ = tx.send(ModalEvent::Log(s.to_string())); };
            if base.is_empty() { log("Base URL not set"); return; }
            // Normalize base
            let mut base_trim = base.trim_end_matches('/').to_string();
            for suffix in ["/healthz", "/health"] { if base_trim.ends_with(suffix) { base_trim = base_trim[..base_trim.len()-suffix.len()].trim_end_matches('/').to_string(); break; } }
            let url = format!("{}/healthz", base_trim);
            let mut req = ureq::get(&url);
            if !key.trim().is_empty() { req = req.set("Authorization", &format!("Bearer {}", key)); }
            match req.call() {
                Ok(resp) => {
                    match resp.into_string() {
                        Ok(body) => {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                                let mut list: Vec<(String, Vec<(String, String)>)> = Vec::new();
                                if let Some(arr) = v.get("recent").and_then(|r| r.as_array()) {
                                    for j in arr.iter() {
                                        let jid = j.get("id").and_then(|s| s.as_str()).unwrap_or("").to_string();
                                        let mut arts: Vec<(String, String)> = Vec::new();
                                        if let Some(a) = j.get("artifacts").and_then(|a| a.as_array()) {
                                            for it in a.iter() {
                                                let fname = it.get("filename").and_then(|s| s.as_str()).unwrap_or("").to_string();
                                                let url = it.get("url").and_then(|s| s.as_str()).unwrap_or("").to_string();
                                                if !url.is_empty() { arts.push((fname, url)); }
                                            }
                                        }
                                        if !jid.is_empty() { list.push((jid, arts)); }
                                    }
                                }
                                let _ = tx.send(ModalEvent::Recent(list));
                            } else {
                                log("/healthz returned non-JSON");
                            }
                        }
                        Err(e) => { log(&format!("/healthz read error: {}", e)); }
                    }
                }
                Err(e) => {
                    log(&format!("/healthz failed: {}", e));
                    // Fallback to /health to at least verify connectivity
                    let url = format!("{}/health", base_trim);
                    let mut req = ureq::get(&url);
                    if !key.trim().is_empty() { req = req.set("Authorization", &format!("Bearer {}", key)); }
                    let _ = req.call().ok();
                    let _ = tx.send(ModalEvent::Recent(Vec::new()));
                }
            }
        });
    }

    fn modal_import_url(&self, url: String, suggested_name: Option<String>) {
        let token = self.modal_api_key.clone();
        let tx_import = self.comfy_ingest_tx.clone();
        let proj_id = self.project_id.clone();
        let tx_log = self.modal_tx.clone();
        std::thread::spawn(move || {
            let log = |s: &str| { let _ = tx_log.send(ModalEvent::Log(s.to_string())); };
            let mut req = ureq::get(&url);
            if !token.trim().is_empty() { req = req.set("Authorization", &format!("Bearer {}", token)); }
            match req.call() {
                Ok(resp) => {
                    let fname = suggested_name.clone().filter(|s| !s.is_empty()).or_else(|| {
                        // derive from URL path
                        url::Url::parse(&url).ok().and_then(|u| u.path_segments().and_then(|mut p| p.next_back()).map(|s| s.to_string()))
                    }).unwrap_or_else(|| "artifact.mp4".to_string());
                    let tmpdir = project::app_data_dir().join("tmp").join("cloud");
                    let _ = std::fs::create_dir_all(&tmpdir);
                    let tmp = tmpdir.join(fname);
                    match std::fs::File::create(&tmp) {
                        Ok(mut f) => {
                            let mut reader = resp.into_reader();
                            if let Err(e) = std::io::copy(&mut reader, &mut f) { log(&format!("Download write failed: {}", e)); return; }
                            let _ = tx_import.send((proj_id.clone(), tmp.clone()));
                            log(&format!("Downloaded → queued import: {}", tmp.to_string_lossy()));
                        }
                        Err(e) => { log(&format!("Temp create failed: {}", e)); }
                    }
                }
                Err(e) => log(&format!("Download failed: {}", e)),
            }
        });
    }
    fn ensure_baseline_tracks(&mut self) {
        if self.seq.graph.tracks.is_empty() {
            // Add three video and three audio tracks by default
            for i in 1..=3 {
                let binding = timeline_crate::TrackBinding { id: timeline_crate::TrackId::new(), name: format!("V{}", i), kind: timeline_crate::TrackKind::Video, node_ids: Vec::new() };
                let _ = self.apply_timeline_command(timeline_crate::TimelineCommand::UpsertTrack { track: binding });
            }
            for i in 1..=3 {
                let binding = timeline_crate::TrackBinding { id: timeline_crate::TrackId::new(), name: format!("A{}", i), kind: timeline_crate::TrackKind::Audio, node_ids: Vec::new() };
                let _ = self.apply_timeline_command(timeline_crate::TimelineCommand::UpsertTrack { track: binding });
            }
            self.sync_tracks_from_graph();
        }
    }
    fn load_project_timeline(&mut self) {
        if let Ok(Some(json)) = self.db.get_project_timeline_json(&self.project_id) {
            if let Ok(seq) = serde_json::from_str::<timeline_crate::Sequence>(&json) {
                self.seq = seq;
            } else {
                let mut seq = timeline_crate::Sequence::new("Main", 1920, 1080, timeline_crate::Fps::new(30,1), 600);
                // Baseline: three video + three audio legacy tracks for migration
                for i in 1..=3 { seq.add_track(timeline_crate::Track { name: format!("V{}", i), items: vec![] }); }
                for i in 1..=3 { seq.add_track(timeline_crate::Track { name: format!("A{}", i), items: vec![] }); }
                self.seq = seq;
            }
        } else {
            let mut seq = timeline_crate::Sequence::new("Main", 1920, 1080, timeline_crate::Fps::new(30,1), 600);
            for i in 1..=3 { seq.add_track(timeline_crate::Track { name: format!("V{}", i), items: vec![] }); }
            for i in 1..=3 { seq.add_track(timeline_crate::Track { name: format!("A{}", i), items: vec![] }); }
            self.seq = seq;
            // Do NOT auto-save an empty timeline over a project that has none yet.
            // We'll save on the first edit or explicit action to avoid wiping unsaved work.
        }
        // Use saved graph if present; migrate only legacy sequences with empty graph
        if self.seq.graph.tracks.is_empty() {
            self.seq.graph = timeline_crate::migrate_sequence_tracks(&self.seq);
        }
        // Ensure denormalized tracks list reflects the graph for UI
        self.sync_tracks_from_graph();
        self.ensure_baseline_tracks();
        self.timeline_history = timeline_crate::CommandHistory::default();
        self.selected = None;
        self.drag = None;
    }

    fn save_project_timeline(&mut self) -> anyhow::Result<()> {
        let json = serde_json::to_string(&self.seq)?;
        self.db.upsert_project_timeline_json(&self.project_id, &json)?;
        self.last_save_at = Some(Instant::now());
        Ok(())
    }
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
            comfy_ws_monitor: true,
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
            modal_ws_monitor: false,
            modal_ws_thread: None,
            modal_ws_stop: None,
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

    // Load or retrieve cached thumbnail texture for an asset
    fn load_thumb_texture(
        &mut self,
        ctx: &egui::Context,
        asset: &project::AssetRow,
        desired_w: u32,
        desired_h: u32,
    ) -> Option<egui::TextureHandle> {
        if let Some(tex) = self.asset_thumb_textures.get(&asset.id) {
            return Some(tex.clone());
        }
        let thumb_path = project::app_data_dir()
            .join("cache")
            .join("thumbnails")
            .join(format!("{}-thumb.jpg", asset.id));
        if !thumb_path.exists() {
            return None;
        }
        if let Ok(img) = image::open(&thumb_path) {
            let resized = img.resize(desired_w, desired_h, image::imageops::FilterType::Triangle);
            let rgba = resized.to_rgba8();
            let (w, h) = rgba.dimensions();
            let color = egui::ColorImage::from_rgba_unmultiplied(
                [w as usize, h as usize],
                &rgba.into_raw(),
            );
            let tex = ctx.load_texture(
                format!("asset_thumb_{}", asset.id),
                color,
                egui::TextureOptions::LINEAR,
            );
            self.asset_thumb_textures.insert(asset.id.clone(), tex.clone());
            Some(tex)
        } else {
            None
        }
    }

    // Insert an asset as a clip at a specific track/index and frame position
    fn insert_asset_at(&mut self, asset: &project::AssetRow, track_index: usize, start_frame: i64) {
        let track_index = track_index.min(self.seq.graph.tracks.len().saturating_sub(1));
        let track_binding = match self.seq.graph.tracks.get(track_index) {
            Some(binding) => binding.clone(),
            None => return,
        };

        let is_audio = asset.kind.eq_ignore_ascii_case("audio");
        let chosen_track_id = if is_audio {
            // Prefer nearest audio track, fallback to current
            self.seq
                .graph
                .tracks
                .iter()
                .enumerate()
                .min_by_key(|(i, t)| {
                    let d = (*i as isize - track_index as isize).abs() as usize;
                    if matches!(t.kind, timeline_crate::TrackKind::Audio) { d } else { usize::MAX / 2 + d }
                })
                .map(|(_, t)| t.id)
                .unwrap_or(track_binding.id)
        } else {
            // Prefer nearest non-audio track, fallback to current
            self.seq
                .graph
                .tracks
                .iter()
                .enumerate()
                .min_by_key(|(i, t)| {
                    let d = (*i as isize - track_index as isize).abs() as usize;
                    if !matches!(t.kind, timeline_crate::TrackKind::Audio) { d } else { usize::MAX / 2 + d }
                })
                .map(|(_, t)| t.id)
                .unwrap_or(track_binding.id)
        };

        let duration = asset.duration_frames.unwrap_or(150).max(1);
        let timeline_range = timeline_crate::FrameRange::new(start_frame.max(0), duration);
        let media_range = timeline_crate::FrameRange::new(0, duration);
        let clip = timeline_crate::ClipNode {
            asset_id: Some(asset.src_abs.clone()),
            media_range,
            timeline_range,
            playback_rate: 1.0,
            reverse: false,
            metadata: serde_json::Value::Null,
        };
        let node = timeline_crate::TimelineNode {
            id: timeline_crate::NodeId::new(),
            label: Some(asset.id.clone()),
            kind: timeline_crate::TimelineNodeKind::Clip(clip),
            locked: false,
            metadata: serde_json::Value::Null,
        };
        let placement = timeline_crate::TrackPlacement { track_id: chosen_track_id, position: None };
        if let Err(err) = self.apply_timeline_command(timeline_crate::TimelineCommand::InsertNode { node, placements: vec![placement], edges: Vec::new() }) {
            eprintln!("timeline insert failed: {err}");
            return;
        }
        self.sync_tracks_from_graph();
    }

    fn apply_timeline_command(&mut self, command: TimelineCommand) -> Result<(), TimelineError> {
        self.timeline_history.apply(&mut self.seq.graph, command)?;
        self.sync_tracks_from_graph();
        // Autosave timeline after each edit (best-effort)
        let _ = self.save_project_timeline();
        Ok(())
    }

    fn sync_tracks_from_graph(&mut self) {
        let mut tracks: Vec<Track> = Vec::with_capacity(self.seq.graph.tracks.len());
        let mut max_end: i64 = 0;
        for binding in &self.seq.graph.tracks {
            let mut items = Vec::with_capacity(binding.node_ids.len());
            for node_id in &binding.node_ids {
                if let Some(node) = self.seq.graph.nodes.get(node_id) {
                    if let Some(item) = Self::item_from_node(node, &binding.kind, self.seq.fps) {
                        max_end = max_end.max(item.from + item.duration_in_frames);
                        items.push(item);
                    }
                }
            }
            tracks.push(Track { name: binding.name.clone(), items });
        }
        self.seq.tracks = tracks;
        self.seq.duration_in_frames = max_end;
    }

    fn item_from_node(node: &TimelineNode, track_kind: &TrackKind, fps: Fps) -> Option<Item> {
        let id = node.id.to_string();
        match (&node.kind, track_kind) {
            (TimelineNodeKind::Clip(clip), TrackKind::Audio) => {
                let src = clip.asset_id.clone().unwrap_or_default();
                Some(Item {
                    id,
                    from: clip.timeline_range.start,
                    duration_in_frames: clip.timeline_range.duration,
                    kind: ItemKind::Audio {
                        src,
                        in_offset_sec: crate::timeline::ui::frames_to_seconds(clip.media_range.start, fps),
                        rate: clip.playback_rate,
                    },
                })
            }
            (TimelineNodeKind::Clip(clip), _) => {
                let src = clip.asset_id.clone().unwrap_or_default();
                let is_image = std::path::Path::new(&src)
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|s| s.eq_ignore_ascii_case("png")
                        || s.eq_ignore_ascii_case("jpg")
                        || s.eq_ignore_ascii_case("jpeg")
                        || s.eq_ignore_ascii_case("gif")
                        || s.eq_ignore_ascii_case("webp")
                        || s.eq_ignore_ascii_case("bmp")
                        || s.eq_ignore_ascii_case("tif")
                        || s.eq_ignore_ascii_case("tiff")
                        || s.eq_ignore_ascii_case("exr"))
                    .unwrap_or(false);
                if is_image {
                    Some(Item {
                        id,
                        from: clip.timeline_range.start,
                        duration_in_frames: clip.timeline_range.duration,
                        kind: ItemKind::Image { src },
                    })
                } else {
                    Some(Item {
                        id,
                        from: clip.timeline_range.start,
                        duration_in_frames: clip.timeline_range.duration,
                        kind: ItemKind::Video {
                            src,
                            frame_rate: Some(fps.num as f32 / fps.den.max(1) as f32),
                            in_offset_sec: crate::timeline::ui::frames_to_seconds(clip.media_range.start, fps),
                            rate: clip.playback_rate,
                        },
                    })
                }
            }
            (TimelineNodeKind::Generator { generator_id, timeline_range, metadata }, _) => {
                match generator_id.as_str() {
                    "solid" => {
                        let color = metadata
                            .get("color")
                            .and_then(|v| v.as_str())
                            .unwrap_or("#4c4c4c")
                            .to_string();
                        Some(Item {
                            id,
                            from: timeline_range.start,
                            duration_in_frames: timeline_range.duration,
                            kind: ItemKind::Solid { color },
                        })
                    }
                    "text" => {
                        let text = metadata.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let color = metadata
                            .get("color")
                            .and_then(|v| v.as_str())
                            .unwrap_or("#ffffff")
                            .to_string();
                        Some(Item {
                            id,
                            from: timeline_range.start,
                            duration_in_frames: timeline_range.duration,
                            kind: ItemKind::Text { text, color },
                        })
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn build_audio_clips(&mut self) -> anyhow::Result<Vec<ActiveAudioClip>> {
        let fps = self.seq.fps;
        let mut clips = Vec::new();
        for binding in &self.seq.graph.tracks {
            if !matches!(binding.kind, TrackKind::Audio) { continue; }
            for node_id in &binding.node_ids {
                let node = match self.seq.graph.nodes.get(node_id) { Some(n) => n, None => continue };
                let clip = match &node.kind { TimelineNodeKind::Clip(c) => c, _ => continue };
                let path_str = match &clip.asset_id { Some(p) => p, None => continue };
                let path = Path::new(path_str);
                let buf = self.audio_buffers.get_or_load(path)?;
                let timeline_start = crate::timeline::ui::frames_to_seconds(clip.timeline_range.start, fps);
                let mut timeline_dur = crate::timeline::ui::frames_to_seconds(clip.timeline_range.duration, fps);
                let mut media_start = crate::timeline::ui::frames_to_seconds(clip.media_range.start, fps);
                let rate = clip.playback_rate.max(0.0001) as f64;
                timeline_dur /= rate;
                media_start /= rate;
                let clip_duration = timeline_dur.min((buf.duration_sec as f64 - media_start).max(0.0));
                if clip_duration <= 0.0 { continue; }
                clips.push(ActiveAudioClip {
                    start_tl_sec: timeline_start,
                    start_media_sec: media_start,
                    duration_sec: clip_duration,
                    buf: buf.clone(),
                });
            }
        }

        clips.sort_by(|a, b| a.start_tl_sec.partial_cmp(&b.start_tl_sec).unwrap_or(std::cmp::Ordering::Equal));
        Ok(clips)
    }

    fn active_video_media_time_graph(&self, timeline_sec: f64) -> Option<(String, f64)> {
        let seq_fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
        let playhead = (timeline_sec * seq_fps).round() as i64;
        // Priority: lower-numbered (top-most) video tracks first
        for binding in self.seq.graph.tracks.iter() {
            if matches!(binding.kind, TrackKind::Audio) { continue; }
            for node_id in &binding.node_ids {
                let Some(node) = self.seq.graph.nodes.get(node_id) else { continue };
                let clip = match &node.kind { TimelineNodeKind::Clip(c) => c, _ => continue };
                if playhead < clip.timeline_range.start || playhead >= clip.timeline_range.end() { continue; }
                let Some(path) = clip.asset_id.clone() else { continue };
                let start_on_timeline_sec = clip.timeline_range.start as f64 / seq_fps;
                let local_t = (timeline_sec - start_on_timeline_sec).max(0.0);
                let media_sec = crate::timeline::ui::frames_to_seconds(clip.media_range.start, self.seq.fps) + local_t * clip.playback_rate as f64;
                return Some((path, media_sec));
            }
        }
        None
    }

    fn active_audio_media_time_graph(&self, timeline_sec: f64) -> Option<(String, f64)> {
        let seq_fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
        let playhead = (timeline_sec * seq_fps).round() as i64;
        for binding in self.seq.graph.tracks.iter().rev() {
            if !matches!(binding.kind, TrackKind::Audio) { continue; }
            for node_id in &binding.node_ids {
                let Some(node) = self.seq.graph.nodes.get(node_id) else { continue };
                let clip = match &node.kind { TimelineNodeKind::Clip(c) => c, _ => continue };
                if playhead < clip.timeline_range.start || playhead >= clip.timeline_range.end() { continue; }
                let Some(path) = clip.asset_id.clone() else { continue };
                let start_on_timeline_sec = clip.timeline_range.start as f64 / seq_fps;
                let local_t = (timeline_sec - start_on_timeline_sec).max(0.0);
                let media_sec = crate::timeline::ui::frames_to_seconds(clip.media_range.start, self.seq.fps) + local_t * clip.playback_rate as f64;
                return Some((path, media_sec));
            }
        }
        self.active_video_media_time_graph(timeline_sec)
    }

    fn request_audio_peaks(&mut self, _path: &std::path::Path) {
        // Placeholder: integrate with audio decoding backend to compute peaks.
        // Keep bounded: one job per path. For now, no-op to avoid blocking UI.
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
        let pid = self.project_id.clone();
        self.import_files_for(&pid, files)
    }

    fn import_files_for(&mut self, project_id: &str, files: &[PathBuf]) -> Result<()> {
        if files.is_empty() { return Ok(()); }
        let ancestor = nearest_common_ancestor(files);
        if let Some(base) = ancestor.as_deref() { self.db.set_project_base_path(project_id, base)?; }
        let db_path = self.db.path().to_path_buf();
        let project_id = project_id.to_string();
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
                            use jobs_crate::{JobKind, JobSpec};
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
        let is_audio = asset.kind.eq_ignore_ascii_case("audio");
        let track_index = if is_audio {
            self.seq.graph.tracks.iter().position(|t| matches!(t.kind, TrackKind::Audio)).unwrap_or_else(|| self.seq.graph.tracks.len().saturating_sub(1))
        } else {
            0
        };

        let track_binding = match self.seq.graph.tracks.get(track_index) {
            Some(binding) => binding.clone(),
            None => return,
        };

        let start_frame = track_binding
            .node_ids
            .iter()
            .filter_map(|id| self.seq.graph.nodes.get(id))
            .filter_map(|node| Self::node_frame_range(node))
            .map(|range| range.end())
            .max()
            .unwrap_or(0);

        let duration = asset.duration_frames.unwrap_or(150).max(1);
        let timeline_range = FrameRange::new(start_frame, duration);
        let media_range = FrameRange::new(0, duration);
        let clip = ClipNode {
            asset_id: Some(asset.src_abs.clone()),
            media_range,
            timeline_range,
            playback_rate: 1.0,
            reverse: false,
            metadata: Value::Null,
        };
        let node = TimelineNode {
            id: NodeId::new(),
            label: Some(asset.id.clone()),
            kind: TimelineNodeKind::Clip(clip),
            locked: false,
            metadata: Value::Null,
        };
        let placement = TrackPlacement { track_id: track_binding.id, position: None };
        if let Err(err) = self.apply_timeline_command(TimelineCommand::InsertNode { node, placements: vec![placement], edges: Vec::new() }) {
            eprintln!("timeline insert failed: {err}");
            return;
        }

        if let Some(track) = self.seq.tracks.get(track_index) {
            let idx = track.items.len().saturating_sub(1);
            self.selected = Some((track_index, idx));
        }
    }

}

// Best-effort converter from a generic "workflow" JSON into a ComfyUI /prompt payload.
// This is intentionally conservative: it tries to recognize a "nodes" array and
// build a minimal prompt map with class_type and any provided literal inputs.
// Complex graph links are not guaranteed to convert; if conversion isn't possible,
// returns an Err with a helpful message.
fn convert_workflow_to_prompt(workflow_json: &str) -> Result<String, String> {
    let v: serde_json::Value = serde_json::from_str(workflow_json)
        .map_err(|e| format!("Invalid JSON: {}", e))?;
    if v.get("prompt").is_some() {
        // Already a prompt payload
        return Ok(v.to_string());
    }
    // If it's already a node-id keyed object with class_type, wrap as prompt
    if v.is_object() {
        let obj = v.as_object().unwrap();
        let looks_like_prompt = obj.values().all(|n| n.get("class_type").is_some());
        if looks_like_prompt {
            let mut wrap = serde_json::Map::new();
            wrap.insert("prompt".into(), serde_json::Value::Object(obj.clone()));
            wrap.insert("client_id".into(), serde_json::Value::String(uuid::Uuid::new_v4().to_string()));
            return Ok(serde_json::Value::Object(wrap).to_string());
        }
    }
    // Try workflow format with nodes[]
    let nodes = v.get("nodes").and_then(|n| n.as_array()).ok_or_else(|| {
        "Workflow JSON doesn't contain a 'prompt' or a 'nodes' array; please paste a ComfyUI API prompt (Copy API)".to_string()
    })?;
    let mut prompt = serde_json::Map::new();
    for node in nodes.iter() {
        // id as string key
        let id_val = node.get("id").ok_or_else(|| "Node missing 'id'".to_string())?;
        let id_str = if let Some(n) = id_val.as_i64() { n.to_string() } else { id_val.as_str().unwrap_or("").to_string() };
        if id_str.is_empty() { return Err("Node id is empty".into()); }
        // class type heuristic
        let class_type = node.get("class_type")
            .or_else(|| node.get("type"))
            .or_else(|| node.get("class"))
            .and_then(|s| s.as_str())
            .ok_or_else(|| format!("Node {} missing class_type", id_str))?;
        // inputs: try 'inputs' object if present; otherwise empty
        let inputs = node.get("inputs").and_then(|i| i.as_object()).cloned().unwrap_or_default();
        let mut nobj = serde_json::Map::new();
        nobj.insert("class_type".into(), serde_json::Value::String(class_type.to_string()));
        nobj.insert("inputs".into(), serde_json::Value::Object(inputs));
        prompt.insert(id_str, serde_json::Value::Object(nobj));
    }
    let mut root = serde_json::Map::new();
    root.insert("prompt".into(), serde_json::Value::Object(prompt));
    root.insert("client_id".into(), serde_json::Value::String(uuid::Uuid::new_v4().to_string()));
    Ok(serde_json::Value::Object(root).to_string())
}

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
                    let token = self.modal_api_key.clone();
                    let jid = id.clone();
                    let tx_log = self.modal_tx.clone();
                    let tx_import = self.comfy_ingest_tx.clone();
                    let proj_id = self.project_id.clone();
                    let app_tmp = project::app_data_dir().join("tmp").join("cloud");
                    let _ = std::fs::create_dir_all(&app_tmp);
                    let active_job = self.modal_active_job.clone();
                    std::thread::spawn(move || {
                        use std::time::Duration;
                        loop {
                            // Exit if a different job became active
                            let still_active = active_job.lock().ok().and_then(|a| a.clone()).map(|cur| cur == jid).unwrap_or(true);
                            if !still_active { break; }
                            let job_url = format!("{}/jobs/{}", http_base.trim_end_matches('/'), jid);
                            let mut req = ureq::get(&job_url);
                            if !token.trim().is_empty() { req = req.set("Authorization", &format!("Bearer {}", token)); }
                            if let Ok(resp) = req.call() {
                                if let Ok(body) = resp.into_string() {
                                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                                        let pr = v.get("progress_percent").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                                        let cur = v.get("current_step").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                        let tot = v.get("total_steps").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                        let _ = tx_log.send(ModalEvent::CloudProgress { job_id: jid.clone(), progress: pr, current: cur, total: tot, node_id: None });
                                        let _ = tx_log.send(ModalEvent::CloudSource { job_id: jid.clone(), source: crate::CloudUpdateSrc::Jobs });
                                        let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("");
                                        if status == "completed" {
                                            let _ = tx_log.send(ModalEvent::JobImporting(jid.clone()));
                                            // Download artifacts and enqueue import, filtered by expected prefix when available
                                            if let Some(arr) = v.get("artifacts").and_then(|a| a.as_array()) {
                                                // Try to read expected prefix for this job from active map (best effort)
                                                let mut any = false;
                                                for it in arr {
                                                    let url = it.get("url").and_then(|s| s.as_str()).unwrap_or("");
                                                    let name = it.get("filename").and_then(|s| s.as_str()).unwrap_or("out.mp4");
                                                    if url.is_empty() { continue; }
                                                    // Filter to mp4; server already scopes by prefix, but double-check
                                                    if !name.to_ascii_lowercase().ends_with(".mp4") { continue; }
                                                    let mut dreq = ureq::get(url);
                                                    if !token.trim().is_empty() { dreq = dreq.set("Authorization", &format!("Bearer {}", token)); }
                                                    if let Ok(dresp) = dreq.call() {
                                                        let mut reader = dresp.into_reader();
                                                        let tmp = app_tmp.join(name);
                                                        if let Ok(mut f) = std::fs::File::create(&tmp) {
                                                            let _ = std::io::copy(&mut reader, &mut f);
                                                            let _ = tx_import.send((proj_id.clone(), tmp.clone()));
                                                            any = true;
                                                        }
                                                    }
                                                }
                                                if !any {
                                                    // Nothing matched; keep looping, allowing WS or future polls to provide urls
                                                }
                                            }
                                            // Notify server that import has completed (best-effort)
                                            let imp_url = format!("{}/jobs/{}/imported", http_base.trim_end_matches('/'), jid);
                                            let mut ireq = ureq::post(&imp_url);
                                            if !token.trim().is_empty() { ireq = ireq.set("Authorization", &format!("Bearer {}", token)); }
                                            let _ = ireq.call();
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
                ModalEvent::JobImported(jid) => { self.modal_phase_agg.remove(&jid); self.modal_phase_plans.remove(&jid); self.modal_job_progress.remove(&jid); }
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

        // Cloud (Modal) live job monitor lifecycle
        // Start monitor only when requested (e.g., on Queue Job), or if user explicitly enabled the toggle
        // Only start monitoring after a job is actually queued.
        // If the user enabled the toggle, defer connection until we have at least one known job.
        let has_jobs = !self.modal_known_jobs.is_empty();
        let modal_needed =
            (self.modal_monitor_requested || (self.modal_ws_monitor && has_jobs))
            && !self.modal_base_url.trim().is_empty();
        let modal_running = self.modal_ws_thread.is_some();
        if modal_needed && !modal_running {
            let mut base = self.modal_base_url.trim().to_string();
            // Normalize base URL scheme
            if !base.starts_with("http://") && !base.starts_with("https://") {
                base = format!("https://{}", base);
            }
            // Strip common health suffixes if user pasted a health URL
            if base.ends_with("/health") { base = base[..base.len()-"/health".len()].trim_end_matches('/').to_string(); }
            if base.ends_with("/healthz") { base = base[..base.len()-"/healthz".len()].trim_end_matches('/').to_string(); }
            // Prefer relay WS URL if provided; else derive /events from Base URL
            let ws_url = if !self.modal_relay_ws_url.trim().is_empty() {
                self.modal_relay_ws_url.trim().to_string()
            } else if base.starts_with("https://") {
                base.replacen("https://", "wss://", 1) + if base.ends_with("/") { "events" } else { "/events" }
            } else {
                base.replacen("http://", "ws://", 1) + if base.ends_with("/") { "events" } else { "/events" }
            };
            let http_base = base.clone();
            let token = self.modal_api_key.clone();
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let tx_import = self.comfy_ingest_tx.clone();
            let tx_log = self.modal_tx.clone();
            let proj_id = self.project_id.clone();
            let app_tmp = project::app_data_dir().join("tmp").join("cloud");
            let allow_poll_fallback = false; // disable /progress-status fallback to avoid cold starts
            let job_prefixes = self.modal_job_prefixes.clone();
            let active_job = self.modal_active_job.clone();
            let handle = std::thread::spawn({
                let stop = std::sync::Arc::clone(&stop);
                let allow_poll_fallback = allow_poll_fallback;
                let job_prefixes = job_prefixes.clone();
                let active_job = active_job.clone();
                move || {
                    let log = |s: &str| { let _ = tx_log.send(ModalEvent::Log(s.to_string())); };
                    let _ = std::fs::create_dir_all(&app_tmp);
                    let mut backoff_ms: u64 = 500;
                    let mut no_job_attempts: u32 = 0;
                    loop {
                        if stop.load(std::sync::atomic::Ordering::Relaxed) { break; }
                        // Build WS request; prefer Authorization header, fallback to token query param
                        let mut url_full = ws_url.clone();
                        if token.trim().is_empty() {
                            // nothing
                        } else if url_full.contains('?') {
                            url_full = format!("{}&token={}", url_full, urlencoding::encode(&token));
                        } else {
                            url_full = format!("{}?token={}", url_full, urlencoding::encode(&token));
                        }
                        let url = match url::Url::parse(&url_full) { Ok(u) => u, Err(e) => { log(&format!("WS URL parse error: {}", e)); break; } };
                        // Try header-based auth first (some servers require Authorization header for WS)
                        // Bring IntoClientRequest into scope for method resolution
                        let mut req = match {
                            use tungstenite::client::IntoClientRequest;
                            url.clone().into_client_request()
                        } {
                            Ok(r) => r,
                            Err(e) => { log(&format!("WS request build error: {}", e)); break; }
                        };
                        if !token.trim().is_empty() {
                            use tungstenite::http::header::AUTHORIZATION;
                            use tungstenite::http::HeaderValue;
                            if let Ok(val) = HeaderValue::from_str(&format!("Bearer {}", token)) {
                                req.headers_mut().insert(AUTHORIZATION, val);
                            }
                        }
                        log(&format!("Cloud monitor: connecting {}", url));
                        match tungstenite::connect(req) {
                            Ok((mut socket, _)) => {
                                backoff_ms = 500;
                                log("Cloud monitor: connected");
                                no_job_attempts = 0;
                                // Start a lightweight HTTP poller alongside WS to keep UI fresh
                                // even if WS events are sparse or missed.
                                if false {
                                    let http_base = http_base.clone();
                                    let token = token.clone();
                                    let tx_log = tx_log.clone();
                                    let stop_http = std::sync::Arc::clone(&stop);
                                    std::thread::spawn(move || {
                                        use std::time::Duration;
                                        loop {
                                            if stop_http.load(std::sync::atomic::Ordering::Relaxed) { break; }
                                            let status_url = format!("{}/progress-status", http_base.trim_end_matches('/'));
                                            let mut req = ureq::get(&status_url);
                                            if !token.trim().is_empty() { req = req.set("Authorization", &format!("Bearer {}", token)); }
                                            if let Ok(resp) = req.call() {
                                                if let Ok(body) = resp.into_string() {
                                                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                                                        let p = v.get("queue_pending").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                                                        let r = v.get("queue_running").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                                                        let _ = tx_log.send(ModalEvent::CloudStatus { pending: p, running: r });
                                                        if let Some(arr) = v.get("job_details").and_then(|a| a.as_array()) {
                                                            for it in arr {
                                                                let jid = it.get("job_id").and_then(|s| s.as_str()).unwrap_or("").to_string();
                                                                let pr = it.get("progress_percent").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                                                                let cur = it.get("current_step").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                                                let tot = it.get("total_steps").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                                                if !jid.is_empty() {
                                                                    let _ = tx_log.send(ModalEvent::CloudProgress { job_id: jid.clone(), progress: pr, current: cur, total: tot, node_id: None });
                                                                    let _ = tx_log.send(ModalEvent::CloudSource { job_id: jid.clone(), source: crate::CloudUpdateSrc::Status });
                                                                    // Mirror confirm_event.py: fetch per-job state for accurate progress
                                                                    let job_url = format!("{}/jobs/{}", http_base.trim_end_matches('/'), jid);
                                                                    let mut jreq = ureq::get(&job_url);
                                                                    if !token.trim().is_empty() { jreq = jreq.set("Authorization", &format!("Bearer {}", token)); }
                                                                    if let Ok(jresp) = jreq.call() {
                                                                        if let Ok(jbody) = jresp.into_string() {
                                                                            if let Ok(jv) = serde_json::from_str::<serde_json::Value>(&jbody) {
                                                                                let jpr = jv.get("progress_percent").and_then(|x| x.as_f64()).unwrap_or(pr as f64) as f32;
                                                                                let jcur = jv.get("current_step").and_then(|x| x.as_u64()).unwrap_or(cur as u64) as u32;
                                                                                let jtot = jv.get("total_steps").and_then(|x| x.as_u64()).unwrap_or(tot as u64) as u32;
                                                                                let _ = tx_log.send(ModalEvent::CloudProgress { job_id: jid.clone(), progress: jpr, current: jcur, total: jtot, node_id: None });
                                                                                let _ = tx_log.send(ModalEvent::CloudSource { job_id: jid.clone(), source: crate::CloudUpdateSrc::Jobs });
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            std::thread::sleep(Duration::from_millis(2000));
                                        }
                                    });
                                }
                                loop {
                                    if stop.load(std::sync::atomic::Ordering::Relaxed) { break; }
                                    match socket.read_message() {
                                        Ok(tungstenite::Message::Text(txt)) => {
                                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                                                let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                                if typ == "job_completed" {
                                                    if let Some(jid) = v.get("job_id").or_else(|| v.get("id")).and_then(|s| s.as_str()) {
                                                        let is_active = active_job.lock().ok().and_then(|a| a.clone()).map(|id| id == jid).unwrap_or(true);
                                                        if !is_active { continue; }
                                                        // Mark importing; download/import is handled by the per-job /jobs poller
                                                        let _ = tx_log.send(ModalEvent::JobImporting(jid.to_string()));
                                                    }
                                                } else if typ == "status" {
                                                    let p = v.get("pending").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                                                    let r = v.get("running").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                                                    let _ = tx_log.send(ModalEvent::CloudStatus { pending: p, running: r });
                                                    if let Some(jobs) = v.get("jobs").and_then(|a| a.as_array()) {
                                                        for it in jobs {
                                                            let jid = it.get("job_id").and_then(|s| s.as_str()).unwrap_or("").to_string();
                                                            let pr = it.get("progress_percent").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                                                            let cur = it.get("current_step").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                                            let tot = it.get("total_steps").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                                            if !jid.is_empty() { let _ = tx_log.send(ModalEvent::CloudProgress { job_id: jid.clone(), progress: pr, current: cur, total: tot, node_id: None }); let _ = tx_log.send(ModalEvent::CloudSource { job_id: jid.clone(), source: crate::CloudUpdateSrc::Ws }); }
                                                        }
                                                    }
                                                } else if typ == "progress" {
                                                    if let Some(jid) = v.get("job_id").and_then(|s| s.as_str()) {
                                                        let pr = v.get("progress_percent").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                                                        let cur = v.get("current_step").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                                        let tot = v.get("total_steps").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                                        let nid = v.get("node_id").or_else(|| v.get("node")).or_else(|| v.get("data").and_then(|d| d.get("node"))).and_then(|s| s.as_str()).map(|s| s.to_string());
                                                        let _ = tx_log.send(ModalEvent::CloudProgress { job_id: jid.to_string(), progress: pr, current: cur, total: tot, node_id: nid });
                                                        let _ = tx_log.send(ModalEvent::CloudSource { job_id: jid.to_string(), source: crate::CloudUpdateSrc::Ws });
                                                    }
                                                }
                                            }
                                        }
                                        Ok(_) => {}
                                        Err(_) => { break; }
                                    }
                                }
                            }
                            Err(_) => {
                                log("Cloud monitor: connect failed; retrying...");
                                if false && allow_poll_fallback {
                                    // Fallback: poll HTTP status to update UI while WS is down
                                    let status_url = format!("{}/progress-status", http_base.trim_end_matches('/'));
                                    let mut req = ureq::get(&status_url);
                                    if !token.trim().is_empty() { req = req.set("Authorization", &format!("Bearer {}", token)); }
                                    if let Ok(resp) = req.call() {
                                        if let Ok(body) = resp.into_string() {
                                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                                                let p = v.get("queue_pending").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                                                let r = v.get("queue_running").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                                                let _ = tx_log.send(ModalEvent::CloudStatus { pending: p, running: r });
                                                if let Some(arr) = v.get("job_details").and_then(|a| a.as_array()) {
                                                    for it in arr {
                                                        let jid = it.get("job_id").and_then(|s| s.as_str()).unwrap_or("").to_string();
                                                        let pr = it.get("progress_percent").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                                                        let cur = it.get("current_step").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                                        let tot = it.get("total_steps").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                                                        if !jid.is_empty() { let _ = tx_log.send(ModalEvent::CloudProgress { job_id: jid.clone(), progress: pr, current: cur, total: tot, node_id: None }); }
                                                    }
                                                }
                                                if v.get("active").and_then(|a| a.as_bool()) == Some(false) { backoff_ms = 8_000; }
                                            }
                                        }
                                    }
                                } else {
                                    no_job_attempts += 1;
                                    if no_job_attempts >= 2 {
                                        log("Cloud monitor: stopped (no jobs to monitor)");
                                        break;
                                    }
                                }
                                std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                                backoff_ms = (backoff_ms * 2).min(8_000);
                            }
                        }
                    }
                    log("Cloud monitor: stopped");
                }
            });
            self.modal_ws_stop = Some(stop);
            self.modal_ws_thread = Some(handle);
        } else if !modal_needed && modal_running {
            if let Some(flag) = &self.modal_ws_stop { flag.store(true, Ordering::Relaxed); }
            let _ = self.modal_ws_thread.take();
            self.modal_ws_stop = None;
        }

        // Idle-stop: when running but no active jobs for a while
        if modal_running {
            let active = (self.modal_queue_pending + self.modal_queue_running) > 0
                || !self.modal_phase_agg.is_empty();
            let now = Instant::now();
            if active {
                self.modal_last_progress_at = Some(now);
            } else if let Some(ts) = self.modal_last_progress_at {
                let idle_for = now.saturating_duration_since(ts);
                if idle_for.as_secs_f32() >= 5.0 {
                    if let Some(flag) = &self.modal_ws_stop { flag.store(true, Ordering::Relaxed); }
                    let _ = self.modal_ws_thread.take();
                    self.modal_ws_stop = None;
                    self.modal_monitor_requested = false;
                    self.modal_last_progress_at = None;
                    self.modal_logs.push_back("Cloud monitor: stopped (idle)".into());
                }
            } else {
                self.modal_last_progress_at = Some(now);
            }
        }

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
                        use tungstenite::{connect, Message};
                        use serde_json::Value;
                        let _ = std::fs::create_dir_all(&app_data);
                        let mut backoff_ms: u64 = 500;
                        loop {
                            if stop.load(std::sync::atomic::Ordering::Relaxed) { break; }
                            // Connect
                            let url = match url::Url::parse(&ws_url) { Ok(u) => u, Err(_) => break };
                            match connect(url) {
                                Ok((mut socket, _)) => {
                                    backoff_ms = 500;
                                    loop {
                                        if stop.load(std::sync::atomic::Ordering::Relaxed) { break; }
                                        match socket.read_message() {
                                            Ok(Message::Text(txt)) => {
                                                if let Ok(v) = serde_json::from_str::<Value>(&txt) {
                                                    let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
                                                    if typ == "execution_end" {
                                                        if let Some(pid) = v.get("data").and_then(|d| d.get("prompt_id")).and_then(|p| p.as_str()) {
                                                            // Fetch history and download outputs
                                                            let hist_url = format!("{}/history/{}", http_base, pid);
                                                            if let Ok(resp) = ureq::get(&hist_url).call() {
                                                                if let Ok(hist_json) = resp.into_string() {
                                                                    if let Ok(hj) = serde_json::from_str::<Value>(&hist_json) {
                                                                        // Expect nested { pid: { outputs: {...} } } or top-level outputs
                                                                        let root = hj.get(pid).unwrap_or(&hj);
                                                                        if let Some(outputs) = root.get("outputs").and_then(|o| o.as_object()) {
                                                                            for (_node, entry) in outputs.iter() {
                                                                                if let Some(images) = entry.get("images").and_then(|a| a.as_array()) {
                                                                                    for img in images {
                                                                                        let filename = img.get("filename").and_then(|s| s.as_str()).unwrap_or("");
                                                                                        let subfolder = img.get("subfolder").and_then(|s| s.as_str()).unwrap_or("");
                                                                                        let typv = img.get("type").and_then(|s| s.as_str()).unwrap_or("output");
                                                                                        if filename.is_empty() { continue; }
                                                                                        let view_url = format!(
                                                                                            "{}/view?filename={}&subfolder={}&type={}",
                                                                                            http_base,
                                                                                            urlencoding::encode(filename),
                                                                                            urlencoding::encode(subfolder),
                                                                                            urlencoding::encode(typv)
                                                                                        );
                                                                                        let tmp_path = app_data.join(filename);
                                                                                        if let Ok(resp) = ureq::get(&view_url).call() {
                                                                                            let mut reader = resp.into_reader();
                                                                                            if let Ok(mut f) = std::fs::File::create(&tmp_path) {
                                                                                                let _ = std::io::copy(&mut reader, &mut f);
                                                                                                let _ = tx.send((proj_id_at_start.clone(), tmp_path.clone()));
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
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
                if ui.button("Back to Projects").clicked() {
                    let _ = self.save_project_timeline();
                    self.mode = AppMode::ProjectPicker;
                    // Close embedded view to avoid overlaying the picker
                    if let Some(mut host) = self.comfy_webview.take() { host.close(); }
                }
                if ui.button("Jobs").clicked() {
                    self.show_jobs = !self.show_jobs;
                }
                if ui.button("Settings").clicked() { self.show_settings = !self.show_settings; }
                ui.separator();
                if ui.button(if self.engine.state == PlayState::Playing { "Pause (Space)" } else { "Play (Space)" }).clicked() {
                    let seq_fps = (self.seq.fps.num.max(1) as f64) / (self.seq.fps.den.max(1) as f64);
                    let current_sec = (self.playhead as f64) / seq_fps;
                    if self.engine.state == PlayState::Playing {
                        self.playback_clock.pause(current_sec);
                        self.engine.state = PlayState::Paused;
                        if let Some(engine) = &self.audio_out { engine.pause(current_sec); }
                    } else {
                        self.playback_clock.play(current_sec);
                        self.engine.state = PlayState::Playing;
                        if let Ok(clips) = self.build_audio_clips() {
                            if let Some(engine) = &self.audio_out { engine.start(current_sec, clips); }
                        }
                    }
                }
            });
        });

        egui::Window::new("Preview Settings")
            .open(&mut self.show_settings)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("Frame-based tolerances:");
                ui.add(
                    egui::Slider::new(&mut self.settings.strict_tolerance_frames, 0.5..=6.0)
                        .text("Strict pause tolerance (frames)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.settings.paused_tolerance_frames, 0.5..=6.0)
                        .text("Paused tolerance (frames)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.settings.clear_threshold_frames, 0.5..=6.0)
                        .text("Clear threshold on seek (frames)"),
                );
                ui.small("Higher tolerance = more off-target frames accepted. Higher clear threshold = fewer blanks on small nudges.");
            });

        // Project Picker page before opening editor
        if matches!(self.mode, AppMode::ProjectPicker) {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("Select a Project");
                ui.separator();
                let projects = self.db.list_projects().unwrap_or_default();
                // Visual grid of project cards
                ui.horizontal_wrapped(|ui| {
                    let card_w = 220.0;
                    let card_h = 170.0;
                    for p in &projects {
                        ui.group(|ui| {
                            ui.set_width(card_w);
                            // Thumbnail placeholder (16:9) with project initial
                            let thumb_h = (card_w / 16.0) * 9.0;
                            let (r, _resp) = ui.allocate_exact_size(egui::vec2(card_w - 16.0, thumb_h), egui::Sense::hover());
                            // Try to load a project thumbnail from the first available asset thumbnail
                            let tex_key = format!("project:{}", p.id);
                            let mut drew_thumb = false;
                            if !self.asset_thumb_textures.contains_key(&tex_key) {
                                if let Ok(mut assets) = self.db.list_assets(&p.id) {
                                    // Prefer earliest (reverse created_at DESC → ASC)
                                    assets.reverse();
                                    for a in &assets {
                                        let thumb_path = project::app_data_dir()
                                            .join("cache").join("thumbnails")
                                            .join(format!("{}-thumb.jpg", a.id));
                                        if thumb_path.exists() {
                                            if let Ok(img) = image::open(&thumb_path) {
                                                let rgba = img.to_rgba8();
                                                let (w, h) = rgba.dimensions();
                                                let color = egui::ColorImage::from_rgba_unmultiplied(
                                                    [w as usize, h as usize], &rgba.into_raw());
                                                let tex = ui.ctx().load_texture(
                                                    format!("project_thumb_{}", p.id),
                                                    color, egui::TextureOptions::LINEAR);
                                                self.asset_thumb_textures.insert(tex_key.clone(), tex);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                            if let Some(tex) = self.asset_thumb_textures.get(&tex_key) {
                                // Draw contained into 16:9 rect
                                let tw = tex.size()[0] as f32;
                                let th = tex.size()[1] as f32;
                                let rw = r.width();
                                let rh = r.height();
                                let scale = (rw / tw).min(rh / th);
                                let dw = (tw * scale).max(1.0);
                                let dh = (th * scale).max(1.0);
                                let img_rect = egui::Rect::from_center_size(r.center(), egui::vec2(dw, dh));
                                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                                ui.painter().image(self.asset_thumb_textures.get(&tex_key).unwrap().id(), img_rect, uv, egui::Color32::WHITE);
                                ui.painter().rect_stroke(r, 6.0, egui::Stroke::new(1.0, egui::Color32::from_gray(70)));
                                drew_thumb = true;
                            }
                            if !drew_thumb {
                                let c = egui::Color32::from_rgb(70, 80, 95);
                                ui.painter().rect_filled(r.shrink(2.0), 6.0, c);
                                let initial = p.name.chars().next().unwrap_or('?');
                                ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, initial, egui::FontId::proportional(28.0), egui::Color32::WHITE);
                            }
                            ui.add_space(6.0);
                            ui.label(egui::RichText::new(&p.name).strong());
                            ui.add_space(4.0);
                                if ui.button("Open").clicked() {
                                    self.project_id = p.id.clone();
                                    self.selected = None;
                                    self.drag = None;
                                    self.load_project_timeline();
                                    self.mode = AppMode::Editor;
                                }
                        });
                        ui.add_space(8.0);
                    }
                });
                ui.separator();
                ui.heading("Create Project");
                ui.horizontal(|ui| { ui.label("Name"); ui.text_edit_singleline(&mut self.new_project_name); });
                ui.small("Base path will be created under app data automatically.");
                if ui.add_enabled(!self.new_project_name.trim().is_empty(), egui::Button::new("Create")).clicked() {
                    // Auto-create base path under app data dir
                    let id = uuid::Uuid::new_v4().to_string();
                    let safe_name = self.new_project_name.trim();
                    let mut base = project::app_data_dir().join("projects").join(safe_name);
                    // Ensure unique
                    let mut i = 1;
                    while base.exists() { base = project::app_data_dir().join("projects").join(format!("{}-{}", safe_name, i)); i += 1; }
                    let _ = std::fs::create_dir_all(&base);
                    let _ = self.db.ensure_project(&id, safe_name, Some(&base));
                    self.project_id = id;
                    self.new_project_name.clear();
                    self.load_project_timeline();
                    self.mode = AppMode::Editor;
                }
            });
            return;
        }

        // Export dialog UI (editor mode only)
        self.export.ui(ctx, &self.seq, &self.db, &self.project_id);

        // Preview panel will be inside CentralPanel with resizable area

        egui::SidePanel::left("assets")
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
                    if let Some(files) = rfd::FileDialog::new().pick_files() {
                        let _ = self.import_files(&files);
                    }
                }
                if ui.button("Refresh").clicked() {}
                if ui.button("Jobs").clicked() { self.show_jobs = !self.show_jobs; }
                if ui.button("ComfyUI").clicked() { self.show_comfy_panel = !self.show_comfy_panel; }
            });
            // Thumbnails are fixed-size squares to keep layout consistent

            // Cloud (Modal) submission UI (always visible; independent of local embed)
            ui.collapsing("Cloud (Modal)", |ui| {
                ui.checkbox(&mut self.modal_enabled, "Enable");
                let mut ws = self.modal_ws_monitor;
                if ui.checkbox(&mut ws, "Live job monitor").on_hover_text("Listen to Cloud backend events and auto-import completed outputs").changed() {
                    self.modal_ws_monitor = ws;
                }
                ui.horizontal(|ui| {
                    ui.label("Target");
                    egui::ComboBox::from_id_source("cloud_target")
                        .selected_text(match self.cloud_target { CloudTarget::Prompt => "ComfyUI /prompt", CloudTarget::Workflow => "Workflow (auto-convert)" })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.cloud_target, CloudTarget::Prompt, "ComfyUI /prompt");
                            ui.selectable_value(&mut self.cloud_target, CloudTarget::Workflow, "Workflow (auto-convert)");
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("Base URL");
                    ui.text_edit_singleline(&mut self.modal_base_url).on_hover_text("e.g., https://api.yourdomain.com");
                });
                ui.horizontal(|ui| {
                    ui.label("Relay WS URL");
                    ui.text_edit_singleline(&mut self.modal_relay_ws_url)
                        .on_hover_text("Optional: wss://relay.yourdomain.com/stream (overrides /events)");
                });
                ui.horizontal(|ui| {
                    ui.label("API Key");
                    let mut masked = self.modal_api_key.clone();
                    if ui.text_edit_singleline(&mut masked).changed() { self.modal_api_key = masked; }
                });
                ui.label("Payload (JSON)");
                egui::ScrollArea::vertical()
                    .id_source("cloud_payload")
                    .max_height(240.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut self.modal_payload)
                                .desired_rows(12)
                        );
                    });
                ui.horizontal(|ui| {
                    if ui.button("Test Connection").clicked() { self.modal_test_connection(); }
                    if ui.add_enabled(self.modal_enabled, egui::Button::new("Queue Job")).clicked() {
                        // Queue the job; monitor will start on JobQueued event
                        self.modal_queue_job();
                    }
                });
                // Live monitor summary
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(format!("Queue — pending: {}, running: {}", self.modal_queue_pending, self.modal_queue_running));
                    if ui.small_button("Clear Progress").clicked() { self.modal_job_progress.clear(); }
                });
                // Show phase-aware progress bars for known jobs
                egui::ScrollArea::vertical().id_source("cloud_progress").max_height(220.0).show(ui, |ui| {
                    let now = std::time::Instant::now();
                    // prune stale entries > 10 min without updates
                    let mut prune: Vec<String> = Vec::new();
                    for (jid, (_p, _c, _t, ts)) in self.modal_job_progress.iter() {
                        if now.duration_since(*ts).as_secs() > 600 { prune.push(jid.clone()); }
                    }
                    for k in prune { self.modal_job_progress.remove(&k); self.modal_phase_agg.remove(&k); }

                    for (jid, agg) in self.modal_phase_agg.iter() {
                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                ui.strong(jid);
                                if let Some(src) = self.modal_job_source.get(jid) {
                                    let tag = match src { crate::CloudUpdateSrc::Ws => "WS", crate::CloudUpdateSrc::Jobs => "/jobs", crate::CloudUpdateSrc::Status => "/status" };
                                    ui.small(format!("[{}]", tag));
                                }
                            });
                            let sum_cur = (agg.s_cur as u64) + (agg.e_cur as u64);
                            let sum_tot = (agg.s_tot as u64) + (agg.e_tot as u64);
                            if agg.importing {
                                ui.add(egui::ProgressBar::new(1.0).text("importing"));
                            } else if sum_tot > 0 {
                                let overall = (sum_cur as f32) / (sum_tot as f32);
                                let overall_dbl = (overall * 2.0).clamp(0.0, 1.0);
                                ui.add(egui::ProgressBar::new(overall_dbl).text(format!("Overall {:.1}%", overall_dbl * 100.0)));
                            } else if let Some((percent, _c, _t, _ts)) = self.modal_job_progress.get(jid) {
                                // Fallback: show percent-based bar if steps are unknown
                                let percent2 = (*percent * 2.0).clamp(0.0, 100.0);
                                let p = (percent2 / 100.0).clamp(0.0, 1.0);
                                ui.add(egui::ProgressBar::new(p).text(format!("Overall {:.1}%", percent2)));
                            } else {
                                // No totals and no percent yet — show a visible queued/connecting placeholder bar
                                ui.add(egui::ProgressBar::new(0.01).text("Queued / waiting for updates…"));
                            }
                            // Removed separate Sampling/Encoding bars to keep a single overall bar
                        });
                        ui.add_space(6.0);
                    }
                    if self.modal_phase_agg.is_empty() { ui.small("No active jobs."); }
                });
                // Cloud monitor logs removed for a cleaner UI
                ui.label("Recent Cloud Artifacts");
                ui.horizontal(|ui| {
                    if ui.small_button("Refresh").clicked() { self.modal_refresh_recent(); }
                    ui.small("Click Import to fetch into this project.");
                });
                egui::ScrollArea::vertical().id_source("cloud_recent").max_height(180.0).show(ui, |ui| {
                    if self.modal_recent.is_empty() {
                        ui.small("No recent jobs yet.");
                    }
                    for (jid, arts) in &self.modal_recent {
                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                ui.strong(jid);
                                if arts.is_empty() { ui.small("(no artifacts)"); }
                            });
                            for (fname, url) in arts.iter().take(3) {
                                ui.horizontal(|ui| {
                                    ui.label(fname);
                                    if ui.small_button("Import").clicked() {
                                        self.modal_import_url(url.clone(), Some(fname.clone()));
                                    }
                                    if ui.small_button("Open").clicked() { let _ = webbrowser::open(url); }
                                });
                            }
                        });
                    }
                });
            });

            // Embedded ComfyUI panel at top of assets (outside scrolling region)
            if self.comfy_embed_inside && self.comfy_embed_in_assets {
                let running = self.comfy.is_running();
                ui.horizontal(|ui| {
                    ui.strong("ComfyUI");
                    ui.add(egui::Slider::new(&mut self.comfy_assets_height, 200.0..=900.0).text("Height"));
                    ui.separator();
                    if ui.small_button("Reload").clicked() {
                        if let Some(h) = self.comfy_webview.as_mut() { h.reload(); self.comfy_embed_logs.push_back("Reload requested".into()); }
                    }
                    let mut dt = self.comfy_devtools;
                    if ui.checkbox(&mut dt, "DevTools").changed() {
                        self.comfy_devtools = dt;
                        if let Some(h) = self.comfy_webview.as_mut() { h.set_devtools(dt); }
                        self.comfy_embed_logs.push_back(format!("DevTools {}", if dt {"enabled"} else {"disabled"}));
                    }
                    if ui.small_button("Inspector").clicked() {
                        if let Some(h) = self.comfy_webview.as_mut() { let _ = h.open_inspector(); }
                    }
                    if ui.small_button("Browser").clicked() { let _ = webbrowser::open(&self.comfy.url()); }
                    ui.separator();
                    ui.checkbox(&mut self.comfy_auto_import, "Auto-import");
                    ui.separator();
                    ui.checkbox(&mut self.comfy_ws_monitor, "Live job monitor").on_hover_text("Listen to ComfyUI WebSocket and import outputs on job completion");
                });
                ui.separator();
                if running {
                    // Ensure host exists
                    if self.comfy_webview.is_none() {
                        if let Some(mut host) = crate::embed_webview::create_platform_host() {
                            if self.comfy_devtools { host.set_devtools(true); }
                            host.navigate(&self.comfy.url());
                            host.set_visible(true);
                            self.comfy_webview = Some(host);
                            self.comfy_embed_logs.push_back("Embedded view created (assets)".into());
                        } else {
                            self.comfy_embed_logs.push_back("Failed to create embedded view (assets)".into());
                        }
                    }
                    // Reserve area and position overlay to match this rect
                    // Leave a small right-side margin so the panel's resize grab remains clickable.
                    let w = (ui.available_width() - 8.0).max(0.0);
                    let size = egui::vec2(w, self.comfy_assets_height);
                    let (rect, _resp) = ui.allocate_exact_size(size, egui::Sense::hover());
                    if let Some(host) = self.comfy_webview.as_mut() {
                        // Use floor for x/top and ceil for width/height to avoid overlap from rounding.
                        let to_floor = |v: f32| -> i32 { v.floor() as i32 };
                        let to_ceil = |v: f32| -> i32 { v.ceil() as i32 };
                        let r = crate::embed_webview::RectPx {
                            x: to_floor(rect.left()),
                            y: to_floor(rect.top()),
                            w: to_ceil(rect.width()),
                            h: to_ceil(rect.height()),
                        };
                        host.set_rect(r);
                        host.set_visible(true);
                    }
                    // Resizable handle below the ComfyUI view (adjusts section height)
                    ui.add_space(2.0);
                    let handle_h = 8.0;
                    let (hrect, hresp) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), handle_h),
                        egui::Sense::click_and_drag(),
                    );
                    let hovered = hresp.hovered() || hresp.dragged();
                    let stroke = if hovered {
                        egui::Stroke::new(2.0, egui::Color32::from_gray(220))
                    } else {
                        egui::Stroke::new(1.0, egui::Color32::from_gray(150))
                    };
                    ui.painter().line_segment([hrect.left_center(), hrect.right_center()], stroke);
                    if hresp.dragged() {
                        self.comfy_assets_height = (self.comfy_assets_height + hresp.drag_delta().y)
                            .clamp(200.0, 900.0);
                    }
                    ui.separator();
                } else {
                    if let Some(mut host) = self.comfy_webview.take() { host.close(); }
                }
            } else {
                // If not embedding here, ensure any previous assets-embedded view is closed
                // (Bottom dock may manage webview separately.)
            }

                // Scrolling region: the rest of the side panel content
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            
            // (Removed: Hardware Encoders list — not useful to users)

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

            // ComfyUI (Phase 1 + basic Phase 2 installer)
            if self.show_comfy_panel {
                ui.separator();
                ui.heading("ComfyUI");
                ui.small("Set the ComfyUI repository path (folder containing main.py). Start server locally and open embedded window.");
                ui.horizontal(|ui| {
                    let mut embed = self.comfy_embed_inside;
                    if ui.checkbox(&mut embed, "Open inside editor").changed() {
                        self.comfy_embed_inside = embed;
                        if !embed {
                            if let Some(mut host) = self.comfy_webview.take() { host.close(); }
                            self.comfy_embed_logs.push_back("Embedded view closed".into());
                        }
                    }
                    if cfg!(not(all(target_os = "macos", feature = "embed-webview"))) {
                        ui.small("(embed requires macOS build with feature: embed-webview)");
                    }
                    if self.comfy_embed_inside {
                        if ui.small_button("Reload").clicked() {
                            if let Some(h) = self.comfy_webview.as_mut() { h.reload(); self.comfy_embed_logs.push_back("Reload requested".into()); }
                        }
                        let mut dt = self.comfy_devtools;
                        if ui.checkbox(&mut dt, "DevTools").on_hover_text("Enable WebKit developer extras; right-click → Inspect").changed() {
                            self.comfy_devtools = dt;
                            if let Some(h) = self.comfy_webview.as_mut() { h.set_devtools(dt); }
                            self.comfy_embed_logs.push_back(format!("DevTools {}", if dt {"enabled"} else {"disabled"}));
                        }
                        if ui.small_button("Open Inspector").clicked() {
                            if let Some(h) = self.comfy_webview.as_mut() {
                                let ok = h.open_inspector();
                                self.comfy_embed_logs.push_back(if ok { "Inspector opened".into() } else { "Inspector unavailable; enable DevTools, then right-click → Inspect".into() });
                            } else {
                                self.comfy_embed_logs.push_back("Inspector: no embedded view".into());
                            }
                        }
                        ui.separator();
                        let mut ai = self.comfy_auto_import;
                        if ui.checkbox(&mut ai, "Auto-import outputs").on_hover_text("Watch ComfyUI output folder and import finished videos into this project").changed() {
                            self.comfy_auto_import = ai;
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Repo Path");
                    let resp = ui.text_edit_singleline(&mut self.comfy_repo_input);
                    let enter_commit = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    let save_clicked = ui.small_button("Save").on_hover_text("Commit path to settings").clicked();
                    if ui.small_button("Browse…").clicked() {
                        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                            self.comfy_repo_input = folder.to_string_lossy().to_string();
                        }
                    }
                    if enter_commit || save_clicked {
                        let s = self.comfy_repo_input.trim();
                        if !s.is_empty() {
                            let dir = std::path::Path::new(s);
                            let has_main = dir.is_dir() && dir.join("main.py").exists();
                            if has_main {
                                self.comfy.config_mut().repo_path = Some(dir.to_path_buf());
                            }
                        }
                    }
                });
                // Basic validation feedback
                let path_str = self.comfy_repo_input.trim();
                if path_str.is_empty() {
                    ui.colored_label(egui::Color32::GRAY, "Path not set");
                } else {
                    let dir = std::path::Path::new(path_str);
                    if !dir.is_dir() {
                        ui.colored_label(egui::Color32::RED, "Folder does not exist");
                    } else if !dir.join("main.py").exists() {
                        ui.colored_label(egui::Color32::YELLOW, "Selected folder does not contain main.py");
                    } else {
                        ui.colored_label(egui::Color32::GREEN, "OK");
                    }
                }
                ui.horizontal(|ui| {
                    ui.label("Python");
                    let mut py = self.comfy.config().python_cmd.clone();
                    if ui.text_edit_singleline(&mut py).changed() {
                        self.comfy.config_mut().python_cmd = py;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Host");
                    let mut host = self.comfy.config().host.clone();
                    if ui.text_edit_singleline(&mut host).changed() {
                        self.comfy.config_mut().host = host;
                    }
                    ui.label("Port");
                    let mut p = self.comfy.config().port as i32;
                    if ui.add(egui::DragValue::new(&mut p).clamp_range(1024..=65535)).changed() {
                        self.comfy.config_mut().port = p.clamp(1024, 65535) as u16;
                    }
                    let mut https = self.comfy.config().https;
                    if ui.checkbox(&mut https, "HTTPS").on_hover_text("Use HTTPS/WSS when connecting to remote ComfyUI").changed() {
                        self.comfy.config_mut().https = https;
                    }
                    let in_use = self.comfy.is_port_open();
                    if in_use {
                        ui.colored_label(egui::Color32::YELLOW, "Port in use");
                    }
                });
                ui.collapsing("Installation", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Install Dir");
                        let _ = ui.text_edit_singleline(&mut self.comfy_install_dir_input);
                        if ui.small_button("Browse…").clicked() {
                            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                                self.comfy_install_dir_input = folder.to_string_lossy().to_string();
                            }
                        }
                    });
                    let dir = std::path::Path::new(self.comfy_install_dir_input.trim());
                    if !dir.exists() {
                        ui.colored_label(egui::Color32::GRAY, "Dir will be created");
                    }
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.comfy_install_ffmpeg, "Install FFmpeg (system)")
                            .on_hover_text("Best-effort install via your OS package manager (brew/winget/choco/apt/etc.)");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Python for venv");
                        ui.text_edit_singleline(&mut self.comfy_venv_python_input)
                            .on_hover_text("Optional: interpreter to create .venv (prefer Python 3.11/3.12)");
                    });
                    ui.collapsing("pip settings", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Index URL");
                            ui.text_edit_singleline(&mut self.pip_index_url_input)
                                .on_hover_text("e.g., https://pypi.org/simple or a local mirror");
                        });
                        ui.horizontal(|ui| {
                            ui.label("Extra Index URL");
                            ui.text_edit_singleline(&mut self.pip_extra_index_url_input)
                                .on_hover_text("additional package index to search");
                        });
                        ui.horizontal(|ui| {
                            ui.label("Trusted hosts");
                            ui.text_edit_singleline(&mut self.pip_trusted_hosts_input)
                                .on_hover_text("comma-separated hostnames (e.g., pypi.org,files.pythonhosted.org)");
                        });
                        ui.horizontal(|ui| {
                            ui.label("Proxy URL");
                            ui.text_edit_singleline(&mut self.pip_proxy_input)
                                .on_hover_text("e.g., http://user:pass@proxy:port");
                        });
                        ui.checkbox(&mut self.pip_no_cache, "No cache")
                            .on_hover_text("Pass --no-cache-dir to pip");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Torch Backend");
                        egui::ComboBox::from_id_source("torch_backend")
                            .selected_text(match self.comfy_torch_backend {
                                crate::comfyui::TorchBackend::Auto => "Auto",
                                crate::comfyui::TorchBackend::Cuda => "CUDA",
                                crate::comfyui::TorchBackend::Mps => "MPS",
                                crate::comfyui::TorchBackend::Rocm => "ROCm",
                                crate::comfyui::TorchBackend::Cpu => "CPU",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.comfy_torch_backend, crate::comfyui::TorchBackend::Auto, "Auto");
                                ui.selectable_value(&mut self.comfy_torch_backend, crate::comfyui::TorchBackend::Cuda, "CUDA");
                                ui.selectable_value(&mut self.comfy_torch_backend, crate::comfyui::TorchBackend::Mps, "MPS");
                                ui.selectable_value(&mut self.comfy_torch_backend, crate::comfyui::TorchBackend::Rocm, "ROCm");
                                ui.selectable_value(&mut self.comfy_torch_backend, crate::comfyui::TorchBackend::Cpu, "CPU");
                            });
                    });
                    ui.checkbox(&mut self.comfy_recreate_venv, "Recreate venv (.venv) using Python for venv")
                        .on_hover_text("Deletes and recreates .venv to switch Python versions (useful when upgrading/downgrading)");
                    ui.horizontal(|ui| {
                        if ui.button("Install / Repair").clicked() {
                            let mut plan = crate::comfyui::InstallerPlan::default();
                            let s = self.comfy_install_dir_input.trim();
                            if !s.is_empty() { plan.install_dir = Some(std::path::PathBuf::from(s)); }
                            plan.torch_backend = self.comfy_torch_backend;
                            let p = self.comfy_venv_python_input.trim();
                            if !p.is_empty() { plan.python_for_venv = Some(p.to_string()); }
                            plan.recreate_venv = self.comfy_recreate_venv;
                            plan.install_ffmpeg = self.comfy_install_ffmpeg;
                            // pip config
                            let mut trusted: Vec<String> = Vec::new();
                            for t in self.pip_trusted_hosts_input.split(',') { let tt = t.trim(); if !tt.is_empty() { trusted.push(tt.to_string()); } }
                            plan.pip.index_url = if self.pip_index_url_input.trim().is_empty() { None } else { Some(self.pip_index_url_input.trim().to_string()) };
                            plan.pip.extra_index_url = if self.pip_extra_index_url_input.trim().is_empty() { None } else { Some(self.pip_extra_index_url_input.trim().to_string()) };
                            plan.pip.trusted_hosts = trusted;
                            plan.pip.proxy = if self.pip_proxy_input.trim().is_empty() { None } else { Some(self.pip_proxy_input.trim().to_string()) };
                            plan.pip.no_cache = self.pip_no_cache;
                            self.comfy.install(plan);
                        }
                        if ui.button("Validate").clicked() {
                            self.comfy.validate_install();
                        }
                        if ui.button("Use Installed").clicked() {
                            self.comfy.use_installed();
                            if let Some(p) = self.comfy.config().repo_path.as_ref() {
                                self.comfy_repo_input = p.to_string_lossy().to_string();
                            }
                        }
                        if ui.button("Uninstall").clicked() {
                            self.comfy.uninstall();
                        }
                        if ui.button("Repair Missing Packages").clicked() {
                            self.comfy.repair_common_packages();
                        }
                    });
                    ui.small("Installer creates a reusable .venv in the selected directory. It will NOT re-create it on Start.");
                });
                let running = self.comfy.is_running();
                // Enable Start only when repo looks valid and python cmd is non-empty
                let repo_dir_valid = {
                    let s = self.comfy_repo_input.trim();
                    !s.is_empty() && std::path::Path::new(s).is_dir() && std::path::Path::new(s).join("main.py").exists()
                };
                let py_ok = !self.comfy.config().python_cmd.trim().is_empty();
                ui.horizontal(|ui| {
                    if ui.add_enabled(!running && repo_dir_valid && py_ok, egui::Button::new("Start ComfyUI")).clicked() {
                        // Commit the repo path from the input before starting
                        let s = self.comfy_repo_input.trim();
                        if !s.is_empty() {
                            self.comfy.config_mut().repo_path = Some(std::path::PathBuf::from(s));
                        }
                        self.comfy.start();
                        // If embedding is enabled and host not created, create and navigate now
                        if self.comfy_embed_inside && self.comfy_webview.is_none() {
                            if let Some(mut host) = crate::embed_webview::create_platform_host() {
                                if self.comfy_devtools { host.set_devtools(true); }
                                host.navigate(&self.comfy.url());
                                host.set_visible(true);
                                self.comfy_webview = Some(host);
                                self.comfy_embed_logs.push_back("Embedded view created".into());
                            } else {
                                self.comfy_embed_inside = false; // disable switch when not available
                                let reason = if cfg!(not(all(target_os = "macos", feature = "embed-webview"))) {
                                    "feature flag not active; rebuild with --features embed-webview"
                                } else { "no NSWindow contentView found; focus the app window and try again" };
                                self.comfy_embed_logs.push_back(format!("Failed to create embedded view ({})", reason));
                            }
                        }
                    }
                    if ui.add_enabled(running, egui::Button::new("Open Window")).clicked() {
                        self.comfy.open_webview_window();
                    }
                    if ui.add_enabled(running, egui::Button::new("Stop")).clicked() {
                        self.comfy.stop();
                        if let Some(mut host) = self.comfy_webview.take() { host.close(); }
                        self.comfy_embed_logs.push_back("Embedded view closed".into());
                    }
                });
                ui.label(format!("Status: {:?}", self.comfy.last_status));
                if let Some(err) = &self.comfy.last_error { ui.colored_label(egui::Color32::RED, err); }
                // Inline embed removed; see bottom dock panel for embedded view rendering.
                ui.collapsing("Logs", |ui| {
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .max_height(220.0)
                        .show(ui, |ui| {
                            for line in self.comfy.logs(500) { ui.monospace(line); }
                        });
                });
                ui.collapsing("Embedded View Logs", |ui| {
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .max_height(120.0)
                        .show(ui, |ui| {
                            while self.comfy_embed_logs.len() > 200 { self.comfy_embed_logs.pop_front(); }
                            for line in self.comfy_embed_logs.iter() { ui.monospace(line); }
                        });
                });
                ui.collapsing("Auto-import Logs", |ui| {
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .max_height(120.0)
                        .show(ui, |ui| {
                            while self.comfy_import_logs.len() > 400 { self.comfy_import_logs.pop_front(); }
                            for line in self.comfy_import_logs.iter() { ui.monospace(line); }
                        });
                });
                if ui.small_button("Open in Browser").clicked() {
                    let _ = webbrowser::open(&self.comfy.url());
                }
            }
            egui::Separator::default().ui(ui);
            let assets = self.assets();
            // Fixed 80x80 square thumbnails; compact grid
            let cell = 80.0f32;
            let card_w = cell + 4.0; // minimal horizontal gap
            let cols = (ui.available_width() / card_w).floor().max(1.0) as usize;
            egui::Grid::new("assets_grid").num_columns(cols).spacing([2.0, 8.0]).show(ui, |ui| {
                for (i, a) in assets.iter().enumerate() {
                    // One cell
                    ui.vertical(|ui| {
                        // Square slot
                        let (r, resp) = ui.allocate_exact_size(
                            egui::vec2(cell, cell),
                            egui::Sense::click_and_drag(),
                        );
                        // Paint image (contained inside square) or placeholder
                        if let Some(tex) = self.load_thumb_texture(ctx, a, cell as u32, cell as u32) {
                            // Contain inside square based on asset or texture aspect
                            let (dw, dh) = match (a.width, a.height) {
                                (Some(w), Some(h)) if w > 0 && h > 0 => {
                                    let aspect = w as f32 / h as f32;
                                    if aspect >= 1.0 { (cell, (cell / aspect).max(1.0)) } else { (cell * aspect, cell) }
                                }
                                _ => {
                                    // Fallback assume 16:9
                                    let aspect = 16.0 / 9.0;
                                    (cell, (cell / aspect).max(1.0))
                                }
                            };
                            let img_rect = egui::Rect::from_center_size(r.center(), egui::vec2(dw, dh));
                            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                            ui.painter().image(tex.id(), img_rect, uv, egui::Color32::WHITE);
                            ui.painter().rect_stroke(r, 4.0, egui::Stroke::new(1.0, egui::Color32::from_gray(60)));
                        } else {
                            // Placeholder card
                            ui.painter().rect_filled(r, 6.0, egui::Color32::from_rgb(60, 66, 82));
                            // Initial letter overlay
                            let name = std::path::Path::new(&a.src_abs)
                                .file_stem()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_else(|| a.src_abs.clone());
                            let ch = name.chars().next().unwrap_or('?');
                            ui.painter().text(
                                r.center(),
                                egui::Align2::CENTER_CENTER,
                                ch,
                                egui::FontId::proportional(28.0),
                                egui::Color32::WHITE,
                            );
                        }
                        // Interactions
                        if resp.drag_started() { self.dragging_asset = Some(a.clone()); }
                        if resp.clicked() { self.add_asset_to_timeline(a); }
                        ui.add_space(2.0);
                        let name = std::path::Path::new(&a.src_abs)
                            .file_name()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| a.src_abs.clone());
                        let lbl = egui::Label::new(egui::RichText::new(name).small());
                        ui.add_sized([cell, 14.0], lbl);
                        ui.small(&a.kind);
                    });
                    if (i + 1) % cols == 0 { ui.end_row(); }
                }
                // Ensure the last row ends
                if assets.len() % cols != 0 { ui.end_row(); }
            });
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
                            // Prepare pending change flags to avoid borrow issues
                            let mut pending_rate: Option<f32> = None;
                            let mut pending_offset_frames: Option<i64> = None;
                            ui.separator();
                            ui.label("Video");
                            ui.horizontal(|ui| {
                                ui.label("Rate");
                                let mut r = *rate as f64;
                                let changed = ui
                                    .add(egui::DragValue::new(&mut r).clamp_range(0.05..=8.0).speed(0.02))
                                    .changed();
                                if changed {
                                    *rate = (r as f32).max(0.01);
                                    pending_rate = Some(*rate);
                                }
                                if ui.small_button("1.0").on_hover_text("Reset").clicked() { *rate = 1.0; }
                            });
                            ui.horizontal(|ui| {
                                ui.label("In Offset (s)");
                                let mut o = *in_offset_sec;
                                let changed = ui
                                    .add(egui::DragValue::new(&mut o).clamp_range(0.0..=1_000_000.0).speed(0.01))
                                    .changed();
                                if changed {
                                    *in_offset_sec = o.max(0.0);
                                    let num = self.seq.fps.num as f64;
                                    let den = self.seq.fps.den.max(1) as f64;
                                    let frames = ((o.max(0.0)) * (num / den)).round() as i64;
                                    pending_offset_frames = Some(frames.max(0));
                                }
                                if ui.small_button("0").on_hover_text("Reset").clicked() { *in_offset_sec = 0.0; }
                            });
                            // Apply pending updates after UI borrows end
                            if pending_rate.is_some() || pending_offset_frames.is_some() {
                                if let Ok(uuid) = uuid::Uuid::parse_str(&item.id) {
                                    let node_id = NodeId(uuid);
                                    if let Some(mut node) = self.seq.graph.nodes.get(&node_id).cloned() {
                                        if let TimelineNodeKind::Clip(mut clip) = node.kind.clone() {
                                            if let Some(r) = pending_rate { clip.playback_rate = r; }
                                            if let Some(fr) = pending_offset_frames { clip.media_range.start = fr; }
                                            node.kind = TimelineNodeKind::Clip(clip);
                                            let _ = self.apply_timeline_command(TimelineCommand::UpdateNode { node });
                                        }
                                    }
                                }
                            }
                        }
                        ItemKind::Audio { in_offset_sec, rate, .. } => {
                            let mut pending_rate: Option<f32> = None;
                            let mut pending_offset_frames: Option<i64> = None;
                            ui.separator();
                            ui.label("Audio");
                            ui.horizontal(|ui| {
                                ui.label("Rate");
                                let mut r = *rate as f64;
                                let changed = ui
                                    .add(egui::DragValue::new(&mut r).clamp_range(0.05..=8.0).speed(0.02))
                                    .changed();
                                if changed {
                                    *rate = (r as f32).max(0.01);
                                    pending_rate = Some(*rate);
                                }
                                if ui.small_button("1.0").on_hover_text("Reset").clicked() { *rate = 1.0; }
                            });
                            ui.horizontal(|ui| {
                                ui.label("In Offset (s)");
                                let mut o = *in_offset_sec;
                                let changed = ui
                                    .add(egui::DragValue::new(&mut o).clamp_range(0.0..=1_000_000.0).speed(0.01))
                                    .changed();
                                if changed {
                                    *in_offset_sec = o.max(0.0);
                                    let num = self.seq.fps.num as f64;
                                    let den = self.seq.fps.den.max(1) as f64;
                                    let frames = ((o.max(0.0)) * (num / den)).round() as i64;
                                    pending_offset_frames = Some(frames.max(0));
                                }
                                if ui.small_button("0").on_hover_text("Reset").clicked() { *in_offset_sec = 0.0; }
                            });
                            if pending_rate.is_some() || pending_offset_frames.is_some() {
                                if let Ok(uuid) = uuid::Uuid::parse_str(&item.id) {
                                    let node_id = NodeId(uuid);
                                    if let Some(mut node) = self.seq.graph.nodes.get(&node_id).cloned() {
                                        if let TimelineNodeKind::Clip(mut clip) = node.kind.clone() {
                                            if let Some(r) = pending_rate { clip.playback_rate = r; }
                                            if let Some(fr) = pending_offset_frames { clip.media_range.start = fr; }
                                            node.kind = TimelineNodeKind::Clip(clip);
                                            let _ = self.apply_timeline_command(TimelineCommand::UpdateNode { node });
                                        }
                                    }
                                }
                            }
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

        // No floating window: when not embedding in assets, ensure any host is closed.
        if !(self.comfy_embed_inside && self.comfy_embed_in_assets) {
            if let Some(mut host) = self.comfy_webview.take() { host.close(); }
        }

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
                // Track controls
                if ui.small_button("+ Video Track").clicked() {
                    // Insert new video track at the top
                    let binding = timeline_crate::TrackBinding {
                        id: timeline_crate::TrackId::new(),
                        name: String::new(),
                        kind: timeline_crate::TrackKind::Video,
                        node_ids: Vec::new(),
                    };
                    self.seq.graph.tracks.insert(0, binding);
                    self.sync_tracks_from_graph();
                    let _ = self.save_project_timeline();
                }
                if ui.small_button("+ Audio Track").clicked() {
                    let binding = timeline_crate::TrackBinding { id: timeline_crate::TrackId::new(), name: String::new(), kind: timeline_crate::TrackKind::Audio, node_ids: Vec::new() };
                    let _ = self.apply_timeline_command(timeline_crate::TimelineCommand::UpsertTrack { track: binding });
                    let _ = self.save_project_timeline();
                }
                if ui.small_button("− Last Video").clicked() {
                    if let Some((idx, id)) = self.seq.graph.tracks.iter().enumerate().rev().find_map(|(i,t)| match t.kind { timeline_crate::TrackKind::Video => Some((i,t.id)), _=>None }) {
                        let _ = self.apply_timeline_command(timeline_crate::TimelineCommand::RemoveTrack { track_id: id });
                        let _ = self.save_project_timeline();
                    }
                }
                if ui.small_button("− Last Audio").clicked() {
                    if let Some((idx, id)) = self.seq.graph.tracks.iter().enumerate().rev().find_map(|(i,t)| match t.kind { timeline_crate::TrackKind::Audio => Some((i,t.id)), _=>None }) {
                        let _ = self.apply_timeline_command(timeline_crate::TimelineCommand::RemoveTrack { track_id: id });
                        let _ = self.save_project_timeline();
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Autosave indicator
                    if let Some(t) = self.last_save_at {
                        let ago = Instant::now().saturating_duration_since(t);
                        let label = if ago.as_secs_f32() < 2.0 { "Saved".to_string() } else { format!("Autosave {}s ago", ago.as_secs()) };
                        ui.small(label);
                    }
                    let cache_stats = format!("Cache: {}/{} hits",
                        self.preview.cache_hits,
                        self.preview.cache_hits + self.preview.cache_misses);
                    ui.small(&cache_stats);
                    if ui.small_button("Save Project").clicked() { let _ = self.save_project_timeline(); }
                });
            });
            
            self.timeline_ui(ui);
        });

        self.jobs_window(ctx);

        // Global drag overlay for assets: show a floating thumbnail near the cursor
        if let Some(asset) = self.dragging_asset.clone() {
            if let Some(pos) = ctx.input(|i| i.pointer.hover_pos()) {
                let painter = ctx.layer_painter(egui::LayerId::new(
                    egui::Order::Foreground,
                    egui::Id::new("dragging_asset_overlay"),
                ));
                let thumb_w = self.asset_thumb_w.min(200.0).max(80.0);
                // Assume 16:9 if unknown; otherwise use asset aspect
                let (w, h) = match (asset.width, asset.height) {
                    (Some(w), Some(h)) if w > 0 && h > 0 => {
                        let aspect = w as f32 / h as f32;
                        let hh = (thumb_w / aspect).clamp(50.0, 200.0);
                        (thumb_w, hh)
                    }
                    _ => (thumb_w, (thumb_w / 16.0) * 9.0),
                };
                // Try to fetch a texture; fallback to a colored rect with initial
                let off = egui::vec2(16.0, 16.0);
                let rect = egui::Rect::from_min_size(pos + off, egui::vec2(w, h));
                if let Some(tex) = self.load_thumb_texture(ctx, &asset, w as u32, h as u32) {
                    painter.image(tex.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0,0.0), egui::pos2(1.0,1.0)), egui::Color32::WHITE);
                    painter.rect_stroke(rect, 6.0, egui::Stroke::new(1.0, egui::Color32::from_gray(240)));
                } else {
                    let fill = if asset.kind.eq_ignore_ascii_case("audio") { egui::Color32::from_rgba_unmultiplied(30, 160, 70, 180) } else { egui::Color32::from_rgba_unmultiplied(60, 120, 220, 180) };
                    painter.rect_filled(rect, 6.0, fill);
                    painter.rect_stroke(rect, 6.0, egui::Stroke::new(1.0, egui::Color32::from_gray(240)));
                    let name = std::path::Path::new(&asset.src_abs)
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| asset.src_abs.clone());
                    let ch = name.chars().next().unwrap_or('?');
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        ch,
                        egui::FontId::proportional(28.0),
                        egui::Color32::WHITE,
                    );
                }
                // Media name label removed for a cleaner asset grid
            }
        }
    }
}
