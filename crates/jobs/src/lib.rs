use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
    time::Duration,
};
use thiserror::Error;
use uuid::Uuid;

use media_io::{generate_proxy, generate_thumbnail, generate_waveform};
use project::{app_data_dir, JobRow, ProjectDb};

#[derive(Debug, Error)]
pub enum JobError {
    #[error("worker stopped")]
    Stopped,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum JobKind {
    Waveform,
    Thumbnails,
    Proxy,
    SeekIndex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobSpec {
    pub asset_id: String,
    pub kind: JobKind,
    pub priority: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Running,
    Progress(f32),
    Done,
    Failed(String),
    Canceled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobEvent {
    pub id: String,
    pub asset_id: String,
    pub kind: JobKind,
    pub status: JobStatus,
}

#[derive(Clone)]
pub struct JobsHandle {
    tx_submit: Sender<(String, JobSpec)>,
    tx_cancel: Sender<String>,
    pub rx_events: Receiver<JobEvent>,
}

pub struct JobsRuntime {
    _queue: Arc<Mutex<VecDeque<(String, JobSpec)>>>,
    rx_submit: Receiver<(String, JobSpec)>,
    rx_cancel: Receiver<String>,
    tx_events: Sender<JobEvent>,
    _workers: Vec<thread::JoinHandle<()>>,
    canceled: Arc<Mutex<HashSet<String>>>,
    _db_path: Arc<PathBuf>,
}

impl JobsRuntime {
    pub fn start(db_path: PathBuf, num_workers: usize) -> JobsHandle {
        let (tx_submit, rx_submit) = unbounded::<(String, JobSpec)>();
        let (tx_cancel, rx_cancel) = unbounded::<String>();
        let (tx_events, rx_events) = unbounded::<JobEvent>();
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        let canceled = Arc::new(Mutex::new(HashSet::new()));

        let db_path_arc = Arc::new(db_path);
        // Preload pending jobs from DB
        {
            if let Ok(db) = ProjectDb::open_or_create(db_path_arc.as_ref()) {
                let _ = db.reset_running_jobs();
                if let Ok(rows) = db.list_pending_jobs() {
                    let mut q = queue.lock();
                    for row in rows {
                        if let Some(spec) = job_spec_from_row(&row) {
                            q.push_back((row.id.clone(), spec));
                        }
                    }
                }
            }
        }

        let runtime = JobsRuntime {
            _queue: queue.clone(),
            rx_submit,
            rx_cancel,
            tx_events: tx_events.clone(),
            _workers: Vec::new(),
            canceled: canceled.clone(),
            _db_path: db_path_arc.clone(),
        };
        runtime.spawn_workers(num_workers, queue.clone(), db_path_arc.clone());

        // Feeder thread
        {
            let q = queue.clone();
            let canceled = canceled.clone();
            let rx_s = runtime.rx_submit.clone();
            let rx_c = runtime.rx_cancel.clone();
            let tx_e = runtime.tx_events.clone();
            thread::spawn(move || loop {
                crossbeam_channel::select! {
                    recv(rx_s) -> msg => {
                        if let Ok((id, spec)) = msg {
                            if canceled.lock().contains(&id) { continue; }
                            q.lock().push_back((id.clone(), spec.clone()));
                            let _ = tx_e.send(JobEvent { id, asset_id: spec.asset_id.clone(), kind: spec.kind, status: JobStatus::Pending });
                        }
                        else { break; }
                    }
                    recv(rx_c) -> msg => {
                        if let Ok(id) = msg { canceled.lock().insert(id); }
                        else { break; }
                    }
                    default(Duration::from_millis(10)) => {}
                }
            });
        }

        JobsHandle {
            tx_submit,
            tx_cancel,
            rx_events,
        }
    }

    fn spawn_workers(
        &self,
        n: usize,
        queue: Arc<Mutex<VecDeque<(String, JobSpec)>>>,
        db_path: Arc<PathBuf>,
    ) {
        for _ in 0..n {
            let q = queue.clone();
            let tx_e = self.tx_events.clone();
            let canceled = self.canceled.clone();
            let db_path = db_path.clone();
            thread::spawn(move || {
                let db = loop {
                    match ProjectDb::open_or_create(db_path.as_ref()) {
                        Ok(db) => break db,
                        Err(e) => {
                            eprintln!("[jobs] failed to open db at {:?}: {e}", db_path);
                            thread::sleep(Duration::from_secs(1));
                        }
                    }
                };
                loop {
                    let job_opt = {
                        let mut ql = q.lock();
                        let mut found: Option<(String, JobSpec)> = None;
                        while let Some((id, spec)) = ql.pop_front() {
                            if !canceled.lock().contains(&id) {
                                found = Some((id, spec));
                                break;
                            }
                        }
                        found
                    };
                    if let Some((id, spec)) = job_opt {
                        if canceled.lock().contains(&id) {
                            let _ = tx_e.send(JobEvent {
                                id,
                                asset_id: spec.asset_id.clone(),
                                kind: spec.kind,
                                status: JobStatus::Canceled,
                            });
                            continue;
                        }
                        let _ = tx_e.send(JobEvent {
                            id: id.clone(),
                            asset_id: spec.asset_id.clone(),
                            kind: spec.kind,
                            status: JobStatus::Running,
                        });
                        let _ = db.update_job_status(&id, "running");

                        let result = execute_job(&db, &spec);
                        match result {
                            Ok(_) => {
                                let _ = db.update_job_status(&id, "done");
                                let _ = tx_e.send(JobEvent {
                                    id,
                                    asset_id: spec.asset_id,
                                    kind: spec.kind,
                                    status: JobStatus::Done,
                                });
                            }
                            Err(e) => {
                                let _ = db.update_job_status(&id, "failed");
                                let _ = tx_e.send(JobEvent {
                                    id,
                                    asset_id: spec.asset_id,
                                    kind: spec.kind,
                                    status: JobStatus::Failed(e.to_string()),
                                });
                            }
                        }
                    } else {
                        thread::sleep(Duration::from_millis(10));
                    }
                }
            });
        }
    }
}

impl JobsHandle {
    pub fn enqueue(&self, spec: JobSpec) -> String {
        let id = Uuid::new_v4().to_string();
        let _ = self.tx_submit.send((id.clone(), spec));
        id
    }

    pub fn cancel_by_asset(&self, _asset_id: &str) {
        // stub: a real impl would track and cancel
    }

    pub fn cancel_job(&self, job_id: &str) {
        let _ = self.tx_cancel.send(job_id.to_string());
    }
}

fn job_spec_from_row(row: &JobRow) -> Option<JobSpec> {
    let kind = parse_job_kind(&row.kind)?;
    Some(JobSpec {
        asset_id: row.asset_id.clone(),
        kind,
        priority: row.priority,
    })
}

fn parse_job_kind(kind: &str) -> Option<JobKind> {
    match kind {
        "waveform" => Some(JobKind::Waveform),
        "thumbs" | "thumbnail" => Some(JobKind::Thumbnails),
        "proxy" => Some(JobKind::Proxy),
        "seekidx" | "seek_index" => Some(JobKind::SeekIndex),
        _ => None,
    }
}

fn execute_job(db: &ProjectDb, spec: &JobSpec) -> Result<()> {
    let asset = db.get_asset(&spec.asset_id).context("load asset for job")?;
    let source_path = Path::new(&asset.src_abs);
    let cache_root = app_data_dir().join("cache");
    fs::create_dir_all(&cache_root)?;

    match spec.kind {
        JobKind::Waveform => {
            let waveform_dir = cache_root.join("waveforms");
            fs::create_dir_all(&waveform_dir)?;
            let out = waveform_dir.join(format!("{}-wf.bin", asset.id));
            let data = generate_waveform(source_path, 2048).context("generate waveform")?;
            let mut file = fs::File::create(&out)?;
            for sample in data {
                file.write_all(&sample.to_le_bytes())?;
            }
            db.update_asset_analysis(&asset.id, Some(out.as_path()), None, None, None)?;
        }
        JobKind::Thumbnails => {
            if !asset.kind.eq_ignore_ascii_case("video")
                && !asset.kind.eq_ignore_ascii_case("image")
            {
                return Err(anyhow!("thumbnail job requires video or image asset"));
            }
            let thumb_dir = cache_root.join("thumbnails");
            fs::create_dir_all(&thumb_dir)?;
            let out = thumb_dir.join(format!("{}-thumb.jpg", asset.id));
            let (width, height) = choose_thumb_dimensions(&asset);
            let capture_sec = choose_capture_time(&asset);
            generate_thumbnail(source_path, &out, capture_sec, width, height)
                .context("generate thumbnail")?;
            db.update_asset_analysis(&asset.id, None, Some(out.as_path()), None, None)?;
        }
        JobKind::Proxy => {
            if !asset.kind.eq_ignore_ascii_case("video") {
                return Err(anyhow!("proxy job requires video asset"));
            }
            let proxy_dir = cache_root.join("proxies");
            fs::create_dir_all(&proxy_dir)?;
            let out = proxy_dir.join(format!("{}-proxy.mp4", asset.id));
            let (width, height) = choose_proxy_dimensions(&asset);
            generate_proxy(source_path, &out, width, height, 6_000).context("generate proxy")?;
            db.update_asset_analysis(&asset.id, None, None, Some(out.as_path()), None)?;
        }
        JobKind::SeekIndex => {
            let idx_dir = cache_root.join("seek_index");
            fs::create_dir_all(&idx_dir)?;
            let out = idx_dir.join(format!("{}-seek.json", asset.id));
            fs::write(&out, b"{}")?;
            db.update_asset_analysis(&asset.id, None, None, None, Some(out.as_path()))?;
        }
    }

    Ok(())
}

fn choose_thumb_dimensions(asset: &project::AssetRow) -> (u32, u32) {
    let width = asset.width.unwrap_or(1280).max(1) as u32;
    let height = asset.height.unwrap_or(720).max(1) as u32;
    let max_dim = 512u32;
    if width <= max_dim && height <= max_dim {
        (width, height)
    } else {
        let scale = (max_dim as f32 / width as f32).min(max_dim as f32 / height as f32);
        let w = (width as f32 * scale).round().max(1.0) as u32;
        let h = (height as f32 * scale).round().max(1.0) as u32;
        (w, h)
    }
}

fn choose_proxy_dimensions(asset: &project::AssetRow) -> (u32, u32) {
    let width = asset.width.unwrap_or(1920).max(1) as u32;
    let height = asset.height.unwrap_or(1080).max(1) as u32;
    let target = 960u32;
    if width <= target && height <= target {
        (width, height)
    } else {
        let scale = (target as f32 / width as f32).min(target as f32 / height as f32);
        let w = (width as f32 * scale).round().max(1.0) as u32;
        let h = (height as f32 * scale).round().max(1.0) as u32;
        (w, h)
    }
}

fn choose_capture_time(asset: &project::AssetRow) -> f64 {
    let fps = match (asset.fps_num, asset.fps_den) {
        (Some(n), Some(d)) if d != 0 => Some(n as f64 / d as f64),
        _ => None,
    };
    if let (Some(frames), Some(fps)) = (asset.duration_frames, fps) {
        if fps > 0.0 {
            return (frames as f64 / fps) * 0.5;
        }
    }
    0.0
}
