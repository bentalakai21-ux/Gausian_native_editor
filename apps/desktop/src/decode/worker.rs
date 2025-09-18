use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use crossbeam_channel as channel;
use native_decoder::{create_decoder, DecoderConfig, YuvPixFmt as NativeYuvPixFmt};

use media_io::YuvPixFmt;

pub(crate) const PREFETCH_BUDGET_PER_TICK: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlayState {
    Paused,
    Seeking,
    Playing,
    Scrubbing,
}

pub(crate) struct EngineState {
    pub(crate) state: PlayState,
    pub(crate) rate: f32, // 1.0 by default
    pub(crate) target_pts: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VideoProps {
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) fps: f64,
    pub(crate) fmt: YuvPixFmt,
}

#[derive(Clone)]
pub(crate) enum FramePayload {
    Cpu { y: Arc<[u8]>, uv: Arc<[u8]> },
}

#[derive(Clone)]
pub(crate) struct VideoFrameOut {
    pub(crate) pts: f64,
    pub(crate) props: VideoProps,
    pub(crate) payload: FramePayload,
}

pub(crate) enum DecodeCmd {
    Play { start_pts: f64, rate: f32 },
    Seek { target_pts: f64 },
    Pause,
    Stop,
}

pub(crate) struct LatestFrameSlot(pub(crate) Arc<Mutex<Option<VideoFrameOut>>>);

pub(crate) struct DecodeWorkerRuntime {
    #[allow(dead_code)]
    pub(crate) handle: thread::JoinHandle<()>,
    pub(crate) cmd_tx: channel::Sender<DecodeCmd>,
    pub(crate) slot: LatestFrameSlot,
}

pub(crate) fn spawn_worker(path: &str) -> DecodeWorkerRuntime {
    use channel::{unbounded, Receiver, Sender};
    let (cmd_tx, cmd_rx) = unbounded::<DecodeCmd>();
    let slot = LatestFrameSlot(Arc::new(Mutex::new(None)));
    let slot_for_worker = LatestFrameSlot(slot.0.clone());
    let path = path.to_string();
    let handle = thread::spawn(move || {
        // Initialize decoders
        let cfg_cpu = DecoderConfig {
            hardware_acceleration: true,
            preferred_format: Some(NativeYuvPixFmt::Nv12),
            zero_copy: false,
        };
        let mut cpu_dec = match create_decoder(&path, cfg_cpu) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[worker] create_decoder CPU failed: {e}");
                return;
            }
        };
        // For now, worker outputs CPU NV12/P010 frames only (zero-copy can be added later)

        let props = cpu_dec.get_properties();
        let fps = if props.frame_rate > 0.0 {
            props.frame_rate
        } else {
            30.0
        };
        let frame_dur = if fps > 0.0 { 1.0 / fps } else { 1.0 / 30.0 };

        let mut mode = PlayState::Paused;
        let mut rate: f32 = 1.0;
        let mut anchor_pts: f64 = 0.0;
        let mut anchor_t = Instant::now();
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
                let fmt = match vf.format {
                    NativeYuvPixFmt::Nv12 => YuvPixFmt::Nv12,
                    NativeYuvPixFmt::P010 => YuvPixFmt::P010,
                };
                let y: Arc<[u8]> = Arc::from(vf.y_plane.into_boxed_slice());
                let uv: Arc<[u8]> = Arc::from(vf.uv_plane.into_boxed_slice());
                return Some(VideoFrameOut {
                    pts: vf.timestamp,
                    props: VideoProps {
                        w: vf.width,
                        h: vf.height,
                        fps,
                        fmt,
                    },
                    payload: FramePayload::Cpu { y, uv },
                });
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
                            anchor_t = Instant::now();
                        }
                        rate = r;
                    }
                    DecodeCmd::Seek { target_pts } => {
                        mode = PlayState::Seeking;
                        anchor_pts = target_pts;
                    }
                    DecodeCmd::Pause => {
                        mode = PlayState::Paused;
                    }
                    DecodeCmd::Stop => {
                        running = false;
                    }
                }
            }

            match mode {
                PlayState::Playing => {
                    let dt = anchor_t.elapsed().as_secs_f64();
                    let target = anchor_pts + dt * (rate as f64);
                    if let Some(out) = attempt_decode(target) {
                        eprintln!("[WORKER] out pts={:.3}", out.pts);
                        if let Ok(mut g) = slot_for_worker.0.lock() {
                            *g = Some(out);
                        }
                    }
                    thread::sleep(std::time::Duration::from_millis(4));
                }
                PlayState::Seeking | PlayState::Scrubbing => {
                    let target = anchor_pts;
                    if let Some(out) = attempt_decode(target) {
                        eprintln!("[WORKER] out pts={:.3}", out.pts);
                        if let Ok(mut g) = slot_for_worker.0.lock() {
                            *g = Some(out);
                        }
                    }
                    thread::sleep(std::time::Duration::from_millis(4));
                }
                PlayState::Paused => {
                    thread::sleep(std::time::Duration::from_millis(6));
                }
            }
        }
    });

    DecodeWorkerRuntime {
        handle,
        cmd_tx,
        slot,
    }
}
