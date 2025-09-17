use std::collections::VecDeque;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use cpal::traits::{HostTrait, DeviceTrait, StreamTrait};

pub struct AudioState {
    device_sr: Option<u32>,
    channels: u16,
    stream: Option<cpal::Stream>,
    samples: Arc<Mutex<VecDeque<i16>>>,
    worker: Option<JoinHandle<()>>,
    stop: Option<Arc<AtomicBool>>,
    current: Option<String>,
    current_t: Option<f64>,
    last_seek_at: Option<Instant>,
}

impl AudioState {
    pub fn new() -> Self {
        Self { device_sr: None, channels: 2, stream: None, samples: Arc::new(Mutex::new(VecDeque::with_capacity(48000*4))), worker: None, stop: None, current: None, current_t: None, last_seek_at: None }
    }

    pub fn ensure_playing(&mut self, src_path: Option<&str>, t_sec: f64) {
        if src_path.is_none() { self.stop(); return; }
        let path = src_path.unwrap();
        if self.current.as_deref() != Some(path) || self.stream.is_none() {
            self.stop();
            self.start_output_stream();
            self.start_ffmpeg_reader(path, t_sec);
            self.current = Some(path.to_string());
            self.current_t = Some(t_sec);
            self.last_seek_at = Some(Instant::now());
        }
    }

    /// Preview audio while scrubbing when paused.
    /// Re-seeks at a throttled interval to avoid spawning too often.
    pub fn preview_scrub(&mut self, src_path: Option<&str>, t_sec: f64) {
        if src_path.is_none() { self.stop(); return; }
        let path = src_path.unwrap();
        if self.stream.is_none() {
            self.start_output_stream();
        }
        // Decide whether to (re)start the reader
        let path_changed = self.current.as_deref() != Some(path);
        let need_seek = match self.current_t {
            Some(ct) => (ct - t_sec).abs() > 0.08,
            None => true,
        };
        let throttled = match self.last_seek_at {
            Some(t) => t.elapsed() >= Duration::from_millis(80),
            None => true,
        };
        if path_changed || (need_seek && throttled) || self.worker.is_none() {
            // Restart just the reader thread; keep the output stream
            self.stop_reader_only();
            self.start_ffmpeg_reader(path, t_sec);
            self.current = Some(path.to_string());
            self.current_t = Some(t_sec);
            self.last_seek_at = Some(Instant::now());
        }
    }

    fn start_output_stream(&mut self) {
        let host = cpal::default_host();
        let device = match host.default_output_device() { Some(d) => d, None => return };
        let supported = match device.default_output_config() { Ok(c) => c, Err(_) => return };
        let sr = supported.sample_rate().0;
        self.device_sr = Some(sr);
        let channels = supported.channels();
        self.channels = channels;

        let samples = self.samples.clone();
        let err_fn = |e| eprintln!("cpal stream error: {e}");
        let stream = match supported.sample_format() {
            cpal::SampleFormat::I16 => device.build_output_stream(&supported.config(), move |out: &mut [i16], _| fill_audio_i16(out, channels, &samples), err_fn, None),
            cpal::SampleFormat::U16 => device.build_output_stream(&supported.config(), move |out: &mut [u16], _| fill_audio_u16(out, channels, &samples), err_fn, None),
            cpal::SampleFormat::F32 => device.build_output_stream(&supported.config(), move |out: &mut [f32], _| fill_audio_f32(out, channels, &samples), err_fn, None),
            _ => return,
        };
        if let Ok(stream) = stream { let _ = stream.play(); self.stream = Some(stream); }
    }

    fn start_ffmpeg_reader(&mut self, path: &str, t_sec: f64) {
        let sr = self.device_sr.unwrap_or(48000);
        let ch = self.channels;
        let stop = Arc::new(AtomicBool::new(false));
        let stop_c = stop.clone();
        let out = self.samples.clone();
        let path = path.to_string();
        self.worker = Some(thread::spawn(move || {
            let mut cmd = std::process::Command::new("ffmpeg");
            cmd.arg("-ss").arg(format!("{:.3}", t_sec.max(0.0)))
               .arg("-i").arg(&path)
               .arg("-vn")
               .arg("-ac").arg(format!("{ch}"))
               .arg("-ar").arg(format!("{sr}"))
               .arg("-f").arg("s16le")
               .arg("-")
               .stdin(std::process::Stdio::null())
               .stdout(std::process::Stdio::piped())
               .stderr(std::process::Stdio::null());
            let mut child = match cmd.spawn() { Ok(c) => c, Err(_) => return };
            let mut stdout = child.stdout.take().unwrap();
            let mut buf = [0u8; 4096];
            while !stop_c.load(Ordering::Relaxed) {
                match std::io::Read::read(&mut stdout, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let mut q = out.lock().unwrap();
                        for chunk in buf[..n].chunks_exact(2) {
                            let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                            q.push_back(s);
                        }
                        let max = (sr as usize) * 4; // ~ 4 seconds buffer
                        let len = q.len();
                        if len > max { q.drain(..len-max); }
                    }
                    Err(_) => break,
                }
            }
            let _ = child.kill();
        }));
        self.stop = Some(stop);
    }

    pub fn stop(&mut self) {
        if let Some(s) = &self.stop { s.store(true, Ordering::Relaxed); }
        if let Some(h) = self.worker.take() { let _ = h.join(); }
        self.stop = None;
        self.current = None;
        self.current_t = None;
        self.last_seek_at = None;
    }

    fn stop_reader_only(&mut self) {
        if let Some(s) = &self.stop { s.store(true, Ordering::Relaxed); }
        if let Some(h) = self.worker.take() { let _ = h.join(); }
        self.stop = None;
    }
}

fn fill_audio_i16(out: &mut [i16], _channels: u16, buf: &Arc<Mutex<VecDeque<i16>>>) {
    let mut q = buf.lock().unwrap();
    for s in out.iter_mut() { *s = q.pop_front().unwrap_or(0); }
}
fn fill_audio_u16(out: &mut [u16], _channels: u16, buf: &Arc<Mutex<VecDeque<i16>>>) {
    let mut q = buf.lock().unwrap();
    for s in out.iter_mut() { let v: i16 = q.pop_front().unwrap_or(0); *s = (v as i32 + 32768) as u16; }
}
fn fill_audio_f32(out: &mut [f32], _channels: u16, buf: &Arc<Mutex<VecDeque<i16>>>) {
    let mut q = buf.lock().unwrap();
    for s in out.iter_mut() { let v: i16 = q.pop_front().unwrap_or(0); *s = (v as f32) / 32768.0; }
}
