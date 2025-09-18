use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct AudioBuffer {
    pub samples: Vec<f32>, // interleaved
    pub channels: u16,
    pub sample_rate: u32,
    pub duration_sec: f32,
}

#[derive(Clone)]
pub struct ActiveAudioClip {
    pub start_tl_sec: f64, // timeline position
    pub start_media_sec: f64,
    pub duration_sec: f64,
    pub buf: Arc<AudioBuffer>,
}

struct Mixer {
    device_sr: u32,
    playing: bool,
    // anchor: timeline time when device_frame_cursor==0
    anchor_timeline_sec: f64,
    device_frame_cursor: u64,
    clips: Vec<ActiveAudioClip>,
}

pub struct AudioEngine {
    stream: Option<cpal::Stream>,
    mixer: Arc<Mutex<Mixer>>,
}

impl AudioEngine {
    pub fn new() -> anyhow::Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("no audio device"))?;
        let mut config = device.default_output_config()?.config();
        // Force stereo f32 if needed:
        config.channels = 2;
        let device_sr = config.sample_rate.0;

        let mixer = Arc::new(Mutex::new(Mixer {
            device_sr,
            playing: false,
            anchor_timeline_sec: 0.0,
            device_frame_cursor: 0,
            clips: Vec::new(),
        }));

        let mix_clone = mixer.clone();
        let stream = device.build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                let mut m = mix_clone.lock().unwrap();
                let ch = 2usize;
                let frames = data.len() / ch;
                for i in 0..frames {
                    let t = m.anchor_timeline_sec
                        + (m.device_frame_cursor as f64) / (m.device_sr as f64);
                    let mut l = 0.0f32;
                    let mut r = 0.0f32;
                    if m.playing {
                        for c in &m.clips {
                            if t >= c.start_tl_sec && t < c.start_tl_sec + (c.duration_sec as f64) {
                                let ct = (t - c.start_tl_sec + c.start_media_sec) as f32;
                                let (sl, sr) = sample_stereo(&c.buf, ct, m.device_sr);
                                l += sl;
                                r += sr;
                            }
                        }
                        m.device_frame_cursor += 1;
                    }
                    let idx = i * ch;
                    data[idx] = l;
                    data[idx + 1] = r;
                }
            },
            move |err| eprintln!("audio error: {err:?}"),
            None,
        )?;
        stream.play()?;

        Ok(Self {
            stream: Some(stream),
            mixer,
        })
    }

    // Called when pressing play:
    pub fn start(&self, timeline_now_sec: f64, active: Vec<ActiveAudioClip>) {
        let mut m = self.mixer.lock().unwrap();
        m.clips = active;
        m.anchor_timeline_sec = timeline_now_sec;
        m.device_frame_cursor = 0;
        m.playing = true;
    }
    pub fn pause(&self, timeline_now_sec: f64) {
        let mut m = self.mixer.lock().unwrap();
        m.anchor_timeline_sec = timeline_now_sec;
        m.device_frame_cursor = 0;
        m.playing = false;
    }
    pub fn seek(&self, timeline_now_sec: f64) {
        let mut m = self.mixer.lock().unwrap();
        m.anchor_timeline_sec = timeline_now_sec;
        m.device_frame_cursor = 0;
    }
}

fn sample_stereo(buf: &AudioBuffer, t_sec: f32, _out_sr: u32) -> (f32, f32) {
    if buf.samples.is_empty() {
        return (0.0, 0.0);
    }
    // Assume buffer at buf.sample_rate; simple linear interpolation
    let in_sr = buf.sample_rate;
    let t_in = t_sec * in_sr as f32; // time in input frames (float)
    let i0 = (t_in.floor() as i64).max(0) as usize;
    let i1 = i0.saturating_add(1);
    let frac = t_in - (i0 as f32);
    let ch = buf.channels as usize;

    let fetch = |frame: usize, c: usize| -> f32 {
        let f = frame.min((buf.samples.len() / ch).saturating_sub(1));
        buf.samples[f * ch + c]
    };
    let l0 = fetch(i0, 0);
    let r0 = fetch(i0, if ch > 1 { 1 } else { 0 });
    let l1 = fetch(i1, 0);
    let r1 = fetch(i1, if ch > 1 { 1 } else { 0 });
    let l = l0 + (l1 - l0) * frac;
    let r = r0 + (r1 - r0) * frac;
    (l, r)
}
