use std::{fs::File, io, path::Path};

use anyhow::{anyhow, Context, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use symphonia::core::{
    audio::{SampleBuffer, Signal},
    codecs::DecoderOptions,
    errors::Error,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};
use tracing::debug;

use crate::audio_engine::AudioBuffer;

pub fn decode_audio_to_buffer(path: &Path) -> Result<AudioBuffer> {
    match decode_with_gstreamer(path) {
        Ok(buf) => Ok(buf),
        Err(err) => {
            debug!(
                "GStreamer audio decode failed for {:?}: {err:?}; falling back to Symphonia",
                path
            );
            decode_with_symphonia(path)
        }
    }
}

fn decode_with_gstreamer(path: &Path) -> Result<AudioBuffer> {
    ensure_gst_init()?;

    let pipeline = gst::Pipeline::new();
    let _guard = PipelineGuard(pipeline.clone());

    let src = gst::ElementFactory::make("filesrc")
        .property("location", &path.to_string_lossy().to_string())
        .build()
        .context("make filesrc")?;
    let decodebin = gst::ElementFactory::make("decodebin")
        .build()
        .context("make decodebin")?;
    let convert = gst::ElementFactory::make("audioconvert")
        .build()
        .context("make audioconvert")?;
    let resample = gst::ElementFactory::make("audioresample")
        .build()
        .context("make audioresample")?;
    let caps = gst::Caps::builder("audio/x-raw")
        .field("format", &"F32LE")
        .field("layout", &"interleaved")
        .build();
    let capsfilter = gst::ElementFactory::make("capsfilter")
        .property("caps", &caps)
        .build()
        .context("make capsfilter")?;
    let appsink = gst_app::AppSink::builder()
        .caps(&caps)
        .max_buffers(64)
        .drop(false)
        .build();

    pipeline
        .add_many(&[
            &src,
            &decodebin,
            &convert,
            &resample,
            &capsfilter,
            appsink.upcast_ref(),
        ])
        .context("add pipeline elements")?;

    gst::Element::link_many(&[&convert, &resample, &capsfilter, appsink.upcast_ref()])
        .context("link convert->resample->capsfilter->appsink")?;
    src.link(&decodebin).context("link filesrc->decodebin")?;

    let convert_weak = convert.downgrade();
    decodebin.connect_pad_added(move |_dbin, src_pad| {
        let Some(convert) = convert_weak.upgrade() else {
            return;
        };
        let Some(sink_pad) = convert.static_pad("sink") else {
            return;
        };
        if sink_pad.is_linked() {
            return;
        }
        let _ = src_pad.link(&sink_pad);
    });

    let bus = pipeline
        .bus()
        .ok_or_else(|| anyhow!("GStreamer pipeline has no bus"))?;

    let elem: &gst::Element = appsink.upcast_ref();
    let _ = elem.set_property("sync", &false);

    pipeline
        .set_state(gst::State::Playing)
        .map_err(|e| anyhow!("set PLAYING: {e}"))?;

    let mut samples = Vec::new();
    let mut channels: Option<u16> = None;
    let mut sample_rate: Option<u32> = None;
    let timeout = gst::ClockTime::from_mseconds(50);

    loop {
        if let Some(sample) = appsink.try_pull_sample(Some(timeout)) {
            if channels.is_none() || sample_rate.is_none() {
                if let Some(caps) = sample.caps() {
                    if let Some(structure) = caps.structure(0) {
                        if channels.is_none() {
                            if let Ok(ch) = structure.get::<i32>("channels") {
                                channels = Some(ch.max(1) as u16);
                            }
                        }
                        if sample_rate.is_none() {
                            if let Ok(rate) = structure.get::<i32>("rate") {
                                sample_rate = Some(rate.max(1) as u32);
                            }
                        }
                    }
                }
            }

            let buffer = sample
                .buffer()
                .ok_or_else(|| anyhow!("appsink sample without buffer"))?;
            let map = buffer.map_readable().map_err(|_| anyhow!("map buffer"))?;
            let data = map.as_slice();
            if data.len() % 4 != 0 {
                return Err(anyhow!("unexpected audio buffer size: {}", data.len()));
            }
            for chunk in data.chunks_exact(4) {
                let mut bytes = [0u8; 4];
                bytes.copy_from_slice(chunk);
                samples.push(f32::from_le_bytes(bytes));
            }
            continue;
        }

        if let Some(msg) = bus.timed_pop_filtered(
            Some(timeout),
            &[gst::MessageType::Error, gst::MessageType::Eos],
        ) {
            use gst::MessageView;
            match msg.view() {
                MessageView::Error(err) => {
                    return Err(anyhow!(
                        "GStreamer error from {}: {} ({:?})",
                        err.src()
                            .map(|s| s.path_string())
                            .unwrap_or_else(|| "unknown".into()),
                        err.error(),
                        err.debug()
                    ));
                }
                MessageView::Eos(_) => break,
                _ => {}
            }
        }
    }

    let ch = channels.ok_or_else(|| anyhow!("missing audio channel count"))?;
    let sr = sample_rate.ok_or_else(|| anyhow!("missing audio sample rate"))?;
    if samples.is_empty() {
        return Err(anyhow!("no audio samples produced"));
    }
    let frames = samples.len() as f32 / ch as f32;
    let duration_sec = if sr > 0 { frames / sr as f32 } else { 0.0 };
    debug!(
        "gst audio decoded: path={} channels={} sample_rate={} seconds={:.3} samples={}",
        path.display(),
        ch,
        sr,
        duration_sec,
        samples.len()
    );

    Ok(AudioBuffer {
        samples,
        channels: ch,
        sample_rate: sr,
        duration_sec,
    })
}

fn ensure_gst_init() -> Result<()> {
    use std::sync::OnceLock;

    static GST_INIT: OnceLock<Result<(), String>> = OnceLock::new();
    match GST_INIT.get_or_init(|| {
        gst::init()
            .map_err(|e| format!("gst::init() failed: {e}"))
            .map(|_| ())
    }) {
        Ok(()) => Ok(()),
        Err(err) => Err(anyhow!(err.clone())),
    }
}

struct PipelineGuard(gst::Pipeline);

impl Drop for PipelineGuard {
    fn drop(&mut self) {
        let _ = self.0.set_state(gst::State::Null);
    }
}

fn decode_with_symphonia(path: &Path) -> Result<AudioBuffer> {
    let file = File::open(path).with_context(|| format!("open audio file {:?}", path))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| anyhow!(e))?;
    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("no default audio track"))?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();
    let channels = codec_params
        .channels
        .ok_or_else(|| anyhow!("audio track missing channel info"))?;
    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("audio track missing sample rate"))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| anyhow!(e))?;

    let mut samples: Vec<f32> = Vec::new();

    loop {
        match format.next_packet() {
            Ok(packet) => {
                if packet.track_id() != track_id {
                    continue;
                }
                let decoded = match decoder.decode(&packet) {
                    Ok(buf) => buf,
                    Err(Error::DecodeError(_)) => continue,
                    Err(err) => return Err(anyhow!(err)),
                };
                let mut sample_buf =
                    SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                sample_buf.copy_interleaved_ref(decoded);
                samples.extend_from_slice(sample_buf.samples());
            }
            Err(Error::IoError(err)) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(Error::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(Error::DecodeError(_)) => continue,
            Err(err) => return Err(anyhow!(err)),
        }
    }

    let channel_count = channels.count().max(1) as u16;
    let total_frames = samples.len() as f32 / channel_count as f32;
    let duration_sec = total_frames / sample_rate as f32;

    debug!(
        "symphonia audio decoded: path={} channels={} sample_rate={} seconds={:.3} samples={}",
        path.display(),
        channel_count,
        sample_rate,
        duration_sec,
        samples.len()
    );

    Ok(AudioBuffer {
        samples,
        channels: channel_count,
        sample_rate,
        duration_sec,
    })
}
