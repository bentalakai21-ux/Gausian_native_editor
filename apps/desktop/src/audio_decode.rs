use std::{fs::File, io, path::Path};

use anyhow::{anyhow, Context, Result};
use symphonia::core::{
    audio::{SampleBuffer, Signal},
    codecs::DecoderOptions,
    errors::Error,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

use crate::audio_engine::AudioBuffer;

pub fn decode_audio_to_buffer(path: &Path) -> Result<AudioBuffer> {
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

    Ok(AudioBuffer {
        samples,
        channels: channel_count,
        sample_rate,
        duration_sec,
    })
}
