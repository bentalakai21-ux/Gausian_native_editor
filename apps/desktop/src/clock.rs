use std::time::Instant;

#[derive(Default, Debug, Clone)]
pub(crate) struct PlaybackClock {
    pub(crate) playing: bool,
    pub(crate) rate: f64, // 1.0 = normal
    pub(crate) anchor_instant: Option<Instant>,
    pub(crate) anchor_timeline_sec: f64, // timeline time at anchor
}

impl PlaybackClock {
    pub(crate) fn play(&mut self, current_timeline_sec: f64) {
        self.playing = true;
        self.anchor_timeline_sec = current_timeline_sec;
        self.anchor_instant = Some(Instant::now());
    }
    pub(crate) fn pause(&mut self, current_timeline_sec: f64) {
        self.playing = false;
        self.anchor_timeline_sec = current_timeline_sec;
        self.anchor_instant = None;
    }
    pub(crate) fn set_rate(&mut self, rate: f64, current_timeline_sec: f64) {
        // re-anchor to avoid jumps
        self.anchor_timeline_sec = current_timeline_sec;
        self.anchor_instant = Some(Instant::now());
        self.rate = rate;
    }
    pub(crate) fn now(&self) -> f64 {
        if self.playing {
            let dt = self.anchor_instant.unwrap().elapsed().as_secs_f64();
            self.anchor_timeline_sec + dt * self.rate
        } else {
            self.anchor_timeline_sec
        }
    }
    pub(crate) fn seek_to(&mut self, timeline_sec: f64) {
        self.anchor_timeline_sec = timeline_sec;
        if self.playing {
            self.anchor_instant = Some(Instant::now());
        }
    }
}
