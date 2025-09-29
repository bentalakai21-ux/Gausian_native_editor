use super::App;

pub(super) fn ensure_baseline_tracks(app: &mut App) {
    if app.seq.graph.tracks.is_empty() {
        for i in 1..=3 {
            let binding = timeline_crate::TrackBinding {
                id: timeline_crate::TrackId::new(),
                name: format!("V{}", i),
                kind: timeline_crate::TrackKind::Video,
                node_ids: Vec::new(),
            };
            let _ = app.apply_timeline_command(timeline_crate::TimelineCommand::UpsertTrack {
                track: binding,
            });
        }
        for i in 1..=3 {
            let binding = timeline_crate::TrackBinding {
                id: timeline_crate::TrackId::new(),
                name: format!("A{}", i),
                kind: timeline_crate::TrackKind::Audio,
                node_ids: Vec::new(),
            };
            let _ = app.apply_timeline_command(timeline_crate::TimelineCommand::UpsertTrack {
                track: binding,
            });
        }
        app.sync_tracks_from_graph();
    }
}

pub(super) fn load_project_timeline(app: &mut App) {
    if let Ok(Some(json)) = app.db.get_project_timeline_json(&app.project_id) {
        if let Ok(seq) = serde_json::from_str::<timeline_crate::Sequence>(&json) {
            app.seq = seq;
        } else {
            let mut seq = timeline_crate::Sequence::new(
                "Main",
                1920,
                1080,
                timeline_crate::Fps::new(30, 1),
                600,
            );
            for i in 1..=3 {
                seq.add_track(timeline_crate::Track {
                    name: format!("V{}", i),
                    items: vec![],
                });
            }
            for i in 1..=3 {
                seq.add_track(timeline_crate::Track {
                    name: format!("A{}", i),
                    items: vec![],
                });
            }
            app.seq = seq;
        }
    } else {
        let mut seq = timeline_crate::Sequence::new(
            "Main",
            1920,
            1080,
            timeline_crate::Fps::new(30, 1),
            600,
        );
        for i in 1..=3 {
            seq.add_track(timeline_crate::Track {
                name: format!("V{}", i),
                items: vec![],
            });
        }
        for i in 1..=3 {
            seq.add_track(timeline_crate::Track {
                name: format!("A{}", i),
                items: vec![],
            });
        }
        app.seq = seq;
    }
    if app.seq.graph.tracks.is_empty() {
        app.seq.graph = timeline_crate::migrate_sequence_tracks(&app.seq);
    }
    super::app_timeline::sync_tracks_from_graph_impl(app);
    ensure_baseline_tracks(app);
    app.timeline_history = timeline_crate::CommandHistory::default();
    app.selected = None;
    app.drag = None;
}

pub(super) fn save_project_timeline_impl(app: &mut App) -> anyhow::Result<()> {
    let json = serde_json::to_string(&app.seq)?;
    app.db
        .upsert_project_timeline_json(&app.project_id, &json)?;
    app.last_save_at = Some(std::time::Instant::now());
    Ok(())
}

pub(super) fn save_project_timeline(app: &mut App) -> anyhow::Result<()> {
    save_project_timeline_impl(app)
}

// Thin App method wrappers to keep app.rs small
impl App {
    pub(crate) fn ensure_baseline_tracks(&mut self) {
        self::ensure_baseline_tracks(self)
    }

    pub(crate) fn load_project_timeline(&mut self) {
        self::load_project_timeline(self)
    }

    pub(crate) fn save_project_timeline(&mut self) -> anyhow::Result<()> {
        self::save_project_timeline(self)
    }
}
