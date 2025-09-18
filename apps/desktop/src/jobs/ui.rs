use std::collections::BTreeMap;

use egui::{self};
use egui_extras::TableBuilder;

use crate::jobs_crate::{JobEvent, JobStatus};
use crate::App;

impl App {
    pub(crate) fn poll_jobs(&mut self) {
        if let Some(j) = &self.jobs {
            while let Ok(ev) = j.rx_events.try_recv() {
                let status_str = match &ev.status {
                    JobStatus::Pending => "pending",
                    JobStatus::Running => "running",
                    JobStatus::Progress(_) => "progress",
                    JobStatus::Done => "done",
                    JobStatus::Failed(_) => "failed",
                    JobStatus::Canceled => "canceled",
                };
                let _ = self.db.update_job_status(&ev.id, status_str);
                self.job_events.push(ev);
                if self.job_events.len() > 300 {
                    self.job_events.remove(0);
                }
            }
        }
    }

    pub(crate) fn jobs_window(&mut self, ctx: &egui::Context) {
        if !self.show_jobs {
            return;
        }

        egui::Window::new("Jobs")
            .open(&mut self.show_jobs)
            .resizable(true)
            .show(ctx, |ui| {
                ui.label("Background Jobs");
                let mut latest: BTreeMap<String, JobEvent> = BTreeMap::new();
                for ev in &self.job_events {
                    latest.insert(ev.id.clone(), ev.clone());
                }
                TableBuilder::new(ui)
                    .striped(true)
                    .column(egui_extras::Column::auto())
                    .column(egui_extras::Column::auto())
                    .column(egui_extras::Column::auto())
                    .column(egui_extras::Column::remainder())
                    .header(18.0, |mut h| {
                        h.col(|ui| {
                            ui.strong("Job");
                        });
                        h.col(|ui| {
                            ui.strong("Asset");
                        });
                        h.col(|ui| {
                            ui.strong("Kind");
                        });
                        h.col(|ui| {
                            ui.strong("Status");
                        });
                    })
                    .body(|mut b| {
                        for (_id, ev) in latest.iter() {
                            b.row(20.0, |mut r| {
                                r.col(|ui| {
                                    ui.monospace(&ev.id[..8.min(ev.id.len())]);
                                });
                                r.col(|ui| {
                                    ui.monospace(&ev.asset_id[..8.min(ev.asset_id.len())]);
                                });
                                r.col(|ui| {
                                    ui.label(format!("{:?}", ev.kind));
                                });
                                r.col(|ui| {
                                    match &ev.status {
                                        JobStatus::Progress(p) => {
                                            ui.add(egui::ProgressBar::new(*p).show_percentage());
                                        }
                                        status => {
                                            ui.label(format!("{:?}", status));
                                        }
                                    }
                                    if !matches!(
                                        ev.status,
                                        JobStatus::Done
                                            | JobStatus::Failed(_)
                                            | JobStatus::Canceled
                                    ) {
                                        if ui.small_button("Cancel").clicked() {
                                            if let Some(j) = &self.jobs {
                                                j.cancel_job(&ev.id);
                                            }
                                        }
                                    }
                                });
                            });
                        }
                    });
            });
    }
}
