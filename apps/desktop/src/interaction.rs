use timeline::{NodeId, TimelineNode, TrackId};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DragMode {
    Move,
    TrimStart,
    TrimEnd,
}

#[derive(Clone, Debug)]
pub struct DragState {
    pub original_track_index: usize,
    pub current_track_index: usize,
    pub mode: DragMode,
    pub start_mouse_x: f32,
    pub orig_from: i64,
    pub orig_dur: i64,
    pub node_id: NodeId,
    pub original_node: TimelineNode,
    pub original_track_id: TrackId,
    pub original_position: usize,
    pub asset_id: Option<String>,
    pub linked: Vec<LinkedDragNode>,
}

#[derive(Clone, Debug)]
pub struct LinkedDragNode {
    pub node_id: NodeId,
    pub original_node: TimelineNode,
    pub original_track_id: TrackId,
    pub original_track_index: usize,
    pub current_track_index: usize,
    pub original_position: usize,
    pub orig_from: i64,
    pub orig_dur: i64,
    pub orig_media_start: i64,
}
