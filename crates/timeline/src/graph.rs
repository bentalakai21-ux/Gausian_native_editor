use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt};
use uuid::Uuid;

use crate::Frame;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(transparent)]
pub struct NodeId(pub Uuid);

impl NodeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(transparent)]
pub struct TrackId(pub Uuid);

impl TrackId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for TrackId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(transparent)]
pub struct LaneId(pub Uuid);

impl LaneId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for LaneId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FrameRange {
    pub start: Frame,
    pub duration: Frame,
}

impl FrameRange {
    pub fn new(start: Frame, duration: Frame) -> Self {
        Self { start, duration }
    }
    pub fn end(&self) -> Frame {
        self.start + self.duration
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClipNode {
    pub asset_id: Option<String>,
    pub media_range: FrameRange,
    pub timeline_range: FrameRange,
    #[serde(default = "default_playback_rate")]
    pub playback_rate: f32,
    #[serde(default)]
    pub reverse: bool,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

fn default_playback_rate() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransitionNode {
    pub duration: Frame,
    #[serde(default)]
    pub kind: TransitionKind,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TransitionKind {
    Dissolve,
    Wipe,
    Slide,
    Custom(String),
}

impl Default for TransitionKind {
    fn default() -> Self {
        Self::Dissolve
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutomationLane {
    pub id: LaneId,
    pub target: AutomationTarget,
    #[serde(default)]
    pub interpolation: AutomationInterpolation,
    pub keyframes: Vec<AutomationKeyframe>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutomationTarget {
    pub node: NodeId,
    pub parameter: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AutomationInterpolation {
    Step,
    Linear,
    Bezier,
}

impl Default for AutomationInterpolation {
    fn default() -> Self {
        Self::Linear
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutomationKeyframe {
    pub frame: Frame,
    pub value: f64,
    #[serde(default)]
    pub easing: KeyframeEasing,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum KeyframeEasing {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Custom { in_tangent: f32, out_tangent: f32 },
}

impl Default for KeyframeEasing {
    fn default() -> Self {
        Self::Linear
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimelineNode {
    pub id: NodeId,
    pub label: Option<String>,
    pub kind: TimelineNodeKind,
    #[serde(default)]
    pub locked: bool,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TimelineNodeKind {
    Clip(ClipNode),
    Transition(TransitionNode),
    Generator {
        generator_id: String,
        timeline_range: FrameRange,
        #[serde(default)]
        metadata: serde_json::Value,
    },
    Effect {
        #[serde(default)]
        metadata: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimelineEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Sequential,
    Layer,
    TransitionInput,
    Automation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrackBinding {
    pub id: TrackId,
    pub name: String,
    pub kind: TrackKind,
    #[serde(default)]
    pub node_ids: Vec<NodeId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TrackKind {
    Video,
    Audio,
    Automation,
    Custom(String),
}

impl Default for TrackKind {
    fn default() -> Self {
        Self::Video
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimelineGraph {
    pub version: u16,
    pub nodes: HashMap<NodeId, TimelineNode>,
    pub edges: Vec<TimelineEdge>,
    #[serde(default)]
    pub tracks: Vec<TrackBinding>,
    #[serde(default)]
    pub automation: Vec<AutomationLane>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl Default for TimelineGraph {
    fn default() -> Self {
        Self {
            version: 1,
            nodes: HashMap::new(),
            edges: Vec::new(),
            tracks: Vec::new(),
            automation: Vec::new(),
            metadata: serde_json::Value::Null,
        }
    }
}
