//! Timeline visualization for profiling events
//! プロファイリングイベントのタイムライン可視化

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Timeline for tracking profiling events
/// プロファイリングイベント追跡用タイムライン
#[derive(Debug, Clone)]
pub struct Timeline {
    /// Timeline events
    /// タイムラインイベント
    events: Vec<TimelineEvent>,
    /// Start time of profiling
    /// プロファイリング開始時刻
    start_time: Option<Instant>,
    /// Event stack for nested operations
    /// ネストされた操作用イベントスタック
    event_stack: Vec<usize>,
}

/// Individual timeline event
/// 個別タイムラインイベント
#[derive(Debug, Clone)]
pub struct TimelineEvent {
    /// Event name
    pub name: String,
    /// Event category
    pub category: EventCategory,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Option<Instant>,
    /// Duration
    pub duration: Option<Duration>,
    /// Thread ID
    pub thread_id: std::thread::ThreadId,
    /// Process ID
    pub process_id: u32,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Parent event index
    pub parent_idx: Option<usize>,
}

/// Event categories for grouping
/// グループ化用イベントカテゴリ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventCategory {
    /// CPU operation
    Cpu,
    /// GPU kernel
    Gpu,
    /// Memory operation
    Memory,
    /// Data transfer
    DataTransfer,
    /// Synchronization
    Synchronization,
    /// User annotation
    UserAnnotation,
}

impl Timeline {
    /// Create new timeline
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            start_time: None,
            event_stack: Vec::new(),
        }
    }

    /// Add event to timeline
    pub fn add_event(&mut self, name: &str, start_time: Instant, category: Option<EventCategory>) {
        if self.start_time.is_none() {
            self.start_time = Some(start_time);
        }

        let parent_idx = self.event_stack.last().cloned();
        
        let event = TimelineEvent {
            name: name.to_string(),
            category: category.unwrap_or(EventCategory::Cpu),
            start_time,
            end_time: None,
            duration: None,
            thread_id: std::thread::current().id(),
            process_id: std::process::id(),
            metadata: HashMap::new(),
            parent_idx,
        };

        self.events.push(event);
        self.event_stack.push(self.events.len() - 1);
    }

    /// End an event
    pub fn end_event(&mut self, name: &str, end_time: Instant) {
        // Find the most recent matching event in the stack
        if let Some(idx) = self.event_stack.iter().rposition(|&i| {
            self.events.get(i).map_or(false, |e| e.name == name)
        }) {
            let event_idx = self.event_stack.remove(idx);
            if let Some(event) = self.events.get_mut(event_idx) {
                event.end_time = Some(end_time);
                event.duration = Some(end_time.duration_since(event.start_time));
            }
        }
    }

    /// Add metadata to the last event
    pub fn add_metadata(&mut self, key: String, value: String) {
        if let Some(event) = self.events.last_mut() {
            event.metadata.insert(key, value);
        }
    }

    /// Clear timeline
    pub fn clear(&mut self) {
        self.events.clear();
        self.start_time = None;
        self.event_stack.clear();
    }

    /// Export timeline to Chrome tracing format
    pub fn export_chrome_trace(&self) -> String {
        let mut trace_events = Vec::new();
        let base_time = self.start_time.unwrap_or_else(Instant::now);

        for event in &self.events {
            let start_us = event.start_time.duration_since(base_time).as_micros() as f64;
            let duration_us = event.duration
                .map(|d| d.as_micros() as f64)
                .unwrap_or(0.0);

            let trace_event = ChromeTraceEvent {
                name: event.name.clone(),
                cat: format!("{:?}", event.category),
                ph: "X".to_string(), // Complete event
                ts: start_us,
                dur: duration_us,
                pid: event.process_id,
                tid: format!("{:?}", event.thread_id),
                args: event.metadata.clone(),
            };

            trace_events.push(trace_event);
        }

        let trace = ChromeTrace {
            traceEvents: trace_events,
            displayTimeUnit: "ms".to_string(),
        };

        serde_json::to_string_pretty(&trace).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get timeline summary
    pub fn get_summary(&self) -> TimelineSummary {
        let total_events = self.events.len();
        let completed_events = self.events.iter()
            .filter(|e| e.end_time.is_some())
            .count();
        
        let total_duration = if let (Some(start), Some(last_event)) = 
            (self.start_time, self.events.last()) {
            last_event.end_time
                .or(Some(Instant::now()))
                .map(|end| end.duration_since(start))
        } else {
            None
        };

        let events_by_category = self.events.iter()
            .fold(HashMap::new(), |mut acc, event| {
                *acc.entry(format!("{:?}", event.category)).or_insert(0) += 1;
                acc
            });

        TimelineSummary {
            total_events,
            completed_events,
            total_duration,
            events_by_category,
        }
    }
}

/// Timeline summary
/// タイムラインサマリー
#[derive(Debug, Clone)]
pub struct TimelineSummary {
    /// Total number of events
    pub total_events: usize,
    /// Number of completed events
    pub completed_events: usize,
    /// Total duration
    pub total_duration: Option<Duration>,
    /// Events grouped by category
    pub events_by_category: HashMap<String, usize>,
}

/// Chrome trace event format
/// Chromeトレースイベント形式
#[derive(Serialize, Deserialize)]
struct ChromeTraceEvent {
    name: String,
    cat: String,
    ph: String,
    ts: f64,
    dur: f64,
    pid: u32,
    tid: String,
    args: HashMap<String, String>,
}

/// Chrome trace format
/// Chromeトレース形式
#[derive(Serialize, Deserialize)]
struct ChromeTrace {
    traceEvents: Vec<ChromeTraceEvent>,
    displayTimeUnit: String,
}

/// Flame graph data structure
/// フレームグラフデータ構造
#[derive(Debug, Clone)]
pub struct FlameGraphData {
    /// Root node
    pub root: FlameGraphNode,
    /// Total time
    pub total_time: Duration,
}

/// Flame graph node
/// フレームグラフノード
#[derive(Debug, Clone)]
pub struct FlameGraphNode {
    /// Node name
    pub name: String,
    /// Self time
    pub self_time: Duration,
    /// Total time including children
    pub total_time: Duration,
    /// Child nodes
    pub children: Vec<FlameGraphNode>,
}

impl Timeline {
    /// Generate flame graph data
    pub fn generate_flame_graph(&self) -> Option<FlameGraphData> {
        if self.events.is_empty() {
            return None;
        }

        // Build tree structure
        let mut root = FlameGraphNode {
            name: "root".to_string(),
            self_time: Duration::ZERO,
            total_time: Duration::ZERO,
            children: Vec::new(),
        };

        // Find root events (no parent)
        let root_events: Vec<_> = self.events.iter()
            .enumerate()
            .filter(|(_, e)| e.parent_idx.is_none())
            .collect();

        for (idx, event) in root_events {
            if let Some(duration) = event.duration {
                let node = self.build_flame_graph_node(idx, &self.events);
                root.children.push(node);
                root.total_time += duration;
            }
        }

        let total_time = root.total_time;
        Some(FlameGraphData {
            root,
            total_time,
        })
    }

    fn build_flame_graph_node(&self, event_idx: usize, events: &[TimelineEvent]) -> FlameGraphNode {
        let event = &events[event_idx];
        let mut node = FlameGraphNode {
            name: event.name.clone(),
            self_time: event.duration.unwrap_or(Duration::ZERO),
            total_time: event.duration.unwrap_or(Duration::ZERO),
            children: Vec::new(),
        };

        // Find child events
        for (idx, child_event) in events.iter().enumerate() {
            if child_event.parent_idx == Some(event_idx) {
                let child_node = self.build_flame_graph_node(idx, events);
                node.self_time = node.self_time.saturating_sub(child_node.total_time);
                node.children.push(child_node);
            }
        }

        node
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_timeline_basic() {
        let mut timeline = Timeline::new();
        
        let start = Instant::now();
        timeline.add_event("operation1", start, Some(EventCategory::Cpu));
        
        thread::sleep(Duration::from_millis(10));
        let end = Instant::now();
        timeline.end_event("operation1", end);
        
        assert_eq!(timeline.events.len(), 1);
        assert!(timeline.events[0].duration.is_some());
    }

    #[test]
    fn test_nested_events() {
        let mut timeline = Timeline::new();
        
        let start1 = Instant::now();
        timeline.add_event("outer", start1, Some(EventCategory::Cpu));
        
        let start2 = Instant::now();
        timeline.add_event("inner", start2, Some(EventCategory::Cpu));
        
        timeline.end_event("inner", Instant::now());
        timeline.end_event("outer", Instant::now());
        
        assert_eq!(timeline.events.len(), 2);
        assert_eq!(timeline.events[1].parent_idx, Some(0));
    }

    #[test]
    fn test_chrome_trace_export() {
        let mut timeline = Timeline::new();
        
        timeline.add_event("test_op", Instant::now(), Some(EventCategory::Cpu));
        thread::sleep(Duration::from_millis(5));
        timeline.end_event("test_op", Instant::now());
        
        let trace = timeline.export_chrome_trace();
        assert!(trace.contains("test_op"));
        assert!(trace.contains("traceEvents"));
    }

    #[test]
    fn test_flame_graph() {
        let mut timeline = Timeline::new();
        
        let start = Instant::now();
        timeline.add_event("parent", start, Some(EventCategory::Cpu));
        timeline.add_event("child1", start, Some(EventCategory::Cpu));
        timeline.end_event("child1", start + Duration::from_millis(5));
        timeline.add_event("child2", start + Duration::from_millis(5), Some(EventCategory::Cpu));
        timeline.end_event("child2", start + Duration::from_millis(10));
        timeline.end_event("parent", start + Duration::from_millis(10));
        
        let flame_graph = timeline.generate_flame_graph().unwrap();
        assert_eq!(flame_graph.root.children.len(), 1);
        assert_eq!(flame_graph.root.children[0].children.len(), 2);
    }
}