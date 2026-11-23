use super::{
    bounded_redacted_output, dedup_paths, ok_or_flag, parse_ts, title_from_messages, Adapter,
    Discovered,
};
use crate::model::{EditEvent, EditKind, Message, Role, Session, ToolCall, ToolResult};
use crate::util::short_id;
use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

mod history;
pub use history::CodexHistory;
#[cfg(test)]
pub(crate) use history::THREAD_SCANS;

/// Codex CLI stores one JSONL rollout per session under
/// `~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl`, and compresses
/// older ones in place to `.jsonl.zst`.
/// Lines carry a `type` plus a `payload`; the schema has shifted across
/// versions, so both `response_item` and `event_msg` shapes are handled.
///
/// Archiving a thread moves its rollout to `~/.codex/archived_sessions/`, laid
/// out the same way; [`Codex::archived_in`] reads that directory.
///
/// `Codex::default()` reads the stock `~/.codex` install. An embedder that
/// runs several Codex installs in different homes builds one adapter per
/// install with [`Codex::in_home`].
pub struct Codex {
    /// Explicit rollout directory, or `None` for `dir` under the stock home.
    root: Option<PathBuf>,
    dir: &'static str,
}

impl Default for Codex {
    fn default() -> Self {
        Codex {
            root: None,
            dir: "sessions",
        }
    }
}

impl Codex {
    /// An adapter for the Codex install rooted at `home` (e.g. `~/.codex2`).
    /// The sessions sub-directory layout is the adapter's business, not the
    /// caller's.
    pub fn in_home(home: impl Into<PathBuf>) -> Self {
        Codex {
            root: Some(home.into().join("sessions")),
            dir: "sessions",
        }
    }

    /// The archived rollouts of the stock install.
    pub fn archived() -> Self {
        Codex {
            root: None,
            dir: "archived_sessions",
        }
    }

    /// The archived rollouts of the Codex install rooted at `home`.
    pub fn archived_in(home: impl Into<PathBuf>) -> Self {
        Codex {
            root: Some(home.into().join("archived_sessions")),
            dir: "archived_sessions",
        }
    }
}

// Deserialize just the routing tags. Unknown fields (including tool payloads
// and compaction history) are skipped without materializing them as Values.
#[derive(Deserialize)]
struct Routing {
    #[serde(rename = "type")]
    kind: String,
    payload: Option<PayloadRouting>,
}

#[derive(Deserialize)]
struct PayloadRouting {
    #[serde(rename = "type")]
    kind: Option<String>,
    item: Option<ItemRouting>,
    call_id: Option<String>,
    turn_id: Option<String>,
}

#[derive(Deserialize)]
struct ItemRouting {
    id: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    status: Option<String>,
}

impl Routing {
    fn tags(&self) -> (&str, Option<&str>, Option<&str>) {
        let payload = self.payload.as_ref();
        (
            &self.kind,
            payload.and_then(|p| p.kind.as_deref()),
            payload
                .and_then(|p| p.item.as_ref())
                .and_then(|i| i.kind.as_deref()),
        )
    }
}

fn dialogue_route(tags: (&str, Option<&str>, Option<&str>)) -> bool {
    matches!(
        tags,
        ("message", _, _)
            | ("response_item", Some("message"), _)
            | ("event_msg", Some("user_message" | "agent_message"), _)
            | (
                "event_msg",
                Some("item_completed"),
                Some("UserMessage" | "AgentMessage")
            )
    )
}

fn turn_boundary<'a>(
    kind: &str,
    payload_kind: Option<&str>,
    turn_id: Option<&'a str>,
) -> Option<&'a str> {
    if kind != "event_msg" {
        return None;
    }
    match payload_kind {
        Some("task_started" | "turn_started") => Some(turn_id.unwrap_or_default()),
        Some("task_complete" | "turn_complete" | "turn_aborted") => Some(""),
        _ => None,
    }
}

fn rollout_suffix(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let stem = name
        .strip_suffix(".jsonl.zst")
        .or_else(|| name.strip_suffix(".jsonl"))?;
    let rollout = stem.strip_prefix("rollout-")?;
    let (prefix, id) = rollout.split_at_checked(rollout.len().checked_sub(36)?)?;
    (prefix.ends_with('-')
        && id.bytes().enumerate().all(|(i, b)| {
            if matches!(i, 8 | 13 | 18 | 23) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        }))
    .then(|| id.to_owned())
}

impl Adapter for Codex {
    fn native_id(&self, key: &str) -> Option<String> {
        let path = Path::new(key);
        crate::util::find_in_jsonl_header(path, |v| {
            if v["type"] != "session_meta" {
                return None;
            }
            super::nonblank(&v["payload"]["id"]).map(str::to_owned)
        })
        .or_else(|| rollout_suffix(path))
    }

    fn name(&self) -> &'static str {
        "codex"
    }

    fn root(&self) -> Option<PathBuf> {
        match &self.root {
            Some(root) => Some(root.clone()),
            None => Some(dirs::home_dir()?.join(".codex").join(self.dir)),
        }
    }

    /// Every install speaks only for the rows under the root it scans.
    fn reconcile_scope(&self) -> Option<String> {
        super::root_scope(self.root().as_deref())
    }

    fn discover(&self) -> Discovered {
        let Some(root) = self.root() else {
            return Vec::new().into();
        };
        if !root.exists() {
            return Vec::new().into(); // no store on this machine - normal
        }
        let mut had_error = false;
        let files = WalkDir::new(root)
            .into_iter()
            .filter_map(|e| ok_or_flag(e, &mut had_error))
            .filter(|e| e.file_type().is_file())
            .filter(|e| {
                let name = e.file_name().to_string_lossy();
                name.starts_with("rollout-")
                    && (name.ends_with(".jsonl") || name.ends_with(".jsonl.zst"))
            })
            .map(|e| e.into_path())
            .collect();
        Discovered { files, had_error }
    }

    fn parse(&self, path: &Path) -> Result<Session> {
        // Over the byte cap: window (head+tail) instead of dropping the session.
        // Past the cap, the conversation is still read from the middle; only
        // its tool calls and output are windowed.
        let (lines, windowed) = crate::util::numbered_session_lines(path)?;

        let mut messages: Vec<Message> = Vec::new();
        let mut evidence = RolloutEvidence::default();
        let mut streamed = if windowed {
            evidence.scan(path, &lines)?
        } else {
            Vec::new()
        }
        .into_iter()
        .peekable();
        let mut mirrors = DialogueMirrors::default();
        // An MCP call's failure is stated only in its `mcp_tool_call_end` event;
        // the `function_call_output` carries the text alone.
        let mut failed_mcp_calls: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut started = None;
        let mut ended = None;
        let mut observe_timestamp = |ts| {
            if let Some(ts) = ts {
                started = started.or(Some(ts));
                ended = Some(ts);
            }
        };
        // Some rollouts record an MCP call only as its end event. Those are
        // indexed from the event, unless a `function_call` already stands for
        // the same call - by id, or by identical arguments under another id.
        let (function_call_ids, function_call_args) = function_calls(&lines);

        for (ordinal, line) in &lines {
            while streamed
                .peek()
                .is_some_and(|(position, _)| position <= ordinal)
            {
                let (_, message) = streamed.next().unwrap();
                observe_timestamp(message.ts);
                messages.push(message);
            }
            let Ok(v) = serde_json::from_str::<Value>(line) else {
                continue;
            };

            let ts = v
                .get("timestamp")
                .and_then(Value::as_str)
                .and_then(parse_ts);
            observe_timestamp(ts);

            if !windowed {
                evidence.observe(&v);
                let tags = (
                    v["type"].as_str().unwrap_or_default(),
                    v["payload"]["type"].as_str(),
                    v["payload"]["item"]["type"].as_str(),
                );
                if dialogue_route(tags) {
                    if let Ok(row) = serde_json::from_str::<StreamedDialogue>(line) {
                        if let Some((dialogue, turn)) = row.project(tags) {
                            evidence.saw_prompt |= dialogue.role == Role::User;
                            if mirrors.first_occurrence(turn, &dialogue) {
                                messages.push(dialogue.into_message(ts));
                            }
                        }
                    }
                    continue;
                }
                if v["type"] == "event_msg"
                    && v["payload"]["type"] == "item_completed"
                    && v["payload"]["item"]["type"] == "FileChange"
                {
                    let item = &v["payload"]["item"];
                    evidence.file_change(line, item["id"].as_str(), item["status"].as_str(), ts);
                }
                if let Some(turn) = turn_boundary(
                    v["type"].as_str().unwrap_or_default(),
                    v["payload"]["type"].as_str(),
                    v["payload"]["turn_id"].as_str(),
                ) {
                    mirrors.begin(turn);
                }
            }

            match v["type"].as_str() {
                Some("response_item") => {
                    match v.pointer("/payload/type").and_then(Value::as_str) {
                        Some("function_call") => {
                            let payload = v.get("payload").unwrap_or(&Value::Null);
                            let raw_args = payload.get("arguments").and_then(Value::as_str);
                            let mut args = if let Some(raw_args) = raw_args {
                                serde_json::from_str(raw_args)
                                    .unwrap_or_else(|_| json!({"arguments": raw_args}))
                            } else {
                                json!({})
                            };
                            redact_json_strings(&mut args);
                            messages.push(Message::tool_call(
                                ts,
                                ToolCall {
                                    id: call_id(payload),
                                    name: payload
                                        .get("name")
                                        .and_then(Value::as_str)
                                        .unwrap_or("?")
                                        .to_owned(),
                                    args,
                                },
                            ));
                        }
                        Some("custom_tool_call") => {
                            let payload = v.get("payload").unwrap_or(&Value::Null);
                            let input = payload.get("input").and_then(Value::as_str).unwrap_or("");
                            let is_patch = input.contains("*** Begin Patch");
                            let (name, mut args) = if is_patch {
                                let patch = input
                                    .split_once("*** Begin Patch")
                                    .map(|(_, patch)| format!("*** Begin Patch{patch}"))
                                    .unwrap_or_else(|| input.to_owned())
                                    .replace("\\n", "\n");
                                ("apply_patch", json!({"patch": patch}))
                            } else if let Some(cmd) = exec_command_cmd(input) {
                                // The `exec` tool runs a JS snippet; the common shape is
                                // `await tools.exec_command({cmd:"..."})`. Surface the
                                // command itself, not the wrapper around it.
                                ("exec_command", json!({"cmd": cmd}))
                            } else {
                                ("exec", json!({"input": input}))
                            };
                            redact_json_strings(&mut args);
                            messages.push(Message::tool_call(
                                ts,
                                ToolCall {
                                    id: call_id(payload),
                                    name: if name != "exec" {
                                        name.to_owned()
                                    } else {
                                        payload
                                            .get("name")
                                            .and_then(Value::as_str)
                                            .unwrap_or(name)
                                            .to_owned()
                                    },
                                    args,
                                },
                            ));
                        }
                        Some("function_call_output" | "custom_tool_call_output") => {
                            let payload = v.get("payload").unwrap_or(&Value::Null);
                            let (output, is_error) = payload
                                .get("output")
                                .map(unwrap_output_value)
                                .unwrap_or_default();
                            messages.push(Message::tool_result(
                                ts,
                                ToolResult {
                                    call_id: call_id(payload),
                                    text: bounded_redacted_output(&output),
                                    is_error,
                                },
                            ));
                        }
                        // Reasoning and other rollout bookkeeping are not user
                        // or tool content and remain excluded from the index.
                        _ => {}
                    }
                }
                Some("event_msg") if v["payload"]["type"] == "mcp_tool_call_end" => {
                    let result = v.pointer("/payload/result");
                    let failed = result.is_some_and(|r| {
                        r.get("Err").is_some()
                            || r.pointer("/Ok/isError").and_then(Value::as_bool) == Some(true)
                    });
                    let id = v.pointer("/payload/call_id").and_then(Value::as_str);
                    if let (true, Some(id)) = (failed, id) {
                        failed_mcp_calls.insert(id.to_owned());
                    }
                    let mut args = v
                        .pointer("/payload/invocation/arguments")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    let paired = id.is_some_and(|id| function_call_ids.contains(id))
                        || function_call_args.contains(&args);
                    if !paired {
                        redact_json_strings(&mut args);
                        let name = v
                            .pointer("/payload/invocation/tool")
                            .and_then(Value::as_str)
                            .unwrap_or("?")
                            .to_owned();
                        messages.push(Message::tool_call(
                            ts,
                            ToolCall {
                                id: id.map(str::to_owned),
                                name,
                                args,
                            },
                        ));
                        messages.push(Message::tool_result(
                            ts,
                            ToolResult {
                                call_id: id.map(str::to_owned),
                                text: bounded_redacted_output(&mcp_result_text(result)),
                                is_error: failed,
                            },
                        ));
                    }
                }
                _ => {}
            }
        }

        for (_, message) in streamed {
            observe_timestamp(message.ts);
            messages.push(message);
        }

        for message in &mut messages {
            if let Some(crate::model::ToolEvent::Part(crate::model::ToolPart::Result(result))) =
                &mut message.tool
            {
                if result
                    .call_id
                    .as_ref()
                    .is_some_and(|id| failed_mcp_calls.contains(id))
                {
                    result.is_error = true;
                }
            }
        }
        let title = if windowed {
            format!("[large] {}", title_from_messages(&messages))
        } else {
            title_from_messages(&messages)
        };

        let operational_edited_paths = evidence.operational_paths();
        let touched = dedup_paths(
            evidence
                .patches
                .applied()
                .chain(evidence.completed_paths.iter().cloned())
                .collect(),
        );

        Ok(Session {
            id: short_id(&path.to_string_lossy()),
            tool: self.name(),
            path: path.to_path_buf(),
            project: evidence.cwd.clone().unwrap_or_default(),
            started,
            ended,
            title,
            subagent: evidence.subagent,
            messages,
            raw_cwd: evidence.cwd,
            operational_edited_paths,
            compacted_prefix: evidence.compacted_prefix,
            touched,
            edits: evidence.edits,
        })
    }
}

#[derive(Clone, Copy)]
enum DialogueSource {
    Response = 0,
    LegacyEvent = 1,
    Completed = 2,
}

struct Dialogue {
    role: Role,
    source: Option<DialogueSource>,
    text: String,
    key: [u8; 32],
}

impl Dialogue {
    fn from_parts<'a>(
        role: Role,
        source: Option<DialogueSource>,
        parts: impl IntoIterator<Item = &'a str>,
    ) -> Option<Self> {
        let text = parts
            .into_iter()
            .filter(|text| !text.trim().is_empty() && (role != Role::User || !is_boilerplate(text)))
            .flat_map(|text| {
                // Model-input attachments carry separate XML-like label blocks.
                // Skip those labels while keeping the user's adjacent text.
                text.split('\n').filter(move |line| {
                    role != Role::User
                        || !matches!(source, Some(DialogueSource::Response))
                        || !attachment_label(line.trim())
                })
            })
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_owned();
        let normalized: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let key = Sha256::digest(normalized.as_bytes()).into();
        (!text.is_empty()).then_some(Self {
            role,
            source,
            text,
            key,
        })
    }

    fn into_message(self, ts: Option<chrono::DateTime<chrono::Utc>>) -> Message {
        Message {
            role: self.role,
            text: self.text,
            ts,
            is_thought: false,
            tool: None,
        }
    }
}

fn attachment_label(text: &str) -> bool {
    matches!(text, "<audio>" | "</audio>" | "<image>" | "</image>")
        || (text.ends_with('>')
            && (text.starts_with("<audio name=") || text.starts_with("<image name=")))
}

fn role_name(name: &str) -> Option<Role> {
    match name {
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        _ => None,
    }
}

#[derive(Default)]
struct DialogueMirrors {
    current_id: Option<String>,
    boundary: bool,
    current: TurnDialogue,
    turns: HashMap<String, TurnDialogue>,
}

impl DialogueMirrors {
    fn begin(&mut self, turn: &str) {
        self.current_id = (!turn.is_empty()).then(|| turn.to_owned());
        self.boundary = self.current_id.is_some();
        self.current = TurnDialogue::default();
        if let Some(id) = &self.current_id {
            self.turns.entry(id.clone()).or_default();
        }
    }

    fn first_occurrence(&mut self, turn: Option<&str>, message: &Dialogue) -> bool {
        let Some(source) = message.source else {
            return true;
        };
        let explicit = turn.filter(|id| !id.is_empty());
        // A newly seen explicit ID can arrive after its untagged mirrors.
        // Once known, it also lets late records resolve to their own turn.
        if let Some(id) = explicit {
            if !self.turns.contains_key(id) {
                let turn = if self.current_id.is_none() {
                    std::mem::take(&mut self.current)
                } else {
                    TurnDialogue::default()
                };
                self.turns.insert(id.to_owned(), turn);
                if self.current_id.as_deref() != Some(id) {
                    self.boundary = false;
                }
                self.current_id = Some(id.to_owned());
            }
            return self
                .turns
                .entry(id.to_owned())
                .or_default()
                .observe(message, source, false);
        }
        if let Some(id) = &self.current_id {
            let turn = self.turns.entry(id.clone()).or_default();
            let fresh_prompt = message.role == Role::User
                && !turn.assistant.is_empty()
                && turn.would_emit(message, source);
            if self.boundary || !fresh_prompt {
                return turn.observe(message, source, false);
            }
            self.current_id = None;
        }
        self.current.observe(message, source, true)
    }
}

#[derive(Default)]
struct TurnDialogue {
    user: HashMap<[u8; 32], [usize; 3]>,
    assistant: HashMap<[u8; 32], [usize; 3]>,
}

impl TurnDialogue {
    fn would_emit(&self, dialogue: &Dialogue, source: DialogueSource) -> bool {
        let by_text = if dialogue.role == Role::User {
            &self.user
        } else {
            &self.assistant
        };
        by_text.get(&dialogue.key).is_none_or(|counts| {
            counts[source as usize] == counts.iter().copied().max().unwrap_or_default()
        })
    }

    fn observe(&mut self, dialogue: &Dialogue, source: DialogueSource, implicit: bool) -> bool {
        let source = source as usize;
        let by_text = if dialogue.role == Role::User {
            &mut self.user
        } else {
            &mut self.assistant
        };
        let counts = by_text.entry(dialogue.key).or_default();
        // The nth occurrence from a source matches the nth occurrence of
        // this text in any other source. Emit at the first record's position.
        let first = counts[source] == counts.iter().copied().max().unwrap_or_default();
        counts[source] += 1;
        // Without explicit turn markers, a new prompt after an assistant
        // response starts a new turn. Late mirrors of that prompt still match
        // the original occurrence before we move the boundary.
        if implicit && dialogue.role == Role::User && first && !self.assistant.is_empty() {
            self.assistant.clear();
            self.user.clear();
            self.user.entry(dialogue.key).or_default()[source] = 1;
        }
        first
    }
}

#[derive(Deserialize)]
struct StreamedDialogue {
    timestamp: Option<String>,
    role: Option<String>,
    turn_id: Option<String>,
    content: Option<Vec<TextBlock>>,
    payload: Option<StreamedPayload>,
}

#[derive(Deserialize)]
struct TextBlock {
    #[serde(default)]
    text: Value,
}

#[derive(Deserialize)]
struct StreamedPayload {
    turn_id: Option<String>,
    role: Option<String>,
    content: Option<Vec<TextBlock>>,
    message: Option<String>,
    item: Option<StreamedItem>,
}

#[derive(Deserialize)]
struct StreamedItem {
    turn_id: Option<String>,
    content: Option<Vec<TextBlock>>,
}

impl StreamedDialogue {
    fn project(
        &self,
        tags: (&str, Option<&str>, Option<&str>),
    ) -> Option<(Dialogue, Option<&str>)> {
        let blocks = |role, source, content: &[TextBlock]| {
            Dialogue::from_parts(
                role,
                source,
                content.iter().filter_map(|block| block.text.as_str()),
            )
        };
        let payload = self.payload.as_ref();
        let turn = payload
            .and_then(|p| p.turn_id.as_deref())
            .or_else(|| payload?.item.as_ref()?.turn_id.as_deref())
            .or(self.turn_id.as_deref());
        let dialogue = match tags {
            ("message", _, _) => blocks(
                role_name(self.role.as_deref()?)?,
                None,
                self.content.as_deref()?,
            )?,
            ("response_item", Some("message"), _) => {
                let payload = payload?;
                blocks(
                    role_name(payload.role.as_deref()?)?,
                    Some(DialogueSource::Response),
                    payload.content.as_deref()?,
                )?
            }
            ("event_msg", Some("user_message" | "agent_message"), _) => {
                let role = if tags.1 == Some("user_message") {
                    Role::User
                } else {
                    Role::Assistant
                };
                Dialogue::from_parts(
                    role,
                    Some(DialogueSource::LegacyEvent),
                    [payload?.message.as_deref()?],
                )?
            }
            ("event_msg", Some("item_completed"), Some("UserMessage" | "AgentMessage")) => {
                let role = if tags.2 == Some("UserMessage") {
                    Role::User
                } else {
                    Role::Assistant
                };
                blocks(
                    role,
                    Some(DialogueSource::Completed),
                    payload?.item.as_ref()?.content.as_deref()?,
                )?
            }
            _ => return None,
        };
        Some((dialogue, turn))
    }
}

#[derive(Default)]
struct RolloutEvidence {
    cwd: Option<String>,
    subagent: bool,
    paginated: bool,
    compacted_prefix: bool,
    saw_prompt: bool,
    patches: Patches,
    completed_paths: Vec<String>,
    has_file_change: bool,
    edits: Vec<EditEvent>,
}

impl RolloutEvidence {
    // Read compact dialogue and evidence from the whole file. Only emitted
    // text from the middle counts against the dialogue budget.
    fn scan(
        &mut self,
        path: &Path,
        raw_window: &[(usize, String)],
    ) -> Result<Vec<(usize, Message)>> {
        let mut dialogue = Vec::new();
        let mut mirrors = DialogueMirrors::default();
        let mut remaining = crate::util::middle_kept_bytes();
        let mut ordinal = 0;
        crate::util::for_each_line(path, |line| {
            let position = ordinal;
            ordinal += 1;
            let Ok(route) = serde_json::from_str::<Routing>(line) else {
                return;
            };
            let tags = route.tags();
            if let Some(turn) = turn_boundary(
                tags.0,
                tags.1,
                route.payload.as_ref().and_then(|p| p.turn_id.as_deref()),
            ) {
                mirrors.begin(turn);
            }
            if !dialogue_route(tags) {
                self.scan_evidence(&route, line);
                return;
            }
            let Ok(row) = serde_json::from_str::<StreamedDialogue>(line) else {
                return;
            };
            let Some((message, turn)) = row.project(tags) else {
                return;
            };
            self.saw_prompt |= message.role == Role::User;
            if !mirrors.first_occurrence(turn, &message) {
                return;
            }
            if raw_window
                .binary_search_by_key(&position, |(i, _)| *i)
                .is_err()
            {
                let bytes = message.text.len() as u64;
                if bytes > remaining {
                    return;
                }
                remaining -= bytes;
            }
            let ts = row.timestamp.as_deref().and_then(parse_ts);
            dialogue.push((position, message.into_message(ts)));
        })?;
        Ok(dialogue)
    }

    fn scan_evidence(&mut self, route: &Routing, line: &str) {
        match route.tags() {
            ("compacted", _, _)
            | ("response_item", Some("compaction"), _)
            | ("event_msg", Some("item_completed"), Some("ContextCompaction")) => {
                self.compacted_prefix |= !self.saw_prompt;
                return;
            }
            ("event_msg", Some("item_completed"), Some("FileChange")) => {
                if let Some(item) = route.payload.as_ref().and_then(|p| p.item.as_ref()) {
                    self.file_change(line, item.id.as_deref(), item.status.as_deref(), None);
                }
                return;
            }
            ("response_item", Some("function_call_output" | "custom_tool_call_output"), _) => {
                if !route
                    .payload
                    .as_ref()
                    .and_then(|p| p.call_id.as_ref())
                    .is_some_and(|id| self.patches.single.contains_key(id))
                {
                    return;
                }
            }
            ("session_meta", _, _)
            | ("response_item", Some("function_call" | "custom_tool_call"), _) => {}
            _ => return,
        }
        if let Ok(record) = serde_json::from_str::<Value>(line) {
            self.observe(&record);
        }
    }

    fn observe(&mut self, record: &Value) {
        let payload = &record["payload"];
        match (record["type"].as_str(), payload["type"].as_str()) {
            (Some("session_meta"), _) => {
                self.subagent |= is_subagent_meta(record);
                self.paginated |= payload["history_mode"] == "paginated";
                if self.cwd.is_none() {
                    self.cwd = super::nonblank(&payload["cwd"]).map(str::to_owned);
                }
            }
            (Some("compacted"), _) | (Some("response_item"), Some("compaction")) => {
                self.compacted_prefix |= !self.saw_prompt;
            }
            (Some("event_msg"), Some("item_completed"))
                if payload["item"]["type"] == "ContextCompaction" =>
            {
                self.compacted_prefix |= !self.saw_prompt;
            }
            (Some("response_item"), Some("function_call" | "custom_tool_call")) => {
                self.patches.call(payload);
            }
            (Some("response_item"), Some("function_call_output" | "custom_tool_call_output"))
                if payload["call_id"]
                    .as_str()
                    .is_some_and(|id| self.patches.single.contains_key(id)) =>
            {
                let (output, is_error) = payload
                    .get("output")
                    .map(unwrap_output_value)
                    .unwrap_or_default();
                self.patches.result(payload, &output, is_error);
            }
            _ => {}
        }
    }

    fn file_change(
        &mut self,
        line: &str,
        id: Option<&str>,
        status: Option<&str>,
        ts: Option<chrono::DateTime<chrono::Utc>>,
    ) {
        self.has_file_change = true;
        if let (Some(id), Some("failed" | "declined")) = (id, status) {
            self.patches.failed.insert(id.to_owned());
        }
        if status != Some("completed") {
            return;
        }
        let Ok(CompletedChange { timestamp, payload }) =
            serde_json::from_str::<CompletedChange>(line)
        else {
            return;
        };
        let ts = ts.or_else(|| timestamp.as_deref().and_then(parse_ts));
        for (raw, change) in payload.item.changes.0 {
            self.completed_paths.push(raw.clone());
            let diff = super::nonblank(&change["diff"])
                .or_else(|| super::nonblank(&change["unified_diff"]))
                .or_else(|| super::nonblank(&change["content"]));
            if let (Some(path), Some(diff)) = (super::clean_path(&raw), diff) {
                let kind = match change["type"].as_str().or_else(|| change["kind"].as_str()) {
                    Some("add" | "write" | "create") => EditKind::Write,
                    _ => EditKind::Edit,
                };
                self.edits.push(EditEvent {
                    path,
                    kind,
                    snippet: super::redacted_truncate(diff, 200),
                    ts,
                });
            }
        }
    }

    fn operational_paths(&self) -> Vec<String> {
        let paths = if self.paginated || self.has_file_change {
            self.completed_paths.clone()
        } else {
            self.patches.confirmed().collect()
        };
        super::dedup_raw_paths(paths)
    }
}

#[derive(Deserialize)]
struct CompletedChange {
    #[serde(alias = "time")]
    timestamp: Option<String>,
    payload: ChangePayload,
}

#[derive(Deserialize)]
struct ChangePayload {
    item: ChangedItem,
}

#[derive(Deserialize)]
struct ChangedItem {
    changes: OrderedChanges,
}

struct OrderedChanges(Vec<(String, Value)>);

impl<'de> Deserialize<'de> for OrderedChanges {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct ChangesVisitor;
        impl<'de> serde::de::Visitor<'de> for ChangesVisitor {
            type Value = OrderedChanges;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a Codex FileChange map")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut changes = Vec::new();
                while let Some(change) = map.next_entry()? {
                    changes.push(change);
                }
                Ok(OrderedChanges(changes))
            }
        }
        deserializer.deserialize_map(ChangesVisitor)
    }
}

/// Extract the files an apply_patch touched from the raw call arguments. The
/// patch format names each file on a header line - `*** Add File: path`,
/// `*** Update File: path`, `*** Delete File: path`, `*** Move to: path` -
/// regardless of whether the call arrives as a dedicated apply_patch function
/// or a shell command wrapping a heredoc. Newlines may be JSON-escaped (\\n)
/// when the patch is embedded in an arguments string, so handle both.
fn collect_patched_paths(args: &str, out: &mut Vec<String>) {
    const MARKERS: [&str; 4] = [
        "*** Add File: ",
        "*** Update File: ",
        "*** Delete File: ",
        "*** Move to: ",
    ];
    for line in args.lines() {
        let line = line.trim_start();
        for m in MARKERS {
            if let Some(rest) = line.strip_prefix(m) {
                if !rest.is_empty() {
                    out.push(rest.to_owned());
                }
            }
        }
    }
}

/// The files one `response_item` payload patched: an apply_patch carried by a
/// function call's arguments or by a custom tool call's input.
fn touched_by(payload: &Value, out: &mut Vec<String>) {
    let kind = payload["type"].as_str();
    let args = match kind {
        Some("function_call") => payload["arguments"].as_str(),
        Some("custom_tool_call") => payload["input"].as_str(),
        _ => None,
    };
    let Some(args) = args else { return };
    if !args.contains("*** ") {
        return;
    }
    if kind == Some("function_call") {
        let parsed = serde_json::from_str::<Value>(args).ok();
        let native = parsed.as_ref().and_then(|v| {
            ["patch", "input", "cmd", "command"]
                .iter()
                .find_map(|field| v[*field].as_str())
        });
        if let Some(native) = native.filter(|text| text.contains('\n')) {
            collect_patched_paths(native, out);
            return;
        }
    }
    if let Some((_, patch)) = args.split_once("*** Begin Patch") {
        if payload["name"] == "apply_patch" && patch.contains('\n') {
            collect_patched_paths(patch, out);
        } else {
            collect_patched_paths(&patch.replace("\\n", "\n"), out);
        }
    } else if kind == Some("function_call") {
        collect_patched_paths(&args.replace("\\n", "\n"), out);
    }
}

/// The files a rollout's patches name, by call, less those of the calls that
/// did not apply. apply_patch checks every hunk before it writes, so a patch
/// that failed wrote none of its files.
#[derive(Default)]
struct Patches {
    calls: Vec<(Option<String>, Vec<String>)>,
    /// Calls that carry one patch, so their result is that patch's; true for
    /// apply_patch itself, whose error flag is the patch's error.
    single: HashMap<String, bool>,
    failed: HashSet<String>,
    confirmed: HashSet<String>,
}

impl Patches {
    const VERIFY_FAILED: &'static str = "apply_patch verification failed";

    fn call(&mut self, payload: &Value) {
        let mut paths = Vec::new();
        touched_by(payload, &mut paths);
        if paths.is_empty() {
            return;
        }
        let id = call_id(payload);
        let text = payload
            .get("input")
            .or_else(|| payload.get("arguments"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if let (Some(id), 1) = (&id, text.matches("*** Begin Patch").count()) {
            let direct = payload.get("name").and_then(Value::as_str) == Some("apply_patch");
            self.single.insert(id.clone(), direct);
        }
        self.calls.push((id, paths));
    }

    /// A script or shell command can fail after its patch applied, or print
    /// apply_patch's error text from a log, so it failed only if it failed with
    /// that error.
    fn result(&mut self, payload: &Value, output: &str, is_error: bool) {
        let Some(id) = call_id(payload) else {
            return;
        };
        let Some(&direct) = self.single.get(&id) else {
            return;
        };
        let verify_failed = output.contains(Self::VERIFY_FAILED);
        let failed = if direct {
            is_error
                || verify_failed
                || output.trim_start().starts_with("aborted by user")
                || output.trim_start().starts_with("patch rejected by user")
        } else {
            is_error && verify_failed
        };
        if failed {
            self.failed.insert(id);
        } else if !is_error && !output.trim().is_empty() {
            self.confirmed.insert(id);
        }
    }

    fn applied(&self) -> impl Iterator<Item = String> + '_ {
        self.calls
            .iter()
            .filter(|(id, _)| id.as_ref().is_none_or(|id| !self.failed.contains(id)))
            .flat_map(|(_, paths)| paths.iter().cloned())
    }

    fn confirmed(&self) -> impl Iterator<Item = String> + '_ {
        self.calls
            .iter()
            .filter(|(id, _)| {
                id.as_ref()
                    .is_some_and(|id| self.confirmed.contains(id) && !self.failed.contains(id))
            })
            .flat_map(|(_, paths)| paths.iter().cloned())
    }
}

fn call_id(payload: &Value) -> Option<String> {
    payload
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
}

/// The ids and parsed arguments of every `function_call` in a rollout.
fn function_calls(lines: &[(usize, String)]) -> (std::collections::HashSet<String>, Vec<Value>) {
    let mut ids = std::collections::HashSet::new();
    let mut args = Vec::new();
    for (_, line) in lines {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if v.get("type").and_then(Value::as_str) != Some("response_item") {
            continue;
        }
        let payload = v.get("payload").unwrap_or(&Value::Null);
        if !matches!(
            payload.get("type").and_then(Value::as_str),
            Some("function_call" | "custom_tool_call")
        ) {
            continue;
        }
        if let Some(id) = payload.get("call_id").and_then(Value::as_str) {
            ids.insert(id.to_owned());
        }
        if let Some(parsed) = payload
            .get("arguments")
            .and_then(Value::as_str)
            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        {
            args.push(parsed);
        }
    }
    (ids, args)
}

/// The text an MCP end event's result carries: its text blocks, or the error.
fn mcp_result_text(result: Option<&Value>) -> String {
    let Some(result) = result else {
        return String::new();
    };
    if let Some(err) = result.get("Err") {
        return err.as_str().map_or_else(|| err.to_string(), str::to_owned);
    }
    result
        .pointer("/Ok/content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// Flatten a result payload's `output` to text and an error flag. A string is
/// unwrapped as one chunk. The `exec` tool returns an array: a preamble block
/// (`Script completed\nWall time ...\nOutput:`), then one JSON chunk per
/// `exec_command` the script ran, each with its own `exit_code`, sometimes
/// with other text blocks between. The preamble is dropped, every chunk is
/// unwrapped, and any nonzero exit code or a failed script marks the result.
fn unwrap_output_value(output: &Value) -> (String, bool) {
    match output {
        Value::String(text) => unwrap_output(text),
        Value::Array(blocks) => {
            let mut parts = Vec::new();
            let mut is_error = false;
            for text in blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(Value::as_str))
            {
                if text.starts_with("Script completed\n") || text.starts_with("Script failed") {
                    is_error |= text.starts_with("Script failed");
                    continue;
                }
                let (chunk, chunk_error) = unwrap_output(text);
                is_error |= chunk_error;
                if !chunk.is_empty() {
                    parts.push(chunk);
                }
            }
            (parts.join("\n"), is_error)
        }
        _ => (String::new(), false),
    }
}

/// Unwrap Codex's JSON-encoded command result while preserving ordinary text.
/// Both current top-level exit_code and the older metadata.exit_code shape are
/// used to mark failed calls.
fn unwrap_output(text: &str) -> (String, bool) {
    let Ok(Value::Object(object)) = serde_json::from_str::<Value>(text) else {
        return unwrap_text_output(text).unwrap_or_else(|| (text.to_owned(), false));
    };
    let Some(output) = object.get("output").and_then(Value::as_str) else {
        return (text.to_owned(), false);
    };
    let exit_code = object.get("exit_code").and_then(Value::as_i64).or_else(|| {
        object
            .get("metadata")
            .and_then(|metadata| metadata.get("exit_code"))
            .and_then(Value::as_i64)
    });
    (output.to_owned(), exit_code.is_some_and(|code| code != 0))
}

/// Codex also writes command results as plain text: a header with
/// `Exit code: N` or `Process exited with code N`, then `Output:` and the
/// output itself.
fn unwrap_text_output(text: &str) -> Option<(String, bool)> {
    let (header, output) = text
        .split_once("\nOutput:\n")
        .or_else(|| text.strip_suffix("\nOutput:").map(|header| (header, "")))?;
    let exit_code = header.lines().find_map(|line| {
        line.strip_prefix("Exit code: ")
            .or_else(|| line.strip_prefix("Process exited with code "))
            .and_then(|code| code.trim().parse::<i64>().ok())
    })?;
    Some((output.to_owned(), exit_code != 0))
}

fn redact_json_strings(value: &mut Value) {
    match value {
        Value::String(text) => *text = crate::redact::redact(text).into_owned(),
        Value::Array(values) => values.iter_mut().for_each(redact_json_strings),
        Value::Object(values) => values.values_mut().for_each(redact_json_strings),
        _ => {}
    }
}

/// Codex wraps instructions and environment dumps in pseudo-XML tags and
/// repeats them in every session. Indexing them buries real matches.
fn is_boilerplate(text: &str) -> bool {
    let t = text.trim_start();
    t.starts_with("<user_instructions>")
        || t.starts_with("<environment_context>")
        || t.starts_with("<ENVIRONMENT_CONTEXT>")
        || t.starts_with("<turn_context>")
        || t.starts_with("# AGENTS.md instructions")
        || t.starts_with("<INSTRUCTIONS>")
}

fn push(
    messages: &mut Vec<Message>,
    role: Role,
    text: &str,
    ts: Option<chrono::DateTime<chrono::Utc>>,
) {
    let text = text.trim();
    if !text.is_empty() {
        messages.push(Message {
            role,
            text: text.to_owned(),
            ts,
            is_thought: false,
            tool: None,
        });
    }
}

/// Whether a rollout's `session_meta` line marks a sub-agent thread. Shared
/// with the index, which reclassifies rows parsed before this was read.
pub(crate) fn is_subagent_meta(v: &Value) -> bool {
    v.pointer("/payload/thread_source").and_then(Value::as_str) == Some("subagent")
        || v.pointer("/payload/source/subagent").is_some()
}

/// Pull the command strings out of an `exec` tool snippet of the form
/// `tools.exec_command({cmd:"..."})`, decoding JS string escapes. A script
/// that runs several commands yields them joined with `; `. Returns None for
/// any other snippet so the caller keeps the raw input.
fn exec_command_cmd(input: &str) -> Option<String> {
    let cmds: Vec<String> = input
        .split("tools.exec_command(")
        .skip(1)
        .filter_map(js_cmd_literal)
        .collect();
    (!cmds.is_empty()).then(|| cmds.join("; "))
}

/// Decode the `cmd` string literal from one `exec_command` argument object.
fn js_cmd_literal(call: &str) -> Option<String> {
    let (_, after_key) = call.split_once("cmd")?;
    let rest = after_key.trim_start().strip_prefix(':')?.trim_start();
    let mut chars = rest.chars();
    let quote = chars.next()?;
    if !matches!(quote, '"' | '\'' | '`') {
        return None;
    }
    let mut cmd = String::new();
    while let Some(c) = chars.next() {
        match c {
            c if c == quote => return (!cmd.trim().is_empty()).then_some(cmd),
            '\\' => match chars.next()? {
                'n' => cmd.push('\n'),
                't' => cmd.push('\t'),
                'r' => cmd.push('\r'),
                other => cmd.push(other),
            },
            c => cmd.push(c),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dialogue_fixture(source: usize, role: Role, text: &str, turn: Option<&str>) -> Value {
        let blocks = text
            .split('\n')
            .map(|text| json!({"text":text}))
            .collect::<Vec<_>>();
        let (kind, mut payload) = match (source, role) {
            (0, _) => (
                "response_item",
                json!({"type":"message","role":role.label(),"content":blocks}),
            ),
            (1, Role::User) => ("event_msg", json!({"type":"user_message","message":text})),
            (1, Role::Assistant) => ("event_msg", json!({"type":"agent_message","message":text})),
            (2, Role::User | Role::Assistant) => (
                "event_msg",
                json!({"type":"item_completed","item":{
                    "type":if role == Role::User { "UserMessage" } else { "AgentMessage" },
                    "content":blocks
                }}),
            ),
            _ => unreachable!(),
        };
        if let Some(turn) = turn {
            payload["turn_id"] = json!(turn);
        }
        json!({"type":kind,"payload":payload})
    }

    fn write_rollout(file: &Path, rows: &[Value]) {
        std::fs::write(
            file,
            rows.iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
    }

    #[test]
    fn dialogue_mirrors_preserve_same_turn_repeats_and_first_arrival_order() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("rollout-test.jsonl");
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let mut records = vec![
                json!({"type":"event_msg","payload":{"type":"turn_started","turn_id":"first"}}),
            ];
            // Two same-text occurrences from the first source, including a
            // late canonical mirror after the intervening assistant reply.
            for source in order {
                records.push(dialogue_fixture(
                    source,
                    Role::User,
                    "repeat",
                    Some("first"),
                ));
                records.push(dialogue_fixture(
                    source,
                    Role::Assistant,
                    "same\nreply",
                    Some("first"),
                ));
                records.push(dialogue_fixture(
                    source,
                    Role::User,
                    "repeat",
                    Some("first"),
                ));
                records.push(dialogue_fixture(
                    source,
                    Role::Assistant,
                    "same\nreply",
                    Some("first"),
                ));
            }
            for source in order.into_iter().rev() {
                records.push(dialogue_fixture(source, Role::User, "repeat", Some("next")));
                records.push(dialogue_fixture(
                    source,
                    Role::Assistant,
                    "same\nreply",
                    Some("next"),
                ));
            }
            write_rollout(&file, &records);
            let session = Codex::default().parse(&file).unwrap();
            let actual = session
                .messages
                .iter()
                .map(|m| (m.role, m.text.as_str()))
                .collect::<Vec<_>>();
            assert_eq!(
                actual,
                [(Role::User, "repeat"), (Role::Assistant, "same\nreply")].repeat(3),
                "source order {order:?}"
            );
        }
    }

    #[test]
    fn dialogue_adopts_untagged_mirrors_before_late_turn_ids() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("rollout-test.jsonl");
        let mut records = Vec::new();
        for id in ["first", "next"] {
            records.extend([
                dialogue_fixture(1, Role::User, "repeat", None),
                dialogue_fixture(1, Role::Assistant, "same reply", None),
                dialogue_fixture(0, Role::User, "repeat", None),
                dialogue_fixture(2, Role::User, "repeat", Some(id)),
                dialogue_fixture(2, Role::Assistant, "same reply", Some(id)),
                dialogue_fixture(0, Role::Assistant, "same reply", None),
            ]);
        }
        write_rollout(&file, &records);
        let s = Codex::default().parse(&file).unwrap();
        let actual = s
            .messages
            .iter()
            .map(|m| (m.role, m.text.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            [(Role::User, "repeat"), (Role::Assistant, "same reply")].repeat(2)
        );
    }

    #[test]
    fn declined_and_failed_file_changes_cancel_the_raw_patch_mirror() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("rollout-test.jsonl");
        let call = |id, path| {
            json!({
                "type":"response_item","payload":{"type":"custom_tool_call","name":"apply_patch",
                    "call_id":id,"input":format!("*** Begin Patch\n*** Add File: {path}\n+content\n*** End Patch")}
            })
        };
        let result = json!({
            "type":"response_item","payload":{"type":"custom_tool_call_output",
                "call_id":"legacy","output":"patch rejected by user"}
        });
        let settle = |id, status, path: &str| {
            json!({
                "type":"event_msg","payload":{"type":"item_completed","item":{
                    "type":"FileChange","id":id,"status":status,"changes":{path:{}}}}
            })
        };
        let records = [
            call("declined", "declined.rs"),
            settle("declined", "declined", "declined.rs"),
            settle("failed", "failed", "failed.rs"),
            call("failed", "failed.rs"),
            call("legacy", "legacy.rs"),
            result,
            call("success", "success.rs"),
            settle("success", "completed", "success.rs"),
        ];
        write_rollout(&file, &records);
        for window in [None, Some((140, 30, 30))] {
            crate::util::TEST_WINDOW.with(|w| w.set(window));
            let session = Codex::default().parse(&file).unwrap();
            crate::util::TEST_WINDOW.with(|w| w.set(None));
            assert_eq!(session.touched, ["success.rs"]);
            assert_eq!(session.operational_edited_paths, ["success.rs"]);
        }
    }

    #[test]
    fn dialogue_reconciles_attachment_labels_and_legacy_chunk_concatenation() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("rollout-test.jsonl");
        let mut response = dialogue_fixture(0, Role::User, "Describe this", Some("one"));
        response["payload"]["content"] = json!([
            {"type":"input_text","text":"Describe this"},
            {"type":"input_text","text":"<audio name=[Audio #1] path=\"/local.wav\">"},
            {"type":"input_audio","input_audio":{"data":"QUJD","format":"wav"}},
            {"type":"input_text","text":"</audio>"}
        ]);
        let mut completion = dialogue_fixture(2, Role::User, "Describe this", Some("one"));
        completion["payload"]["item"]["content"] = json!([
            {"type":"text","text":"Describe this"},
            {"type":"localAudio","path":"/local.wav"}
        ]);
        let attachment = [
            response,
            completion,
            dialogue_fixture(1, Role::User, "Describe this", Some("one")),
        ];
        let chunks = [
            dialogue_fixture(0, Role::User, "Explain this\ncode", Some("two")),
            dialogue_fixture(1, Role::User, "Explain thiscode", Some("two")),
            dialogue_fixture(2, Role::User, "Explain this\ncode", Some("two")),
        ];
        for order in [[0, 1, 2], [1, 2, 0], [2, 0, 1]] {
            let records = order
                .iter()
                .map(|&i| attachment[i].clone())
                .chain(order.iter().map(|&i| chunks[i].clone()))
                .collect::<Vec<_>>();
            write_rollout(&file, &records);
            for window in [None, Some((150, 30, 30))] {
                crate::util::TEST_WINDOW.with(|w| w.set(window));
                let session = Codex::default().parse(&file).unwrap();
                crate::util::TEST_WINDOW.with(|w| w.set(None));
                let prompts = session
                    .messages
                    .iter()
                    .map(|m| m.text.as_str())
                    .collect::<Vec<_>>();
                let first_chunks = if order[0] == 1 {
                    "Explain thiscode"
                } else {
                    "Explain this\ncode"
                };
                assert_eq!(prompts, ["Describe this", first_chunks]);
            }
        }
    }

    #[test]
    fn windowed_dialogue_budget_excludes_media_and_mirrors() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("rollout-test.jsonl");
        let first = "A".repeat(95);
        let mut response = dialogue_fixture(0, Role::User, &first, Some("one"));
        response["payload"]["content"].as_array_mut().unwrap().extend([
            json!({"type":"input_image","image_url":format!("data:image/png;base64,{}", "A".repeat(8000))}),
            json!({"type":"input_audio","input_audio":{"data":"A".repeat(8000)}}),
        ]);
        let records = [
            json!({"type":"session_meta","payload":{"cwd":"/repo"}}),
            response,
            dialogue_fixture(1, Role::User, &first, Some("one")),
            dialogue_fixture(2, Role::User, &first, Some("one")),
            dialogue_fixture(2, Role::Assistant, "middle reply", Some("one")),
            dialogue_fixture(2, Role::User, "later middle prompt", Some("two")),
            json!({"type":"response_item","payload":{"type":"function_call_output","output":"tail".repeat(100)}}),
        ];
        write_rollout(&file, &records);
        crate::util::TEST_WINDOW.with(|w| w.set(Some((240, 60, 60))));
        crate::util::TEST_MIDDLE_BYTES.with(|w| w.set(Some(150)));
        let session = Codex::default().parse(&file).unwrap();
        crate::util::TEST_WINDOW.with(|w| w.set(None));
        crate::util::TEST_MIDDLE_BYTES.with(|w| w.set(None));
        assert!(session.title.starts_with("[large]"));
        assert_eq!(
            session
                .messages
                .iter()
                .map(|m| m.text.as_str())
                .collect::<Vec<_>>(),
            [first.as_str(), "middle reply", "later middle prompt"]
        );
    }

    #[test]
    fn any_file_change_disables_legacy_operational_patch_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("rollout-test.jsonl");
        let legacy = [
            json!({"type":"response_item","payload":{"type":"custom_tool_call","call_id":"x","name":"apply_patch",
                "input":"*** Begin Patch\n*** Update File: raw  \n@@\n-old\n+new\n*** End Patch"}}),
            json!({"type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"x","output":"Success"}}),
        ];
        write_rollout(&file, &legacy);
        let adapter = Codex::default();
        assert_eq!(
            adapter.parse(&file).unwrap().operational_edited_paths,
            ["raw  "]
        );
        for item in [
            json!({"type":"event_msg","payload":{"type":"item_completed","item":{"type":"FileChange","status":"failed","changes":{"unrelated spelling":{}}}}}),
            json!({"type":"event_msg","payload":{"type":"item_completed","item":{"type":"FileChange","status":"declined","changes":{"raw  ":{}}}}}),
            json!({"type":"event_msg","payload":{"type":"item_completed","item":{"type":"FileChange"}}}),
            json!({"type":"session_meta","payload":{"history_mode":"paginated"}}),
        ] {
            let mut records = legacy.to_vec();
            records.push(item);
            write_rollout(&file, &records);
            let s = adapter.parse(&file).unwrap();
            assert!(
                s.operational_edited_paths.is_empty(),
                "{:?}",
                s.operational_edited_paths
            );
            assert_eq!(s.touched, ["raw"]);
        }
    }

    #[test]
    fn windowed_paginated_rollout_separates_large_payloads_from_metadata_and_dialogue() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("rollout-test.jsonl");
        let records = vec![
            json!({"type":"session_meta","payload":{"cwd":" "}}),
            json!({"type":"compacted","payload":{"replacement_history":[{"type":"message","role":"user","content":[{"text":"synthetic"}]}]}}),
            json!({"type":"session_meta","payload":{"cwd":"/middle-only  ","history_mode":"paginated"}}),
            json!({"type":"event_msg","payload":{"type":"item_completed","item":{"type":"CommandExecution","output":"x".repeat(255)}}}),
            json!({"type":"event_msg","payload":{"type":"item_completed","item":{"type":"McpToolCall","output":"x".repeat(255)}}}),
            json!({"type":"event_msg","payload":{"type":"item_completed","item":{"type":"FileChange","status":"completed","changes":{"done  ":{"diff":"x".repeat(770)}}}}}),
            json!({"type":"event_msg","payload":{"type":"item_completed","item":{"type":"FileChange","status":"failed","changes":{"failed":{}}}}}),
            dialogue_fixture(2, Role::User, "middle prompt", None),
            dialogue_fixture(2, Role::Assistant, "middle reply", None),
            json!({"type":"response_item","payload":{"type":"function_call_output","output":"end".repeat(100)}}),
        ];
        let mut lines = records.iter().map(Value::to_string).collect::<Vec<_>>();
        let payload = &records[7]["payload"];
        lines[7] = format!(r#"{{ "payload" : {payload}, "type" : "event_msg" }}"#);
        std::fs::write(&file, lines.join("\n")).unwrap();
        crate::util::TEST_WINDOW.with(|w| w.set(Some((400, 70, 70))));
        crate::util::TEST_MIDDLE_BYTES.with(|w| w.set(Some(400)));
        let session = Codex::default().parse(&file).unwrap();
        crate::util::TEST_WINDOW.with(|w| w.set(None));
        crate::util::TEST_MIDDLE_BYTES.with(|w| w.set(None));
        assert!(session.title.starts_with("[large]"));
        assert!(session.compacted_prefix);
        assert_eq!(session.raw_cwd.as_deref(), Some("/middle-only  "));
        assert_eq!(session.operational_edited_paths, ["done  "]);
        assert_eq!(session.touched, ["done"]);
        assert_eq!(session.edits[0].path, "done");
        assert_eq!(
            session
                .messages
                .iter()
                .map(|m| m.text.as_str())
                .collect::<Vec<_>>(),
            ["middle prompt", "middle reply"]
        );
    }

    #[test]
    fn compaction_markers_only_describe_missing_prefixes() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("rollout-test.jsonl");
        let prompt = dialogue_fixture(1, Role::User, "real prompt", None);
        for marker in [
            json!({"type":"compacted","payload":{"replacement_history":[{"type":"message","role":"user","content":[{"text":"synthetic"}]}]}}),
            json!({"type":"response_item","payload":{"type":"compaction","encrypted_content":"secret"}}),
            json!({"type":"event_msg","payload":{"type":"item_completed","item":{"type":"ContextCompaction"}}}),
        ] {
            for (records, prefix) in [
                (vec![marker.clone(), prompt.clone()], true),
                (vec![prompt.clone(), marker], false),
            ] {
                write_rollout(&file, &records);
                let session = Codex::default().parse(&file).unwrap();
                assert_eq!(session.compacted_prefix, prefix);
                assert_eq!(session.raw_cwd, None);
                assert_eq!(session.messages.len(), 1);
                assert_eq!(session.messages[0].text, "real prompt");
            }
        }
    }

    /// An embedder points one adapter at each Codex install. The adapter must
    /// find that install's rollouts and claim only that install's rows for
    /// deletion reconciliation.
    #[test]
    fn in_home_discovers_that_installs_rollouts_and_scopes_reconciliation() {
        let home = tempfile::tempdir().unwrap();
        let day = home
            .path()
            .join("sessions")
            .join("2026")
            .join("01")
            .join("01");
        std::fs::create_dir_all(&day).unwrap();
        let file = day.join("rollout-2026-01-01T10-00-00-abc.jsonl");
        std::fs::write(
            &file,
            "{\"type\":\"session_meta\",\"payload\":{\"cwd\":\"/repo\"}}\n",
        )
        .unwrap();

        let adapter = Codex::in_home(home.path());
        let found = adapter.discover();
        assert!(!found.had_error);
        assert_eq!(found.files, vec![file.clone()]);

        let scope = adapter
            .reconcile_scope()
            .expect("an explicit install is scoped");
        assert!(
            file.to_string_lossy().starts_with(&scope),
            "{scope} must be a prefix of the discovered {}",
            file.display()
        );
        assert_eq!(
            Codex::default().reconcile_scope(),
            crate::adapters::root_scope(Codex::default().root().as_deref()),
            "the stock adapter speaks only for its own install"
        );
    }

    /// Codex marks a rollout a sub-agent's in its session_meta
    /// (`thread_source: "subagent"`, `source.subagent`). The adapter called
    /// every rollout a main session, so on one machine 2,925 of 4,725
    /// rollouts - sub-agent threads whose task arrives encrypted - filled
    /// `list`, `projects` and the recall hook as "(no user prompt)" rows.
    #[test]
    fn a_subagent_rollout_is_parsed_as_a_subagent() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, meta: &str| {
            let file = dir.path().join(name);
            std::fs::write(
                &file,
                format!(
                    "{{\"timestamp\":\"2026-10-07T00:00:00Z\",\"type\":\"session_meta\",\"payload\":{meta}}}\n\
                     {{\"timestamp\":\"2026-10-07T00:00:01Z\",\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":\"working\"}}]}}}}\n"
                ),
            )
            .unwrap();
            file
        };
        let main = write("main.jsonl", r#"{"cwd":"/repo","source":"cli"}"#);
        let by_thread = write(
            "sub1.jsonl",
            r#"{"cwd":"/repo","thread_source":"subagent","agent_path":"/root/worker"}"#,
        );
        let by_source = write(
            "sub2.jsonl",
            r#"{"cwd":"/repo","source":{"subagent":{"thread_spawn":{"depth":1}}}}"#,
        );
        let codex = Codex::default();
        assert!(!codex.parse(&main).unwrap().subagent);
        assert!(codex.parse(&by_thread).unwrap().subagent);
        assert!(codex.parse(&by_source).unwrap().subagent);
    }

    /// A patch Codex did not apply edited nothing. apply_patch checks every
    /// hunk before it writes, so a call that failed verification, or that the
    /// user aborted, wrote none of the files it names. Each output below is a
    /// shape real rollouts carry.
    fn failed_patch_rollout() -> String {
        let patch = |id: &str, name: &str, input: &str| {
            json!({"type":"response_item","payload":{"type":"custom_tool_call","call_id":id,"name":name,"input":input}}).to_string()
        };
        let output = |id: &str, output: Value| {
            json!({"type":"response_item","payload":{"type":"custom_tool_call_output","call_id":id,"output":output}}).to_string()
        };
        let one = |path: &str| {
            format!("*** Begin Patch\n*** Update File: {path}\n@@\n-a\n+b\n*** End Patch")
        };
        let script_failed = |error: &str| {
            json!([
                {"type":"input_text","text":"Script failed\nWall time 0.3 seconds\nOutput:\n"},
                {"type":"input_text","text":format!("Script error:\n{error}")}
            ])
        };
        [
            json!({"type":"session_meta","payload":{"cwd":"/repo"}}).to_string(),
            patch(
                "verify",
                "apply_patch",
                "*** Begin Patch\n*** Update File: src/a.rs\n@@\n-a\n+b\n*** Update File: src/b.rs\n@@\n-a\n+b\n*** End Patch",
            ),
            output(
                "verify",
                json!("apply_patch verification failed: Failed to find expected lines in src/b.rs:\na"),
            ),
            patch("exit", "apply_patch", &one("src/c.rs")),
            output(
                "exit",
                json!(r#"{"output":"Failed to find expected lines in src/c.rs","metadata":{"exit_code":1,"duration_seconds":0.0}}"#),
            ),
            patch("text", "apply_patch", &one("src/t.rs")),
            output(
                "text",
                json!("Exit code: 1\nWall time: 0 seconds\nOutput:\nFailed to find expected lines in src/t.rs"),
            ),
            patch("process", "apply_patch", &one("src/u.rs")),
            output(
                "process",
                json!("Process exited with code 1\nOutput:\nFailed to read file to update src/u.rs"),
            ),
            patch("abort", "apply_patch", &one("src/d.rs")),
            output("abort", json!("aborted by user after 4.2s")),
            patch(
                "script",
                "exec",
                &format!("text(await tools.apply_patch({:?}))", one("src/e.rs")),
            ),
            output(
                "script",
                script_failed("apply_patch verification failed: Failed to find expected lines in src/e.rs:\na"),
            ),
            // Applied: these stay.
            patch("ok", "apply_patch", &one("src/ok.rs")),
            output(
                "ok",
                json!(r#"{"output":"Success. Updated the following files:\nM src/ok.rs\n","metadata":{"exit_code":0,"duration_seconds":0.0}}"#),
            ),
            patch("pending", "apply_patch", &one("src/pending.rs")),
            // A script that failed elsewhere may have applied its patch, and of
            // two patches in one script the first may have applied.
            patch(
                "other",
                "exec",
                &format!("text(await tools.apply_patch({:?})); boom()", one("src/f.rs")),
            ),
            output("other", script_failed("ReferenceError: boom is not defined")),
            // A script that applied its patch and then printed the phrase.
            patch(
                "printed",
                "exec",
                &format!("text(await tools.apply_patch({:?})); text(log)", one("src/p.rs")),
            ),
            output(
                "printed",
                json!([
                    {"type":"input_text","text":"Script completed\nWall time 0.2 seconds\nOutput:\n"},
                    {"type":"input_text","text":"Success. Updated the following files:\nM src/p.rs\nlog: apply_patch verification failed: Failed to find expected lines"}
                ]),
            ),
            patch(
                "two",
                "exec",
                &format!(
                    "await tools.apply_patch({:?}); await tools.apply_patch({:?})",
                    one("src/g.rs"),
                    one("src/h.rs")
                ),
            ),
            output(
                "two",
                script_failed("apply_patch verification failed: Failed to find expected lines in src/h.rs:\na"),
            ),
            // Failed once, applied on the retry.
            patch("first", "apply_patch", &one("src/r.rs")),
            output(
                "first",
                json!("apply_patch verification failed: Failed to find expected lines in src/r.rs:\na"),
            ),
            patch("retry", "apply_patch", &one("src/r.rs")),
            output(
                "retry",
                json!(r#"{"output":"Success. Updated the following files:\nM src/r.rs\n","metadata":{"exit_code":0}}"#),
            ),
        ]
        .join("\n")
            + "\n"
    }

    const APPLIED: [&str; 7] = [
        "src/f.rs",
        "src/g.rs",
        "src/h.rs",
        "src/ok.rs",
        "src/p.rs",
        "src/pending.rs",
        "src/r.rs",
    ];

    fn sorted(mut paths: Vec<String>) -> Vec<String> {
        paths.sort();
        paths
    }

    #[test]
    fn a_patch_codex_did_not_apply_edited_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("rollout-2026-10-01T00-00-00-abc.jsonl");
        std::fs::write(&file, failed_patch_rollout()).unwrap();
        let session = Codex::default().parse(&file).unwrap();
        assert_eq!(sorted(session.touched), APPLIED);
    }

    /// The same when the patches sit in the middle of a windowed rollout.
    #[test]
    fn a_windowed_rollout_drops_the_patches_codex_did_not_apply() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("rollout-2026-10-01T00-00-00-abc.jsonl");
        let filler = |i: usize| {
            json!({"type":"response_item","payload":{"type":"function_call_output","call_id":format!("x{i}"),"output":"x".repeat(200)}}).to_string() + "\n"
        };
        let mut body = (0..10).map(filler).collect::<String>();
        body.push_str(&failed_patch_rollout());
        body.extend((10..20).map(filler));
        std::fs::write(&file, body).unwrap();
        crate::util::TEST_WINDOW.with(|w| w.set(Some((2_000, 600, 600))));
        let session = Codex::default().parse(&file).unwrap();
        crate::util::TEST_WINDOW.with(|w| w.set(None));
        assert!(session.title.starts_with("[large]"), "{}", session.title);
        assert_eq!(sorted(session.touched), APPLIED);
    }

    /// The same for a Codex rollout: its patches in the middle of a windowed
    /// file are still the files it edited.
    #[test]
    fn a_windowed_rollout_keeps_every_file_it_patched() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("rollout-2026-10-01T00-00-00-abc.jsonl");
        let mut body =
            String::from("{\"type\":\"session_meta\",\"payload\":{\"cwd\":\"/repo\"}}\n");
        for i in 0..30 {
            body.push_str(&format!(
                "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"custom_tool_call\",\"call_id\":\"c{i}\",\"name\":\"apply_patch\",\"input\":\"*** Begin Patch\\n*** Update File: src/f{i}.rs\\n@@\\n-a\\n+b\\n*** End Patch\"}}}}\n"
            ));
            body.push_str(&format!(
                "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"function_call\",\"call_id\":\"s{i}\",\"name\":\"shell\",\"arguments\":\"{{\\\"cmd\\\":\\\"apply_patch <<EOF\\\\n*** Begin Patch\\\\n*** Add File: src/g{i}.rs\\\\n+x\\\\n*** End Patch\\\\nEOF\\\"}}\"}}}}\n"
            ));
        }
        std::fs::write(&file, body).unwrap();
        crate::util::TEST_WINDOW.with(|w| w.set(Some((2_000, 600, 600))));
        let session = Codex::default().parse(&file).unwrap();
        crate::util::TEST_WINDOW.with(|w| w.set(None));
        assert!(session.title.starts_with("[large]"), "{}", session.title);
        assert_eq!(session.touched.len(), 60, "{:?}", session.touched);
        assert!(session.touched.contains(&"src/f15.rs".to_string()));
        assert!(session.touched.contains(&"src/g15.rs".to_string()));
    }

    /// Only conversation is kept from the middle. A `compacted` record carries
    /// the whole replacement history, so it contains `"type":"message"` too;
    /// kept, 205 of them filled the 64 MB budget of a real 1.1 GB rollout and
    /// 5025 of its 6960 conversation lines were dropped. A tool's output that
    /// merely mentions `user_message` is not conversation either.
    #[test]
    fn only_conversation_records_are_kept_from_the_middle() {
        let t = |line: &str| dialogue_route(serde_json::from_str::<Routing>(line).unwrap().tags());
        assert!(t(
            r#"{"timestamp":"t","type":"event_msg","payload":{"type":"user_message","message":"hi"}}"#
        ));
        assert!(t(
            r#"{"timestamp":"t","type":"event_msg","payload":{"type":"agent_message","message":"ok"}}"#
        ));
        assert!(t(
            r#"{"timestamp":"t","type":"response_item","payload":{"type":"message","role":"user","content":[]}}"#
        ));
        assert!(t(
            r#"{"type":"response_item","payload":{"type":"message","role":"assistant","content":[]}}"#
        ));
        // Older rollouts wrote a message at the top level.
        assert!(t(
            r#"{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}"#
        ));
        assert!(!t(
            r#"{"timestamp":"t","type":"compacted","payload":{"message":"","replacement_history":[{"type":"message","role":"user","content":[]}]}}"#
        ));
        assert!(!t(
            r#"{"timestamp":"t","type":"response_item","payload":{"type":"function_call_output","call_id":"c","output":{"content":[{"type":"message","text":"from a tool"}]}}}"#
        ));
    }

    /// The same for a Codex rollout's conversation.
    #[test]
    fn a_windowed_rollout_keeps_the_conversation_in_between() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("rollout-2026-10-01T00-00-00-abc.jsonl");
        let mut body =
            String::from("{\"type\":\"session_meta\",\"payload\":{\"cwd\":\"/repo\"}}\n");
        for i in 0..30 {
            body.push_str(&format!(
                "{{\"type\":\"event_msg\",\"payload\":{{\"type\":\"user_message\",\"message\":\"question number {i}\"}}}}\n"
            ));
            body.push_str(&format!(
                "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"input_text\",\"text\":\"question number {i}\"}}]}}}}\n"
            ));
            body.push_str(&format!(
                "{{\"type\":\"response_item\",\"payload\":{{\"type\":\"function_call_output\",\"call_id\":\"c{i}\",\"output\":\"bulky output {i}\"}}}}\n"
            ));
            body.push_str(&format!(
                "{{\"type\":\"event_msg\",\"payload\":{{\"type\":\"agent_message\",\"message\":\"answer number {i}\"}}}}\n"
            ));
        }
        std::fs::write(&file, body).unwrap();
        crate::util::TEST_WINDOW.with(|w| w.set(Some((2_000, 600, 600))));
        let session = Codex::default().parse(&file).unwrap();
        crate::util::TEST_WINDOW.with(|w| w.set(None));
        assert!(session.title.starts_with("[large]"), "{}", session.title);
        let text = |role: Role| -> Vec<String> {
            session
                .messages
                .iter()
                .filter(|m| m.role == role)
                .map(|m| m.text.clone())
                .collect()
        };
        let questions: Vec<String> = (0..30).map(|i| format!("question number {i}")).collect();
        let answers: Vec<String> = (0..30).map(|i| format!("answer number {i}")).collect();
        assert_eq!(text(Role::User), questions, "each prompt once");
        assert_eq!(text(Role::Assistant), answers);
    }
}
