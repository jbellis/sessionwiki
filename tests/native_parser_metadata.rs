use rusqlite::{params, Connection};
use serde_json::{json, Value};
use sessionwiki::adapters::{
    self, Adapter, Antigravity, ClaudeCode, Codex, CodexHistory, Dsh, GrokBuild, KimiCode, Muse,
};
use sessionwiki::model::Role;
use std::{fs, path::Path};

fn write_lines(path: &Path, rows: &[Value]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut body = rows
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    body.push('\n');
    fs::write(path, body).unwrap();
}

fn native_id(adapter: &dyn Adapter, path: &Path) -> Option<String> {
    adapter.native_id(path.to_str().unwrap())
}

#[test]
fn index_schema_stays_at_nine() {
    assert_eq!(sessionwiki::index::SCHEMA_VERSION, 9);
}

#[test]
fn claude_uses_literal_primary_cwd_and_completed_edits_per_log() {
    let tmp = tempfile::tempdir().unwrap();
    let adapter = ClaudeCode::in_home(tmp.path());
    let log = tmp.path().join("projects/sanitized/LitERAL.jsonl");
    let child = log.with_file_name("LitERAL/subagents/nested/agent.jsonl");
    let tool = |id, name, input| {
        json!({
            "type":"assistant","message":{"content":[{"type":"tool_use","id":id,"name":name,"input":input}]}
        })
    };
    let result = |id, error| {
        json!({
            "type":"user","message":{"content":[{"type":"tool_result","tool_use_id":id,"is_error":error}]}
        })
    };
    let summary = json!({"type":"user","isMeta":true,"message":{"isCompactSummary":true,"content":"hidden summary"}});
    let prompt = json!({"type":"user","message":{"content":"main prompt"}});
    write_lines(
        &log,
        &[
            json!({"type":"system","cwd":"  "}),
            summary.clone(),
            json!({"type":"file-history-delta","trackingPath":"/delta/a "}),
            json!({"type":"file-history-delta","trackingPath":"dir/b ","backup":{"realParentDir":"/backup"}}),
            tool("write", "Edit", json!({"file_path":"/main/a "})),
            tool("pending", "Write", json!({"file_path":"/pending"})),
            tool("failed", "NotebookEdit", json!({"notebook_path":"/failed"})),
            tool(
                "view",
                "str_replace_based_edit_tool",
                json!({"command":"view","path":"/read"}),
            ),
            tool(
                "replace",
                "str_replace_based_edit_tool",
                json!({"command":"str_replace","path":"/changed  "}),
            ),
            result("write", false),
            result("failed", true),
            result("view", false),
            result("replace", false),
            json!({"type":"system","cwd":"/Native/Path "}),
            prompt.clone(),
            json!({"type":"system","cwd":"/Later"}),
        ],
    );
    write_lines(
        &child,
        &[
            tool("write", "Write", json!({"file_path":"/unpaired-child"})),
            result("pending", false),
            tool("child", "Write", json!({"file_path":"/child "})),
            result("child", false),
            tool("dupe", "Write", json!({"file_path":"/main/a "})),
            result("dupe", false),
            json!({"type":"user","message":{"content":"child-only prompt"}}),
        ],
    );
    #[cfg(unix)]
    {
        let outside = tmp.path().join("escape.jsonl");
        write_lines(
            &outside,
            &[json!({"type":"file-history-delta","trackingPath":"/symlink"})],
        );
        std::os::unix::fs::symlink(&outside, child.with_file_name("escape.jsonl")).unwrap();
    }
    let s = adapter.parse(&log).unwrap();
    assert_eq!(s.raw_cwd.as_deref(), Some("/Native/Path "));
    assert_eq!(native_id(&adapter, &log).as_deref(), Some("LitERAL"));
    assert_eq!(
        s.operational_edited_paths,
        [
            "/delta/a ",
            "/backup/b ",
            "/main/a ",
            "/changed  ",
            "/child "
        ]
    );
    assert!(!s.touched.iter().any(|path| path == "/read"));
    assert!(s.compacted_prefix);
    let prompts: Vec<_> = s
        .messages
        .iter()
        .filter(|m| m.role == Role::User)
        .map(|m| m.text.as_str())
        .collect();
    assert_eq!(prompts, ["main prompt"]);
    write_lines(&log, &[prompt, summary]);
    let s = adapter.parse(&log).unwrap();
    assert!(!s.compacted_prefix);
    assert_eq!(s.raw_cwd, None);
    assert_eq!(s.project, "sanitized");
    assert_eq!(
        native_id(&adapter, &log.with_file_name("Missing.jsonl")).as_deref(),
        Some("Missing")
    );
}

#[test]
fn claude_multiedit_requires_a_success_result_in_its_own_log() {
    let tmp = tempfile::tempdir().unwrap();
    let main = tmp.path().join("session.jsonl");
    let child = tmp.path().join("session/subagents/nested/child.jsonl");
    let call = |id, path| {
        json!({"type":"assistant","message":{"content":[{
            "type":"tool_use","name":"MultiEdit","id":id,"input":{"file_path":path,"edits":[]}
        }]}})
    };
    let result = |id, is_error| {
        json!({"type":"user","message":{"content":[{
            "type":"tool_result","tool_use_id":id,"is_error":is_error
        }]}})
    };
    write_lines(
        &main,
        &[
            call("ok", "main "),
            result("ok", false),
            call("unpaired", "no-result"),
            call("error", "failed"),
            result("error", true),
        ],
    );
    write_lines(
        &child,
        &[
            result("unpaired", false),
            call("ok", "child "),
            result("ok", false),
        ],
    );
    let adapter = ClaudeCode::in_home(tmp.path());
    assert_eq!(
        adapter.parse(&main).unwrap().operational_edited_paths,
        ["main ", "child "]
    );
}

fn completed(item: Value) -> Value {
    json!({"type":"event_msg","payload":{"type":"item_completed","item":item}})
}

#[test]
fn codex_reads_paginated_items_and_literal_compressed_identity() {
    const SUFFIX: &str = "A1111111-AbCd-7890-aBcD-ABCDEF123456";
    let tmp = tempfile::tempdir().unwrap();
    let adapter = Codex::in_home(tmp.path());
    let path = tmp
        .path()
        .join(format!("rollout-2026-11-22T10-00-00-{SUFFIX}.jsonl.zst"));
    let rows = [
        json!({"type":"session_meta","payload":{"id":"Literal-Header-ID","cwd":" ","history_mode":"paginated"}}),
        json!({"type":"session_meta","payload":{"cwd":"/Native/Codex "}}),
        completed(json!({"type":"ContextCompaction","id":"compacted"})),
        completed(
            json!({"type":"UserMessage","content":[{"text":"first"},{"text":""},{"image":"skip"},{"text":"second"}]}),
        ),
        completed(json!({"type":"AgentMessage","content":[{"text":"reply"},{"text":"continued"}]})),
        completed(
            json!({"type":"FileChange","status":"completed","changes":{"z-last ":{"type":"update","unified_diff":"-old\n+new"},"a-first":{}}}),
        ),
        completed(json!({"type":"FileChange","status":"failed","changes":{"bad":{}}})),
        completed(json!({"type":"FileChange","status":"declined","changes":{"declined":{}}})),
        completed(json!({"type":"FileChange","changes":{"pending":{}}})),
        completed(json!({"type":"FileChange","status":"completed","changes":["array"]})),
    ];
    let lines = rows[..5].iter().map(Value::to_string).chain(
        [r#"{"type":"event_msg","payload":{"type":"item_completed","item":{"type":"FileChange","status":"completed","changes":{"z-last ":{"type":"update","unified_diff":"-old\n+new"},"a-first":{}}}}}"#.to_owned()]
    ).chain(rows[6..].iter().map(Value::to_string));
    let body = lines.collect::<Vec<_>>().join("\n");
    fs::write(
        &path,
        ruzstd::encoding::compress_to_vec(
            body.as_bytes(),
            ruzstd::encoding::CompressionLevel::Fastest,
        ),
    )
    .unwrap();
    let s = adapter.parse(&path).unwrap();
    assert_eq!(s.raw_cwd.as_deref(), Some("/Native/Codex "));
    assert_eq!(
        native_id(&adapter, &path).as_deref(),
        Some("Literal-Header-ID")
    );
    assert!(s.compacted_prefix);
    assert_eq!(
        s.messages
            .iter()
            .map(|m| (m.role, m.text.as_str()))
            .collect::<Vec<_>>(),
        [
            (Role::User, "first\nsecond"),
            (Role::Assistant, "reply\ncontinued")
        ]
    );
    assert_eq!(s.operational_edited_paths, ["z-last ", "a-first"]);
    assert_eq!(s.touched, ["z-last", "a-first"]);
    assert_eq!(s.edits.len(), 1);
    assert_eq!(s.edits[0].path, "z-last");
    assert!(s.edits[0].snippet.contains("+new"));
    fs::remove_file(&path).unwrap();
    assert_eq!(native_id(&adapter, &path).as_deref(), Some(SUFFIX));
    assert_eq!(
        native_id(&adapter, &path.with_file_name("rollout-invalid.jsonl.zst")),
        None
    );
}

#[test]
fn codex_history_keys_and_registry_cwd_are_literal() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("thread_history_1.sqlite");
    let conn = Connection::open(&db).unwrap();
    conn.execute_batch("CREATE TABLE thread_items (thread_id TEXT, rollout_ordinal INTEGER, created_at_ms INTEGER, item_json TEXT)").unwrap();
    let item = json!({"type":"fileChange","cwd":"/item ","status":"completed","changes":[{"path":"edited ","type":"update"}]});
    conn.execute(
        "INSERT INTO thread_items VALUES ('LitERAThread', 0, 1, ?)",
        [item.to_string()],
    )
    .unwrap();
    let state = Connection::open(tmp.path().join("state_1.sqlite")).unwrap();
    state
        .execute_batch(
            "CREATE TABLE threads (id TEXT, cwd TEXT, source TEXT);
        INSERT INTO threads VALUES ('LitERAThread', '/Native/Registry ', 'cli');",
        )
        .unwrap();
    let adapter = CodexHistory::in_home(tmp.path());
    let key = format!("{}\u{1f}LitERAThread", db.display());
    let s = adapter.parse_key(&key).unwrap();
    assert_eq!(adapter.native_id(&key).as_deref(), Some("LitERAThread"));
    assert_eq!(s.raw_cwd.as_deref(), Some("/Native/Registry "));
    assert_eq!(s.operational_edited_paths, ["edited "]);
    state.execute("UPDATE threads SET cwd = '  '", []).unwrap();
    assert_eq!(
        adapter.parse_key(&key).unwrap().raw_cwd.as_deref(),
        Some("/item ")
    );
    assert_eq!(
        adapter
            .native_id("/missing.sqlite\u{1f}Archived")
            .as_deref(),
        Some("Archived")
    );
    assert_eq!(adapter.native_id("/missing.sqlite"), None);
}

#[test]
fn kimi_recursive_wires_use_per_file_ids_and_literal_cwd() {
    let tmp = tempfile::tempdir().unwrap();
    let adapter = KimiCode::in_home(tmp.path());
    let dir = tmp.path().join("sessions/KiMi");
    let wire = dir.join("agents/main/wire.jsonl");
    let event = |ev| json!({"type":"context.append_loop_event","event":ev});
    let call =
        |id, args| event(json!({"type":"tool.call","toolCallId":id,"name":"Edit","args":args}));
    let result =
        |id, error| event(json!({"type":"tool.result","toolCallId":id,"result":{"isError":error}}));
    let prompt = |text| json!({"type":"turn.prompt","origin":{"kind":"user"},"input":text});
    let marker = json!({"type":"context.compacted"});
    write_lines(
        &wire,
        &[
            marker.clone(),
            prompt("main"),
            call("shared", json!({"path":"main ","file_path":"wrong"})),
            call("main-only", json!({"path":"unpaired"})),
            result("shared", false),
        ],
    );
    write_lines(
        &dir.join("agents/nested/x/edited.jsonl"),
        &[
            result("main-only", false),
            call("shared", json!({"path":"failed"})),
            result("shared", true),
            call("child", json!({"file_path":"child  "})),
            result("child", false),
            prompt("child"),
        ],
    );
    #[cfg(unix)]
    {
        let outside = tmp.path().join("outside");
        write_lines(
            &outside.join("escaped.jsonl"),
            &[
                call("escape", json!({"path":"symlink"})),
                result("escape", false),
            ],
        );
        std::os::unix::fs::symlink(&outside, dir.join("agents/escaped")).unwrap();
    }
    let state = dir.join("state.json");
    fs::write(
        &state,
        json!({"workDir":" ","cwd":"/Literal/../Kimi  "}).to_string(),
    )
    .unwrap();
    write_lines(
        &tmp.path().join("session_index.jsonl"),
        &[json!({"sessionId":"KiMi","sessionDir":dir,"workDir":"/index/../Other "})],
    );
    let s = adapter.parse(&dir).unwrap();
    assert_eq!(s.raw_cwd.as_deref(), Some("/Literal/../Kimi  "));
    assert_eq!(native_id(&adapter, &dir).as_deref(), Some("KiMi"));
    assert_eq!(s.operational_edited_paths, ["main ", "child  "]);
    assert!(s.compacted_prefix);
    assert_eq!(
        s.messages
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.text.as_str())
            .collect::<Vec<_>>(),
        ["main"]
    );
    fs::remove_file(state).unwrap();
    write_lines(&wire, &[prompt("main"), marker]);
    let s = adapter.parse(&dir).unwrap();
    assert_eq!(s.raw_cwd.as_deref(), Some("/index/../Other "));
    assert!(!s.compacted_prefix);
    assert_eq!(
        native_id(&adapter, &dir.with_file_name("Archived")).as_deref(),
        Some("Archived")
    );
}

#[test]
fn grok_thought_and_write_evidence_survive_native_compaction() {
    let tmp = tempfile::tempdir().unwrap();
    let adapter = GrokBuild::in_home(tmp.path());
    let dir = tmp.path().join("sessions/%2FDecoded%20Cwd/GroK-ID");
    let log = dir.join("chat_history.jsonl");
    let marker =
        json!({"type":"user","synthetic_reason":"compaction_meta","content":"hidden compaction"});
    let prompt = json!({"type":"user","content":"actual prompt"});
    write_lines(
        &log,
        &[
            marker.clone(),
            prompt.clone(),
            json!({"type":"reasoning","summary":[{"type":"summary_text","text":"Think"}]}),
            json!({"type":"assistant","tool_calls":[
                {"name":"search_replace","id":"ok","arguments":"{\"file_path\":\"ok \"}"},
                {"name":"search_replace","id":"fail","arguments":{"file_path":"failed"}},
                {"name":"search_replace","id":"pending","arguments":{"file_path":"pending"}}
            ]}),
            json!({"type":"tool_result","tool_call_id":"fail","is_error":true}),
            json!({"type":"tool_result","tool_call_id":"ok"}),
        ],
    );
    let summary = dir.join("summary.json");
    fs::write(&summary, json!({"info":{"cwd":"../Native  "}}).to_string()).unwrap();
    let s = adapter.parse(&dir).unwrap();
    assert_eq!(s.raw_cwd.as_deref(), Some("../Native  "));
    assert_eq!(native_id(&adapter, &dir).as_deref(), Some("GroK-ID"));
    assert_eq!(s.operational_edited_paths, ["ok "]);
    assert!(s.compacted_prefix);
    assert!(s
        .messages
        .iter()
        .any(|m| m.role == Role::Assistant && m.is_thought && m.text == "Think"));
    assert!(s.messages.iter().all(|m| m.text != "hidden compaction"));
    fs::remove_file(summary).unwrap();
    write_lines(&log, &[prompt, marker]);
    let s = adapter.parse(&dir).unwrap();
    assert_eq!(s.raw_cwd.as_deref(), Some("/Decoded Cwd"));
    assert!(!s.compacted_prefix);
}

#[test]
fn antigravity_cli_keys_use_the_conversation_and_native_workspace() {
    let tmp = tempfile::tempdir().unwrap();
    let adapter = Antigravity::cli_in_home(tmp.path());
    let dir = tmp.path().join("antigravity-cli/brain/CoNversation");
    let log = dir.join(".system_generated/logs/transcript_full.jsonl");
    let steps = [
        json!({"type":"PLANNER_RESPONSE","step_index":1,"tool_calls":[{"name":"write_to_file","arguments":{"Cwd":"/tool-cwd","TargetFile":"/edited "}}]}),
        json!({"type":"GENERIC","step_index":2,"status":"DONE","cwd":"/Native/Project "}),
    ];
    write_lines(&log, &steps);
    assert_eq!(native_id(&adapter, &log).as_deref(), Some("CoNversation"));
    assert_eq!(native_id(&adapter, &dir).as_deref(), Some("CoNversation"));
    let s = adapter.parse(&log).unwrap();
    assert_eq!(s.raw_cwd.as_deref(), Some("/Native/Project "));
    assert_eq!(s.operational_edited_paths, ["/edited "]);
    write_lines(&log, &steps[..1]);
    assert_eq!(adapter.parse(&log).unwrap().raw_cwd, None);
    fs::remove_file(&log).unwrap();
    assert_eq!(native_id(&adapter, &log).as_deref(), Some("CoNversation"));
}

#[test]
fn muse_private_copies_resolve_recorded_stream_id() {
    let tmp = tempfile::tempdir().unwrap();
    let adapter = Muse::in_sessions_root(tmp.path());
    let path = tmp.path().join("private/session.jsonl");
    write_lines(
        &path,
        &[json!({
            "schema_version":1,"stream":{"kind":"session","id":"Native-MuSE"},
            "payload_type":"runtime.session.metadata","payload_schema_version":1,
            "payload":{"record":{"workspace_root":"/Muse/Root "}}
        })],
    );
    assert_eq!(native_id(&adapter, &path).as_deref(), Some("Native-MuSE"));
    assert_eq!(
        adapter.parse(&path).unwrap().raw_cwd.as_deref(),
        Some("/Muse/Root ")
    );
    let gone = tmp.path().join("Archived-MuSE/session.jsonl");
    assert_eq!(native_id(&adapter, &gone).as_deref(), Some("Archived-MuSE"));
}

#[test]
fn dsh_native_decodes_ids_and_acp_uses_its_file_header() {
    let tmp = tempfile::tempdir().unwrap();
    let native = Dsh::in_home(tmp.path());
    let dir = tmp.path().join("sessions/project/ID~002F~D83D~DE00.ext");
    let log = dir.join("session.v2.jsonl");
    write_lines(
        &log,
        &[
            json!({"type":"session","version":2,"id":"ID/😀.ext","cwd":"/Dsh/Native ","createdAt":1,"delegationDepth":0,"isSeeded":false}),
            json!({"type":"tool/call","seq":1,"time":2,"data":{"callId":"x","name":"write","arguments":{"file_path":"dsh-edited ","content":"body"}}}),
        ],
    );
    assert_eq!(native_id(&native, &dir).as_deref(), Some("ID/😀.ext"));
    let s = native.parse(&dir).unwrap();
    assert_eq!(s.raw_cwd.as_deref(), Some("/Dsh/Native "));
    assert_eq!(s.operational_edited_paths, ["dsh-edited "]);
    fs::remove_file(log).unwrap();
    assert_eq!(native_id(&native, &dir).as_deref(), Some("ID/😀.ext"));
    assert_eq!(
        native_id(&native, &dir.with_file_name("bad~encoding")),
        None
    );
    let acp = Dsh::acp_default();
    let path = tmp.path().join("Archive-Stem.json");
    fs::write(&path,json!({"id":"ACP-Literal","cwd":"/Dsh/ACP ","messages":[{"role":"user","content":"hello"}]}).to_string()).unwrap();
    assert_eq!(native_id(&acp, &path).as_deref(), Some("ACP-Literal"));
    assert_eq!(
        acp.parse(&path).unwrap().raw_cwd.as_deref(),
        Some("/Dsh/ACP ")
    );
    fs::remove_file(&path).unwrap();
    assert_eq!(native_id(&acp, &path).as_deref(), Some("Archive-Stem"));
}

#[test]
fn opencode_directory_distinguishes_null_and_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let adapter = adapters::by_name("opencode").unwrap();
    let db = tmp.path().join("opencode.db");
    let conn = Connection::open(&db).unwrap();
    conn.execute_batch(
        "CREATE TABLE session(id TEXT PRIMARY KEY, parent_id TEXT, directory TEXT, title TEXT, time_created INTEGER, time_updated INTEGER);
         CREATE TABLE message(id TEXT, session_id TEXT, time_created INTEGER, data TEXT);
         CREATE TABLE part(id TEXT, message_id TEXT, session_id TEXT, time_created INTEGER, data TEXT);"
    ).unwrap();
    for (id, cwd) in [
        ("LitERAL", Some("/OpenCode/SQL ")),
        ("NoCwd", None),
        ("Empty", Some("")),
    ] {
        conn.execute(
            "INSERT INTO session VALUES (?1, NULL, ?2, 'title', 10, 20)",
            params![id, cwd],
        )
        .unwrap();
        let key = format!("{}\u{1f}{id}", db.display());
        assert_eq!(adapter.native_id(&key).as_deref(), Some(id));
        assert_eq!(adapter.parse_key(&key).unwrap().raw_cwd.as_deref(), cwd);
    }
    let legacy = tmp.path().join("Legacy-Key.json");
    fs::write(
        &legacy,
        json!({"id":"Legacy-Literal","directory":"/Legacy "} ).to_string(),
    )
    .unwrap();
    assert_eq!(
        native_id(adapter.as_ref(), &legacy).as_deref(),
        Some("Legacy-Literal")
    );
    assert_eq!(
        adapter.parse(&legacy).unwrap().raw_cwd.as_deref(),
        Some("/Legacy ")
    );
    fs::write(&legacy, r#"{"id":"Legacy-Literal"}"#).unwrap();
    assert_eq!(adapter.parse(&legacy).unwrap().raw_cwd, None);
    fs::remove_file(&legacy).unwrap();
    assert_eq!(
        native_id(adapter.as_ref(), &legacy).as_deref(),
        Some("Legacy-Key")
    );
    assert_eq!(
        adapter.native_id("missing.db\u{1f}Archive").as_deref(),
        Some("Archive")
    );
}
