//! Optional swapdex integration: attribute each session to the ACCOUNT profile
//! that was active when it started, by reading swapdex's switch timeline
//! (read-only). No swapdex on the machine -> no events -> no badges, silently.
//!
//! The join mirrors swapdex's own `sessions` attribution: a session belongs to
//! the last activating event (`use`, `restore`, or `serve`) for its tool with ts
//! <= session start, unless a later `serve-off` cleared that state. A session
//! that predates every event stays unattributed (None) - a missing badge, never
//! a guess.

use serde_json::Value;
use std::path::PathBuf;

pub struct SwitchEvent {
    pub ts: i64,
    pub tool: String,
    pub account: String,
    /// What swapdex recorded: `use`/`restore` move where new sessions start,
    /// `serve` hands turns to an account without moving them, while `serve-off`
    /// explicitly clears that attribution until a later activating event.
    pub action: String,
}

/// swapdex's timeline location (same `dirs::data_dir` convention swapdex
/// uses). `SESSIONWIKI_SWAPDEX_TIMELINE` overrides for tests.
fn timeline_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("SESSIONWIKI_SWAPDEX_TIMELINE") {
        return Some(p.into());
    }
    Some(dirs::data_dir()?.join("swapdex").join("timeline.jsonl"))
}

/// Parse the timeline defensively: keep activating events and the explicit
/// `serve-off` boundary, and skip malformed lines.
///
/// `serve` counts. It was dropped here as "not a switch", which was true of the
/// event and false of the question: on a machine where switching goes through
/// swapdex's proxy, `serve` is the ONLY record of which account was live, so
/// every claude-code session on this one was badged with nothing while 190
/// serves named three accounts, and codex sessions carried a `use` from months
/// before the account actually changed.
///
/// The actions are listed rather than "anything swapdex writes": this reads
/// another program's file, and an action it adds later need not mean an account
/// went live.
pub fn load_events() -> Vec<SwitchEvent> {
    let Some(path) = timeline_path() else {
        return Vec::new();
    };
    // The bound was claimed and not taken: the whole file was read into memory
    // and only the PARSE was capped at 4000 lines. swapdex trims the timeline to
    // ~1000 events, but its size is another program's business, and a hand-edited
    // or runaway file would be read whole on every query. 1 MB is about eight
    // thousand of these lines, so the cap is far above what the producer writes
    // and far below what would hurt.
    let Ok(text) = crate::util::read_tail(&path, 1024 * 1024) else {
        return Vec::new();
    };
    parse_events(&text)
}

/// The parsing half, separated from the read so which lines survive it can be
/// tested without a file or an environment variable.
fn parse_events(text: &str) -> Vec<SwitchEvent> {
    let mut out = Vec::new();
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        // A line written before swapdex recorded actions carries no field, and
        // those were all switches. A present value of another JSON type is
        // malformed, not legacy.
        let action = match v.get("action") {
            None => "use",
            Some(Value::String(action)) => action,
            Some(_) => continue,
        };
        if !matches!(action, "use" | "restore" | "serve" | "serve-off") {
            continue;
        }
        if let (Some(ts), Some(tool), Some(account)) =
            (v["ts"].as_i64(), v["tool"].as_str(), v["account"].as_str())
        {
            let account: String = account.chars().filter(|c| !c.is_control()).collect();
            // Active-account events need a real badge. `serve-off` deliberately
            // carries an empty account in swapdex's cross-repo protocol; keep
            // it as a boundary rather than dropping or treating it as a badge.
            if action != "serve-off" && account.is_empty() {
                continue;
            }
            out.push(SwitchEvent {
                ts,
                tool: tool.to_string(),
                action: action.to_string(),
                // Strip control chars at the source: every consumer (CLI
                // badge, web, JSON) then gets a terminal-safe name.
                account,
            });
        }
    }
    out
}

/// The profile active when a session of `tool` started: the newest event for
/// that tool at or before it, whatever kind. Equal timestamps use append order,
/// matching the timeline's event-log semantics. `serve-off` returns None and
/// does not fall through to an older account. None also covers no prior event or
/// no swapdex timeline, so a missing badge is still never a guess.
pub fn account_for(
    events: &[SwitchEvent],
    tool: &str,
    started_rfc3339: Option<&str>,
) -> Option<String> {
    let started = chrono::DateTime::parse_from_rfc3339(started_rfc3339?)
        .ok()?
        .timestamp();
    events
        .iter()
        .enumerate()
        .filter(|(_, e)| e.tool == tool && e.ts <= started)
        .max_by_key(|(order, e)| (e.ts, *order))
        .and_then(|(_, e)| match e.action.as_str() {
            "use" | "restore" | "serve" => Some(e.account.clone()),
            "serve-off" => None,
            _ => None,
        })
}

/// Fill `account` on freshly queried rows. One timeline read per query call;
/// zero work when no swapdex timeline exists.
pub fn annotate<'a, I: IntoIterator<Item = &'a mut crate::index::SessionRow>>(rows: I) {
    let events = load_events();
    if events.is_empty() {
        return;
    }
    for r in rows {
        r.account = account_for(&events, &r.tool, r.started.as_deref());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(ts: i64, tool: &str, account: &str) -> SwitchEvent {
        SwitchEvent {
            ts,
            tool: tool.into(),
            account: account.into(),
            action: "use".into(),
        }
    }

    fn at(ts: i64) -> String {
        chrono::DateTime::from_timestamp(ts, 0)
            .unwrap()
            .to_rfc3339()
    }

    fn serve(ts: i64, tool: &str, account: &str) -> SwitchEvent {
        SwitchEvent {
            action: "serve".into(),
            ..ev(ts, tool, account)
        }
    }

    #[test]
    fn the_parser_keeps_serve_lines() {
        // Real lines from the timeline this was found on.
        let text = concat!(
            r#"{"ts":1787186784,"tool":"claude-code","account":"rnd","action":"serve"}"#,
            "\n",
            r#"{"ts":1784504808,"tool":"codex","account":"codex","action":"use"}"#,
            "\n",
            r#"{"ts":1788488191,"tool":"codex","account":"work","action":"serve","by":"swapdex"}"#,
            "\n",
            r#"{"ts":1,"tool":"codex","account":"legacy"}"#,
            "\n",
            r#"{"ts":3,"tool":"codex","account":"","action":"serve-off"}"#,
            "\n",
            r#"not json"#,
            "\n",
            r#"{"ts":2,"tool":"codex","account":"nope","action":"something-new"}"#,
            "\n",
            r#"{"ts":4,"tool":"codex","account":"typed-null","action":null}"#,
        );
        let events = parse_events(text);
        let kept: Vec<(&str, &str)> = events
            .iter()
            .map(|e| (e.action.as_str(), e.account.as_str()))
            .collect();
        assert_eq!(
            kept,
            vec![
                ("serve", "rnd"),
                ("use", "codex"),
                ("serve", "work"),
                ("use", "legacy"),
                ("serve-off", ""),
            ],
            "serve boundaries and legacy uses are kept; unknown actions are not"
        );
    }

    #[test]
    fn serve_off_clears_attribution_until_a_later_activation() {
        let off = SwitchEvent {
            ts: 200,
            tool: "codex".into(),
            account: String::new(),
            action: "serve-off".into(),
        };
        let events = vec![serve(100, "codex", "payer"), off];
        assert_eq!(
            account_for(&events, "codex", Some(&at(250))),
            None,
            "serve-off is a state boundary, not an empty account badge"
        );

        for action in ["use", "restore", "serve"] {
            let mut resumed = events
                .iter()
                .map(|e| SwitchEvent {
                    ts: e.ts,
                    tool: e.tool.clone(),
                    account: e.account.clone(),
                    action: e.action.clone(),
                })
                .collect::<Vec<_>>();
            resumed.push(SwitchEvent {
                ts: 300,
                tool: "codex".into(),
                account: "home".into(),
                action: action.into(),
            });
            assert_eq!(
                account_for(&resumed, "codex", Some(&at(350))).as_deref(),
                Some("home"),
                "a later {action} restores attribution"
            );
        }
    }

    #[test]
    fn equal_timestamps_use_append_order() {
        let events = parse_events(concat!(
            r#"{"ts":100,"tool":"codex","account":"payer","action":"serve"}"#,
            "\n",
            r#"{"ts":100,"tool":"codex","account":"","action":"serve-off"}"#,
            "\n",
            r#"{"ts":100,"tool":"codex","account":"home","action":"use"}"#,
        ));
        assert_eq!(
            account_for(&events[..2], "codex", Some(&at(100))),
            None,
            "the later appended serve-off wins a timestamp tie"
        );
        assert_eq!(
            account_for(&events, "codex", Some(&at(100))).as_deref(),
            Some("home"),
            "a still-later use at the same timestamp wins"
        );
    }

    #[test]
    fn a_serve_is_evidence_of_who_was_active() {
        // The machine this was found on: claude-code has never been `use`d -
        // switching goes through swapdex's proxy, which writes `serve` - and
        // codex's last `use` is months older than its last serve.
        let only_serves = vec![serve(200, "claude-code", "kong")];
        assert_eq!(
            account_for(&only_serves, "claude-code", Some(&at(300))).as_deref(),
            Some("kong"),
            "190 serves naming three accounts is not 'no information'"
        );

        let stale_use = vec![ev(100, "codex", "codex"), serve(200, "codex", "work")];
        assert_eq!(
            account_for(&stale_use, "codex", Some(&at(300))).as_deref(),
            Some("work"),
            "the newest evidence wins, not the oldest kind of event"
        );
        // Before any event for that tool, still no badge rather than a guess.
        assert_eq!(account_for(&stale_use, "codex", Some(&at(50))), None);
    }

    #[test]
    fn attributes_to_the_last_switch_before_start() {
        let events = vec![
            ev(100, "codex", "work"),
            ev(200, "codex", "personal"),
            ev(150, "claude-code", "work"),
        ];
        // codex session started at t=250 -> personal (the t=200 switch).
        let started = chrono::DateTime::from_timestamp(250, 0)
            .unwrap()
            .to_rfc3339();
        assert_eq!(
            account_for(&events, "codex", Some(&started)).as_deref(),
            Some("personal")
        );
        // claude session at t=250 -> work (its own tool's events only).
        assert_eq!(
            account_for(&events, "claude-code", Some(&started)).as_deref(),
            Some("work")
        );
        // A session that predates every switch stays unattributed.
        let early = chrono::DateTime::from_timestamp(50, 0)
            .unwrap()
            .to_rfc3339();
        assert_eq!(account_for(&events, "codex", Some(&early)), None);
        // Unknown start time -> None, never a guess.
        assert_eq!(account_for(&events, "codex", None), None);
    }
}
