use sessionwiki::adapters;
use sessionwiki::model::{Role, Session};
use std::path::{Path, PathBuf};

fn fake_openai_key(fill: char) -> String {
    format!("{}{}", "sk-", fill.to_string().repeat(45))
}

fn parse(tool: &str, path: &Path) -> Session {
    adapters::by_name(tool)
        .expect("adapter exists")
        .parse(path)
        .expect("fixture parses")
}

fn write_json(path: &Path, value: &serde_json::Value) {
    std::fs::create_dir_all(path.parent().expect("fixture has a parent")).unwrap();
    std::fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

fn prodex_task(root: &Path, id: &str, prompt: &str) -> PathBuf {
    let path = root.join(".bridge/tasks").join(format!("{id}.json"));
    write_json(
        &path,
        &serde_json::json!({
            "id": id,
            "title": "GPT Pro consult",
            "prompt": prompt,
        }),
    );
    path
}

#[test]
fn common_title_redacts_before_the_eighty_character_cap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rollout-test.jsonl");
    let secret = fake_openai_key('Z');
    let prompt = format!("{} {secret}", "T".repeat(65));
    let event = serde_json::json!({
        "type": "event_msg",
        "payload": {"type": "user_message", "message": prompt},
    });
    std::fs::write(&path, format!("{event}\n")).unwrap();

    let session = parse("codex", &path);

    assert!(
        !session.title.contains(&secret[..12]),
        "title: {}",
        session.title
    );
    assert_eq!(session.messages[0].text, prompt);
    assert!(session.messages[0].text.contains(&secret));
    assert!(std::fs::read_to_string(path).unwrap().contains(&secret));
}

#[test]
fn prodex_title_redacts_before_its_own_eighty_character_cap() {
    let dir = tempfile::tempdir().unwrap();
    let secret = fake_openai_key('P');
    let prompt = format!("{} {secret}", "T".repeat(65));
    let path = prodex_task(dir.path(), "task_20260914_120000_redaction-title", &prompt);

    let session = parse("prodex", &path);

    assert!(
        !session.title.contains(&secret[..12]),
        "title: {}",
        session.title
    );
    assert_eq!(session.messages[0].text, prompt);
    assert!(session.messages[0].text.contains(&secret));
    assert!(std::fs::read_to_string(path).unwrap().contains(&secret));
}

#[test]
fn prodex_redacts_a_multiline_pem_before_selecting_the_title_line() {
    let dir = tempfile::tempdir().unwrap();
    let prompt = concat!(
        "Inspect -----BEGIN OPENSSH PRIVATE KEY-----\n",
        "c3ludGhldGljLWtleS1ib2R5\n",
        "-----END OPENSSH PRIVATE KEY----- then preserve this suffix"
    );
    let path = prodex_task(dir.path(), "task_20260914_120100_multiline-title", prompt);

    let session = parse("prodex", &path);

    assert!(
        !session.title.contains("BEGIN OPENSSH"),
        "title: {}",
        session.title
    );
    assert!(
        session.title.contains("then preserve this suffix"),
        "title: {}",
        session.title
    );
    assert_eq!(session.messages[0].text, prompt);
}

#[test]
fn json_encoded_tool_preview_redacts_before_the_preview_cap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("continue-session.json");
    let secret = fake_openai_key('J');
    let raw_arguments = serde_json::json!({
        "provider": format!("{} {secret}", "R".repeat(275))
    })
    .to_string();
    write_json(
        &path,
        &serde_json::json!({
            "title": "New Session",
            "history": [
                {"message": {"role": "user", "content": "safe prompt"}},
                {"message": {
                    "role": "assistant",
                    "content": "done",
                    "toolCalls": [{
                        "function": {
                            "name": "provider_call",
                            "arguments": raw_arguments,
                        }
                    }]
                }}
            ]
        }),
    );

    let session = parse("continue", &path);
    let tool = session
        .messages
        .iter()
        .find(|message| message.role == Role::Tool)
        .expect("tool preview exists");

    assert!(
        !tool.text.contains(&secret[..10]),
        "tool preview: {}",
        tool.text
    );
    assert!(std::fs::read_to_string(path).unwrap().contains(&secret));
}

#[test]
fn claude_edit_snippet_redacts_before_the_snippet_cap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("claude-session.jsonl");
    let secret = fake_openai_key('E');
    let content = format!("{} {secret}", "N".repeat(190));
    let user = serde_json::json!({
        "type": "user",
        "message": {"content": "safe prompt"},
    });
    let assistant = serde_json::json!({
        "type": "assistant",
        "message": {"content": [{
            "type": "tool_use",
            "name": "Write",
            "input": {"file_path": "src/safe.rs", "content": content},
        }]},
    });
    std::fs::write(&path, format!("{user}\n{assistant}\n")).unwrap();

    let session = parse("claude-code", &path);

    assert_eq!(session.edits.len(), 1);
    assert!(
        !session.edits[0].snippet.contains(&secret[..8]),
        "edit snippet: {}",
        session.edits[0].snippet
    );
    assert!(std::fs::read_to_string(path).unwrap().contains(&secret));
}

#[test]
fn prodex_answer_redacts_before_the_sixty_four_kib_cap() {
    const ANSWER_CAP: usize = 64 * 1024;

    let dir = tempfile::tempdir().unwrap();
    let id = "task_20260914_120200_answer-boundary";
    let task = prodex_task(dir.path(), id, "safe prompt");
    let artifact = dir
        .path()
        .join(".bridge/artifacts/pro-consults")
        .join(format!("{id}.md"));
    let secret = fake_openai_key('A');
    let answer = format!("{} {secret}", "A".repeat(ANSWER_CAP - 12));
    std::fs::create_dir_all(artifact.parent().unwrap()).unwrap();
    std::fs::write(&artifact, &answer).unwrap();

    let session = parse("prodex", &task);
    let answer_message = session
        .messages
        .iter()
        .find(|message| message.role == Role::Assistant)
        .expect("answer artifact becomes an assistant message");

    assert!(
        !answer_message.text.contains(&secret[..8]),
        "bounded answer leaked a credential prefix"
    );
    assert!(std::fs::read_to_string(artifact).unwrap().contains(&secret));
}
