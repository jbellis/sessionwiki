//! prodex discovery goes through the machine-wide bridges registry that
//! prodex >=0.11.0 maintains. Missing roots are normal (a registered repo may
//! be deleted); no registry means no prodex on this machine.

use rusqlite::Connection;
use sessionwiki::adapters;
use std::sync::Mutex;

static LOCK: Mutex<()> = Mutex::new(());

#[test]
fn discovers_tasks_across_registered_bridge_roots() {
    let _g = LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    // Two registered roots: one real (fixture copy), one deleted.
    let repo = dir.path().join("repo-a");
    let tasks = repo.join(".bridge").join("tasks");
    std::fs::create_dir_all(&tasks).unwrap();
    std::fs::write(
        tasks.join("task_20260707_090000_x.json"),
        r#"{"id":"task_20260707_090000_x","title":"t","prompt":"p"}"#,
    )
    .unwrap();
    let registry = dir.path().join("bridges.json");
    std::fs::write(
        &registry,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "roots": [repo.to_str().unwrap(), "/no/such/repo-anywhere"]
        }))
        .unwrap(),
    )
    .unwrap();
    std::env::set_var("SESSIONWIKI_PRODEX_REGISTRY", &registry);

    let adapter = adapters::by_name("prodex").unwrap();
    let d = adapter.discover();
    assert_eq!(d.files.len(), 1, "one task found, missing root skipped");
    assert!(!d.had_error, "a deleted registered repo is not an error");

    std::env::remove_var("SESSIONWIKI_PRODEX_REGISTRY");
}

#[test]
fn no_registry_means_no_prodex() {
    let _g = LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var(
        "SESSIONWIKI_PRODEX_REGISTRY",
        dir.path().join("absent.json"),
    );
    let adapter = adapters::by_name("prodex").unwrap();
    let d = adapter.discover();
    assert!(d.files.is_empty());
    assert!(!d.had_error);
    std::env::remove_var("SESSIONWIKI_PRODEX_REGISTRY");
}

#[test]
fn malformed_or_structurally_invalid_registry_is_partial_discovery() {
    let _g = LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let registry = dir.path().join("bridges.json");
    std::env::set_var("SESSIONWIKI_PRODEX_REGISTRY", &registry);

    for contents in [
        "{not json",
        r#"{"schema_version":1}"#,
        r#"{"schema_version":1,"roots":"not-an-array"}"#,
        r#"{"schema_version":1,"roots":[42]}"#,
    ] {
        std::fs::write(&registry, contents).unwrap();
        let d = adapters::by_name("prodex").unwrap().discover();
        assert!(d.files.is_empty());
        assert!(
            d.had_error,
            "invalid registry must suppress deletion reconciliation: {contents}"
        );
    }

    std::fs::write(&registry, [0xff]).unwrap();
    let d = adapters::by_name("prodex").unwrap().discover();
    assert!(d.had_error, "a registry read failure is partial discovery");

    std::fs::write(&registry, r#"{"roots":[]}"#).unwrap();
    let d = adapters::by_name("prodex").unwrap().discover();
    assert!(d.files.is_empty());
    assert!(
        !d.had_error,
        "a valid empty registry is a clean no-session store"
    );

    std::fs::remove_file(&registry).unwrap();
    std::fs::create_dir(&registry).unwrap();
    let d = adapters::by_name("prodex").unwrap().discover();
    assert!(d.had_error, "an unreadable registry is a partial discovery");
    std::env::remove_var("SESSIONWIKI_PRODEX_REGISTRY");
}

#[test]
fn invalid_registered_task_store_is_partial_discovery() {
    let _g = LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("repo");
    std::fs::create_dir_all(repo.join(".bridge")).unwrap();
    std::fs::write(repo.join(".bridge/tasks"), "not a directory").unwrap();
    let registry = dir.path().join("bridges.json");
    std::fs::write(
        &registry,
        serde_json::to_vec(&serde_json::json!({"roots": [repo]})).unwrap(),
    )
    .unwrap();
    std::env::set_var("SESSIONWIKI_PRODEX_REGISTRY", &registry);

    let d = adapters::by_name("prodex").unwrap().discover();
    assert!(d.files.is_empty());
    assert!(
        d.had_error,
        "a broken tasks store is not an empty clean store"
    );
    std::env::remove_var("SESSIONWIKI_PRODEX_REGISTRY");
}

fn archived(conn: &Connection) -> bool {
    conn.query_row(
        "SELECT archived_at IS NOT NULL FROM files WHERE tool='prodex'",
        [],
        |r| r.get(0),
    )
    .unwrap()
}

fn sync_prodex(data: &std::path::Path, registry: &std::path::Path) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_sessionwiki"))
        .args(["sync", "--tool", "prodex"])
        .env("SESSIONWIKI_DATA", data)
        .env("SESSIONWIKI_PRODEX_REGISTRY", registry)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "sync failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn corrupt_registry_does_not_archive_but_later_real_deletion_does() {
    let _g = LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    let repo = dir.path().join("repo");
    let tasks = repo.join(".bridge/tasks");
    std::fs::create_dir_all(&tasks).unwrap();
    let task = tasks.join("task_20260707_090000_x.json");
    std::fs::write(
        &task,
        r#"{"id":"task_20260707_090000_x","title":"t","prompt":"p"}"#,
    )
    .unwrap();
    let registry = dir.path().join("bridges.json");
    let valid = serde_json::to_vec(&serde_json::json!({"roots": [repo]})).unwrap();
    std::fs::write(&registry, &valid).unwrap();

    sync_prodex(&data, &registry);
    let conn = Connection::open(data.join("index.db")).unwrap();
    assert!(!archived(&conn), "valid task starts live");

    std::fs::write(&registry, "{not json").unwrap();
    sync_prodex(&data, &registry);
    assert!(
        !archived(&conn),
        "a corrupt registry is incomplete discovery, not task deletion"
    );

    std::fs::write(&registry, valid).unwrap();
    std::fs::remove_file(task).unwrap();
    sync_prodex(&data, &registry);
    assert!(
        archived(&conn),
        "a real deletion after clean discovery archives"
    );
}

#[test]
fn disappearing_registry_is_absent_store_not_task_deletion() {
    let _g = LOCK.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    let repo = dir.path().join("repo");
    let tasks = repo.join(".bridge/tasks");
    std::fs::create_dir_all(&tasks).unwrap();
    std::fs::write(
        tasks.join("task_20260707_090000_x.json"),
        r#"{"id":"task_20260707_090000_x","title":"t","prompt":"p"}"#,
    )
    .unwrap();
    let registry = dir.path().join("bridges.json");
    std::fs::write(
        &registry,
        serde_json::to_vec(&serde_json::json!({"roots": [repo]})).unwrap(),
    )
    .unwrap();

    sync_prodex(&data, &registry);
    let conn = Connection::open(data.join("index.db")).unwrap();
    assert!(!archived(&conn), "valid task starts live");

    std::fs::remove_file(&registry).unwrap();
    sync_prodex(&data, &registry);
    assert!(
        !archived(&conn),
        "a missing registry means the entire store is absent, not that every task was deleted"
    );
}
