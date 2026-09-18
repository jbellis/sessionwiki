//! Find the matching passages inside one already-loaded session.
//!
//! `search` finds sessions; this finds the messages within a session, the way
//! `grep` finds lines within a file. Matching is a case-insensitive substring
//! search over NFC-normalised, redacted text. That reproduces what the index
//! found: its full-text table uses a trigram tokenizer, which is substring
//! matching for queries of three or more characters, and shorter queries
//! already go through a `LIKE` scan. Redaction is the same `redact` pass
//! `brief_markdown` makes, so a credential that never reaches a briefing never
//! reaches a grep hit either.

use chrono::{DateTime, Utc};

use crate::model::{Role, Session};

/// How much of a session a grep run returns.
#[derive(Debug, Clone)]
pub struct GrepOpts {
    /// Messages of context kept on each side of a matching message.
    pub context_messages: usize,
    /// Characters kept per message, windowed around its first match with a
    /// quarter of the budget as lead-in. 0 keeps the whole message.
    pub chars: usize,
    /// Stop after this many matching messages. `None` returns all of them.
    pub max_matches: Option<usize>,
    /// Roles whose messages may anchor a hit. Messages of other roles still
    /// appear as context around a hit.
    pub anchor_roles: Vec<Role>,
}

impl Default for GrepOpts {
    fn default() -> Self {
        Self {
            context_messages: 0,
            chars: 0,
            max_matches: None,
            anchor_roles: vec![Role::User, Role::Assistant, Role::Tool],
        }
    }
}

/// One message returned by a grep run: either a match or context beside one.
#[derive(Debug, Clone)]
pub struct GrepHit {
    /// The message's index within the session.
    pub i: usize,
    pub role: Role,
    pub ts: Option<DateTime<Utc>>,
    /// The message text as returned: redacted, NFC-normalised, trimmed, and
    /// windowed to `GrepOpts::chars`.
    pub text: String,
    /// Byte ranges of the matches inside `text`. Empty for a context message.
    pub matches: Vec<(usize, usize)>,
    /// True when `text` is a window of a longer message.
    pub truncated: bool,
    /// Messages skipped since the previous returned message. Non-zero only on
    /// the first message of a group.
    pub omitted_before: usize,
}

/// The result of one grep run over one session.
#[derive(Debug, Clone, Default)]
pub struct GrepResult {
    pub hits: Vec<GrepHit>,
    /// Messages after the last returned one that were not shown.
    pub omitted_after: usize,
}

/// The passages of `session` that contain `pattern`.
///
/// Every matching message comes back with `opts.context_messages` neighbours on
/// each side; overlapping groups are merged and each group's first message says
/// how many messages were skipped before it.
pub fn grep_session(session: &Session, pattern: &str, opts: &GrepOpts) -> GrepResult {
    let needle = crate::util::nfc(pattern.trim()).to_lowercase();
    if needle.is_empty() || session.messages.is_empty() {
        return GrepResult::default();
    }
    let texts: Vec<String> = session
        .messages
        .iter()
        .map(|message| crate::redact::redact(&crate::util::nfc(message.text.trim())).into_owned())
        .collect();
    let found: Vec<Vec<(usize, usize)>> = texts
        .iter()
        .zip(&session.messages)
        .map(|(text, message)| {
            if opts.anchor_roles.contains(&message.role) {
                matches_in(text, &needle)
            } else {
                Vec::new()
            }
        })
        .collect();

    // Merge each match's context window into groups of consecutive messages.
    // Windows one apart are merged too: "0 messages omitted" is noise.
    let last = texts.len() - 1;
    let anchors = (0..texts.len())
        .filter(|index| !found[*index].is_empty())
        .take(opts.max_matches.unwrap_or(usize::MAX));
    let mut groups: Vec<(usize, usize)> = Vec::new();
    for index in anchors {
        let start = index.saturating_sub(opts.context_messages);
        let end = (index + opts.context_messages).min(last);
        match groups.last_mut() {
            Some(previous) if start <= previous.1 + 1 => previous.1 = previous.1.max(end),
            _ => groups.push((start, end)),
        }
    }
    if groups.is_empty() {
        return GrepResult::default();
    }

    let mut hits: Vec<GrepHit> = Vec::new();
    let mut previous_end: Option<usize> = None;
    for (start, end) in &groups {
        let omitted = match previous_end {
            Some(previous) => start - previous - 1,
            None => *start,
        };
        for index in *start..=*end {
            let (text, matches, truncated) = excerpt(&texts[index], &found[index], opts.chars);
            hits.push(GrepHit {
                i: index,
                role: session.messages[index].role,
                ts: session.messages[index].ts,
                text,
                matches,
                truncated,
                omitted_before: if index == *start { omitted } else { 0 },
            });
        }
        previous_end = Some(*end);
    }
    GrepResult {
        hits,
        omitted_after: last - previous_end.unwrap_or(last),
    }
}

/// Byte ranges of every non-overlapping case-insensitive occurrence of an
/// already-lowercased needle.
///
/// Lowercasing can change a string's length (`İ` lowercases to two chars), so
/// the search carries a map from each lowercased byte back to the byte that
/// starts the character it came from. The returned ranges are therefore
/// offsets into `text` itself, on character boundaries.
fn matches_in(text: &str, needle: &str) -> Vec<(usize, usize)> {
    let mut lowered = String::with_capacity(text.len());
    let mut origin: Vec<usize> = Vec::with_capacity(text.len() + 1);
    for (index, character) in text.char_indices() {
        let before = lowered.len();
        lowered.extend(character.to_lowercase());
        origin.resize(origin.len() + (lowered.len() - before), index);
    }
    origin.push(text.len());

    let mut hits: Vec<(usize, usize)> = Vec::new();
    let mut from = 0;
    while let Some(offset) = lowered[from..].find(needle) {
        let start = from + offset;
        from = start + needle.len();
        let begin = origin[start];
        let mut end = origin[from];
        if end <= begin {
            // The whole match sat inside one character's lowercase expansion.
            end = text[begin..]
                .chars()
                .next()
                .map_or(begin, |character| begin + character.len_utf8());
        }
        hits.push((begin, end));
    }
    hits
}

/// One message capped at `chars` characters, keeping the window around its
/// first match, with the hit ranges rebased onto what is kept.
fn excerpt(
    text: &str,
    hits: &[(usize, usize)],
    chars: usize,
) -> (String, Vec<(usize, usize)>, bool) {
    let total = text.chars().count();
    if chars == 0 || total <= chars {
        return (text.to_owned(), hits.to_vec(), false);
    }
    // A quarter of the budget of lead-in, so the hit reads in context rather
    // than starting the excerpt.
    let first = hits
        .first()
        .map_or(0, |(start, _)| text[..*start].chars().count());
    let mut window_start = first.saturating_sub(chars / 4);
    window_start = window_start.min(total - chars);
    let begin = byte_of_char(text, window_start);
    let end = byte_of_char(text, window_start + chars);
    let kept = hits
        .iter()
        .filter_map(|(start, stop)| {
            let start = (*start).max(begin);
            let stop = (*stop).min(end);
            if start < stop {
                Some((start - begin, stop - begin))
            } else {
                None
            }
        })
        .collect();
    (text[begin..end].to_owned(), kept, true)
}

fn byte_of_char(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .nth(char_index)
        .map_or(text.len(), |(offset, _)| offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Message;
    use std::path::PathBuf;

    fn session(messages: Vec<(Role, &str)>) -> Session {
        Session {
            id: "0123456789abcdef".into(),
            tool: "claude-code",
            path: PathBuf::from("/sessions/0123456789abcdef"),
            project: "/home/dev/project".into(),
            started: DateTime::from_timestamp_millis(1_700_000_000_000),
            ended: None,
            title: "the session".into(),
            subagent: false,
            messages: messages
                .into_iter()
                .map(|(role, text)| Message {
                    role,
                    text: text.to_owned(),
                    ts: None,
                })
                .collect(),
            touched: Vec::new(),
            edits: Vec::new(),
        }
    }

    fn opts(context_messages: usize, chars: usize) -> GrepOpts {
        GrepOpts {
            context_messages,
            chars,
            ..Default::default()
        }
    }

    /// A hit is found whatever the case of the pattern or of the transcript,
    /// and the reported range covers the matched text in the returned message.
    #[test]
    fn locates_case_insensitive_matches() {
        let session = session(vec![
            (Role::User, "Make the Tests green"),
            (Role::Assistant, "the tests are green now"),
        ]);

        let found = grep_session(&session, "TESTS", &opts(0, 4_000));

        assert_eq!(found.hits.len(), 2, "both messages contain the pattern");
        assert_eq!(found.hits[0].role, Role::User);
        assert_eq!(found.hits[0].i, 0);
        let (start, end) = found.hits[0].matches[0];
        assert_eq!(&found.hits[0].text[start..end], "Tests");
        let (start, end) = found.hits[1].matches[0];
        assert_eq!(&found.hits[1].text[start..end], "tests");
        assert!(!found.hits[0].truncated);
        assert_eq!(found.omitted_after, 0);
    }

    /// Context messages come back around each hit, with the gap between two
    /// groups counted rather than silently closed.
    #[test]
    fn keeps_context_and_marks_omissions() {
        let session = session(vec![
            (Role::User, "zero"),
            (Role::Assistant, "one needle one"),
            (Role::Tool, "two"),
            (Role::User, "three"),
            (Role::Assistant, "four"),
            (Role::Tool, "five"),
            (Role::User, "six needle six"),
            (Role::Assistant, "seven"),
            (Role::User, "eight"),
        ]);

        let found = grep_session(&session, "needle", &opts(1, 4_000));

        let shown: Vec<(usize, Role, &str, usize)> = found
            .hits
            .iter()
            .map(|hit| (hit.i, hit.role, hit.text.as_str(), hit.omitted_before))
            .collect();
        assert_eq!(
            shown,
            vec![
                (0, Role::User, "zero", 0),
                (1, Role::Assistant, "one needle one", 0),
                (2, Role::Tool, "two", 0),
                (5, Role::Tool, "five", 2),
                (6, Role::User, "six needle six", 0),
                (7, Role::Assistant, "seven", 0),
            ]
        );
        assert_eq!(found.omitted_after, 1, "the last message is not shown");
        assert!(found.hits[0].matches.is_empty(), "context has no matches");
    }

    /// With Tool left out of the anchor roles, a pattern that only occurs in
    /// tool output finds nothing, and a tool message beside a real match still
    /// comes back as context.
    #[test]
    fn excluded_roles_never_anchor_but_still_give_context() {
        let session = session(vec![
            (Role::User, "make it build"),
            (Role::Tool, "cargo build --needle"),
            (Role::Assistant, "it builds"),
        ]);
        let without_tools = GrepOpts {
            context_messages: 1,
            chars: 4_000,
            anchor_roles: vec![Role::User, Role::Assistant],
            ..Default::default()
        };

        let only_in_a_tool = grep_session(&session, "needle", &without_tools);
        assert!(
            only_in_a_tool.hits.is_empty(),
            "tool output must not anchor a passage, got {:?}",
            only_in_a_tool.hits
        );

        let beside_a_match = grep_session(&session, "builds", &without_tools);
        let shown: Vec<(Role, bool)> = beside_a_match
            .hits
            .iter()
            .map(|hit| (hit.role, !hit.matches.is_empty()))
            .collect();
        assert_eq!(
            shown,
            vec![(Role::Tool, false), (Role::Assistant, true)],
            "a tool message is still context around a real match"
        );

        let all_roles = grep_session(&session, "needle", &opts(1, 4_000));
        assert_eq!(
            all_roles.hits.len(),
            3,
            "the default anchors on tool output too"
        );
    }

    /// A long message is cut down to the caller's budget around its first hit,
    /// not from the start, so the match is always in what comes back.
    #[test]
    fn a_window_keeps_the_first_hit() {
        let filler = "x".repeat(4_000);
        let session = session(vec![(Role::User, &format!("{filler} needle {filler}"))]);

        let found = grep_session(&session, "needle", &opts(0, 100));

        let hit = &found.hits[0];
        assert!(hit.truncated);
        assert_eq!(hit.text.chars().count(), 100);
        assert_eq!(hit.matches.len(), 1, "the windowed text keeps its hit");
        let (start, end) = hit.matches[0];
        assert_eq!(&hit.text[start..end], "needle");
        assert!(
            start >= 20,
            "the window keeps lead-in before the hit, got {start}"
        );
    }

    /// `max_matches` stops the run after that many matching messages, while
    /// the context after the last one still comes back.
    #[test]
    fn max_matches_stops_after_that_many_matching_messages() {
        let session = session(vec![
            (Role::User, "needle one"),
            (Role::Assistant, "plain"),
            (Role::User, "needle two"),
            (Role::Assistant, "after"),
            (Role::User, "needle three"),
        ]);

        let found = grep_session(
            &session,
            "needle",
            &GrepOpts {
                context_messages: 1,
                max_matches: Some(2),
                ..Default::default()
            },
        );

        assert_eq!(
            found.hits.iter().map(|hit| hit.i).collect::<Vec<_>>(),
            vec![0, 1, 2, 3],
            "two matches plus their context, and nothing from the third"
        );
        assert_eq!(found.omitted_after, 1);
    }

    /// An empty pattern or an empty session finds nothing rather than
    /// everything.
    #[test]
    fn an_empty_pattern_finds_nothing() {
        let one = session(vec![(Role::User, "anything")]);
        let empty = session(Vec::new());
        assert!(grep_session(&one, "   ", &GrepOpts::default())
            .hits
            .is_empty());
        assert!(grep_session(&empty, "needle", &GrepOpts::default())
            .hits
            .is_empty());
    }
}
