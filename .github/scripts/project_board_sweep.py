#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Scheduled sweep of the Forge project board (org Project 54).

Sets two date fields on board items whose content is an issue of one of the six
Forge repositories:

- "Last Updated": the UTC date of the issue's own last change (updatedAt).
  Written only when it differs from the board value.
- "Work Started": when the issue is open, Status is "In Progress" and Work
  Started is empty, the UTC date of the Status value's last change. This is the
  rule of .github/scripts/bulk_update_issues/bulk_update_issues.py.

Standard library only. The token is read from PROJECT_BOARD_TOKEN and never
printed. No issue title or body is requested, so none can be printed.

Logs: this repository is public, so its run logs are public, and the board is
private. By default the script prints totals only (one number for all skipped
items). The skipped items by reason and the per-change lines are printed only
when SWEEP_LOG_DETAILS=1 (for a dry run on a private copy or a local run).
Error messages never hold item data.

Guard: before the first write of a run, the script reads that issue's updatedAt.
After the first successful write it reads it again after about 5 seconds, and a
second time about 30 seconds after the write. The run continues only when both
reads show no move. If it moved, a board write changes the issue's own
last-change time, and writing Last Updated would destroy its meaning. Then the
run stops at once with exit 8 and makes no further write. A failed read after a
write also stops the run with exit 8. A failed read before any write is exit 9:
no write was made, and the next run tries again.

Environment:
  PROJECT_BOARD_TOKEN  GitHub token with organization_projects write.
  SWEEP_MODE           "dry-run" (default: count the changes, write nothing) or "write".
  SWEEP_MAX_WRITES     Writes per run, 1 to 300. Default 20.
  SWEEP_LOG_DETAILS    "1" prints the skipped items by reason and one line per change.
                       Not set in the workflow.

Exit codes: 0 done; 2 bad setup or auth error; 3 rate limit (stopped, the next
run continues); 4 network or server error; 5 GraphQL error; 6 no issue content
readable (a missing permission); 7 the board's fields changed; 8 guard: a board
write moved the issue's updatedAt, or a check read after the first write failed
(disable the workflow and tell OSPO); 9 the read before the first write failed
(no write was made; the next run tries again).
"""

import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

API_URL = "https://api.github.com/graphql"
USER_AGENT = "tt-forge-project-board-sweep"

# Org Project 54 and its fields (ids from the board, read 2026-11-17).
PROJECT_ID = "PVT_kwDOA9MHEM4AjeTl"
LAST_UPDATED_NAME = "Last Updated"
LAST_UPDATED_FIELD_ID = "PVTF_lADOA9MHEM4AjeTlzgiiU18"
WORK_STARTED_NAME = "Work Started"
WORK_STARTED_FIELD_ID = "PVTF_lADOA9MHEM4AjeTlzgzZQtk"
STATUS_NAME = "Status"
IN_PROGRESS = "In Progress"

REPOSITORIES = frozenset(
    {
        "tenstorrent/tt-forge",
        "tenstorrent/tt-mlir",
        "tenstorrent/tt-forge-onnx",
        "tenstorrent/tt-torch",
        "tenstorrent/tt-xla",
        "tenstorrent/tt-blacksmith",
    }
)

DEFAULT_MAX_WRITES = 20  # per run, when SWEEP_MAX_WRITES is not set; the next run continues
MAX_WRITES_LIMIT = 300  # SWEEP_MAX_WRITES above this is refused
MAX_WRITE_ERRORS = 3  # write errors (not rate limits) skipped per run before the run stops
WRITE_INTERVAL_SECONDS = 1.0
GUARD_WAIT_SECONDS = 5.0  # wait after the first write before the guard's first check read
GUARD_SECOND_WAIT_SECONDS = 25.0  # then wait again: the second check read is about 30 s after the write
PAGE_SIZE = 100
MAX_PAGES = 400  # 40,000 items; the board has about 17,000
MIN_RATE_REMAINING = 1000  # stop early and leave GraphQL budget for other jobs
HTTP_TIMEOUT_SECONDS = 60
SERVER_RETRY_WAITS = (5, 15)  # bounded retries for 5xx and network errors only

EXIT_OK = 0
EXIT_SETUP = 2
EXIT_RATE_LIMIT = 3
EXIT_SERVER = 4
EXIT_GRAPHQL = 5
EXIT_NO_CONTENT = 6
EXIT_FIELDS = 7
EXIT_GUARD = 8
EXIT_GUARD_BASELINE = 9

GUARD_MOVED_MESSAGE = (
    "GUARD: the first board write of this run changed the issue's own last-change time (updatedAt). "
    "Then every Last Updated write would set the sweep date and destroy the field's meaning. "
    "No further write in this run. This job cannot remember this between runs: the team must disable "
    "the workflow 'Project board sweep' now (Actions, Project board sweep, Disable workflow) and tell OSPO "
    "(open an OSPO ticket)."
)
GUARD_READ_FAILED_MESSAGE = (
    "GUARD: could not read the issue again after the first board write, so this run cannot confirm that "
    "a write leaves the issue's last-change time alone. No further write in this run. The team must disable "
    "the workflow 'Project board sweep' now (Actions, Project board sweep, Disable workflow) and tell OSPO "
    "(open an OSPO ticket). OSPO checks it and says when to enable it again."
)
GUARD_BASELINE_FAILED_MESSAGE = (
    "could not read the issue before the first board write. No write was made. "
    "Stopped; the next scheduled run tries again. Cause: "
)
NO_CONTENT_AT_ALL_MESSAGE = (
    "every board item came back without content. The token very probably lacks a permission. "
    "No write was made. Tell OSPO."
)

FIELDS_QUERY = """
query($project: ID!) {
  node(id: $project) {
    ... on ProjectV2 {
      lastUpdated: field(name: "Last Updated") { ... on ProjectV2FieldCommon { id dataType } }
      workStarted: field(name: "Work Started") { ... on ProjectV2FieldCommon { id dataType } }
      status: field(name: "Status") { ... on ProjectV2FieldCommon { id dataType } }
    }
  }
  rateLimit { remaining resetAt }
}
"""

ITEMS_QUERY = """
query($project: ID!, $cursor: String, $first: Int!) {
  node(id: $project) {
    ... on ProjectV2 {
      items(first: $first, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          type
          isArchived
          lastUpdated: fieldValueByName(name: "Last Updated") { ... on ProjectV2ItemFieldDateValue { date } }
          workStarted: fieldValueByName(name: "Work Started") { ... on ProjectV2ItemFieldDateValue { date } }
          status: fieldValueByName(name: "Status") { ... on ProjectV2ItemFieldSingleSelectValue { name updatedAt } }
          content {
            __typename
            ... on Issue { number state updatedAt repository { nameWithOwner } }
          }
        }
      }
    }
  }
  rateLimit { remaining resetAt }
}
"""

ISSUE_UPDATED_QUERY = """
query($item: ID!) {
  node(id: $item) {
    ... on ProjectV2Item {
      content { ... on Issue { updatedAt } }
    }
  }
}
"""

UPDATE_MUTATION = """
mutation($project: ID!, $item: ID!, $field: ID!, $date: Date!) {
  updateProjectV2ItemFieldValue(input: {projectId: $project, itemId: $item, fieldId: $field, value: {date: $date}}) {
    clientMutationId
  }
}
"""


class SweepStop(Exception):
    """Stop the run with an exit code and a message (never a response body, never item data)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Pure logic (unit tested).
# ---------------------------------------------------------------------------


def utc_date(timestamp):
    """ISO 8601 timestamp (GitHub gives UTC with "Z") -> "YYYY-MM-DD" in UTC. None stays None."""
    if not timestamp:
        return None
    text = timestamp.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).date().isoformat()


def parse_max_writes(value):
    """SWEEP_MAX_WRITES -> int. Missing or empty: the default (20). Refuses non-numbers, 0 and values above 300."""
    if value is None or not str(value).strip():
        return DEFAULT_MAX_WRITES
    text = str(value).strip()
    if not text.isdigit():
        raise SweepStop(EXIT_SETUP, "SWEEP_MAX_WRITES must be a whole number from 1 to 300")
    number = int(text)
    if number < 1 or number > MAX_WRITES_LIMIT:
        raise SweepStop(EXIT_SETUP, f"SWEEP_MAX_WRITES={number} is refused; allowed: 1 to {MAX_WRITES_LIMIT}")
    return number


def classify(item):
    """Item kind: "issue", "other-repo-issue", "pull-request", "draft", "archived", "redacted" or "no-content"."""
    item_type = item.get("type")
    content = item.get("content")
    if item_type == "REDACTED":
        return "redacted"
    if item.get("isArchived"):
        return "archived"
    if content is None:
        return "no-content"
    typename = content.get("__typename")
    if typename == "PullRequest":
        return "pull-request"
    if typename == "DraftIssue":
        return "draft"
    if typename != "Issue":
        return "no-content"
    repo = (content.get("repository") or {}).get("nameWithOwner")
    if repo not in REPOSITORIES:
        return "other-repo-issue"
    return "issue"


def _date_value(value):
    if not value:
        return None
    return value.get("date") or None


def wanted_changes(item, today):
    """Changes wanted for one board item, as dicts. `today` is the UTC date used when Status has no time."""
    if classify(item) != "issue":
        return []
    content = item["content"]
    repo = content["repository"]["nameWithOwner"]
    number = content["number"]
    issue_updated = content.get("updatedAt")
    changes = []

    new_last = utc_date(issue_updated)
    old_last = _date_value(item.get("lastUpdated"))
    if new_last and new_last != old_last:
        changes.append(
            {
                "item": item["id"],
                "repo": repo,
                "number": number,
                "field": LAST_UPDATED_NAME,
                "field_id": LAST_UPDATED_FIELD_ID,
                "old": old_last,
                "new": new_last,
                "sort_key": issue_updated or "",
                "issue_updated": issue_updated,
            }
        )

    status = item.get("status") or {}
    old_started = _date_value(item.get("workStarted"))
    if content.get("state") == "OPEN" and status.get("name") == IN_PROGRESS and not old_started:
        # Same rule as bulk_update_issues.py: the date the Status value last changed,
        # today's date if GitHub gives no time.
        new_started = utc_date(status.get("updatedAt")) or today
        changes.append(
            {
                "item": item["id"],
                "repo": repo,
                "number": number,
                "field": WORK_STARTED_NAME,
                "field_id": WORK_STARTED_FIELD_ID,
                "old": None,
                "new": new_started,
                "sort_key": status.get("updatedAt") or "",
                "issue_updated": issue_updated,
            }
        )
    return changes


def unique_changes(changes):
    """Drop repeated (item, field) changes (an item seen twice across pages). Keeps the first."""
    seen = set()
    result = []
    for change in changes:
        key = (change["item"], change["field"])
        if key in seen:
            continue
        seen.add(key)
        result.append(change)
    return result


def order_and_cap(changes, cap=DEFAULT_MAX_WRITES):
    """Work Started first, then Last Updated with the most recently changed issues first. Return (do, left)."""
    started = [c for c in changes if c["field"] == WORK_STARTED_NAME]
    last = [c for c in changes if c["field"] == LAST_UPDATED_NAME]
    started.sort(key=lambda c: (c["sort_key"], c["repo"], c["number"]), reverse=True)
    last.sort(key=lambda c: (c["sort_key"], c["repo"], c["number"]), reverse=True)
    ordered = started + last
    return ordered[:cap], ordered[cap:]


def change_line(change, mode):
    """One detail line. Printed only when SWEEP_LOG_DETAILS=1."""
    verb = "WRITE" if mode == "write" else "WOULD-WRITE"
    return (
        f"{verb} repo={change['repo']} issue={change['number']} field=\"{change['field']}\" "
        f"old={change['old'] or 'empty'} new={change['new']}"
    )


def content_error_items(errors):
    """
    GraphQL errors that only say the content of some items cannot be read
    (FORBIDDEN or NOT_FOUND under items.nodes[i].content). Returns the set of
    item indexes, or None if any error is of another kind (then the run stops).
    """
    indexes = set()
    for err in errors:
        path = err.get("path") or []
        if err.get("type") not in ("FORBIDDEN", "NOT_FOUND"):
            return None
        if len(path) < 5 or path[:3] != ["node", "items", "nodes"] or not isinstance(path[3], int):
            return None
        if path[4] != "content":
            return None
        indexes.add(path[3])
    return indexes


def error_summary(errors):
    """Error types and a count only. GitHub's messages can name items, so they are not printed."""
    types = sorted({str(err.get("type", "unknown")) for err in errors})
    return f"{len(errors)} error(s), types={','.join(types)}"


def is_rate_limited(errors):
    return any(e.get("type") == "RATE_LIMITED" for e in errors)


# ---------------------------------------------------------------------------
# GitHub client.
# ---------------------------------------------------------------------------


class GitHubGraphQL:
    def __init__(self, token, sleep=None, urlopen=None):
        self._token = token
        self._sleep = sleep or time.sleep
        self._urlopen = urlopen or urllib.request.urlopen

    def post(self, query, variables, retry_server_errors=True):
        body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        waits = list(SERVER_RETRY_WAITS) if retry_server_errors else []
        while True:
            request = urllib.request.Request(API_URL, data=body, method="POST")
            request.add_header("Authorization", f"bearer {self._token}")
            request.add_header("Content-Type", "application/json")
            request.add_header("Accept", "application/json")
            request.add_header("User-Agent", USER_AGENT)
            try:
                with self._urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                    data = json.loads(response.read().decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("answer is not a JSON object")
                return data
            except urllib.error.HTTPError as err:
                status = err.code
                headers = err.headers or {}
                message = _github_message(err)
                retry_after = headers.get("Retry-After")
                if retry_after or status == 429 or "rate limit" in message.lower():
                    raise SweepStop(
                        EXIT_RATE_LIMIT,
                        f"rate limit (HTTP {status}, Retry-After={retry_after or 'none'}): {message}. "
                        "Stopped; the next scheduled run continues.",
                    )
                if status in (401, 403):
                    raise SweepStop(EXIT_SETUP, f"auth error (HTTP {status}): {message}")
                if status >= 500 and waits:
                    self._sleep(waits.pop(0))
                    continue
                raise SweepStop(EXIT_SERVER, f"HTTP {status}: {message}")
            except (OSError, http.client.HTTPException, ValueError) as err:
                # OSError includes URLError, ConnectionError, TimeoutError and ssl.SSLError.
                if waits:
                    self._sleep(waits.pop(0))
                    continue
                raise SweepStop(EXIT_SERVER, f"network or answer error: {type(err).__name__}")


def _github_message(err):
    """Only GitHub's short top-level error message, never the raw body."""
    try:
        data = json.loads(err.read().decode("utf-8"))
        return str(data.get("message", ""))[:200]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# The run.
# ---------------------------------------------------------------------------


def check_rate(data):
    rate = data.get("data", {}).get("rateLimit") or {}
    remaining = rate.get("remaining")
    if remaining is not None and remaining < MIN_RATE_REMAINING:
        raise SweepStop(
            EXIT_RATE_LIMIT,
            f"GraphQL budget low (remaining={remaining}, resets {rate.get('resetAt')}). "
            "Stopped; the next scheduled run continues.",
        )
    return remaining


def check_fields(client):
    data = client.post(FIELDS_QUERY, {"project": PROJECT_ID})
    if data.get("errors"):
        if is_rate_limited(data["errors"]):
            raise SweepStop(EXIT_RATE_LIMIT, "GraphQL RATE_LIMITED. Stopped; the next scheduled run continues.")
        raise SweepStop(EXIT_GRAPHQL, f"GraphQL error reading the board fields: {error_summary(data['errors'])}")
    node = (data.get("data") or {}).get("node")
    if not node:
        raise SweepStop(EXIT_FIELDS, "the board was not found or cannot be read with this token")
    expected = {
        "lastUpdated": (LAST_UPDATED_FIELD_ID, "DATE"),
        "workStarted": (WORK_STARTED_FIELD_ID, "DATE"),
    }
    for key, (field_id, data_type) in expected.items():
        field = node.get(key) or {}
        if field.get("id") != field_id or field.get("dataType") != data_type:
            raise SweepStop(EXIT_FIELDS, f"board field {key} is not {field_id} ({data_type}); no write was made")
    if (node.get("status") or {}).get("dataType") != "SINGLE_SELECT":
        raise SweepStop(EXIT_FIELDS, "board field Status is not a single select; no write was made")
    check_rate(data)


def walk_board(client, counts):
    """Yield every board item. Items whose content GitHub refuses get content None."""
    cursor = None
    for _ in range(MAX_PAGES):
        data = client.post(ITEMS_QUERY, {"project": PROJECT_ID, "cursor": cursor, "first": PAGE_SIZE})
        refused = set()
        if data.get("errors"):
            refused = content_error_items(data["errors"])
            if refused is None:
                if is_rate_limited(data["errors"]):
                    raise SweepStop(EXIT_RATE_LIMIT, "GraphQL RATE_LIMITED. Stopped; the next scheduled run continues.")
                raise SweepStop(EXIT_GRAPHQL, f"GraphQL error: {error_summary(data['errors'])}")
        node = (data.get("data") or {}).get("node")
        if not node or not node.get("items"):
            raise SweepStop(EXIT_GRAPHQL, "GraphQL answer has no board items")
        counts["pages"] += 1
        counts["refused"] += len(refused)
        for index, item in enumerate(node["items"]["nodes"] or []):
            if item is None:
                counts["items"] += 1
                counts["no-content"] += 1
                continue
            if index in refused:
                item = dict(item, content=None)
            yield item
        check_rate(data)
        page = node["items"]["pageInfo"] or {}
        if not page.get("hasNextPage"):
            return
        next_cursor = page.get("endCursor")
        if not next_cursor or next_cursor == cursor:
            raise SweepStop(EXIT_GRAPHQL, "GraphQL paging error: more pages but no new cursor; no write was made")
        cursor = next_cursor
    raise SweepStop(EXIT_GRAPHQL, f"more than {MAX_PAGES} pages; no write was made")


def read_issue_updated(client, item_id, after_write):
    """
    The issue's updatedAt for one board item, read now. On any failure: after a write of this
    run, SweepStop(EXIT_GUARD); before any write, SweepStop(EXIT_GUARD_BASELINE) (no write was made).
    """

    def failed(cause):
        if after_write:
            return SweepStop(EXIT_GUARD, GUARD_READ_FAILED_MESSAGE)
        return SweepStop(EXIT_GUARD_BASELINE, GUARD_BASELINE_FAILED_MESSAGE + cause)

    try:
        data = client.post(ISSUE_UPDATED_QUERY, {"item": item_id})
    except SweepStop as stop:
        raise failed(f"exit {stop.code}: {stop.message}")
    if data.get("errors"):
        raise failed(f"GraphQL {error_summary(data['errors'])}")
    content = ((data.get("data") or {}).get("node") or {}).get("content") or {}
    value = content.get("updatedAt")
    if not value:
        raise failed("empty answer")
    return value


def write_change(client, change):
    """Write one field. Returns None on success, or the error types (a short text) for a skipped error."""
    data = client.post(
        UPDATE_MUTATION,
        {"project": PROJECT_ID, "item": change["item"], "field": change["field_id"], "date": change["new"]},
        retry_server_errors=False,
    )
    if data.get("errors"):
        if is_rate_limited(data["errors"]):
            raise SweepStop(EXIT_RATE_LIMIT, "GraphQL RATE_LIMITED on a write. Stopped; the next run continues.")
        return error_summary(data["errors"])
    return None


SKIP_KINDS = ("other-repo-issue", "pull-request", "draft", "archived", "redacted", "no-content")


def run(client, mode, out=print, sleep=None, today=None, cap=DEFAULT_MAX_WRITES, details=False):
    sleep = sleep or time.sleep
    today = today or datetime.now(timezone.utc).date().isoformat()
    counts = {"pages": 0, "items": 0, "issue": 0, "refused": 0}
    counts.update({kind: 0 for kind in SKIP_KINDS})
    written = {LAST_UPDATED_NAME: 0, WORK_STARTED_NAME: 0}
    write_errors = 0
    changes = []
    left = []
    code = EXIT_OK
    try:
        check_fields(client)
        for item in walk_board(client, counts):
            counts["items"] += 1
            counts[classify(item)] += 1
            changes.extend(wanted_changes(item, today))
        changes = unique_changes(changes)

        skipped = sum(counts[kind] for kind in SKIP_KINDS)
        out(
            f"Board read: items={counts['items']} pages={counts['pages']} in_scope={counts['issue']} "
            f"skipped={skipped}"
        )
        if details:
            # By reason only on request: this is board data of a private board, and the run logs are public.
            out(
                f"Skipped by reason: other_repo={counts['other-repo-issue']} pull_request={counts['pull-request']} "
                f"draft={counts['draft']} archived={counts['archived']} redacted={counts['redacted']} "
                f"no_content={counts['no-content']} (of these refused={counts['refused']})"
            )
        if counts["items"] > 0 and counts["redacted"] + counts["no-content"] == counts["items"]:
            raise SweepStop(EXIT_NO_CONTENT, NO_CONTENT_AT_ALL_MESSAGE)
        if counts["items"] > 0 and counts["issue"] == 0:
            raise SweepStop(
                EXIT_NO_CONTENT,
                "no issue of the six repositories could be read on the board. No write was made. Tell OSPO.",
            )

        todo, left = order_and_cap(changes, cap)
        guard_done = False
        attempted = 0
        for index, change in enumerate(todo):
            if mode == "write":
                if index:
                    sleep(WRITE_INTERVAL_SECONDS)
                # A write attempt that returned an error counts as a write here (it may have been
                # applied). Only a failed read before any write attempt of this run is exit 9.
                baseline = None if guard_done else read_issue_updated(client, change["item"], attempted > 0)
                attempted += 1
                error = write_change(client, change)
                if error:
                    write_errors += 1
                    if details:
                        out(f"WRITE-ERROR repo={change['repo']} issue={change['number']} {error}")
                    if write_errors >= MAX_WRITE_ERRORS:
                        raise SweepStop(EXIT_GRAPHQL, f"{write_errors} write errors in this run; stopped")
                    continue
                written[change["field"]] += 1
                if details:
                    out(change_line(change, mode))
                if not guard_done:
                    # Two check reads: about 5 s and about 30 s after the write. Both must show no move.
                    for wait in (GUARD_WAIT_SECONDS, GUARD_SECOND_WAIT_SECONDS):
                        sleep(wait)
                        if read_issue_updated(client, change["item"], True) != baseline:
                            raise SweepStop(EXIT_GUARD, GUARD_MOVED_MESSAGE)
                    guard_done = True
            elif details:
                out(change_line(change, mode))
    except SweepStop as stop:
        out(f"STOP: {stop.message}")
        code = stop.code
    wanted_last = sum(1 for c in changes if c["field"] == LAST_UPDATED_NAME)
    wanted_started = len(changes) - wanted_last
    out(
        f"SUMMARY mode={mode} exit={code} cap={cap} items={counts['items']} "
        f"wanted_last_updated={wanted_last} wanted_work_started={wanted_started} "
        f"written_last_updated={written[LAST_UPDATED_NAME]} written_work_started={written[WORK_STARTED_NAME]} "
        f"write_errors={write_errors} left_for_next_run={len(left)}"
    )
    return code


def main():
    out = lambda line: print(line, flush=True)  # noqa: E731
    mode = os.environ.get("SWEEP_MODE", "dry-run").strip() or "dry-run"
    if mode not in ("dry-run", "write"):
        out("STOP: SWEEP_MODE must be dry-run or write")
        return EXIT_SETUP
    try:
        cap = parse_max_writes(os.environ.get("SWEEP_MAX_WRITES"))
    except SweepStop as stop:
        out(f"STOP: {stop.message}")
        return stop.code
    details = os.environ.get("SWEEP_LOG_DETAILS", "").strip() == "1"
    token = os.environ.get("PROJECT_BOARD_TOKEN", "")
    if not token:
        out("STOP: PROJECT_BOARD_TOKEN is empty")
        return EXIT_SETUP
    return run(GitHubGraphQL(token), mode, out=out, cap=cap, details=details)


if __name__ == "__main__":
    sys.exit(main())
