# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for project_board_sweep.py (no network).

Run from this folder: python3 -m unittest test_project_board_sweep
"""

import contextlib
import io
import json
import os
import unittest
import urllib.error
from unittest import mock

import project_board_sweep as sweep

FAKE_TOKEN = "ghs_FAKEtokenFORtestsONLY0123456789"


def issue_item(
    item_id="I1",
    repo="tenstorrent/tt-mlir",
    number=7,
    state="OPEN",
    updated="2026-11-17T10:00:00Z",
    last=None,
    started=None,
    status=None,
    status_updated=None,
):
    return {
        "id": item_id,
        "type": "ISSUE",
        "isArchived": False,
        "lastUpdated": {"date": last} if last else None,
        "workStarted": {"date": started} if started else None,
        "status": {"name": status, "updatedAt": status_updated} if status else None,
        "content": {
            "__typename": "Issue",
            "number": number,
            "state": state,
            "updatedAt": updated,
            "repository": {"nameWithOwner": repo},
        },
    }


FIELDS_OK = {
    "data": {
        "node": {
            "lastUpdated": {"id": sweep.LAST_UPDATED_FIELD_ID, "dataType": "DATE"},
            "workStarted": {"id": sweep.WORK_STARTED_FIELD_ID, "dataType": "DATE"},
            "status": {"id": "PVTSSF_x", "dataType": "SINGLE_SELECT"},
        },
        "rateLimit": {"remaining": 5000, "resetAt": "2026-11-17T23:00:00Z"},
    }
}


def page(items, has_next=False, cursor="c1", errors=None, remaining=4000):
    data = {
        "data": {
            "node": {"items": {"pageInfo": {"hasNextPage": has_next, "endCursor": cursor}, "nodes": items}},
            "rateLimit": {"remaining": remaining, "resetAt": "2026-11-17T23:00:00Z"},
        }
    }
    if errors:
        data["errors"] = errors
    return data


def issue_updated_answer(value):
    return {"data": {"node": {"content": {"updatedAt": value}}}}


class FakeClient:
    """
    Answers the fields query, then the given pages; records writes.
    The guard's read of one issue answers from `issue_reads` in order; when the
    list is empty it answers the item's walk value (nothing moved).
    """

    def __init__(self, pages, write_answer=None, issue_reads=None):
        self.pages = list(pages)
        self.writes = []
        self.reads = []
        self.calls = []
        self.issue_reads = list(issue_reads or [])
        self.write_answer = write_answer or {"data": {"updateProjectV2ItemFieldValue": {"clientMutationId": None}}}
        self.updated = {}
        for p in self.pages:
            for item in p["data"]["node"]["items"]["nodes"] or []:
                if item and item.get("content"):
                    self.updated[item["id"]] = item["content"].get("updatedAt")

    def post(self, query, variables, retry_server_errors=True):
        if query is sweep.FIELDS_QUERY:
            self.calls.append("fields")
            return FIELDS_OK
        if query is sweep.ITEMS_QUERY:
            self.calls.append("items")
            return self.pages.pop(0)
        if query is sweep.UPDATE_MUTATION:
            self.calls.append("write")
            self.writes.append(variables)
            answer = self.write_answer
            return answer(len(self.writes)) if callable(answer) else answer
        if query is sweep.ISSUE_UPDATED_QUERY:
            self.calls.append("read")
            self.reads.append(variables["item"])
            if self.issue_reads:
                answer = self.issue_reads.pop(0)
                if isinstance(answer, Exception):
                    raise answer
                return answer
            return issue_updated_answer(self.updated.get(variables["item"]))
        raise AssertionError("unexpected query")


def run_quiet(client, mode="write", **kwargs):
    lines = []
    kwargs.setdefault("today", "2026-11-17")
    kwargs.setdefault("sleep", lambda s: None)
    code = sweep.run(client, mode, out=lines.append, **kwargs)
    return code, lines


class UtcDateTest(unittest.TestCase):
    def test_z_timestamp(self):
        self.assertEqual(sweep.utc_date("2026-11-17T23:59:59Z"), "2026-11-17")

    def test_just_after_midnight_utc(self):
        self.assertEqual(sweep.utc_date("2026-11-18T00:00:01Z"), "2026-11-18")

    def test_offset_before_midnight_is_next_day_in_utc(self):
        # 20:30 at -05:00 is 01:30 UTC on the next day.
        self.assertEqual(sweep.utc_date("2026-11-17T20:30:00-05:00"), "2026-11-18")

    def test_offset_after_midnight_is_previous_day_in_utc(self):
        self.assertEqual(sweep.utc_date("2026-11-18T01:00:00+02:00"), "2026-11-17")

    def test_fraction_and_empty(self):
        self.assertEqual(sweep.utc_date("2026-01-01T00:00:00.123Z"), "2026-01-01")
        self.assertIsNone(sweep.utc_date(None))
        self.assertIsNone(sweep.utc_date(""))


class WantedChangesTest(unittest.TestCase):
    TODAY = "2026-11-17"

    def fields(self, item):
        return [(c["field"], c["old"], c["new"]) for c in sweep.wanted_changes(item, self.TODAY)]

    def test_last_updated_differs(self):
        item = issue_item(updated="2026-11-17T10:00:00Z", last="2026-11-01")
        self.assertEqual(self.fields(item), [("Last Updated", "2026-11-01", "2026-11-17")])

    def test_last_updated_same_no_write(self):
        item = issue_item(updated="2026-11-17T10:00:00Z", last="2026-11-17")
        self.assertEqual(self.fields(item), [])

    def test_last_updated_empty(self):
        item = issue_item(updated="2026-11-16T23:59:00Z")
        self.assertEqual(self.fields(item), [("Last Updated", None, "2026-11-16")])

    def test_closed_issue_gets_last_updated_only(self):
        item = issue_item(state="CLOSED", last="2026-01-01", status="In Progress", status_updated="2026-11-10T08:00Z")
        self.assertEqual(self.fields(item), [("Last Updated", "2026-01-01", "2026-11-17")])

    def test_work_started_set_from_status_time(self):
        item = issue_item(last="2026-11-17", status="In Progress", status_updated="2026-11-10T23:30:00Z")
        self.assertEqual(self.fields(item), [("Work Started", None, "2026-11-10")])

    def test_work_started_not_overwritten(self):
        item = issue_item(last="2026-11-17", status="In Progress", status_updated="2026-11-10T08Z", started="x")
        self.assertEqual(self.fields(item), [])

    def test_work_started_other_status(self):
        for status in ("Backlog", "In Review", "Done", None):
            item = issue_item(last="2026-11-17", status=status, status_updated="2026-11-10T08:00:00Z")
            self.assertEqual(self.fields(item), [], status)

    def test_work_started_status_without_time_uses_today(self):
        item = issue_item(last="2026-11-17", status="In Progress", status_updated=None)
        self.assertEqual(self.fields(item), [("Work Started", None, self.TODAY)])

    def test_both_fields(self):
        item = issue_item(last="2026-11-01", status="In Progress", status_updated="2026-11-12T01:00:00Z")
        self.assertEqual(
            self.fields(item),
            [("Last Updated", "2026-11-01", "2026-11-17"), ("Work Started", None, "2026-11-12")],
        )

    def test_out_of_scope_items(self):
        other = issue_item(repo="tenstorrent/tt-metal", last="2020-01-01")
        pr = dict(issue_item(), type="PULL_REQUEST", content={"__typename": "PullRequest"})
        draft = dict(issue_item(), type="DRAFT_ISSUE", content={"__typename": "DraftIssue"})
        archived = dict(issue_item(last="2020-01-01"), isArchived=True)
        redacted = {"id": "R", "type": "REDACTED", "content": None}
        empty = {"id": "N", "type": "ISSUE", "content": None}
        self.assertEqual(sweep.classify(other), "other-repo-issue")
        self.assertEqual(sweep.classify(pr), "pull-request")
        self.assertEqual(sweep.classify(draft), "draft")
        self.assertEqual(sweep.classify(archived), "archived")
        self.assertEqual(sweep.classify(redacted), "redacted")
        self.assertEqual(sweep.classify(empty), "no-content")
        for item in (other, pr, draft, archived, redacted, empty):
            self.assertEqual(sweep.wanted_changes(item, self.TODAY), [])

    def test_all_six_repositories_in_scope(self):
        for repo in sweep.REPOSITORIES:
            self.assertEqual(sweep.classify(issue_item(repo=repo)), "issue")
        self.assertEqual(len(sweep.REPOSITORIES), 6)


class CapTest(unittest.TestCase):
    def test_cap_and_order(self):
        changes = []
        for n in range(5):
            item = issue_item(item_id=f"I{n}", number=n, updated=f"2026-11-1{n}T00:00:00Z", last="2026-01-01")
            changes += sweep.wanted_changes(item, "2026-11-17")
        item = issue_item(item_id="W", number=99, last="2026-11-17", updated="2026-11-17T01:00:00Z")
        item["status"] = {"name": "In Progress", "updatedAt": "2026-11-02T00:00:00Z"}
        changes += sweep.wanted_changes(item, "2026-11-17")
        todo, left = sweep.order_and_cap(changes, cap=3)
        self.assertEqual([c["field"] for c in todo], ["Work Started", "Last Updated", "Last Updated"])
        self.assertEqual([c["number"] for c in todo], [99, 4, 3])
        self.assertEqual(len(left), 3)

    def test_cap_1_with_an_item_that_needs_both_fields(self):
        item = issue_item(last="2026-01-01", status="In Progress", status_updated="2026-11-02T00:00:00Z")
        client = FakeClient([page([item])])
        code, lines = run_quiet(client, cap=1)
        self.assertEqual(code, 0)
        self.assertEqual(len(client.writes), 1)
        self.assertEqual(client.writes[0]["field"], sweep.WORK_STARTED_FIELD_ID)
        self.assertIn("written_last_updated=0 written_work_started=1", lines[-1])
        self.assertIn("left_for_next_run=1", lines[-1])

    def test_run_respects_cap(self):
        items = [issue_item(item_id=f"I{n}", number=n, last="2026-01-01") for n in range(10)]
        client = FakeClient([page(items)])
        code, lines = run_quiet(client, cap=4)
        self.assertEqual(code, 0)
        self.assertEqual(len(client.writes), 4)
        self.assertIn("left_for_next_run=6", lines[-1])

    def test_default_cap_is_20(self):
        self.assertEqual(sweep.DEFAULT_MAX_WRITES, 20)
        self.assertEqual(sweep.parse_max_writes(None), 20)
        self.assertEqual(sweep.parse_max_writes(""), 20)
        self.assertEqual(sweep.parse_max_writes("  "), 20)

    def test_cap_values_accepted(self):
        for text, number in (("1", 1), ("20", 20), (" 300 ", 300), ("150", 150)):
            self.assertEqual(sweep.parse_max_writes(text), number)

    def test_cap_values_refused(self):
        for text in ("301", "1000", "0", "-5", "abc", "20.5", "1e3"):
            with self.assertRaises(sweep.SweepStop) as caught:
                sweep.parse_max_writes(text)
            self.assertEqual(caught.exception.code, sweep.EXIT_SETUP, text)

    def test_duplicate_item_across_pages_written_once(self):
        first = issue_item(item_id="D", number=5, last="2026-01-01")
        client = FakeClient([page([first], has_next=True, cursor="c1"), page([dict(first)], cursor="c2")])
        code, lines = run_quiet(client)
        self.assertEqual(code, 0)
        self.assertEqual(len(client.writes), 1)
        self.assertIn("wanted_last_updated=1", lines[-1])


class GuardTest(unittest.TestCase):
    def items(self, n=3):
        return [issue_item(item_id=f"I{k}", number=k, last="2026-01-01") for k in range(n)]

    def test_not_moved_continues_and_checks_once(self):
        client = FakeClient([page(self.items())])
        sleeps = []
        code, lines = run_quiet(client, sleep=sleeps.append)
        self.assertEqual(code, 0)
        self.assertEqual(len(client.writes), 3)
        self.assertEqual(client.calls, ["fields", "items", "read", "write", "read", "read", "write", "write"])
        self.assertEqual(sleeps, [sweep.GUARD_WAIT_SECONDS, sweep.GUARD_SECOND_WAIT_SECONDS, 1.0, 1.0])

    def test_guard_waits_about_5_and_30_seconds(self):
        self.assertEqual(sweep.GUARD_WAIT_SECONDS, 5.0)
        self.assertEqual(sweep.GUARD_WAIT_SECONDS + sweep.GUARD_SECOND_WAIT_SECONDS, 30.0)

    def test_moved_at_second_check_stops_with_exit_8(self):
        same = issue_updated_answer("2026-11-17T10:00:00Z")
        moved = issue_updated_answer("2026-11-17T23:41:30Z")
        client = FakeClient([page(self.items())], issue_reads=[same, same, moved])
        sleeps = []
        code, lines = run_quiet(client, sleep=sleeps.append)
        self.assertEqual(code, sweep.EXIT_GUARD)
        self.assertEqual(len(client.writes), 1)
        self.assertEqual(client.calls[2:], ["read", "write", "read", "read"])
        self.assertEqual(sleeps, [sweep.GUARD_WAIT_SECONDS, sweep.GUARD_SECOND_WAIT_SECONDS])
        self.assertTrue(any(line.startswith("STOP: GUARD: the first board write") for line in lines))

    def test_second_check_read_fails_stops_with_exit_8(self):
        same = issue_updated_answer("2026-11-17T10:00:00Z")
        error = sweep.SweepStop(sweep.EXIT_SERVER, "HTTP 502: ")
        client = FakeClient([page(self.items())], issue_reads=[same, same, error])
        code, _ = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_GUARD)
        self.assertEqual(len(client.writes), 1)

    def test_moved_stops_at_once_with_exit_8(self):
        before = issue_updated_answer("2026-11-17T10:00:00Z")
        after = issue_updated_answer("2026-11-17T23:41:05Z")
        client = FakeClient([page(self.items())], issue_reads=[before, after])
        code, lines = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_GUARD)
        self.assertEqual(code, 8)
        self.assertEqual(len(client.writes), 1)
        self.assertEqual(client.calls[2:], ["read", "write", "read"])
        stop = next(line for line in lines if line.startswith("STOP: "))
        self.assertIn("disable", stop)
        self.assertIn("OSPO", stop)
        self.assertNotIn("I0", stop)
        self.assertNotIn("tt-mlir", stop)

    def test_reread_fails_stops(self):
        before = issue_updated_answer("2026-11-17T10:00:00Z")
        error = sweep.SweepStop(sweep.EXIT_SERVER, "network or answer error: URLError")
        client = FakeClient([page(self.items())], issue_reads=[before, error])
        code, lines = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_GUARD)
        self.assertEqual(len(client.writes), 1)

    def test_reread_graphql_error_or_empty_stops(self):
        before = issue_updated_answer("2026-11-17T10:00:00Z")
        for bad in ({"errors": [{"type": "NOT_FOUND", "message": "x"}]}, {"data": {"node": None}}):
            client = FakeClient([page(self.items())], issue_reads=[before, bad])
            code, _ = run_quiet(client)
            self.assertEqual(code, sweep.EXIT_GUARD)
            self.assertEqual(len(client.writes), 1)

    def test_reread_fails_message_says_disable(self):
        before = issue_updated_answer("2026-11-17T10:00:00Z")
        client = FakeClient([page(self.items())], issue_reads=[before, {"data": {"node": None}}])
        code, lines = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_GUARD)
        stop = next(line for line in lines if line.startswith("STOP: "))
        self.assertIn("disable", stop)
        self.assertIn("OSPO", stop)

    def test_read_before_write_fails_no_write_exit_9(self):
        for bad in ({"data": {"node": None}}, {"errors": [{"type": "INTERNAL", "message": "PVTI_x"}]}):
            client = FakeClient([page(self.items())], issue_reads=[bad])
            code, lines = run_quiet(client)
            self.assertEqual(code, sweep.EXIT_GUARD_BASELINE)
            self.assertEqual(code, 9)
            self.assertEqual(client.writes, [])
            stop = next(line for line in lines if line.startswith("STOP: "))
            self.assertIn("No write was made", stop)
            self.assertNotIn("GUARD", stop)
            self.assertNotIn("PVTI_x", stop)

    def test_rate_limit_on_read_before_write_is_exit_9(self):
        limit = sweep.SweepStop(sweep.EXIT_RATE_LIMIT, "rate limit (HTTP 429, Retry-After=60): . Stopped")
        client = FakeClient([page(self.items())], issue_reads=[limit])
        code, lines = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_GUARD_BASELINE)
        self.assertEqual(client.writes, [])
        self.assertIn("exit 3", next(line for line in lines if line.startswith("STOP: ")))

    def test_first_write_error_then_guard_on_second_write_with_fresh_baseline(self):
        def answer(n):
            if n == 1:
                return {"errors": [{"type": "UNPROCESSABLE", "message": "no"}]}
            return {"data": {"updateProjectV2ItemFieldValue": {"clientMutationId": None}}}

        client = FakeClient([page(self.items())], write_answer=answer)
        sleeps = []
        code, lines = run_quiet(client, sleep=sleeps.append)
        self.assertEqual(code, 0)
        self.assertEqual(
            client.calls, ["fields", "items", "read", "write", "read", "write", "read", "read", "write"]
        )
        self.assertEqual(client.reads, ["I2", "I1", "I1", "I1"])
        self.assertEqual(sleeps, [1.0, sweep.GUARD_WAIT_SECONDS, sweep.GUARD_SECOND_WAIT_SECONDS, 1.0])
        self.assertIn("write_errors=1 ", lines[-1])

    def test_read_fails_after_a_failed_write_attempt_is_exit_8(self):
        client = FakeClient(
            [page(self.items())],
            write_answer={"errors": [{"type": "UNPROCESSABLE", "message": "no"}]},
            issue_reads=[issue_updated_answer("2026-11-17T10:00:00Z"), {"data": {"node": None}}],
        )
        code, _ = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_GUARD)
        self.assertEqual(len(client.writes), 1)

    def test_guard_covers_work_started_write(self):
        item = issue_item(last="2026-11-17", status="In Progress", status_updated="2026-11-10T08:00:00Z")
        before = issue_updated_answer("2026-11-17T10:00:00Z")
        after = issue_updated_answer("2026-11-17T10:00:09Z")
        client = FakeClient([page([item])], issue_reads=[before, after])
        code, _ = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_GUARD)
        self.assertEqual(client.writes[0]["field"], sweep.WORK_STARTED_FIELD_ID)

    def test_dry_run_has_no_guard_read(self):
        client = FakeClient([page(self.items())])
        code, _ = run_quiet(client, mode="dry-run")
        self.assertEqual(code, 0)
        self.assertEqual(client.reads, [])


class RunTest(unittest.TestCase):
    def test_dry_run_writes_nothing(self):
        items = [issue_item(last="2026-01-01", status="In Progress", status_updated="2026-11-02T00:00:00Z")]
        client = FakeClient([page(items)])
        code, lines = run_quiet(client, mode="dry-run", sleep=lambda s: self.fail("no sleep"))
        self.assertEqual(code, 0)
        self.assertEqual(client.writes, [])
        self.assertIn("wanted_last_updated=1 wanted_work_started=1", lines[-1])
        self.assertIn("written_last_updated=0", lines[-1])

    def test_write_mode_writes_and_paces(self):
        items = [issue_item(item_id=f"I{n}", number=n, last="2026-01-01") for n in range(3)]
        client = FakeClient([page(items, has_next=True), page([issue_item(item_id="J", number=50, last="2026-01-01")])])
        sleeps = []
        code, lines = run_quiet(client, sleep=sleeps.append)
        self.assertEqual(code, 0)
        self.assertEqual(len(client.writes), 4)
        self.assertEqual(sleeps, [sweep.GUARD_WAIT_SECONDS, sweep.GUARD_SECOND_WAIT_SECONDS, 1.0, 1.0, 1.0])
        self.assertEqual(client.writes[0]["field"], sweep.LAST_UPDATED_FIELD_ID)
        self.assertEqual(client.writes[0]["project"], sweep.PROJECT_ID)
        self.assertIn("written_last_updated=4 written_work_started=0", lines[-1])

    def test_all_without_content_fails(self):
        items = [{"id": f"R{n}", "type": "REDACTED", "content": None} for n in range(5)]
        items += [{"id": "N", "type": "ISSUE", "content": None}, None]
        client = FakeClient([page(items)])
        code, lines = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_NO_CONTENT)
        self.assertEqual(client.writes, [])
        stop = next(line for line in lines if line.startswith("STOP: "))
        self.assertIn("every board item came back without content", stop)
        self.assertIn("permission", stop)

    def test_all_refused_by_content_errors_fails_plainly(self):
        items = [issue_item(item_id="A"), issue_item(item_id="B", number=8)]
        errors = [
            {"type": "FORBIDDEN", "path": ["node", "items", "nodes", k, "content"], "message": "x"} for k in (0, 1)
        ]
        client = FakeClient([page(items, errors=errors)])
        code, lines = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_NO_CONTENT)
        self.assertTrue(any("without content" in line for line in lines))

    def test_no_issue_in_scope_but_content_readable_fails(self):
        items = [issue_item(repo="tenstorrent/tt-metal", last="2020-01-01")]
        client = FakeClient([page(items)])
        code, lines = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_NO_CONTENT)
        self.assertFalse(any("without content" in line for line in lines))

    def test_refused_content_errors_are_counted_not_fatal(self):
        items = [issue_item(item_id="A", last="2026-01-01"), issue_item(item_id="B", number=8, last="2026-01-01")]
        errors = [{"type": "FORBIDDEN", "path": ["node", "items", "nodes", 1, "content"], "message": "no access"}]
        client = FakeClient([page(items, errors=errors)])
        code, lines = run_quiet(client, mode="dry-run", details=True)
        self.assertEqual(code, 0)
        self.assertTrue(any("refused=1" in line for line in lines))
        self.assertIn("wanted_last_updated=1", lines[-1])

    def test_null_item_counted(self):
        client = FakeClient([page([None, issue_item(last="2026-01-01")])])
        code, lines = run_quiet(client, mode="dry-run", details=True)
        self.assertEqual(code, 0)
        self.assertIn("items=2 ", lines[0])
        self.assertIn("skipped=1", lines[0])
        self.assertIn("no_content=1", lines[1])

    def test_other_graphql_error_stops(self):
        errors = [{"type": "INTERNAL", "path": ["node"], "message": "boom"}]
        client = FakeClient([page([issue_item()], errors=errors)])
        code, _ = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_GRAPHQL)
        self.assertEqual(client.writes, [])

    def test_partial_data_node_null_stops(self):
        client = FakeClient([])
        client.pages = [{"data": {"node": None, "rateLimit": {"remaining": 4000}}}]
        code, _ = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_GRAPHQL)
        self.assertEqual(client.writes, [])

    def test_rate_limited_error_stops(self):
        errors = [{"type": "RATE_LIMITED", "message": "API rate limit exceeded"}]
        client = FakeClient([page([issue_item()], errors=errors)])
        code, _ = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_RATE_LIMIT)

    def test_rate_limited_on_fields_query_is_exit_3(self):
        client = FakeClient([])
        client.post = lambda q, v, retry_server_errors=True: {"errors": [{"type": "RATE_LIMITED", "message": "x"}]}
        code, _ = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_RATE_LIMIT)

    def test_low_rate_budget_stops_before_writes(self):
        client = FakeClient([page([issue_item(last="2026-01-01")], remaining=sweep.MIN_RATE_REMAINING - 1)])
        code, _ = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_RATE_LIMIT)
        self.assertEqual(client.writes, [])

    def test_more_pages_without_cursor_stops(self):
        for cursor in ("", None):
            client = FakeClient([page([issue_item(last="2026-01-01")], has_next=True, cursor=cursor)])
            code, _ = run_quiet(client)
            self.assertEqual(code, sweep.EXIT_GRAPHQL)
            self.assertEqual(client.writes, [])

    def test_same_cursor_twice_stops(self):
        p1 = page([issue_item(item_id="A", last="2026-01-01")], has_next=True, cursor="c1")
        p2 = page([issue_item(item_id="B", last="2026-01-01")], has_next=True, cursor="c1")
        client = FakeClient([p1, p2])
        code, _ = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_GRAPHQL)
        self.assertEqual(client.writes, [])

    def test_write_error_skipped_then_continues(self):
        items = [issue_item(item_id=f"I{n}", number=n, last="2026-01-01") for n in range(3)]

        def answer(n):
            if n == 1:
                return {"errors": [{"type": "UNPROCESSABLE", "message": "item PVTI_x is archived"}]}
            return {"data": {"updateProjectV2ItemFieldValue": {"clientMutationId": None}}}

        client = FakeClient([page(items)], write_answer=answer)
        code, lines = run_quiet(client)
        self.assertEqual(code, 0)
        self.assertEqual(len(client.writes), 3)
        self.assertIn("write_errors=1", lines[-1])
        self.assertIn("written_last_updated=2", lines[-1])
        self.assertFalse(any("PVTI_x" in line for line in lines))

    def test_three_write_errors_stop(self):
        items = [issue_item(item_id=f"I{n}", number=n, last="2026-01-01") for n in range(5)]
        client = FakeClient([page(items)], write_answer={"errors": [{"type": "FORBIDDEN", "message": "no"}]})
        code, lines = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_GRAPHQL)
        self.assertEqual(len(client.writes), 3)

    def test_rate_limited_write_stops_at_once(self):
        items = [issue_item(item_id=f"I{n}", number=n, last="2026-01-01") for n in range(3)]
        client = FakeClient([page(items)], write_answer={"errors": [{"type": "RATE_LIMITED", "message": "no"}]})
        code, _ = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_RATE_LIMIT)
        self.assertEqual(len(client.writes), 1)

    def test_changed_field_id_stops_before_any_read_of_items(self):
        client = FakeClient([])
        bad = {"data": {"node": dict(FIELDS_OK["data"]["node"], lastUpdated={"id": "OTHER", "dataType": "DATE"})}}
        client.post = lambda q, v, retry_server_errors=True: bad
        code, _ = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_FIELDS)

    def test_no_title_or_body_requested(self):
        for query in (sweep.ITEMS_QUERY, sweep.FIELDS_QUERY, sweep.UPDATE_MUTATION, sweep.ISSUE_UPDATED_QUERY):
            self.assertNotIn("title", query)
            self.assertNotIn("body", query)


class LoggingTest(unittest.TestCase):
    def items(self):
        return [
            issue_item(
                item_id="PVTI_one",
                number=4242,
                last="2026-01-01",
                status="In Progress",
                status_updated="2026-11-02T00:00:00Z",
            ),
            issue_item(item_id="PVTI_two", repo="tenstorrent/tt-metal", number=777, last="2020-01-01"),
        ]

    def test_default_log_is_counts_only(self):
        for mode in ("write", "dry-run"):
            client = FakeClient([page(self.items())])
            code, lines = run_quiet(client, mode=mode)
            self.assertEqual(code, 0)
            self.assertEqual(len(lines), 2, lines)
            self.assertEqual(lines[0], "Board read: items=2 pages=1 in_scope=1 skipped=1")
            self.assertTrue(lines[1].startswith(f"SUMMARY mode={mode} exit=0 cap=20 "))
            for line in lines:
                for secret in ("PVTI_", "4242", "777", "tt-mlir", "tt-metal", "2026-11-02", "2026-01-01"):
                    self.assertNotIn(secret, line)
                for reason in ("other_repo", "pull_request", "draft", "archived", "redacted", "no_content", "refused"):
                    self.assertNotIn(reason, line)

    def test_details_on_prints_skip_reasons(self):
        client = FakeClient([page(self.items())])
        code, lines = run_quiet(client, mode="dry-run", details=True)
        self.assertEqual(code, 0)
        self.assertEqual(
            lines[1],
            "Skipped by reason: other_repo=1 pull_request=0 draft=0 archived=0 redacted=0 no_content=0 "
            "(of these refused=0)",
        )

    def test_details_on_prints_change_lines(self):
        client = FakeClient([page(self.items())])
        code, lines = run_quiet(client, details=True)
        self.assertEqual(code, 0)
        detail = [line for line in lines if line.startswith("WRITE ")]
        self.assertEqual(len(detail), 2)
        expected = r'^WRITE repo=tenstorrent/tt-mlir issue=4242 field="Work Started" old=empty new=2026-11-02$'
        self.assertRegex(detail[0], expected)

    def test_error_summary_has_no_messages(self):
        text = sweep.error_summary([{"type": "NOT_FOUND", "message": "Could not resolve PVTI_secret"}])
        self.assertEqual(text, "1 error(s), types=NOT_FOUND")

    def test_graphql_error_stop_line_has_no_item_data(self):
        errors = [{"type": "INTERNAL", "path": ["node", "items", "nodes", 0], "message": "item PVTI_one issue 4242"}]
        client = FakeClient([page(self.items(), errors=errors)])
        code, lines = run_quiet(client)
        self.assertEqual(code, sweep.EXIT_GRAPHQL)
        for line in lines:
            self.assertNotIn("PVTI_one", line)
            self.assertNotIn("4242", line)


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def http_error(status, message="", headers=None):
    body = io.BytesIO(json.dumps({"message": message}).encode("utf-8"))
    return urllib.error.HTTPError(sweep.API_URL, status, "error", headers or {}, body)


class FakeUrlopen:
    """Returns or raises the given answers in order; records the requests."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return FakeResponse(answer)


class HttpTest(unittest.TestCase):
    def client(self, answers, sleeps=None):
        opener = FakeUrlopen(answers)
        sleep = sleeps.append if sleeps is not None else (lambda s: None)
        return sweep.GitHubGraphQL(FAKE_TOKEN, sleep=sleep, urlopen=opener), opener

    def stop_code(self, client, retry=True):
        with self.assertRaises(sweep.SweepStop) as caught:
            client.post("query", {}, retry_server_errors=retry)
        self.assertNotIn(FAKE_TOKEN, caught.exception.message)
        return caught.exception.code

    def test_ok(self):
        client, opener = self.client([b'{"data": {"x": 1}}'])
        self.assertEqual(client.post("query", {}), {"data": {"x": 1}})
        self.assertEqual(opener.requests[0].get_header("Authorization"), f"bearer {FAKE_TOKEN}")

    def test_401_and_403_are_auth_errors(self):
        for status in (401, 403):
            client, _ = self.client([http_error(status, "Bad credentials")])
            self.assertEqual(self.stop_code(client), sweep.EXIT_SETUP)

    def test_429_is_rate_limit(self):
        client, _ = self.client([http_error(429)])
        self.assertEqual(self.stop_code(client), sweep.EXIT_RATE_LIMIT)

    def test_retry_after_is_rate_limit(self):
        client, _ = self.client([http_error(403, "", {"Retry-After": "60"})])
        self.assertEqual(self.stop_code(client), sweep.EXIT_RATE_LIMIT)

    def test_403_secondary_rate_limit_message(self):
        client, _ = self.client([http_error(403, "You have exceeded a secondary rate limit")])
        self.assertEqual(self.stop_code(client), sweep.EXIT_RATE_LIMIT)

    def test_5xx_then_success(self):
        sleeps = []
        client, _ = self.client([http_error(502), b'{"data": {}}'], sleeps)
        self.assertEqual(client.post("query", {}), {"data": {}})
        self.assertEqual(sleeps, [5])

    def test_5xx_three_times_stops(self):
        sleeps = []
        client, opener = self.client([http_error(502), http_error(503), http_error(500)], sleeps)
        self.assertEqual(self.stop_code(client), sweep.EXIT_SERVER)
        self.assertEqual(sleeps, [5, 15])
        self.assertEqual(len(opener.requests), 3)

    def test_5xx_on_write_is_not_retried(self):
        client, opener = self.client([http_error(502)])
        self.assertEqual(self.stop_code(client, retry=False), sweep.EXIT_SERVER)
        self.assertEqual(len(opener.requests), 1)

    def test_network_error_retried_then_stops(self):
        client, opener = self.client([urllib.error.URLError("down")] * 3)
        self.assertEqual(self.stop_code(client), sweep.EXIT_SERVER)
        self.assertEqual(len(opener.requests), 3)

    def test_non_json_body_stops(self):
        client, _ = self.client([b"<html>oops</html>"] * 3)
        self.assertEqual(self.stop_code(client), sweep.EXIT_SERVER)

    def test_ssl_and_other_os_errors_stop_cleanly(self):
        import ssl

        for error in (ssl.SSLError("bad record"), OSError("reset"), ConnectionResetError(), TimeoutError()):
            client, opener = self.client([error] * 3)
            self.assertEqual(self.stop_code(client), sweep.EXIT_SERVER)
            self.assertEqual(len(opener.requests), 3)

    def test_os_error_while_reading_body_stops_cleanly(self):
        class BrokenResponse(FakeResponse):
            def read(self):
                raise OSError("connection reset while reading")

        class Opener(FakeUrlopen):
            def __call__(self, request, timeout=None):
                self.requests.append(request)
                return BrokenResponse(b"")

        opener = Opener([])
        client = sweep.GitHubGraphQL(FAKE_TOKEN, sleep=lambda s: None, urlopen=opener)
        self.assertEqual(self.stop_code(client), sweep.EXIT_SERVER)
        self.assertEqual(len(opener.requests), 3)

    def test_incomplete_read_stops(self):
        import http.client

        client, _ = self.client([http.client.IncompleteRead(b"x")] * 3)
        self.assertEqual(self.stop_code(client), sweep.EXIT_SERVER)


class MainTest(unittest.TestCase):
    def run_main(self, env, urlopen=None):
        buffer = io.StringIO()
        patches = [mock.patch.dict(os.environ, env, clear=True)]
        if urlopen is not None:
            patches.append(mock.patch("urllib.request.urlopen", urlopen))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(buffer))
            code = sweep.main()
        return code, buffer.getvalue()

    def test_empty_token(self):
        code, output = self.run_main({"SWEEP_MODE": "write"})
        self.assertEqual(code, sweep.EXIT_SETUP)
        self.assertIn("PROJECT_BOARD_TOKEN is empty", output)

    def test_bad_mode(self):
        code, output = self.run_main({"SWEEP_MODE": "yes", "PROJECT_BOARD_TOKEN": FAKE_TOKEN})
        self.assertEqual(code, sweep.EXIT_SETUP)
        self.assertNotIn(FAKE_TOKEN, output)

    def test_bad_cap(self):
        code, output = self.run_main({"SWEEP_MAX_WRITES": "301", "PROJECT_BOARD_TOKEN": FAKE_TOKEN})
        self.assertEqual(code, sweep.EXIT_SETUP)
        self.assertIn("refused", output)

    def test_token_never_printed_on_error_paths(self):
        for answer in (http_error(401, "Bad credentials"), http_error(502), urllib.error.URLError("down")):
            opener = FakeUrlopen([answer] * 3)
            with mock.patch.object(sweep.time, "sleep", lambda s: None):
                code, output = self.run_main(
                    {"SWEEP_MODE": "write", "PROJECT_BOARD_TOKEN": FAKE_TOKEN, "SWEEP_LOG_DETAILS": "1"}, opener
                )
            self.assertNotEqual(code, 0)
            self.assertIn("SUMMARY", output)
            self.assertNotIn(FAKE_TOKEN, output)
            self.assertNotIn("FAKEtoken", output)


if __name__ == "__main__":
    unittest.main()
