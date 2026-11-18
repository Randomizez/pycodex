"""Native notification lifecycle checks; no model or speaker requests."""
from datetime import datetime, timezone
import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from summarize import validate_speech


module_path = Path(__file__).with_name("notifications.py")
spec = importlib.util.spec_from_file_location("native_notifications", module_path)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class FakeSession:
    def __init__(self):
        self.state = {
            "title": "代码修复", "rollout_path": "/private/stable-rollout.jsonl",
            "recorded_rollout_path": "/private/stable-rollout.jsonl",
            "running": False, "spinner": "", "queued_inputs": [],
            "turns": [{"turn_id": "turn-1", "status": "completed",
                       "response": "修复已通过测试，请检查页面。", "prompt": "修复错误"}],
        }
        self.records = []

    def snapshot(self):
        return self.state

    def set_notification_records(self, records):
        self.records = records

    def event(self):
        return {"kind": "turn_completed", "turn_id": "turn-1", "payload": {},
                "snapshot": self.snapshot()}


class PolicyTests(unittest.TestCase):
    def test_disabled_service_creates_no_database(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "workspace"
            service = native.SessionNotifications(prefix, {})
            service.observe("session", FakeSession(), FakeSession().event())
            self.assertFalse(service.enabled)
            self.assertIsNone(service.db)
            self.assertIsNone(service.worker)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_beijing_quiet_hour_boundaries(self):
        for hour, minute, expected in [(0, 59, False), (1, 0, True),
                                        (9, 59, True), (10, 0, False), (23, 0, False)]:
            now = datetime(2026, 11, 17, hour, minute, tzinfo=native.BEIJING)
            self.assertEqual(native.quiet_hours(now), expected)
            self.assertEqual(native.quiet_hours(now.astimezone(timezone.utc)), expected)

    def test_fifty_character_limit_includes_punctuation(self):
        self.assertEqual(validate_speech("小林" + "字" * 47 + "。"),
                         "小林" + "字" * 47 + "。")
        for text in (None, 5, "小林" + "字" * 49, "小林\n请查看", "", "小林，https://example.com"):
            with self.assertRaises(ValueError):
                validate_speech(text)
        self.assertEqual(validate_speech("检查已完成。"), "检查已完成。")
        self.assertEqual(validate_speech("小林，检查已完成。", "小林"), "小林，检查已完成。")
        with self.assertRaises(ValueError):
            validate_speech("检查已完成。", "小林")


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.prefix = Path(self.temporary.name) / "workspace"
        self.config = {"enabled": True, "prepare_command": ["summary"],
                       "deliver_command": ["speaker"]}
        self.service = native.SessionNotifications(self.prefix, self.config)
        self.addAsyncCleanup(self.service.close)
        self.session = FakeSession()
        self.service.bind("session", self.session)
        self.speech = "代码修复已通过测试，请检查页面显示。"
        self.service.command = AsyncMock(side_effect=[
            self.speech, json.dumps({"state": "accepted"})])
        self.quiet = patch.object(native, "quiet_hours", return_value=False)
        self.quiet.start()
        self.addCleanup(self.quiet.stop)

    async def drain(self):
        await self.service.queue.join()

    async def test_structured_messages_supply_the_current_request(self):
        turn = self.session.state["turns"][0]
        del turn["prompt"]
        turn["messages"] = [{"text": "检查部署"}, {"text": "保留已有设置"}]
        self.service.observe("session", self.session, self.session.event())
        await self.drain()
        payload = self.service.command.call_args_list[0].args[1]
        self.assertEqual(payload["latest_request"], "检查部署\n保留已有设置")

    async def test_normal_speech_is_recorded_and_duplicate_not_replayed(self):
        event = self.session.event()
        self.service.observe("session", self.session, event)
        self.service.observe("session", self.session, event)
        await self.drain()
        self.assertEqual(self.service.command.await_count, 2)
        self.assertEqual(len(self.session.records), 1)
        self.assertEqual(self.session.records[0]["speech"], self.speech)
        self.assertEqual(self.session.records[0]["state"], "accepted")
        self.assertEqual(self.session.state["turns"][0]["response"],
                         "修复已通过测试，请检查页面。")

    async def test_compaction_and_command_records_do_not_suppress_latest_result(self):
        event = self.session.event()
        self.service.observe("session", self.session, event)
        self.session.state["turns"].extend([
            {"turn_id": "compact-1", "kind": "compaction", "status": "completed"},
            {"turn_id": "command-1", "kind": "control", "status": "completed"},
        ])
        await self.drain()
        self.assertEqual(self.service.command.await_count, 2)
        self.assertEqual(self.session.records[0]["state"], "accepted")

    async def test_intermediate_events_never_generate_or_create_records(self):
        for kind in ("turn_interrupted", "input_requested", "assistant_delta"):
            self.service.observe("session", self.session,
                                 dict(self.session.event(), kind=kind))
        self.assertIsNone(self.service.queue)
        self.assertEqual(self.service.db.execute(
            "SELECT COUNT(*) FROM notifications").fetchone()[0], 0)
        self.assertEqual(self.service.command.await_count, 0)
        self.service.observe("session", self.session, self.session.event())
        await self.drain()
        self.assertEqual(self.service.command.await_count, 2)
        self.assertEqual(len(self.session.records), 1)

    async def test_queued_or_background_work_does_not_generate_until_finished(self):
        self.session.state["queued_inputs"] = [{"queue": "steer", "prompt": "补充"}]
        self.service.observe("session", self.session, self.session.event())
        self.session.state["queued_inputs"] = []
        self.service.observe("session", self.session, dict(
            self.session.event(), payload={"background_work_count": 1}))
        self.session.state["spinner"] = "idle: sleeping"
        self.service.observe("session", self.session, self.session.event())
        self.assertIsNone(self.service.queue)
        self.assertEqual(self.service.db.execute(
            "SELECT COUNT(*) FROM notifications").fetchone()[0], 0)
        self.session.state["spinner"] = ""
        self.service.observe("session", self.session, self.session.event())
        await self.drain()
        self.assertEqual(self.service.command.await_count, 2)
        self.assertEqual(len(self.session.records), 1)

    async def test_queued_results_with_same_turn_id_only_generate_latest(self):
        failed = dict(self.session.event(), kind="turn_failed")
        # Equal timestamps must not make both events current.
        with patch.object(native.time, "time", return_value=123):
            self.service.observe("session", self.session, failed)
            self.service.observe("session", self.session, self.session.event())
        await self.drain()
        self.assertEqual(self.service.command.await_count, 2)
        self.assertEqual(self.service.command.await_args_list[0].args[1]["kind"],
                         "turn_completed")
        self.assertEqual([row["state"] for row in self.session.records],
                         ["accepted"])

    async def test_new_result_replaces_previous_result_during_generation(self):
        started, release = asyncio.Event(), asyncio.Event()
        failed_speech = "代码修复失败，请检查。"

        async def command(argv, payload, timeout):
            if argv == ["speaker"]:
                return json.dumps({"state": "accepted"})
            if payload["kind"] == "turn_failed":
                started.set()
                await release.wait()
                return failed_speech
            return self.speech

        self.service.command.side_effect = command
        self.service.observe("session", self.session,
                             dict(self.session.event(), kind="turn_failed"))
        await asyncio.wait_for(started.wait(), 1)
        self.service.observe("session", self.session, self.session.event())
        release.set()
        await self.drain()
        self.assertEqual([row["state"] for row in self.session.records],
                         ["superseded", "accepted"])
        self.assertEqual(self.session.records[0]["speech"], failed_speech)
        deliveries = [call.args[1] for call in self.service.command.await_args_list
                      if call.args[0] == ["speaker"]]
        self.assertEqual(deliveries, [{"speech": self.speech}])

    async def test_new_event_while_waiting_for_speaker_replaces_old_speech(self):
        self.service.last_speech_end = native.time.monotonic() + 10
        self.service.command.side_effect = [
            self.speech, self.speech, json.dumps({"state": "accepted"})]

        async def next_event(delay):
            self.service.last_speech_end = 0
            self.service.observe("session", self.session, self.session.event())

        self.service.observe("session", self.session,
                             dict(self.session.event(), kind="turn_failed"))
        with patch.object(native.asyncio, "sleep", side_effect=next_event):
            await self.drain()
        self.assertEqual(self.service.command.await_count, 3)
        self.assertEqual([row["state"] for row in self.session.records],
                         ["superseded", "accepted"])

    async def test_newer_assistant_turn_still_suppresses_old_completion(self):
        self.service.observe("session", self.session, self.session.event())
        self.session.state["turns"].extend([
            {"turn_id": "turn-2", "kind": "assistant", "status": "completed"},
            {"turn_id": "compact-1", "kind": "compaction", "status": "completed"},
        ])
        await self.drain()
        self.assertEqual(self.service.command.await_count, 0)
        self.assertEqual(self.session.records, [])

    async def test_other_session_does_not_replace_this_sessions_notification(self):
        other = FakeSession()
        other.state["recorded_rollout_path"] = "/private/other-rollout.jsonl"
        self.service.bind("other", other)
        self.service.command.side_effect = [
            self.speech, json.dumps({"state": "accepted"}),
            self.speech, json.dumps({"state": "accepted"}),
        ]
        self.service.observe("session", self.session, self.session.event())
        self.service.observe("other", other, other.event())
        with patch.object(native.asyncio, "sleep", new_callable=AsyncMock):
            await self.drain()
        self.assertEqual(self.service.command.await_count, 4)
        self.assertEqual(self.session.records[0]["state"], "accepted")
        self.assertEqual(other.records[0]["state"], "accepted")

    async def test_quiet_time_generates_and_records_content_without_delivery(self):
        with patch.object(native, "quiet_hours", return_value=True):
            self.service.observe("session", self.session, self.session.event())
            await self.drain()
        self.assertEqual(self.service.command.await_count, 1)
        self.assertEqual(self.session.records[0]["speech"], self.speech)
        self.assertEqual(self.session.records[0]["state"], "quiet_hours")
        # Returning to daytime must not queue last night's speech again.
        self.service.observe("session", self.session, self.session.event())
        await self.drain()
        self.assertEqual(self.service.command.await_count, 1)

    async def test_boundary_crossed_while_preparing_is_suppressed(self):
        with patch.object(native, "quiet_hours", side_effect=[False, True]):
            self.service.observe("session", self.session, self.session.event())
            await self.drain()
        self.assertEqual(self.service.command.await_count, 1)
        self.assertEqual(self.session.records[0]["state"], "quiet_hours")

    async def test_device_boundary_suppression_is_recorded(self):
        self.service.command.side_effect = [
            self.speech, json.dumps({"state": "quiet_hours"})]
        self.service.observe("session", self.session, self.session.event())
        await self.drain()
        self.assertEqual(self.session.records[0]["state"], "quiet_hours")
        self.assertEqual(self.session.records[0]["speech"], self.speech)

    async def test_session_resumed_before_generation_creates_no_speech_or_chat_entry(self):
        self.session.state["running"] = True
        self.session.state["spinner"] = "working"
        self.service.observe("session", self.session, self.session.event())
        await self.drain()
        self.assertEqual(self.service.command.await_count, 0)
        self.assertEqual(self.session.records, [])

    async def test_failed_delivery_keeps_speech_in_chat(self):
        self.service.command.side_effect = [self.speech, RuntimeError("private service detail")]
        self.service.observe("session", self.session, self.session.event())
        await self.drain()
        self.assertEqual(self.session.records[0]["state"], "failed")
        self.assertEqual(self.session.records[0]["speech"], self.speech)
        self.assertEqual(self.session.records[0]["error"], "RuntimeError")

    async def test_oversized_summary_never_reaches_speaker(self):
        self.service.command.side_effect = ["字" * 51]
        self.service.observe("session", self.session, self.session.event())
        await self.drain()
        self.assertEqual(self.service.command.await_count, 1)
        self.assertEqual(self.session.records[0]["state"], "failed")

    async def test_records_survive_restart_and_changed_web_session_id(self):
        self.service.observe("session", self.session, self.session.event())
        await self.drain()
        await self.service.close()
        restored = native.SessionNotifications(self.prefix, self.config)
        self.addAsyncCleanup(restored.close)
        restored.bind("new-web-session-id", self.session)
        self.assertEqual(self.session.records[0]["speech"], self.speech)
        self.assertEqual(self.session.records[0]["state"], "accepted")
        restored.observe("new-web-session-id", self.session, self.session.event())
        self.assertIsNone(restored.worker)

    async def test_history_subscription_is_not_a_completion_notification(self):
        self.service.observe("session", self.session, {"type": "hello", "snapshot": self.session.state})
        self.assertIsNone(self.service.worker)
        self.assertEqual(self.session.records, [])


if __name__ == "__main__":
    unittest.main()
