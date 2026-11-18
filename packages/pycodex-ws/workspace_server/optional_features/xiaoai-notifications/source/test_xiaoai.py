"""Behavior checks with isolated credentials and no real device requests."""
import base64
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import xiaoai


class SpeakerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        quiet = patch.object(xiaoai, "quiet_hours", return_value=False)
        quiet.start()
        self.addCleanup(quiet.stop)
        self.speaker = xiaoai.Speaker(self.root / "state")
        self.addCleanup(self.speaker.session.close)
        self.speaker.session = Mock()
        self.speaker.token = {
            "deviceId": "client", "userId": "user", "passToken": "private-pass",
            xiaoai.SID: ["private-security", "private-service"],
        }

    def test_requires_an_explicit_device_selection(self):
        target = {"hardware": "X08A", "deviceID": "target"}
        other = {"hardware": "LX06", "deviceID": "other"}
        with self.assertRaises(xiaoai.SpeakerError):
            self.speaker.selected([target])
        xiaoai.private_json(self.speaker.config_path, {"device_id": "target"})
        self.assertEqual(self.speaker.selected([other, target]), target)
        for devices in ([other], [], [target, dict(target)]):
            with self.assertRaises(xiaoai.SpeakerError):
                self.speaker.selected(devices)
        self.speaker.session.post.assert_not_called()

    def test_saved_selection_cannot_silently_switch_devices(self):
        xiaoai.private_json(self.speaker.config_path, {"device_id": "absent"})
        with self.assertRaises(xiaoai.SpeakerError):
            self.speaker.selected([{"hardware": "X08A", "deviceID": "new"}])

    def test_private_token_file_permissions_and_replacement(self):
        path = self.root / "private" / "token.json"
        xiaoai.private_json(path, {"token": "first"})
        xiaoai.private_json(path, {"token": "second"})
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual(json.loads(path.read_text()), {"token": "second"})
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_tts_is_directed_to_exactly_one_device(self):
        response = Mock(status_code=200)
        response.json.return_value = {"code": 0}
        self.speaker.session.post.return_value = response
        self.speaker.say("target", "测试通知")
        self.speaker.session.post.assert_called_once()
        call = self.speaker.session.post.call_args
        self.assertEqual(call.args[0], xiaoai.MINA + "/remote/ubus")
        self.assertEqual(call.kwargs["data"]["deviceId"], "target")
        self.assertEqual(call.kwargs["data"]["method"], "text_to_speech")
        self.assertEqual(json.loads(call.kwargs["data"]["message"]), {"text": "测试通知"})

    def test_server_error_does_not_replay_potentially_delivered_speech(self):
        response = Mock(status_code=503)
        response.json.return_value = {"code": 1, "message": "private-error"}
        self.speaker.session.post.return_value = response
        with self.assertRaises(xiaoai.SpeakerError) as raised:
            self.speaker.say("target", "测试通知")
        self.assertNotIn("private-error", str(raised.exception))
        self.speaker.session.post.assert_called_once()

    def authorization_request(self, transport, responses):
        self.speaker.session.reset_mock()
        sid = xiaoai.SID if transport == "mina" else "xiaomiio"
        security = base64.b64encode(b"test-security").decode()
        self.speaker.token[sid] = [security, "expired-service"]

        def refresh(*args):
            self.speaker.token[sid] = [security, "renewed-service"]

        self.speaker.refresh_login = Mock(side_effect=refresh)
        if transport == "mina":
            request = self.speaker.session.get
            invoke = self.speaker.devices
        else:
            request = self.speaker.session.post
            invoke = lambda: self.speaker.miot_api("/miotspec/prop/get", {"params": []})
        request.side_effect = responses
        return request, invoke

    def test_html_401_refreshes_the_correct_service_before_parsing(self):
        for transport in ("mina", "miot"):
            with self.subTest(transport=transport):
                expired = Mock(status_code=401)
                expired.json.side_effect = ValueError("private HTML response")
                success = Mock(status_code=200)
                success.json.return_value = {"code": 0, "data": [], "result": []}
                request, invoke = self.authorization_request(transport, [expired, success])
                self.assertEqual(invoke(), [])
                expired.json.assert_not_called()
                expected = () if transport == "mina" else ("xiaomiio",)
                self.speaker.refresh_login.assert_called_once_with(*expected)
                self.assertEqual(request.call_count, 2)
                self.assertEqual(
                    [call.kwargs["cookies"]["serviceToken"] for call in request.call_args_list],
                    ["expired-service", "renewed-service"])

    def test_repeated_html_401_stops_after_one_refresh(self):
        for transport in ("mina", "miot"):
            with self.subTest(transport=transport):
                expired = Mock(status_code=401)
                expired.json.side_effect = ValueError("private HTML response")
                request, invoke = self.authorization_request(transport, [expired, expired])
                with self.assertRaisesRegex(xiaoai.SpeakerError, "HTTP 401") as raised:
                    invoke()
                self.assertNotIn("private", str(raised.exception))
                self.assertEqual(request.call_count, 2)
                self.speaker.refresh_login.assert_called_once()
                expired.json.assert_not_called()

    def test_html_server_error_does_not_refresh_or_retry(self):
        for transport in ("mina", "miot"):
            with self.subTest(transport=transport):
                response = Mock(status_code=503)
                response.json.side_effect = ValueError("private HTML response")
                request, invoke = self.authorization_request(transport, [response])
                with self.assertRaises(xiaoai.SpeakerError) as raised:
                    invoke()
                self.assertNotIn("private", str(raised.exception))
                request.assert_called_once()
                self.speaker.refresh_login.assert_not_called()

    def test_json_authorization_error_still_refreshes(self):
        for transport in ("mina", "miot"):
            with self.subTest(transport=transport):
                expired = Mock(status_code=200)
                expired.json.return_value = {"code": 1, "message": "auth expired"}
                success = Mock(status_code=200)
                success.json.return_value = {"code": 0, "data": [], "result": []}
                request, invoke = self.authorization_request(transport, [expired, success])
                self.assertEqual(invoke(), [])
                self.assertEqual(request.call_count, 2)
                self.speaker.refresh_login.assert_called_once()

    def test_refresh_failure_does_not_retry_the_request(self):
        for transport in ("mina", "miot"):
            with self.subTest(transport=transport):
                expired = Mock(status_code=401)
                request, invoke = self.authorization_request(transport, [expired])
                self.speaker.refresh_login.side_effect = xiaoai.SpeakerError("需要重新扫码")
                with self.assertRaisesRegex(xiaoai.SpeakerError, "需要重新扫码"):
                    invoke()
                request.assert_called_once()
                expired.json.assert_not_called()

    def test_quiet_time_is_rechecked_after_authorization_refresh(self):
        for transport in ("mina", "miot"):
            with self.subTest(transport=transport):
                expired = Mock(status_code=401)
                self.authorization_request(transport, [expired])
                self.speaker.session.post.side_effect = [expired]
                with patch.object(xiaoai, "quiet_hours", side_effect=[False, True]):
                    with self.assertRaises(xiaoai.QuietHours):
                        if transport == "mina":
                            self.speaker.say("target", "通知测试")
                        else:
                            self.speaker.say_miot(
                                {"hardware": "X08C", "miotDID": "target"}, "通知测试")
                self.speaker.session.post.assert_called_once()
                self.speaker.refresh_login.assert_called_once()

    def test_nested_device_rejection_is_not_success(self):
        self.speaker.api = Mock(return_value={"code": 0, "data": {"code": 7}})
        with self.assertRaises(xiaoai.SpeakerError):
            self.speaker.say("target", "测试通知")

    def test_miot_action_uses_the_target_models_actual_service(self):
        self.speaker.miot_api = Mock(return_value={"code": 0})
        self.speaker.say_miot({"hardware": "X08C", "miotDID": "target"}, "通知测试")
        self.speaker.miot_api.assert_called_once_with("/miotspec/action", {
            "params": {"did": "target", "siid": 3, "aiid": 1, "in": ["通知测试"]}})
        self.speaker.say_miot({"hardware": "X08A", "miotDID": "other"}, "测试")
        self.assertEqual(self.speaker.miot_api.call_args.args[1]["params"]["siid"], 5)

    def test_miot_device_error_is_not_cloud_success(self):
        self.speaker.miot_api = Mock(return_value={"code": -704030013})
        with self.assertRaises(xiaoai.SpeakerError):
            self.speaker.say_miot({"hardware": "X08C", "miotDID": "target"}, "测试")
        self.speaker.miot_api.assert_called_once()

    def test_miot_unknown_hardware_is_not_called(self):
        self.speaker.miot_api = Mock()
        with self.assertRaises(xiaoai.SpeakerError):
            self.speaker.say_miot({"hardware": "X4B", "miotDID": "other"}, "测试")
        self.speaker.miot_api.assert_not_called()

    def test_service_authorization_preserves_existing_mina_token(self):
        response = Mock(text=json.dumps({
            "code": 0, "ssecurity": "new-security",
            "location": "https://sts.api.io.mi.com/sts"}))
        self.speaker.session.get.return_value = response
        self.speaker.exchange_token = Mock(return_value="new-service")
        self.speaker.refresh_login("xiaomiio")
        self.assertEqual(self.speaker.token[xiaoai.SID], ["private-security", "private-service"])
        self.assertEqual(self.speaker.token["xiaomiio"], ["new-security", "new-service"])
        self.assertEqual(self.speaker.session.get.call_args.kwargs["params"]["sid"], "xiaomiio")

    def test_empty_and_excessive_speech_rejected_before_network(self):
        for text in ("", "   ", "字" * 201):
            with self.assertRaises(xiaoai.SpeakerError):
                self.speaker.say("target", text)
        self.speaker.session.post.assert_not_called()

    def test_quiet_time_blocks_the_device_request(self):
        with patch.object(xiaoai, "quiet_hours", return_value=True):
            with self.assertRaises(xiaoai.QuietHours):
                self.speaker.say("target", "通知测试")
        self.speaker.session.post.assert_not_called()

    def test_expired_qr_does_not_authorize_or_save(self):
        with self.assertRaises(xiaoai.SpeakerError):
            self.speaker.finish_login({"expires_at": time.time() - 1})
        self.assertFalse(self.speaker.token_path.exists())
        self.speaker.session.get.assert_not_called()

    def test_rejected_qr_never_prints_response_secrets(self):
        response = Mock(text='&&&START&&&{"code": 7, "passToken": "private"}')
        self.speaker.session.get.return_value = response
        with self.assertRaises(xiaoai.SpeakerError) as raised:
            self.speaker.finish_login({"expires_at": time.time() + 30,
                                       "lp": "https://account.xiaomi.com/poll"})
        self.assertNotIn("private", str(raised.exception))
        self.assertFalse(self.speaker.token_path.exists())

    def test_successful_qr_saves_service_token_after_confirmation(self):
        response = Mock(text=json.dumps({
            "userId": 123, "passToken": "private-pass", "ssecurity": "private-security",
            "location": "https://api2.mina.mi.com/sts",
        }))
        self.speaker.session.get.return_value = response
        self.speaker.exchange_token = Mock(return_value="private-service")
        self.speaker.finish_login({"expires_at": time.time() + 30,
                                   "lp": "https://account.xiaomi.com/poll"})
        saved = json.loads(self.speaker.token_path.read_text())
        self.assertEqual(saved[xiaoai.SID][1], "private-service")
        self.assertEqual(saved["userId"], "123")

    def test_only_xiaomi_https_login_urls_are_followed(self):
        for value in ("http://account.xiaomi.com/a", "https://xiaomi.com.evil.test/a",
                      "https://example.com/a", "file:///tmp/a"):
            with self.assertRaises(xiaoai.SpeakerError):
                xiaoai.xiaomi_url(value)
        self.assertEqual(xiaoai.xiaomi_url("https://api2.mina.mi.com/sts"),
                         "https://api2.mina.mi.com/sts")

    def test_canvas_preserves_previous_content_and_removes_completed_qr(self):
        canvas = self.root / "board.html"
        canvas.write_text("<p>previous canvas</p>")
        qr = {"image": b"image bytes", "loginUrl": "https://account.xiaomi.com/qr",
              "expires_at": time.time() + 60}
        xiaoai.render_canvas(canvas, "扫码", qr)
        self.assertIn("board.xiaoai-qr.png", canvas.read_text())
        self.assertEqual((self.root / "board.before-xiaoai.html").read_text(),
                         "<p>previous canvas</p>")
        xiaoai.render_canvas(canvas, "已发送")
        self.assertFalse((self.root / "board.xiaoai-qr.png").exists())
        self.assertNotIn(qr["loginUrl"], canvas.read_text())


if __name__ == "__main__":
    unittest.main()
