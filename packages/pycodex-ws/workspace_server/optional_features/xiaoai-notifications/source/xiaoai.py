#!/usr/bin/env python3
"""Opt-in Xiaomi speaker login and notification utility."""
import argparse
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import hmac
import html
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import sys
import time
from urllib.parse import quote, urlsplit

import requests


ACCOUNT = "https://account.xiaomi.com"
MINA = "https://api2.mina.mi.com"
SID = "micoapi"
USER_AGENT = (
    "MiHome/6.0.103 (com.xiaomi.mihome; build:6.0.103.1; "
    "iOS 14.4.0) Alamofire/6.0.103 MICO/iOSApp/appStore/6.0.103"
)
TEST_TEXT = "这是小爱音箱的通知测试。"
DEFAULT_STATE = Path.home() / ".config" / "pycodex-xiaoai"


class SpeakerError(Exception):
    """A safe-to-display error; never include a raw authentication response."""

class QuietHours(SpeakerError):
    pass


def quiet_hours(now=None):
    zone = timezone(timedelta(hours=8))
    return 1 <= (now or datetime.now(zone)).astimezone(zone).hour < 10


def check_speech_time():
    if quiet_hours():
        raise QuietHours("北京时间01:00至10:00免打扰，未播报。")


def private_json(path, data):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + secrets.token_hex(4) + ".tmp")
    try:
        with os.fdopen(os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                       "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def account_json(response):
    response.raise_for_status()
    text = response.text
    if text.startswith("&&&START&&&"):
        text = text[len("&&&START&&&"):]
    try:
        data = json.loads(text)
    except ValueError:
        raise SpeakerError("小米登录服务返回了非 JSON 响应，请稍后重试。") from None
    if not isinstance(data, dict):
        raise SpeakerError("小米登录服务响应格式异常。")
    return data


def xiaomi_url(value):
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not any(
            host == domain or host.endswith("." + domain)
            for domain in ("xiaomi.com", "mi.com")):
        raise SpeakerError("小米服务返回了无法识别的跳转地址。")
    return value


class Speaker:
    def __init__(self, state_dir=DEFAULT_STATE):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.state_dir.chmod(0o700)
        self.token_path = self.state_dir / "token.json"
        self.config_path = self.state_dir / "speaker.json"
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.token = None
        if self.token_path.exists():
            self.token = json.loads(self.token_path.read_text(encoding="utf-8"))

    def begin_login(self):
        seed = account_json(self.session.get(
            ACCOUNT + "/pass/serviceLogin",
            params={"sid": SID, "_json": "true"}, timeout=(10, 25)))
        if not seed.get("callback") or not seed.get("qs"):
            raise SpeakerError("未能取得小米扫码登录参数。")
        params = {
            "_qrsize": "480", "qs": seed["qs"], "callback": seed["callback"],
            "_hasLogo": "false", "sid": SID, "serviceParam": "",
            "_locale": "zh_CN", "_dc": str(int(time.time() * 1000)),
        }
        qr = account_json(self.session.get(
            ACCOUNT + "/longPolling/loginUrl", params=params, timeout=(10, 25)))
        if not all(qr.get(key) for key in ("qr", "lp", "loginUrl")):
            raise SpeakerError("未能生成小米登录二维码。")
        for key in ("qr", "lp", "loginUrl"):
            xiaomi_url(qr[key])
        response = self.session.get(qr["qr"], timeout=(10, 25))
        response.raise_for_status()
        if not response.headers.get("Content-Type", "").startswith("image/"):
            raise SpeakerError("小米登录二维码不是有效图片。")
        qr["image"] = response.content
        qr["expires_at"] = time.time() + min(int(qr.get("timeout", 300)), 600)
        return qr

    def finish_login(self, qr):
        while time.time() < qr["expires_at"]:
            try:
                response = self.session.get(qr["lp"], timeout=(10, 25))
            except requests.Timeout:
                continue
            data = account_json(response)
            if all(data.get(key) for key in ("userId", "passToken", "ssecurity", "location")):
                self.token = {
                    "deviceId": secrets.token_hex(8).upper(),
                    "userId": str(data["userId"]), "passToken": data["passToken"],
                    SID: [data["ssecurity"], self.exchange_token(data)],
                }
                private_json(self.token_path, self.token)
                return
            if data.get("code") not in (None, 0):
                raise SpeakerError("二维码已失效或登录被取消，请重新生成。")
            # A nonterminal response is not evidence of authorization.
            time.sleep(1)
        raise SpeakerError("登录二维码已过期，请重新生成并扫码。")

    def exchange_token(self, data):
        location = xiaomi_url(data["location"])
        if data.get("nonce") is not None:
            message = "nonce={}&{}".format(data["nonce"], data["ssecurity"])
            signature = base64.b64encode(hashlib.sha1(message.encode()).digest()).decode()
            location += ("&" if "?" in location else "?") + "clientSign=" + quote(signature)
        response = self.session.get(location, timeout=(10, 25))
        response.raise_for_status()
        for item in [response] + list(reversed(response.history)):
            for cookie in item.cookies:
                if cookie.name == "serviceToken":
                    return cookie.value
        raise SpeakerError("扫码已确认，但未取得音箱服务登录态。")

    def refresh_login(self, sid=SID):
        if not self.token:
            raise SpeakerError("尚未登录，请先执行 login。")
        data = account_json(self.session.get(
            ACCOUNT + "/pass/serviceLogin",
            params={"sid": sid, "_json": "true"},
            cookies={"userId": self.token["userId"],
                     "passToken": self.token["passToken"],
                     "deviceId": self.token["deviceId"], "sdkVersion": "3.9"},
            timeout=(10, 25)))
        if data.get("code") != 0 or not data.get("location"):
            raise SpeakerError("小米登录态已失效，需要重新扫码。")
        self.token[sid] = [data["ssecurity"], self.exchange_token(data)]
        if data.get("passToken"):
            self.token["passToken"] = data["passToken"]
        private_json(self.token_path, self.token)

    def api(self, path, data=None):
        if not self.token:
            raise SpeakerError("尚未登录，请先执行 login。")
        for attempt in range(2):
            params = {"requestId": "app_ios_" + secrets.token_hex(15)}
            kwargs = {"cookies": {"userId": self.token["userId"],
                                  "serviceToken": self.token[SID][1]},
                      "timeout": (10, 25)}
            if data is None:
                response = self.session.get(MINA + path, params=params, **kwargs)
            else:
                if data.get("method") == "text_to_speech":
                    check_speech_time()
                response = self.session.post(MINA + path, data=dict(data, **params), **kwargs)
            # Expired service tokens can return an HTML 401 page, not JSON.
            if response.status_code == 401:
                if attempt == 0:
                    self.refresh_login()
                    continue
                raise SpeakerError("小米音箱授权失败（HTTP 401），请重新登录。")
            try:
                result = response.json()
            except ValueError:
                raise SpeakerError("小米音箱服务返回了无法识别的响应。") from None
            if response.status_code == 200 and result.get("code") == 0:
                return result
            if attempt == 0 and "auth" in str(result.get("message", "")).lower():
                self.refresh_login()
                continue
            raise SpeakerError("小米音箱请求失败（HTTP {}，错误码 {}）。".format(
                response.status_code, result.get("code", "unknown")))
        raise SpeakerError("小米音箱授权失败。")

    def devices(self):
        devices = self.api("/admin/v2/device_list?master=0").get("data")
        if not isinstance(devices, list):
            raise SpeakerError("未取得有效的音箱列表。")
        return devices

    def miot_api(self, path, data):
        if not self.token:
            raise SpeakerError("尚未登录，请先执行 login。")
        if "xiaomiio" not in self.token:
            self.refresh_login("xiaomiio")
        for attempt in range(2):
            security, service_token = self.token["xiaomiio"]
            nonce_bytes = secrets.token_bytes(8) + int(time.time() / 60).to_bytes(4, "big")
            nonce = base64.b64encode(nonce_bytes).decode()
            signed_nonce = hashlib.sha256(base64.b64decode(security) + nonce_bytes).digest()
            payload = json.dumps(data)
            message = "&".join([
                path, base64.b64encode(signed_nonce).decode(), nonce, "data=" + payload])
            signature = base64.b64encode(
                hmac.new(signed_nonce, message.encode(), hashlib.sha256).digest()).decode()
            if path == "/miotspec/action":
                check_speech_time()
            response = self.session.post(
                "https://api.io.mi.com/app" + path,
                data={"_nonce": nonce, "data": payload, "signature": signature},
                cookies={"userId": self.token["userId"], "serviceToken": service_token,
                         "PassportDeviceId": self.token["deviceId"]},
                headers={"x-xiaomi-protocal-flag-cli": "PROTOCAL-HTTP2"},
                timeout=(10, 25))
            if response.status_code == 401:
                if attempt == 0:
                    self.refresh_login("xiaomiio")
                    continue
                raise SpeakerError("米家设备授权失败（HTTP 401），请重新登录。")
            try:
                result = response.json()
            except ValueError:
                raise SpeakerError("米家设备服务返回了无法识别的响应。") from None
            if response.status_code == 200 and result.get("code") == 0 and "result" in result:
                return result["result"]
            if attempt == 0 and "auth" in str(result.get("message", "")).lower():
                self.refresh_login("xiaomiio")
                continue
            raise SpeakerError("米家设备请求失败（HTTP {}，错误码 {}）。".format(
                response.status_code, result.get("code", "unknown")))
        raise SpeakerError("米家设备授权失败。")

    def say_miot(self, device, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 200:
            raise SpeakerError("播报内容必须为 1–200 个字符。")
        # X08C and X08A expose the same action at different service indices.
        service_id = {"X08C": 3, "X08A": 5}.get(device.get("hardware"))
        if service_id is None or not device.get("miotDID"):
            raise SpeakerError("这台音箱尚未配置经过核实的 MIoT 文字播报动作。")
        result = self.miot_api("/miotspec/action", {"params": {
            "did": device["miotDID"], "siid": service_id, "aiid": 1, "in": [text]}})
        if not isinstance(result, dict) or result.get("code") != 0:
            code = result.get("code", "unknown") if isinstance(result, dict) else "unknown"
            raise SpeakerError("音箱文字播报动作失败（设备错误码 {}）。".format(code))
        return result

    def selected(self, devices):
        if self.config_path.exists():
            selected_id = json.loads(self.config_path.read_text(encoding="utf-8"))["device_id"]
            matches = [d for d in devices if d.get("deviceID") == selected_id]
        else:
            matches = []
        if len(matches) != 1:
            raise SpeakerError("请先用 devices 查看音箱，再用 select 指定一个设备。")
        return matches[0]

    def say(self, device_id, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 200:
            raise SpeakerError("播报内容必须为 1–200 个字符。")
        message = json.dumps({"text": text}, ensure_ascii=False)
        result = self.api("/remote/ubus", {
            "deviceId": device_id, "method": "text_to_speech",
            "path": "mibrain", "message": message,
        })
        nested = result.get("data")
        if isinstance(nested, dict) and nested.get("code", 0) != 0:
            raise SpeakerError("音箱拒绝了播报请求。")


def render_canvas(path, message, qr=None):
    path = Path(path)
    backup = path.with_name(path.stem + ".before-xiaoai.html")
    if path.exists() and not backup.exists():
        shutil.copyfile(path, backup)
        backup.chmod(0o600)
    image_path = path.with_name(path.stem + ".xiaoai-qr.png")
    content = ""
    countdown = ""
    if qr:
        with os.fdopen(os.open(str(image_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600),
                       "wb") as stream:
            stream.write(qr["image"])
        content = (
            '<img alt="小米账号登录二维码" src="{}?v={}">'
            '<p>请用绑定音箱的小米账号扫码，并在手机上确认登录。</p>'
            '<p><a href="{}" target="_blank" rel="noreferrer">在手机上打开小米登录页</a></p>'
            '<p id="expiry"></p>'
        ).format(html.escape(image_path.name), int(time.time()), html.escape(qr["loginUrl"], quote=True))
        countdown = """<script>
const deadline = %d;
function tick() {
  const left = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
  document.getElementById('expiry').textContent = left
    ? '二维码剩余有效时间：' + left + ' 秒'
    : '二维码已过期，请回到会话告诉我重新生成。';
}
tick(); setInterval(tick, 1000);
</script>""" % int(qr["expires_at"] * 1000)
    elif image_path.exists():
        image_path.unlink()
    document = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>小爱音箱 · 通知试播</title>
<style>
body{margin:0;background:#f5f6f8;color:#242630;font:16px/1.7 system-ui,sans-serif}
main{max-width:620px;margin:5vh auto;padding:32px;background:white;border-radius:20px;text-align:center}
h1{font-size:26px;margin:12px 0} .label{color:#b65b0b;font-size:13px;letter-spacing:.12em}
img{width:min(320px,85%%);height:auto;display:block;margin:24px auto}
a{color:#8a4c0d} .sub{color:#626671;font-size:14px}
</style><main><div class="label">小爱触屏音箱 8</div>
<h1>%s</h1>%s<p class="sub">测试语音：%s</p>
<p class="sub">登录态保存在本机私有目录；密码和令牌不会显示在会话里。</p>
</main>%s</html>""" % (html.escape(message), content, html.escape(TEST_TEXT), countdown)
    path.write_text(document, encoding="utf-8")


def device_summary(device, index):
    return {"index": index, "name": device.get("name", ""),
            "hardware": device.get("hardware", "")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    sub = parser.add_subparsers(dest="command", required=True)
    login = sub.add_parser("login")
    login.add_argument("--canvas", type=Path, required=True)
    login.add_argument("--test", action="store_true")
    sub.add_parser("devices")
    select = sub.add_parser("select")
    select.add_argument("index", type=int, help="1-based index from devices")
    say = sub.add_parser("say")
    say.add_argument("text")
    say.add_argument("--transport", choices=("mina", "miot"), default="miot")
    test = sub.add_parser("test")
    test.add_argument("--transport", choices=("mina", "miot"), default="miot")
    sub.add_parser("deliver", help="workspace-owned delivery; reads a short speech JSON from stdin")
    args = parser.parse_args()
    os.umask(0o077)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    try:
        speaker = Speaker(args.state_dir)
        if args.command == "deliver":
            from summarize import validate_speech
            text = validate_speech(json.load(sys.stdin)["speech"])
            check_speech_time()
            device = speaker.selected(speaker.devices())
            speaker.say_miot(device, text)
            print(json.dumps({"state": "accepted"}, ensure_ascii=False), flush=True)
            return 0
        if args.command == "login":
            qr = speaker.begin_login()
            render_canvas(args.canvas, "扫码登录小米账号", qr)
            print("二维码已显示在 Canvas，等待手机扫码确认。", flush=True)
            speaker.finish_login(qr)
            render_canvas(args.canvas, "小米账号已登录，正在识别音箱")
            print("小米账号登录成功。", flush=True)
        devices = speaker.devices()
        if args.command == "devices" or (args.command == "login" and not args.test):
            print(json.dumps([device_summary(d, i) for i, d in enumerate(devices, 1)],
                             ensure_ascii=False))
            return 0
        if args.command == "select":
            if not 1 <= args.index <= len(devices):
                raise SpeakerError("设备序号超出范围。")
            device = devices[args.index - 1]
        else:
            device = speaker.selected(devices)
        private_json(speaker.config_path, {
            "device_id": device["deviceID"], "name": device.get("name", ""),
            "hardware": device.get("hardware", ""),
        })
        print("目标音箱：" + json.dumps(device_summary(device, devices.index(device) + 1),
                                     ensure_ascii=False), flush=True)
        if args.command == "select":
            return 0
        text = args.text if args.command == "say" else TEST_TEXT
        if getattr(args, "transport", "miot") == "miot":
            speaker.say_miot(device, text)
            print("音箱文字播报动作返回设备码 0，实际出声需听音确认。", flush=True)
        else:
            speaker.say(device["deviceID"], text)
            print("小米服务已接受播报请求，实际出声需听音确认。", flush=True)
        if args.command == "login":
            render_canvas(args.canvas, "已发送测试播报，请留意音箱声音")
        return 0
    except QuietHours as error:
        if args.command == "deliver":
            print(json.dumps({"state": "quiet_hours"}), flush=True)
            return 0
        print(str(error), flush=True)
        return 0
    except (SpeakerError, requests.RequestException, ValueError, KeyError) as error:
        message = str(error) if isinstance(error, SpeakerError) else (
            "连接或响应异常（{}），未确认播报成功。".format(type(error).__name__))
        print(message, file=sys.stderr, flush=True)
        if args.command == "login":
            render_canvas(args.canvas, message)
        return 1


if __name__ == "__main__":
    sys.exit(main())
