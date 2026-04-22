"""微信消息推送 - 通过bridge HTTP API发送"""
import json, sys, urllib.request, os, time
from typing import Optional

BRIDGE_URL = os.getenv("WEIXIN_BRIDGE_URL", "http://localhost:9101")
USER_ID = ""
cred_path = os.path.expanduser("~/.hermes/weixin/credentials.json")
if os.path.exists(cred_path):
    with open(cred_path) as f:
        creds = json.load(f)
        USER_ID = creds.get("user_id", "")


def send_weixin(content: str) -> bool:
    """发送文本消息到微信（简单接口）"""
    if not USER_ID:
        print("[微信] 未配置user_id, 跳过")
        return False
    body = json.dumps({"to": USER_ID, "content": content}).encode()
    req = urllib.request.Request(
        f"{BRIDGE_URL}/send",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            print(f"[微信] 推送成功")
            return True
    except Exception as e:
        print(f"[微信] 推送失败: {e}")
        return False


class WeixinNotifier:
    """微信通知器（供scanner调用）"""

    def __init__(self):
        self._sent_keys: dict = {}  # 去重
        self.bridge_url = BRIDGE_URL
        self.user_id = USER_ID

    def is_configured(self) -> bool:
        return bool(self.user_id)

    def _send(self, text: str) -> bool:
        body = json.dumps({"to": self.user_id, "content": text}).encode()
        req = urllib.request.Request(
            f"{self.bridge_url}/send",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                json.loads(resp.read())
                return True
        except Exception as e:
            print(f"[微信] 推送失败: {e}")
            return False

    def _dedup(self, key: str) -> bool:
        """去重：同一key 1小时内不重复发送"""
        now = time.time()
        expired = [k for k, t in self._sent_keys.items() if now - t > 3600]
        for k in expired:
            del self._sent_keys[k]
        if key in self._sent_keys:
            return False
        self._sent_keys[key] = now
        return True

    def send_signal_text(self, symbol: str, point_type: str, action: str,
                         price: float, stop_loss: float = 0,
                         confidence: float = 0, reason: str = "",
                         target: float = 0, macd_status: str = "",
                         daily_trend: str = "") -> bool:
        """发送单个信号文本消息（供realtime_scanner用）"""
        key = f"{symbol}_{point_type}_{action}"
        if not self._dedup(key):
            return False

        is_sell = action == 'sell'
        is_div = '背驰' in point_type

        if is_div:
            emoji = "💰"
            desc = f"{point_type} 建议减半仓"
        elif is_sell:
            emoji = "⚠️"
            desc = f"{point_type} 建议卖出"
        else:
            emoji = "🟢"
            risk = (price - stop_loss) / price * 100 if stop_loss else 0
            desc = f"{point_type} 止损{stop_loss:.2f}(-{risk:.1f}%)"
            if target:
                desc += f" 目标{target:.2f}(+{(target/price-1)*100:.1f}%)"

        lines = [
            f"{emoji} 缠论信号: {symbol} {point_type}",
            f"价格: {price:.2f}",
            f"置信度: {confidence:.0%}",
        ]
        if macd_status:
            lines.append(f"MACD: {macd_status}")
        if daily_trend:
            lines.append(f"日线趋势: {daily_trend}")
        lines.append(desc)
        if reason:
            lines.append(f"理由: {reason}")
        lines.append(f"时间: {time.strftime('%Y-%m-%d %H:%M')}")

        return self._send("\n".join(lines))

    def send_buy_signals(self, signals, scan_date: str) -> bool:
        """日线买入信号推送（供live_scanner用）"""
        lines = [f"🟢 买入信号 | {scan_date} | {len(signals)}只", ""]
        for i, s in enumerate(signals, 1):
            risk = (s.price - s.stop_loss) / s.price * 100
            lines.append(
                f"{i}. {s.code} [{s.industry}] "
                f"价格{s.price:.2f} 仓位{s.position*100:.1f}% "
                f"止损{s.stop_loss:.2f}(-{risk:.1f}%)"
            )
        lines += ["", "缠论v12策略 Sharpe 3.99", "仅供参考, 需人工确认"]
        return self._send("\n".join(lines))

    def send_hold_signals(self, signals, scan_date: str) -> bool:
        """持仓信号推送（供live_scanner用）"""
        lines = [f"📊 当前持仓 | {scan_date} | {len(signals)}只", ""]
        total_pos = 0
        for i, s in enumerate(signals, 1):
            total_pos += s.position * 100
            lines.append(
                f"{i}. {s.code} [{s.industry}] "
                f"价格{s.price:.2f} 仓位{s.position*100:.1f}%"
            )
        lines += ["", f"总仓位: {total_pos:.1f}%"]
        return self._send("\n".join(lines))

    def send_no_signal(self, scan_date: str) -> bool:
        """无信号推送"""
        return self._send(f"📭 信号扫描 | {scan_date}\n今日无新信号")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: weixin_notify.py <message>")
        sys.exit(1)
    msg = sys.argv[1] if len(sys.argv) == 2 else " ".join(sys.argv[1:])
    ok = send_weixin(msg)
    sys.exit(0 if ok else 1)
