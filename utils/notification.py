#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通知模块 - 支持企业微信、钉钉、飞书、邮件
"""
import os
import requests
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from typing import Optional, List, Dict
from dataclasses import dataclass

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


# ==================== 基础配置 ====================

@dataclass
class WeChatConfig:
    """企业微信机器人配置"""
    webhook_url: str
    mentioned_list: Optional[List[str]] = None
    mentioned_mobile_list: Optional[List[str]] = None


@dataclass
class DingTalkConfig:
    """钉钉机器人配置"""
    webhook_url: str
    secret: Optional[str] = None  # 加签密钥
    at_mobiles: Optional[List[str]] = None
    at_user_ids: Optional[List[str]] = None


@dataclass
class FeishuConfig:
    """飞书机器人配置"""
    webhook_url: str


@dataclass
class EmailConfig:
    """邮件配置"""
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    from_addr: str
    to_addrs: List[str]


# ==================== 企业微信 ====================

class WeChatNotifier:
    """企业微信通知器"""

    def __init__(self, config: WeChatConfig):
        self.config = config
        self.webhook_url = config.webhook_url

    def send_markdown(self, content: str) -> bool:
        """发送Markdown格式消息"""
        data = {
            "msgtype": "markdown",
            "markdown": {
                "content": content
            }
        }

        if self.config.mentioned_list:
            data["markdown"]["mentioned_list"] = self.config.mentioned_list
        if self.config.mentioned_mobile_list:
            data["markdown"]["mentioned_mobile_list"] = self.config.mentioned_mobile_list

        return self._send(data)

    def send_text(self, content: str) -> bool:
        """发送文本消息"""
        data = {
            "msgtype": "text",
            "text": {
                "content": content
            }
        }

        if self.config.mentioned_list:
            data["text"]["mentioned_list"] = self.config.mentioned_list
        if self.config.mentioned_mobile_list:
            data["text"]["mentioned_mobile_list"] = self.config.mentioned_mobile_list

        return self._send(data)

    def _send(self, data: dict) -> bool:
        """发送消息"""
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                self.webhook_url,
                headers=headers,
                data=json.dumps(data, ensure_ascii=False).encode('utf-8'),
                timeout=10
            )

            result = response.json()
            return result.get('errcode') == 0

        except Exception as e:
            print(f"[ERR] 企业微信发送失败: {e}")
            return False


# ==================== 钉钉 ====================

class DingTalkNotifier:
    """钉钉通知器"""

    def __init__(self, config: DingTalkConfig):
        self.config = config
        self.webhook_url = config.webhook_url

    def _get_sign(self, timestamp: int) -> str:
        """计算签名（如果配置了secret）"""
        import hmac
        import hashlib
        import base64
        import urllib.parse

        if not self.config.secret:
            return ""

        secret_enc = self.config.secret.encode('utf-8')
        string_to_sign = f'{timestamp}\n{self.config.secret}'
        string_to_sign_enc = string_to_sign.encode('utf-8')

        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

        return sign

    def send_markdown(self, title: str, content: str) -> bool:
        """发送Markdown消息"""
        import time

        data = {
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": content
            }
        }

        # 添加@信息
        at = {"isAtAll": False}
        if self.config.at_mobiles:
            at["atMobiles"] = self.config.at_mobiles
            at["atUserIds"] = self.config.at_user_ids or []
            data["markdown"]["text"] += f"\n@{', @'.join(self.config.at_mobiles)}"
        data["at"] = at

        # 如果有secret，添加签名和timestamp
        if self.config.secret:
            timestamp = int(time.time() * 1000)
            sign = self._get_sign(timestamp)
            self.webhook_url += f"&timestamp={timestamp}&sign={sign}"

        return self._send(data)

    def send_text(self, content: str) -> bool:
        """发送文本消息"""
        import time

        data = {
            "msgtype": "text",
            "text": {
                "content": content
            }
        }

        at = {"isAtAll": False}
        if self.config.at_mobiles:
            at["atMobiles"] = self.config.at_mobiles
            data["text"]["content"] += f"\n@{', @'.join(self.config.at_mobiles)}"
        data["at"] = at

        if self.config.secret:
            timestamp = int(time.time() * 1000)
            sign = self._get_sign(timestamp)
            self.webhook_url += f"&timestamp={timestamp}&sign={sign}"

        return self._send(data)

    def _send(self, data: dict) -> bool:
        """发送消息"""
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                self.webhook_url,
                headers=headers,
                data=json.dumps(data, ensure_ascii=False).encode('utf-8'),
                timeout=10
            )

            result = response.json()
            return result.get('errcode') == 0

        except Exception as e:
            print(f"[ERR] 钉钉发送失败: {e}")
            return False


# ==================== 飞书 ====================

class FeishuNotifier:
    """飞书通知器"""

    def __init__(self, config: FeishuConfig):
        self.config = config
        self.webhook_url = config.webhook_url

    def send_text(self, content: str) -> bool:
        """发送文本消息"""
        data = {
            "msg_type": "text",
            "content": {
                "text": content
            }
        }

        return self._send(data)

    def send_post(self, title: str, content: str) -> bool:
        """
        发送富文本/卡片消息

        飞书使用卡片消息展示更丰富的内容
        """
        data = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": title
                    },
                    "template": "orange"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "plain_text",
                            "content": content
                        }
                    }
                ]
            }
        }

        return self._send(data)

    def send_card(self, title: str, elements: List[Dict]) -> bool:
        """
        发送卡片消息

        Args:
            title: 卡片标题
            elements: 卡片内容元素列表
                [{"tag": "div", "text": {"tag": "plain_text", "content": "内容"}}]
        """
        data = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": title
                    },
                    "template": "orange"
                },
                "elements": elements
            }
        }

        return self._send(data)

    def _send(self, data: dict) -> bool:
        """发送消息"""
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                self.webhook_url,
                headers=headers,
                data=json.dumps(data, ensure_ascii=False).encode('utf-8'),
                timeout=10
            )

            result = response.json()
            return result.get('code') == 0 or result.get('StatusCode') == 0

        except Exception as e:
            print(f"[ERR] 飞书发送失败: {e}")
            return False


# ==================== 邮件 ====================

class EmailNotifier:
    """邮件通知器"""

    def __init__(self, config: EmailConfig):
        self.config = config

    def send_text(self, subject: str, content: str) -> bool:
        """发送纯文本邮件"""
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['From'] = self.config.from_addr
        msg['To'] = ', '.join(self.config.to_addrs)
        msg['Subject'] = Header(subject, 'utf-8')

        return self._send(msg)

    def send_html(self, subject: str, html_content: str) -> bool:
        """发送HTML邮件"""
        msg = MIMEText(html_content, 'html', 'utf-8')
        msg['From'] = self.config.from_addr
        msg['To'] = ', '.join(self.config.to_addrs)
        msg['Subject'] = Header(subject, 'utf-8')

        return self._send(msg)

    def _send(self, msg: MIMEMultipart) -> bool:
        """发送邮件"""
        try:
            if self.config.smtp_port == 465:
                server = smtplib.SMTP_SSL(self.config.smtp_host, self.config.smtp_port)
            else:
                server = smtplib.SMTP(self.config.smtp_host, self.config.smtp_port)
                server.starttls()

            server.login(self.config.username, self.config.password)
            server.send_message(msg)
            server.quit()

            return True

        except Exception as e:
            print(f"[ERR] 邮件发送失败: {e}")
            return False


# ==================== 通知管理器 ====================

class NotificationManager:
    """统一通知管理器"""

    def __init__(self):
        self.wechat: Optional[WeChatNotifier] = None
        self.dingtalk: Optional[DingTalkNotifier] = None
        self.feishu: Optional[FeishuNotifier] = None
        self.email: Optional[EmailNotifier] = None

    def add_wechat(self, config: WeChatConfig):
        self.wechat = WeChatNotifier(config)

    def add_dingtalk(self, config: DingTalkConfig):
        self.dingtalk = DingTalkNotifier(config)

    def add_feishu(self, config: FeishuConfig):
        self.feishu = FeishuNotifier(config)

    def add_email(self, config: EmailConfig):
        self.email = EmailNotifier(config)

    def send_all(self, title: str, content: str, html_content: str = None):
        """发送到所有已配置的通知渠道"""
        results = {}

        if self.wechat:
            results['wechat'] = self.wechat.send_markdown(f"## {title}\n\n{content}")

        if self.dingtalk:
            results['dingtalk'] = self.dingtalk.send_markdown(title, content)

        if self.feishu:
            results['feishu'] = self.feishu.send_post(title, content)

        if self.email:
            if html_content:
                results['email'] = self.email.send_html(title, html_content)
            else:
                results['email'] = self.email.send_text(title, content)

        return results


# ==================== 配置加载 ====================

def load_notification_config(config_file: str = None) -> NotificationManager:
    """
    加载所有通知配置

    Args:
        config_file: 配置文件路径

    Returns:
        NotificationManager实例
    """
    import os
    from pathlib import Path

    manager = NotificationManager()
    config_paths = [
        Path(config_file) if config_file else None,
        Path.home() / '.claude' / 'notification.json',
        Path('.') / 'config' / 'notification.json',
        Path('.') / 'notification_config.json',
    ]

    config = {}
    for config_path in config_paths:
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                break
            except Exception as e:
                print(f"[WARN] 读取配置失败 {config_path}: {e}")

    # 从环境变量加载
    # 飞书
    feishu_url = os.getenv('FEISHU_WEBHOOK_URL')
    if feishu_url:
        manager.add_feishu(FeishuConfig(webhook_url=feishu_url))

    # 邮件
    email_host = os.getenv('EMAIL_SMTP_HOST')
    if email_host:
        manager.add_email(EmailConfig(
            smtp_host=email_host,
            smtp_port=int(os.getenv('EMAIL_SMTP_PORT', 465)),
            username=os.getenv('EMAIL_USERNAME') or (config.get('email') or {}).get('username'),
            password=os.getenv('EMAIL_PASSWORD') or (config.get('email') or {}).get('password'),
            from_addr=os.getenv('EMAIL_FROM') or (config.get('email') or {}).get('from_addr'),
            to_addrs=os.getenv('EMAIL_TO', '').split(',') or (config.get('email') or {}).get('to_addrs', [])
        ))

    return manager


# ==================== 信号格式化 ====================

def format_2buy_alert(signal, monitor_name: str = "缠论30分钟监控") -> str:
    """格式化2买信号为Markdown"""
    return f"""### {monitor_name} - 发现2买信号

> **股票**: {signal.symbol} {signal.name}
> **时间**: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}
> **价格**: {signal.price:.2f}
> **置信度**: {signal.confidence*100:.0f}%

**交易建议**:
- 止损: {signal.stop_loss:.2f}
- 目标: {signal.target:.2f}

**理由**: {signal.reason}
"""


def format_2buy_html(signal, monitor_name: str = "缠论30分钟监控") -> str:
    """格式化2买信号为HTML（邮件用）"""
    return f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .header {{ background: #ff6b6b; color: white; padding: 15px; }}
        .content {{ padding: 20px; }}
        .info {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .trade {{ background: #e7f3ff; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .reason {{ background: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>{monitor_name} - 发现2买信号</h2>
    </div>
    <div class="content">
        <div class="info">
            <h3>股票信息</h3>
            <p><strong>代码:</strong> {signal.symbol}</p>
            <p><strong>名称:</strong> {signal.name}</p>
            <p><strong>时间:</strong> {signal.timestamp.strftime('%Y-%m-%d %H:%M')}</p>
            <p><strong>价格:</strong> <span style="color:red;font-size:18px">{signal.price:.2f}</span></p>
            <p><strong>置信度:</strong> {signal.confidence*100:.0f}%</p>
        </div>
        <div class="trade">
            <h3>交易建议</h3>
            <p><strong>止损:</strong> {signal.stop_loss:.2f}</p>
            <p><strong>目标:</strong> {signal.target:.2f}</p>
        </div>
        <div class="reason">
            <h3>理由</h3>
            <p>{signal.reason}</p>
        </div>
    </div>
</body>
</html>
"""


def format_summary_alert(signals, monitor_name: str = "缠论30分钟监控") -> str:
    """格式化汇总提醒"""
    if not signals:
        return f"### {monitor_name} - 扫描完成\n\n未发现2买信号"

    lines = [
        f"### {monitor_name} - 发现{len(signals)}个2买信号\n",
        f"**扫描时间**: {signals[0].timestamp.strftime('%Y-%m-%d %H:%M')}\n",
        "**信号列表**:\n"
    ]

    for i, s in enumerate(signals, 1):
        lines.append(
            f"{i}. **{s.symbol} {s.name}**\n"
            f"   - 价格: `{s.price:.2f}`  "
            f"止损: `{s.stop_loss:.2f}`  "
            f"目标: `{s.target:.2f}`  "
            f"置信度: `{s.confidence*100:.0f}%`\n"
            f"   - {s.reason}\n"
        )

    return "".join(lines)
