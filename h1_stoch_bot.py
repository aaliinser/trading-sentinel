#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1 Stochastic Extreme Reversal Bot - Layer 1: Skeleton & State Management
الهدف: إنشاء الهيكل الأساسي ونظام إدارة الحالة دون منطق تداول بعد.
"""
import os, sys, time, json, logging
from datetime import datetime, timezone
from pathlib import Path
import requests

# ═══════════════════════════════════════════════
# 1. الإعدادات العامة (Global Config)
# ═══════════════════════════════════════════════
BOT_NAME = "H1_Stoch_Bot"
STATE_FILE = "state_h1.json"
LOG_LEVEL = logging.INFO

# Telegram Credentials (من متغيرات البيئة)
TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
TG_CHAT = os.getenv("TG_CHAT", "").strip()

# إعدادات اللوجينج
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(BOT_NAME)

# ═══════════════════════════════════════════════
# 2. أدوات مساعدة (Utilities)
# ═══════════════════════════════════════════════
def send_telegram(msg: str):
    """إرسال رسالة نصية إلى تليجرام."""
    if not TG_TOKEN or not TG_CHAT:
        log.warning("Telegram credentials missing. Skipping send.")
        return None
    
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT,
        "text": msg,
        "parse_mode": "Markdown"
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            log.info("Message sent successfully to Telegram.")
            return True
        else:
            log.error(f"Telegram API Error: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        log.error(f"Exception while sending Telegram message: {e}")
        return False

# ═══════════════════════════════════════════════
# 3. إدارة الحالة (State Management)
# ═══════════════════════════════════════════════
class StateManager:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.data = {"pending_trades": [], "history": []}
        self.load()

    def load(self):
        """قراءة ملف الحالة إذا كان موجوداً."""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    # دمج البيانات المحملة مع الهيكل الافتراضي لضمان وجود الحقول
                    self.data.update(loaded_data)
                log.info("State loaded successfully from disk.")
            except Exception as e:
                log.error(f"Failed to load state file: {e}. Starting fresh.")
                self.save() # إعادة إنشاء الملف الفارغ
        else:
            log.info("No existing state file found. Creating new one.")
            self.save()

    def save(self):
        """حفظ الحالة الحالية على القرص."""
        try:
            # كتابة مؤقتة ثم استبدال لضمان سلامة الملف (Atomic Write)
            temp_file = self.filepath.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, default=str)
            
            # استبدال الملف القديم بالجديد
            if self.filepath.exists():
                self.filepath.unlink()
            temp_file.rename(self.filepath)
            
            log.debug("State saved successfully.")
        except Exception as e:
            log.critical(f"CRITICAL ERROR saving state: {e}")
            raise

    def add_pending_trade(self, trade_info: dict):
        """إضافة صفقة جديدة لقائمة الانتظار."""
        self.data["pending_trades"].append(trade_info)
        self.save()
        log.info(f"Added pending trade for {trade_info.get('symbol')}")

    def get_pending_trades(self):
        """استرجاع قائمة الصفقات المعلقة."""
        return self.data.get("pending_trades", [])

    def remove_pending_trade(self, trade_id: str):
        """حذف صفقة من قائمة الانتظار بعد حسمها."""
        self.data["pending_trades"] = [
            t for t in self.data["pending_trades"] 
            if t.get("id") != trade_id
        ]
        self.save()
        log.info(f"Removed resolved trade {trade_id}")

    def archive_trade(self, trade_result: dict):
        """نقل الصفقة المحسومة إلى سجل التاريخ."""
        self.data["history"].append(trade_result)
        # optional: limit history size
        if len(self.data["history"]) > 1000:
            self.data["history"] = self.data["history"][-1000:]
        self.save()

# ═══════════════════════════════════════════════
# 4. الحلقة الرئيسية (Main Loop Placeholder)
# ═══════════════════════════════════════════════
def main():
    log.info("="*50)
    log.info(f"Starting {BOT_NAME} - Layer 1 Initialization")
    log.info("="*50)

    # تهيئة مدير الحالة
    sm = StateManager(STATE_FILE)
    
    # اختبار إرسال رسالة تليجرام
    test_msg = f"✅ **{BOT_NAME} Initialized Successfully!**\nLayer 1 (Skeleton & State) is online.\nTime: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    send_telegram(test_msg)

    # عرض حالة النظام الحالية
    pending_count = len(sm.get_pending_trades())
    log.info(f"Current Pending Trades Count: {pending_count}")

    log.info("Layer 1 Complete. Ready for next layer instructions.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Bot stopped manually by user.")
    except Exception as e:
        log.exception(f"Fatal error occurred: {e}")
        send_telegram(f"🚨 **CRASH ALERT**\nError: `{str(e)[:200]}`")
