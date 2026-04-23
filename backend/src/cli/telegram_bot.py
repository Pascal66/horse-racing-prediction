import asyncio
import telegram
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger("TelegramBot")

HORACE_TOKEN = os.getenv("HORACE_TOKEN")
HORACE_ID = os.getenv("HORACE_ID")
# TELEGRAM_TOKEN = "your_bot_token_here"
# TELEGRAM_CHAT_ID = "your_chat_id_here"

bot = telegram.Bot(token=HORACE_TOKEN) if HORACE_TOKEN else None

async def send_telegram_message(message: str):
    if not bot:
        logger.error("Telegram bot not initialized (missing HORACE_TOKEN)")
        return
    try:
        await bot.send_message(chat_id=HORACE_ID, text=message, parse_mode='HTML')
        logger.info("Telegram message sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

async def main():
    await send_telegram_message("Hello darkness, my old friend")

if __name__ == "__main__":
    asyncio.run(main())