import asyncio
import telegram
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

HORACE_TOKEN = os.getenv("HORACE_TOKEN")
HORACE_ID = os.getenv("HORACE_ID")
# TELEGRAM_TOKEN = "your_bot_token_here"
# TELEGRAM_CHAT_ID = "your_chat_id_here"

bot = telegram.Bot(token=HORACE_TOKEN)

async def send_telegram_message(message):
    await bot.send_message(chat_id=HORACE_ID, text=message)

async def main():
    await send_telegram_message("Hello darkness, my old friend")

if __name__ == "__main__":
    asyncio.run(main())