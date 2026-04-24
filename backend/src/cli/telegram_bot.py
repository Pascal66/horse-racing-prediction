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

# Application.builder().token(BOT_TOKEN).read_timeout(30).write_timeout(30).connect_timeout(30).pool_timeout(30).build()

bot = telegram.Bot(token=HORACE_TOKEN) if HORACE_TOKEN else None

# async def send_telegram_message(message: str):
#     if not bot:
#         logger.error("Telegram bot not initialized (missing HORACE_TOKEN)")
#         return
#     try:
#         await bot.send_message(chat_id=HORACE_ID, text=message, parse_mode='HTML')
#         logger.info("Telegram message sent successfully.")
#     except Exception as e:
#         logger.error(f"Failed to send Telegram message: {e}")

async def send_telegram_message(text: str):
    """Envoie un message via Telegram en gérant proprement le cycle de vie de la boucle d'événements."""
    token = os.getenv("HORACE_TOKEN")
    chat_id = os.getenv("HORACE_ID")
    if not token or not chat_id:
        logger.warning("Configuration Telegram manquante (TOKEN ou CHAT_ID).")
        return

    from telegram import Bot
    from telegram.constants import ParseMode

    try:
        # Utiliser 'async with' pour éviter l'erreur 'Event loop is closed'
        # Cela recrée un client HTTP propre pour chaque appel dans APScheduler
        async with Bot(token=token) as bot:
            await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi Telegram : {str(e)}")

async def main():
    await send_telegram_message("Hello darkness, my old friend")

if __name__ == "__main__":
    asyncio.run(main())