"""
Zhirinovsky Telegram Bot
Powered by OpenRouter (Gemini Flash) + RAG over speeches corpus + live news
"""
import asyncio
import logging
import os
from collections import defaultdict, deque
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import httpx

from prompt import ZHIRINOVSKY_SYSTEM_PROMPT
from news import get_top_news, get_news_for_context
from speeches import search_speeches, reload_corpus

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-flash-1.5")
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Per-user conversation history: user_id -> deque of {"role": ..., "content": ...}
conversation_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_CONTEXT_MESSAGES))


async def call_openrouter(messages: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://github.com/zhirinovsky-bot",
                "X-Title": "ZhirinovskyBot",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.9,
                "max_tokens": 1024,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def build_messages(user_id: int, user_text: str) -> list[dict]:
    """Assemble message list: system prompt + optional context injections + history + user msg."""
    # 1. Fetch relevant speeches for RAG context
    speech_context = search_speeches(user_text, top_k=2)

    # 2. Build system message
    system_content = ZHIRINOVSKY_SYSTEM_PROMPT
    if speech_context:
        system_content += f"\n\n{speech_context}"

    messages: list[dict] = [{"role": "system", "content": system_content}]

    # 3. Inject news digest as assistant "reminder" (hidden from user)
    # This was added at the start of the session or can be refreshed
    history = list(conversation_history[user_id])

    # If history is empty (new session), prepend a news injection
    if not history:
        messages.append({
            "role": "user",
            "content": "[Системное обновление: вот свежие новости для твоего контекста. Учти их при ответах.]"
        })
        messages.append({
            "role": "assistant",
            "content": "Понял, буду держать в голове!"
        })

    # 4. Add conversation history
    messages.extend(history)

    # 5. Add current user message
    messages.append({"role": "user", "content": user_text})

    return messages


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversation_history[user_id].clear()

    # Inject news into the very first exchange
    try:
        news_digest = await get_news_for_context()
    except Exception:
        news_digest = ""

    if news_digest:
        conversation_history[user_id].append({
            "role": "user",
            "content": f"[Новости дня: {news_digest}]"
        })
        conversation_history[user_id].append({
            "role": "assistant",
            "content": "Слышу, слышу! Слежу за обстановкой!"
        })

    greeting = (
        "Владимир Вольфович приветствует вас!\n\n"
        "Спрашивайте — отвечу на всё! Политика, история, жизнь, судьба России — "
        "Жириновский знает ответы на все вопросы!\n\n"
        "Команды:\n"
        "/news — последние новости\n"
        "/clear — начать разговор заново\n"
        "/speeches — перезагрузить базу выступлений"
    )
    await update.message.reply_text(greeting)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversation_history[user_id].clear()
    await update.message.reply_text("Разговор начат заново! Жириновский готов к новой беседе!")


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Загружаю свежие новости...")
    try:
        news = await get_top_news(limit=12)
        await update.message.reply_text(f"ПОСЛЕДНИЕ НОВОСТИ:\n\n{news}")
    except Exception as e:
        logger.error("News fetch error: %s", e)
        await update.message.reply_text("Не удалось загрузить новости. Видимо, враги мешают!")


async def speeches_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    count = reload_corpus()
    await update.message.reply_text(
        f"База выступлений перезагружена! Загружено {count} выступлений. "
        f"Жириновский готов цитировать себя!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_id = user.id
    user_text = update.message.text

    if not user_text:
        return

    logger.info("User %s (%s): %s", user_id, user.username, user_text[:80])

    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        messages = build_messages(user_id, user_text)
        reply = await call_openrouter(messages)
    except httpx.HTTPStatusError as e:
        logger.error("OpenRouter HTTP error: %s", e)
        reply = "Что-то случилось с линией связи! Враги глушат! Попробуйте ещё раз!"
    except Exception as e:
        logger.error("Error: %s", e)
        reply = "Произошла техническая неполадка. Но Жириновский не сдаётся! Попробуйте снова."

    # Save to history
    conversation_history[user_id].append({"role": "user", "content": user_text})
    conversation_history[user_id].append({"role": "assistant", "content": reply})

    # Telegram has 4096 char limit per message
    if len(reply) > 4000:
        for i in range(0, len(reply), 4000):
            await update.message.reply_text(reply[i:i+4000])
    else:
        await update.message.reply_text(reply)


async def post_init(app: Application) -> None:
    """Pre-load corpus and warm up news cache on startup."""
    logger.info("Pre-loading speeches corpus...")
    count = reload_corpus()
    logger.info("Loaded %d speeches", count)

    logger.info("Warming up news cache...")
    try:
        await get_news_for_context()
    except Exception as e:
        logger.warning("News warmup failed: %s", e)


def main() -> None:
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("news", news_command))
    app.add_handler(CommandHandler("speeches", speeches_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting Zhirinovsky bot with model: %s", MODEL)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
