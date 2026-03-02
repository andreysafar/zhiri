"""
Zhirinovsky Telegram Bot
Powered by OpenRouter (Gemini Flash) + RAG over speeches corpus + live news + TTS voice
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
from voice import synthesize

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

# Per-user voice mode toggle
voice_mode: set[int] = set()

# Keywords that trigger voice response for a single message
VOICE_TRIGGERS = ("голосом", "вслух", "скажи", "прочитай вслух", "озвучь")


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
    """Assemble message list: system prompt + RAG speeches + history + user msg."""
    speech_context = search_speeches(user_text, top_k=2)

    system_content = ZHIRINOVSKY_SYSTEM_PROMPT
    if speech_context:
        system_content += f"\n\n{speech_context}"

    messages: list[dict] = [{"role": "system", "content": system_content}]

    history = list(conversation_history[user_id])

    if not history:
        messages.append({
            "role": "user",
            "content": "[Системное обновление: вот свежие новости для твоего контекста. Учти их при ответах.]"
        })
        messages.append({
            "role": "assistant",
            "content": "Понял, буду держать в голове!"
        })

    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def _wants_voice(user_id: int, text: str) -> bool:
    """True if user has voice mode on, or message contains a voice trigger word."""
    if user_id in voice_mode:
        return True
    lower = text.lower()
    return any(kw in lower for kw in VOICE_TRIGGERS)


async def _send_reply(update: Update, context: ContextTypes.DEFAULT_TYPE,
                      reply: str, as_voice: bool) -> None:
    if as_voice:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action="record_voice"
        )
        try:
            audio_buf, mime = await asyncio.get_event_loop().run_in_executor(
                None, synthesize, reply
            )
            if mime == "audio/ogg":
                await update.message.reply_voice(voice=audio_buf)
            else:
                await update.message.reply_audio(audio=audio_buf, filename="zhirik.mp3")
            # Also send text so user can read it
            if len(reply) <= 4000:
                await update.message.reply_text(reply)
            return
        except Exception as e:
            logger.error("TTS error: %s", e)
            await update.message.reply_text("Голос сорвался! Но текст — вот:")

    # Text-only fallback / normal mode
    if len(reply) > 4000:
        for i in range(0, len(reply), 4000):
            await update.message.reply_text(reply[i:i + 4000])
    else:
        await update.message.reply_text(reply)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversation_history[user_id].clear()
    voice_mode.discard(user_id)

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
        "Спрашивайте — отвечу на всё! Политика, история, жизнь, судьба России!\n\n"
        "Команды:\n"
        "/news — последние новости\n"
        "/voice — включить/выключить голосовые ответы\n"
        "/clear — начать разговор заново\n"
        "/speeches — перезагрузить базу выступлений\n\n"
        "Можно написать «скажи голосом» или «озвучь» — отвечу голосом разово."
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


async def voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id in voice_mode:
        voice_mode.discard(user_id)
        await update.message.reply_text("Голосовой режим ВЫКЛЮЧЕН. Говорю текстом.")
    else:
        voice_mode.add(user_id)
        await update.message.reply_text(
            "Голосовой режим ВКЛЮЧЁН! Жириновский будет отвечать голосом!\n"
            "Выключить — /voice"
        )


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

    as_voice = _wants_voice(user_id, user_text)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="record_voice" if as_voice else "typing",
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

    conversation_history[user_id].append({"role": "user", "content": user_text})
    conversation_history[user_id].append({"role": "assistant", "content": reply})

    await _send_reply(update, context, reply, as_voice)


async def post_init(app: Application) -> None:
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
    app.add_handler(CommandHandler("voice", voice_command))
    app.add_handler(CommandHandler("speeches", speeches_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting Zhirinovsky bot with model: %s", MODEL)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
