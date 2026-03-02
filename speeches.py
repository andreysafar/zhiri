"""
Loader and searcher for Zhirinovsky's speeches corpus.

Sources:
  1. Local text files in speeches/ directory (hand-curated or scraped)
  2. Duma transcripts scraper (optional, via scrape_duma.py)
  3. Fallback: built-in curated quotes
"""
import os
import json
import difflib
import re
from pathlib import Path

SPEECHES_DIR = Path(__file__).parent / "speeches"

# Curated famous speeches and quotes — fallback when no corpus is loaded
BUILTIN_CORPUS: list[dict] = [
    {
        "year": 1991,
        "title": "Первое выступление на президентских выборах",
        "text": (
            "Я буду защищать русских людей! Везде — в Прибалтике, в Средней Азии, "
            "в Закавказье! Русский народ — великий народ, и он заслуживает великой страны! "
            "ЛДПР — единственная партия, которая стоит на защите русского человека!"
        ),
    },
    {
        "year": 1993,
        "title": "Выступление в Госдуме о реформах",
        "text": (
            "Гайдар уничтожил страну! Чубайс разграбил страну! А народ живёт хуже, чем при Брежневе! "
            "Я предупреждал — шоковая терапия убьёт Россию. Вот результат: "
            "нищета, безработица, преступность. И они ещё смеют называть это реформами!"
        ),
    },
    {
        "year": 1995,
        "title": "О войне в Чечне",
        "text": (
            "Это позор! Великая армия не может справиться с бандитами! "
            "Потому что политики мешают военным работать! "
            "Дайте армии свободу действий — и через неделю порядок будет восстановлен! "
            "Никто не смеет угрожать территориальной целостности России!"
        ),
    },
    {
        "year": 1999,
        "title": "О расширении НАТО",
        "text": (
            "НАТО идёт на восток! Я говорил об этом десять лет назад! Никто не слушал! "
            "Горбачёв подписал бумаги — и нас предали! Запад никогда не был нашим другом "
            "и никогда им не будет! России нужна своя сила, свои союзники, свои ракеты!"
        ),
    },
    {
        "year": 2003,
        "title": "Предвыборная речь ЛДПР",
        "text": (
            "Я не обещаю вам золотых гор! Я обещаю порядок! Я обещаю безопасность на улицах! "
            "Я обещаю, что русский человек не будет бояться выйти вечером из дома! "
            "Вот чего хочет народ — не абстрактной свободы, а нормальной жизни!"
        ),
    },
    {
        "year": 2007,
        "title": "О мировой политике",
        "text": (
            "Американцы думают, что весь мир — их вотчина. Ошибаются! "
            "Россия встаёт с колен! И те, кто привык топтать нас ногами, "
            "скоро поймут, что медведя лучше не будить. "
            "Мы — великая цивилизация! Тысячелетняя история!"
        ),
    },
    {
        "year": 2011,
        "title": "О коррупции",
        "text": (
            "Коррупция разъедает страну как ржавчина! Чиновники воруют! "
            "Судьи берут взятки! А народ молчит — потому что боится! "
            "Нужны показательные процессы! Нужно вернуть смертную казнь для коррупционеров! "
            "Тогда воровать перестанут!"
        ),
    },
    {
        "year": 2014,
        "title": "О событиях на Украине",
        "text": (
            "Я всегда говорил — Украина искусственное государство! "
            "Созданное большевиками в угоду националистам! "
            "Русские на Украине — наши братья, и мы не можем их бросить! "
            "Запад устроил переворот — теперь пусть расхлёбывает последствия!"
        ),
    },
    {
        "year": 2018,
        "title": "О будущем России",
        "text": (
            "Через двадцать лет Россия будет крупнейшей экономикой Европы! "
            "У нас всё для этого есть — ресурсы, земля, умный народ! "
            "Мешают только враги внутри страны и враги снаружи! "
            "Но мы справимся! Россия всегда справлялась!"
        ),
    },
    {
        "year": 2021,
        "title": "О пандемии COVID-19",
        "text": (
            "Это биологическая война против человечества! "
            "Лаборатории работали годами — и вот результат! "
            "Россия сделала лучшую вакцину в мире — Спутник V! "
            "А Запад колется американскими препаратами и болеет! "
            "Потому что наша наука лучше!"
        ),
    },
    {
        "year": 2022,
        "title": "Последнее большое выступление",
        "text": (
            "Я прожил большую жизнь. Я видел распад СССР — это трагедия. "
            "Я видел, как Россия поднималась. Я верю, что она поднимется снова. "
            "ЛДПР продолжит мой путь. Россия — великая страна, "
            "и она выстоит перед любыми испытаниями!"
        ),
    },
]


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[а-яёa-z]+", text.lower())
    return set(words)


def _score(query_tokens: set[str], speech: dict) -> float:
    speech_tokens = _tokenize(speech["title"] + " " + speech["text"])
    if not speech_tokens:
        return 0.0
    overlap = query_tokens & speech_tokens
    return len(overlap) / (len(query_tokens) + 1)


def load_corpus() -> list[dict]:
    """Load all speeches: from files first, then builtins."""
    corpus = list(BUILTIN_CORPUS)

    if SPEECHES_DIR.exists():
        for fpath in sorted(SPEECHES_DIR.glob("*.json")):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    corpus.extend(data)
                elif isinstance(data, dict):
                    corpus.append(data)
            except Exception:
                pass
        for fpath in sorted(SPEECHES_DIR.glob("*.txt")):
            try:
                text = fpath.read_text(encoding="utf-8").strip()
                corpus.append({
                    "year": int(fpath.stem[:4]) if fpath.stem[:4].isdigit() else 0,
                    "title": fpath.stem,
                    "text": text,
                })
            except Exception:
                pass

    return corpus


_corpus: list[dict] | None = None


def get_corpus() -> list[dict]:
    global _corpus
    if _corpus is None:
        _corpus = load_corpus()
    return _corpus


def search_speeches(query: str, top_k: int = 3) -> str:
    """Return relevant speech excerpts for the given query."""
    corpus = get_corpus()
    query_tokens = _tokenize(query)

    if not query_tokens:
        return ""

    scored = sorted(corpus, key=lambda s: _score(query_tokens, s), reverse=True)
    top = scored[:top_k]

    if not top or _score(query_tokens, top[0]) < 0.03:
        return ""

    lines = ["ВЫДЕРЖКИ ИЗ ВЫСТУПЛЕНИЙ ЖИРИНОВСКОГО (для справки):"]
    for speech in top:
        year = f" ({speech['year']})" if speech.get("year") else ""
        lines.append(f'\n[{speech["title"]}{year}]\n"{speech["text"]}"')

    return "\n".join(lines)


def reload_corpus() -> int:
    """Force reload corpus from disk. Returns count of speeches."""
    global _corpus
    _corpus = load_corpus()
    return len(_corpus)
