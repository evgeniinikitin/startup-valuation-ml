from __future__ import annotations

from datetime import date
from pathlib import Path

from .llm import LLMClient
from .pdf import report_source_excerpt, retrieval_source_excerpt, summary_source_excerpt
from .search import SearchService, deduplicate_results
from .crunchbase_analogs import CrunchbaseAnalogRetriever


def build_startup_summary(llm: LLMClient, startup_text: str) -> str:
    prompt = f"""
Сделай структурированное summary стартапа на русском.
Секции: Продукт, Рынок, Команда, Трекшн, Выручка, Риски.
Сохраняй факты и цифры, ничего не выдумывай.

Текст:
{summary_source_excerpt(startup_text)}
    """.strip()
    return llm.generate(prompt, temperature=0.15, max_output_tokens=2200)


def build_project_brief(llm: LLMClient, startup_text: str) -> str:
    prompt = f"""
Сделай "Паспорт проекта" на русском по извлеченному тексту PDF.
Нужны только факты, без выдумок.
Структура:
- Название проекта
- Проблема
- Решение
- Клиенты/рынок
- Команда
- Текущая стадия
- Выручка/монетизация
- Метрики/тракшн
- Риски
- Что непонятно / каких данных не хватает

Если поле не найдено в тексте, пиши "Не найдено в PDF".

Текст:
{summary_source_excerpt(startup_text)}
    """.strip()
    return llm.generate(prompt, temperature=0.1, max_output_tokens=1800)


def build_search_seed(llm: LLMClient, startup_text: str) -> str:
    prompt = f"""
Сделай короткое retrieval-summary (1-2 абзаца).
Включи: отрасль, бизнес-модель, ICP, pain, ценностное предложение, метрики.

Текст:
{retrieval_source_excerpt(startup_text)}
    """.strip()
    return llm.generate(prompt, temperature=0.1, max_output_tokens=900)


def detect_red_flags(llm: LLMClient, startup_text: str, summary: str) -> str:
    today_iso = date.today().isoformat()
    prompt = f"""
Ты критичный венчурный аналитик.
Найди красные флаги.
Если данных достаточно, НЕ пиши формальную фразу про "недостаток информации".
Если данных не хватает, укажи конкретно какой блок отсутствует.
Текущая дата: {today_iso}. Не называй дату презентации "будущей", если она раньше этой даты.

SUMMARY:
{summary}

SOURCE:
{summary_source_excerpt(startup_text)}

Формат:
🚩 [Флаг 1]: ...
🚩 [Флаг 2]: ...
    """.strip()
    return llm.generate(prompt, temperature=0.15, max_output_tokens=1400)


def _ensure_markdown_table(llm: LLMClient, scorecard: str, shared_context: str, sys: str) -> str:
    # Fast validation: table separator row and at least 4 data rows.
    has_separator = ("|---" in scorecard) or ("| ---" in scorecard)
    row_count = sum(1 for line in scorecard.splitlines() if line.strip().startswith("|"))
    if has_separator and row_count >= 5:
        return scorecard

    repair_prompt = f"""
Перепиши секцию строго в виде markdown-таблицы.
Верни только:
# 2. Скоринг-таблица
| Критерий | Оценка (1-10) | Обоснование |
|---|---:|---|
... минимум 6 строк + ИТОГО

Контекст:
{shared_context}
    """.strip()
    return llm.generate(repair_prompt, system_instruction=sys, temperature=0.1, max_output_tokens=1400)


def _build_shared_context(
    *,
    startup_text: str,
    project_brief: str,
    startup_summary: str,
    db_context: str,
    eval_instruction_path: Path,
) -> tuple[str, str]:
    eval_instruction = (
        eval_instruction_path.read_text(encoding="utf-8")
        if eval_instruction_path.exists()
        else "Инструкция оценки не найдена."
    )
    shared_context = f"""
SOURCE PITCH:
{report_source_excerpt(startup_text)}

PROJECT BRIEF (обязательная опора):
{project_brief}

STRUCTURED SUMMARY:
{startup_summary}

КОНТЕКСТ ИЗ БАЗЫ:
{db_context}

ИНСТРУКЦИЯ ПО ОЦЕНКЕ:
--- START ---
{eval_instruction}
--- END ---
    """.strip()

    sys = (
        "Ты старший венчурный аналитик с жестким реалистичным подходом к valuation. "
        "Опирайся только на факты из source+context и не придумывай цифры. "
        "Если данных не хватает, явно указывай конкретный пробел и снижай уверенность оценки."
    )
    return shared_context, sys


def _context_block(label: str, rows: list[dict], max_items: int = 10, snippet_len: int = 750) -> str:
    if not rows:
        return ""
    lines = [f"### {label}"]
    for i, r in enumerate(rows[:max_items], 1):
        sim = float(r.get("similarity", r.get("hybrid_score", 0)) or 0)
        content = (r.get("content") or "")[:snippet_len]
        lines.append(f"[{label} {i}] score={sim:.3f}\n{content}")
    return "\n\n".join(lines)


def collect_db_context(search: SearchService, llm: LLMClient, startup_text: str) -> tuple[str, list[dict], list[dict], str]:
    # Keep search cheap and deterministic: no extra LLM call here.
    seed = retrieval_source_excerpt(startup_text)

    analog_chunks: list[dict] = []
    analog_modes: list[str] = []
    for sid in (
        "sec_formd_q1_2026",
        "sec_investments_vc",
        "investment_summary.csv",
        "investments_vc.csv",
        "indian.csv",
        "trustmrr_master_database.csv",
    ):
        rows, mode_one = search.search(
            query=seed,
            filter_obj={"source_id": sid},
            desired_top_k=6,
        )
        analog_chunks.extend(rows)
        analog_modes.append(mode_one)
    analogs_raw = analog_chunks
    mode_a = "+".join(sorted(set(analog_modes))) if analog_modes else "n/a"
    market_chunks: list[dict] = []
    market_modes: list[str] = []
    for market_sid in ("markets.csv", "global_market_sizes.csv"):
        rows, mode_one = search.search(
            query=seed,
            filter_obj={"source_id": market_sid},
            desired_top_k=8,
        )
        market_chunks.extend(rows)
        market_modes.append(mode_one)
    markets_raw = market_chunks
    mode_m = "+".join(sorted(set(market_modes))) if market_modes else "n/a"

    analogs = deduplicate_results(analogs_raw)[:10]
    markets = deduplicate_results(markets_raw)[:10]

    crunchbase_analogs: list[dict] = []
    mode_cb = "crunchbase_disabled"
    try:
        crunchbase_analogs, mode_cb = CrunchbaseAnalogRetriever().search(seed, top_k=10)
    except Exception as e:
        mode_cb = f"crunchbase_error:{str(e)[:120]}"

    context = (
        _context_block("АНАЛОГИ", analogs)
        + "\n\n"
        + _context_block("CRUNCHBASE АНАЛОГИ (structured/lexical, без векторизации)", crunchbase_analogs)
        + "\n\n"
        + _context_block("РЫНОК", markets)
    ).strip()
    if not context:
        context = "Релевантные аналоги/рынок в БД не найдены."
    search_mode = f"analogs={mode_a}, crunchbase={mode_cb}, market={mode_m}"
    return context, analogs, markets, search_mode


def generate_report_blocks_1_3(
    *,
    llm: LLMClient,
    shared_context: str,
    sys: str,
) -> str:
    block = llm.generate(
        f"""
Сформируй блоки 1, 2 и 3 итогового отчета:
# 1. Питч и описание проекта
# 2. Скоринг-таблица
# 3. TAM/SAM/SOM и верификация

Требования:
- Для секции 2 используй markdown-таблицу.
- Без воды, только факты и проверяемые выводы.

{shared_context}
        """.strip(),
        system_instruction=sys,
        temperature=0.2,
        max_output_tokens=4200,
    )
    # Try to enforce a valid score table if model got messy.
    if "# 2. Скоринг-таблица" in block:
        # Extract the section roughly and validate table lines.
        score_lines = [ln for ln in block.splitlines() if ln.strip().startswith("|")]
        if len(score_lines) < 5:
            repaired = _ensure_markdown_table(llm, block, shared_context, sys)
            block = block + "\n\n" + repaired
    return block


def generate_report_blocks_4_5(
    *,
    llm: LLMClient,
    shared_context: str,
    sys: str,
) -> str:
    return llm.generate(
        f"""
Сформируй блоки 4 и 5 итогового отчета:
# 4. Сильные и слабые стороны
# 5. Конкурентный анализ

Требования:
- В секции 4: отдельные подзаголовки "Сильные стороны" и "Слабые стороны", минимум 5 пунктов в каждом.
- В секции 5: сравнительная markdown-таблица минимум по 5 конкурентам.

{shared_context}
        """.strip(),
        system_instruction=sys,
        temperature=0.2,
        max_output_tokens=3200,
    )


def generate_report_blocks_6_7(
    *,
    llm: LLMClient,
    shared_context: str,
    sys: str,
    deterministic_context: str = "",
) -> str:
    return llm.generate(
        f"""
Сформируй блоки 6 и 7 итогового отчета:
# 6. Valuation с расчетом и обоснованием
# 7. Рекомендации

Требования:
- В секции 6 используй жесткий реалистичный подход, как в app.py:
  - Без подтвержденной выручки оценка не может быть высокой.
  - Без работающего продукта оценка должна быть в нижнем диапазоне ранней стадии.
  - Мультипликаторы применяй только к фактической выручке, а не к прогнозам.
  - TAM/SAM/SOM сверяй с данными из базы, а не только из презентации.
  - Если рыночные данные не подтверждены, явно снижай оценку на 30-50%.
  - Если есть детерминированный расчет ниже, используй его как первичный якорь valuation.
- В конце секции 6 обязательно JSON-блок:
```json
{{
  "stage": "...",
  "method": "...",
  "valuation_usd": 0,
  "confidence": "low/medium/high",
  "key_factors": ["..."]
}}
```
- В секции 7: конкретный план действий на 0-3 и 3-12 месяцев.

ДЕТЕРМИНИРОВАННЫЙ РАСЧЕТ (ЯКОРЬ):
{deterministic_context}

{shared_context}
        """.strip(),
        system_instruction=sys,
        temperature=0.2,
        max_output_tokens=3200,
    )


def generate_report_section(
    *,
    llm: LLMClient,
    section_id: str,
    shared_context: str,
    sys: str,
) -> str:
    if section_id == "1":
        return llm.generate(
            f"""
Сформируй секцию 1 отчета на русском:
# 1. Питч и описание проекта

Требование: до 8-12 абзацев, структурно и фактологично.

{shared_context}
            """.strip(),
            system_instruction=sys,
            temperature=0.2,
            max_output_tokens=1800,
        )

    if section_id == "2":
        scorecard = llm.generate(
            f"""
Сформируй секцию 2:
# 2. Скоринг-таблица

Только markdown-таблица:
| Критерий | Оценка (1-10) | Обоснование |
и строка ИТОГО.

{shared_context}
            """.strip(),
            system_instruction=sys,
            temperature=0.15,
            max_output_tokens=1400,
        )
        return _ensure_markdown_table(llm, scorecard, shared_context, sys)

    if section_id == "3":
        return llm.generate(
            f"""
Сформируй секцию 3:
# 3. TAM/SAM/SOM и верификация

Требования:
- Сверяй цифры с контекстом из базы.
- Не заявляй неподтвержденные метрики как факты.

{shared_context}
            """.strip(),
            system_instruction=sys,
            temperature=0.2,
            max_output_tokens=1800,
        )

    if section_id == "4":
        return llm.generate(
            f"""
Сформируй секцию 4:
# 4. Сильные и слабые стороны

Требования:
- Отдельные подзаголовки: "Сильные стороны" и "Слабые стороны".
- Минимум 5 пунктов в каждом списке.
- Пункты должны быть конкретными и подтверждаться контекстом.

{shared_context}
            """.strip(),
            system_instruction=sys,
            temperature=0.2,
            max_output_tokens=1800,
        )

    if section_id == "5":
        return llm.generate(
            f"""
Сформируй секцию 5:
# 5. Конкурентный анализ

Требования:
- Используй найденные аналоги из контекста БД.
- Сделай сравнительную markdown-таблицу минимум по 5 конкурентам:
| Компания | Сегмент | Funding/масштаб | Сравнение с проектом |
- После таблицы дай 5-8 выводов.

{shared_context}
            """.strip(),
            system_instruction=sys,
            temperature=0.2,
            max_output_tokens=2200,
        )

    if section_id == "6":
        return llm.generate(
            f"""
Сформируй секцию 6:
# 6. Valuation с расчетом и обоснованием

Обязательно:
- Покажи логику расчета стоимости.
- Применяй реалистичные рамки ранних стадий: без выручки и без валидации рынка оценка должна быть консервативной.
- Мультипликаторы используй только для подтвержденной фактической выручки.
- TAM/SAM/SOM обязательно сверяй с контекстом БД; при слабом подтверждении явно дисконтирай оценку.
- Если данных недостаточно, верни консервативный диапазон и явно объясни допущения.
- В конце добавь JSON-блок:
```json
{{
  "stage": "...",
  "method": "...",
  "valuation_usd": 0,
  "confidence": "low/medium/high",
  "key_factors": ["..."]
}}
```

{shared_context}
            """.strip(),
            system_instruction=sys,
            temperature=0.2,
            max_output_tokens=2200,
        )

    if section_id == "7":
        return llm.generate(
            f"""
Сформируй секцию 7:
# 7. Рекомендации

Требования:
- Раздели на "Краткосрочные (0-3 мес)" и "Среднесрочные (3-12 мес)".
- Минимум 6 конкретных рекомендаций.

{shared_context}
            """.strip(),
            system_instruction=sys,
            temperature=0.2,
            max_output_tokens=1600,
        )

    raise ValueError(f"Unknown section_id: {section_id}")


def generate_full_report(
    *,
    llm: LLMClient,
    startup_text: str,
    project_brief: str,
    startup_summary: str,
    db_context: str,
    eval_instruction_path: Path,
) -> str:
    shared_context, sys = _build_shared_context(
        startup_text=startup_text,
        project_brief=project_brief,
        startup_summary=startup_summary,
        db_context=db_context,
        eval_instruction_path=eval_instruction_path,
    )
    sections: list[str] = []
    for sid in ("1", "2", "3", "4", "5", "6", "7"):
        sections.append(generate_report_section(llm=llm, section_id=sid, shared_context=shared_context, sys=sys))
    return "\n\n".join([s.strip() for s in sections if s.strip()])


def prepare_report_context(
    *,
    startup_text: str,
    project_brief: str,
    startup_summary: str,
    db_context: str,
    eval_instruction_path: Path,
) -> tuple[str, str]:
    return _build_shared_context(
        startup_text=startup_text,
        project_brief=project_brief,
        startup_summary=startup_summary,
        db_context=db_context,
        eval_instruction_path=eval_instruction_path,
    )
