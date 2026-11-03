from __future__ import annotations

from pathlib import Path
import time
import tempfile
import json
import inspect

import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client

from modular_rag.benchmark import (
    is_ai_case,
    load_json_dataset,
    parse_year_from_source,
    run_benchmark,
)
from modular_rag.cache import EmbeddingCache
from modular_rag.config import load_settings
from modular_rag.damodaran_data import fetch_all_rows, load_damodaran_to_sql_tables
from modular_rag.ingest import ingest_document
from modular_rag.llm import LLMClient
from modular_rag.ml_batch_eval import evaluate_ml_csv
from modular_rag.multi_method_valuation import evaluate_multi_method_valuation
from modular_rag.pdf import extract_pdf_text
from modular_rag.report import (
    build_project_brief,
    build_startup_summary,
    collect_db_context,
    detect_red_flags,
    generate_report_blocks_1_3,
    generate_report_blocks_4_5,
    generate_report_blocks_6_7,
    prepare_report_context,
)
from modular_rag.search import SearchService, infer_top_k_from_query
from modular_rag.valuation import (
    evaluate_project_deterministic,
    generate_ai_valuation_diagnostics,
    valuation_result_to_dict,
)


APP_DIR = Path(__file__).resolve().parent
settings = load_settings(APP_DIR)

st.set_page_config(page_title="LocalRAG Modular", page_icon="🧩", layout="wide")
st.title("🧩 LocalRAG Modular")
st.caption("Новая модульная версия: стабильный PDF parsing, hybrid search fallback, минимум магии.")
st.caption(f"Текстовая модель: `{settings.text_model}` (fallback: `{settings.text_fallback_model}`)")


missing = [
    name
    for name, value in [
        ("GEMINI_API_KEY", settings.gemini_api_key),
        ("SUPABASE_URL", settings.supabase_url),
        ("SUPABASE_SERVICE_KEY", settings.supabase_service_key),
    ]
    if not value
]
if missing:
    st.error(f"В `.env` не заданы: {', '.join(missing)}")
    st.stop()

cache = EmbeddingCache(settings.embedding_cache_db_path)
llm = LLMClient(settings, cache)
supabase = create_client(settings.supabase_url, settings.supabase_service_key)
search = SearchService(supabase, llm)


def _safe_eval_valuation(text: str, use_llm: bool = True):
    sig = inspect.signature(evaluate_project_deterministic)
    if "llm" in sig.parameters:
        return evaluate_project_deterministic(text, supabase, llm=llm if use_llm else None)
    return evaluate_project_deterministic(text, supabase)


def create_pdf(md_text: str) -> bytes:
    """Convert Markdown text to PDF bytes using markdown-pdf."""
    from markdown_pdf import MarkdownPdf, Section

    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(md_text))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.save(tmp.name)
        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()
    Path(tmp.name).unlink(missing_ok=True)
    return pdf_bytes


with st.expander("Архитектура", expanded=False):
    st.markdown(
        """
- `modular_rag/pdf.py`: извлечение текста (Gemini-first, fallback на pypdf).
- `modular_rag/search.py`: hybrid search с автоматическим fallback на vector search.
- `modular_rag/report.py`: summary/red flags/report как отдельные независимые блоки.
- `modular_rag/ingest.py`: загрузка данных в `documents`.
- `modular_rag/llm.py`: единая обертка над Gemini (без ротаций ключей).
        """.strip()
    )

tab_ingest, tab_qa, tab_report, tab_valuation, tab_benchmark, tab_damodaran = st.tabs(
    [
        "Ingest PDF",
        "Ask",
        "Startup Report",
        "Valuation Only (Script)",
        "Benchmark (R²)",
        "Damodaran Analytics",
    ]
)


with tab_ingest:
    st.subheader("Индексация PDF")
    pdf = st.file_uploader("PDF для индексации", type=["pdf"], key="ingest_pdf")
    source_id = st.text_input("source_id (опционально)", key="ingest_source_id")
    mode = st.selectbox("Режим извлечения", ["gemini_first", "auto", "pypdf_only"], index=0, key="ingest_mode")

    if st.button("Индексировать", disabled=not pdf):
        pdf_bytes = pdf.getvalue()
        with st.spinner("Извлекаем текст..."):
            extracted = extract_pdf_text(pdf_bytes, llm, mode=mode)

        st.info(
            f"Метод: `{extracted.method}` | pages={extracted.page_count} | "
            f"failed_pages={extracted.failed_pages} | quality={extracted.quality_score:.2f}"
        )
        if not extracted.text.strip():
            st.error("Не удалось извлечь текст.")
            st.stop()

        with st.spinner("Загружаем в базу..."):
            sid, chunks = ingest_document(
                supabase=supabase,
                llm=llm,
                text=extracted.text,
                filename=pdf.name,
                source_id=source_id.strip() or None,
            )
        st.success(f"Готово: source_id={sid}, чанков={chunks}")


with tab_qa:
    st.subheader("Q&A")
    query = st.text_input("Вопрос", key="qa_query")
    source_filter = st.text_input("Фильтр source_id (опционально)", key="qa_source_filter")

    if st.button("Найти ответ", disabled=not query.strip()):
        filter_obj = {"source_id": source_filter.strip()} if source_filter.strip() else {}
        try:
            rows, mode = search.search(query=query, filter_obj=filter_obj)
        except Exception as e:
            st.error(
                "Поиск не сработал. База доступна, но поисковый запрос упал "
                f"на стороне Supabase/PostgREST: {e}"
            )
            st.stop()
        st.caption(f"Search mode: `{mode}` | results={len(rows)}")
        requested_top_k = infer_top_k_from_query(query)
        if len(rows) < requested_top_k and any(k in query.lower() for k in ["funding", "финанс", "инвестиц", "раунд"]):
            st.info(
                f"Найдено {len(rows)} записей с явно указанным funding "
                f"(запрошено top {requested_top_k}). Пустые funding намеренно отфильтрованы."
            )

        if not rows:
            st.info("Ничего не найдено.")
        else:
            for i, r in enumerate(rows, 1):
                sim = float(r.get("similarity", r.get("hybrid_score", 0)) or 0)
                with st.expander(f"{i}. score={sim:.4f}"):
                    st.write(r.get("content") or "")

            context = "\n\n".join([f"[Chunk {i+1}] {r.get('content','')}" for i, r in enumerate(rows)])
            prompt = f"""
Ответь на вопрос по контексту.
Если данных недостаточно, укажи конкретно каких.

Контекст:
{context}

Вопрос:
{query}
            """.strip()
            with st.spinner("Генерируем ответ..."):
                answer = llm.generate(prompt, temperature=0.2, max_output_tokens=3072)
            st.markdown("### Ответ")
            st.markdown(answer)


with tab_report:
    st.subheader("Startup Report")
    startup_pdf = st.file_uploader("Pitch deck PDF", type=["pdf"], key="report_pdf")
    mode = st.selectbox(
        "Режим чтения PDF",
        ["gemini_first", "auto", "pypdf_only"],
        index=1,
        key="report_mode",
    )

    if "auto_report_result" not in st.session_state:
        st.session_state.auto_report_result = ""
    if "auto_red_flags" not in st.session_state:
        st.session_state.auto_red_flags = ""
    if "auto_project_brief" not in st.session_state:
        st.session_state.auto_project_brief = ""
    if "auto_search_mode" not in st.session_state:
        st.session_state.auto_search_mode = ""
    if "auto_report_pdf_bytes" not in st.session_state:
        st.session_state.auto_report_pdf_bytes = None
    if "auto_report_pdf_hash" not in st.session_state:
        st.session_state.auto_report_pdf_hash = ""

    st.caption("Автопайплайн: 4 фазы по 65 секунд минимум (1 минута + 5 секунд запас).")

    if st.button("Сгенерировать отчет (авто 4 фазы)", disabled=not startup_pdf):
        phase_min_seconds = 65
        progress = st.progress(0.0, text="Старт...")
        started_all = time.time()

        # Phase 1: Extract and gather all context
        phase1_start = time.time()
        progress.progress(0.05, text="Фаза 1/4: извлечение данных из PDF и поиск аналогов...")
        extracted = extract_pdf_text(startup_pdf.getvalue(), llm, mode=mode)
        if not extracted.text.strip():
            st.error("Текст не извлечен.")
            st.stop()

        summary = build_startup_summary(llm, extracted.text)
        project_brief = build_project_brief(llm, extracted.text)
        db_context, analogs, markets, search_mode = collect_db_context(search, llm, extracted.text)
        shared_context, sys_prompt = prepare_report_context(
            startup_text=extracted.text,
            project_brief=project_brief,
            startup_summary=summary,
            db_context=db_context,
            eval_instruction_path=settings.eval_instruction_path,
        )
        remain = phase_min_seconds - (time.time() - phase1_start)
        if remain > 0:
            time.sleep(remain)
        progress.progress(0.25, text="Фаза 1/4 завершена.")

        # Phase 2: Blocks 1-3
        phase2_start = time.time()
        progress.progress(0.27, text="Фаза 2/4: генерация блоков 1-3...")
        block_1_3 = generate_report_blocks_1_3(
            llm=llm,
            shared_context=shared_context,
            sys=sys_prompt,
        )
        remain = phase_min_seconds - (time.time() - phase2_start)
        if remain > 0:
            time.sleep(remain)
        progress.progress(0.50, text="Фаза 2/4 завершена.")

        # Phase 3: Blocks 4-5
        phase3_start = time.time()
        progress.progress(0.52, text="Фаза 3/4: генерация блоков 4-5...")
        block_4_5 = generate_report_blocks_4_5(
            llm=llm,
            shared_context=shared_context,
            sys=sys_prompt,
        )
        remain = phase_min_seconds - (time.time() - phase3_start)
        if remain > 0:
            time.sleep(remain)
        progress.progress(0.75, text="Фаза 3/4 завершена.")

        # Phase 4: Blocks 6-7 + Red flags (last)
        phase4_start = time.time()
        progress.progress(0.77, text="Фаза 4/4: генерация блоков 6-7 и red flags...")
        det = _safe_eval_valuation(extracted.text)
        det_payload = valuation_result_to_dict(det)
        det_anchor = json.dumps(
            {
                "stage": det_payload.get("stage"),
                "method": det_payload.get("method"),
                "damodaran_year_used": det_payload.get("damodaran_year_used"),
                "valuation_min_usd": det_payload.get("valuation_min_usd"),
                "valuation_mid_usd": det_payload.get("valuation_mid_usd"),
                "valuation_max_usd": det_payload.get("valuation_max_usd"),
                "confidence": det_payload.get("confidence"),
                "quality_gate_flags": det_payload.get("quality_gate_flags", []),
            },
            ensure_ascii=False,
            indent=2,
        )
        block_6_7 = generate_report_blocks_6_7(
            llm=llm,
            shared_context=shared_context,
            sys=sys_prompt,
            deterministic_context=det_anchor,
        )
        full_report = (block_1_3.strip() + "\n\n" + block_4_5.strip() + "\n\n" + block_6_7.strip()).strip()
        red_flags = detect_red_flags(
            llm,
            extracted.text + "\n\n" + full_report,
            project_brief + "\n\n" + summary,
        )
        remain = phase_min_seconds - (time.time() - phase4_start)
        if remain > 0:
            time.sleep(remain)
        progress.progress(1.0, text="Готово.")

        elapsed = int(time.time() - started_all)
        st.info(
            f"Метод: `{extracted.method}` | pages={extracted.page_count} | failed_pages={extracted.failed_pages} | "
            f"quality={extracted.quality_score:.2f} | search={search_mode} | analogs={len(analogs)} | market={len(markets)} | total={elapsed}s"
        )
        st.session_state.auto_project_brief = project_brief
        st.session_state.auto_red_flags = red_flags
        st.session_state.auto_report_result = full_report
        st.session_state.auto_search_mode = search_mode
        st.session_state.auto_report_pdf_bytes = None
        st.session_state.auto_report_pdf_hash = ""

    if st.session_state.auto_project_brief:
        with st.expander("Нейро-сводка после извлечения PDF", expanded=False):
            st.markdown(st.session_state.auto_project_brief)

    if st.session_state.auto_red_flags:
        st.markdown("### 🚩 Red Flags")
        st.markdown(st.session_state.auto_red_flags)

    if st.session_state.auto_report_result:
        st.markdown("### 📄 Full Report")
        st.markdown(st.session_state.auto_report_result)
        report_body = ""
        if st.session_state.auto_red_flags:
            report_body += "## 🚩 Red Flags\n\n" + st.session_state.auto_red_flags + "\n\n"
        report_body += st.session_state.auto_report_result

        report_hash = str(hash(report_body))
        if st.session_state.auto_report_pdf_hash != report_hash or st.session_state.auto_report_pdf_bytes is None:
            with st.spinner("Подготавливаем PDF отчета..."):
                st.session_state.auto_report_pdf_bytes = create_pdf(report_body)
            st.session_state.auto_report_pdf_hash = report_hash

        st.download_button(
            label="📥 Скачать отчет (PDF)",
            data=st.session_state.auto_report_pdf_bytes,
            file_name="startup_report.pdf",
            mime="application/pdf",
        )


with tab_valuation:
    st.subheader("Valuation Only (детерминированный расчет)")
    st.caption(
        "Этот режим считает стоимость проекта скриптом и формулами "
        "(Damodaran SQL + deal terms + benchmark anchors). "
        "LLM используется только для улучшенного извлечения стадии/аудита, а не для самой математики."
    )

    valuation_pdf = st.file_uploader("PDF проекта (опционально)", type=["pdf"], key="valuation_pdf")
    valuation_mode = st.selectbox(
        "Режим чтения PDF",
        ["auto", "gemini_first", "pypdf_only"],
        index=2,
        key="valuation_mode",
    )
    valuation_text = st.text_area(
        "Или вставьте текст проекта вручную",
        height=220,
        key="valuation_text_manual",
        placeholder="Вставьте описание проекта, метрики, условия раунда (если есть)...",
    )

    if "valuation_last_payload" not in st.session_state:
        st.session_state.valuation_last_payload = None
    if "valuation_last_input_text" not in st.session_state:
        st.session_state.valuation_last_input_text = ""
    if "valuation_last_extraction_note" not in st.session_state:
        st.session_state.valuation_last_extraction_note = ""
    if "valuation_ai_audit" not in st.session_state:
        st.session_state.valuation_ai_audit = ""
    if "valuation_multi_method_payload" not in st.session_state:
        st.session_state.valuation_multi_method_payload = None
    if "valuation_multi_method_error" not in st.session_state:
        st.session_state.valuation_multi_method_error = ""

    run_disabled = (valuation_pdf is None) and (not valuation_text.strip())
    if st.button("Рассчитать valuation (script only)", disabled=run_disabled):
        source_parts: list[str] = []
        extraction_note = "manual_text"

        if valuation_pdf is not None:
            extracted = extract_pdf_text(valuation_pdf.getvalue(), llm, mode=valuation_mode)
            source_parts.append(extracted.text or "")
            extraction_note = (
                f"{extracted.method} | pages={extracted.page_count} | "
                f"failed_pages={extracted.failed_pages} | quality={extracted.quality_score:.2f}"
            )

        if valuation_text.strip():
            source_parts.append(valuation_text.strip())

        full_text = "\n\n".join([p for p in source_parts if p.strip()])
        if not full_text.strip():
            st.error("Нет входных данных для расчета valuation.")
            st.stop()

        result = _safe_eval_valuation(full_text)
        payload = valuation_result_to_dict(result)
        st.session_state.valuation_last_payload = payload
        st.session_state.valuation_last_input_text = full_text
        st.session_state.valuation_last_extraction_note = extraction_note
        st.session_state.valuation_ai_audit = ""
        st.session_state.valuation_multi_method_payload = None
        st.session_state.valuation_multi_method_error = ""
        try:
            st.session_state.valuation_multi_method_payload = evaluate_multi_method_valuation(full_text, payload, llm=llm)
        except Exception as e:
            st.session_state.valuation_multi_method_error = str(e)

    payload = st.session_state.valuation_last_payload
    if payload:
        extraction_note = st.session_state.valuation_last_extraction_note

        st.info(f"Input prepared: {extraction_note}")
        st.markdown("### Result")
        st.markdown(
            f"- `stage`: **{payload['stage']}**\n"
            f"- `method`: **{payload['method']}**\n"
            f"- `valuation range (USD)`: **{payload['valuation_min_usd']:,.0f} – {payload['valuation_max_usd']:,.0f}**\n"
            f"- `mid (USD)`: **{payload['valuation_mid_usd']:,.0f}**\n"
            f"- `confidence`: **{payload['confidence']}**"
        )

        st.markdown("### JSON")
        st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")

        st.markdown("### Mini Report")
        st.code(payload.get("mini_report", ""), language="text")

        st.markdown("### Multi-Method Valuation (experimental)")
        st.caption(
            "Новый слой оценки по нескольким методам. Он не заменяет основной script-only результат "
            "и при ошибке не влияет на рабочий расчёт."
        )
        multi_payload = st.session_state.valuation_multi_method_payload
        multi_error = st.session_state.valuation_multi_method_error
        if multi_payload:
            st.markdown(
                f"- `version`: **{multi_payload['version']}**\n"
                f"- `stage_family`: **{multi_payload['stage_family']}**\n"
                f"- `valuation range (USD)`: **{multi_payload['valuation_min_usd']:,.0f} – {multi_payload['valuation_max_usd']:,.0f}**\n"
                f"- `mid (USD)`: **{multi_payload['valuation_mid_usd']:,.0f}**\n"
                f"- `confidence`: **{multi_payload['confidence']}**"
            )
            methods_df = pd.DataFrame(multi_payload.get("methods") or [])
            if not methods_df.empty:
                display_methods_df = methods_df.drop(columns=["details"], errors="ignore")
                st.dataframe(
                    display_methods_df[
                        [
                            "name",
                            "weight",
                            "value_min_usd",
                            "value_mid_usd",
                            "value_max_usd",
                            "confidence",
                            "rationale",
                        ]
                    ],
                    use_container_width=True,
                )
                ml_rows = [
                    method
                    for method in (multi_payload.get("methods") or [])
                    if method.get("name") == "ml_market_benchmark_adjusted"
                ]
                if ml_rows:
                    ml_details = (ml_rows[0].get("details") or {})
                    explain = ml_details.get("explainability") or {}
                    ml_pred = ml_details.get("ml_prediction") or {}
                    with st.expander("ML benchmark explanation", expanded=True):
                        raw_mid = ((explain.get("raw_ml") or {}).get("valuation_mid_usd"))
                        cal = explain.get("calibration") or {}
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Raw ML", f"${raw_mid:,.0f}" if raw_mid else "n/a")
                        c2.metric("Calibrated ML", f"${(cal.get('calibrated_mid_usd') or 0):,.0f}")
                        c3.metric("ML range min", f"${(cal.get('range_min_usd') or 0):,.0f}")
                        c4.metric("ML range max", f"${(cal.get('range_max_usd') or 0):,.0f}")
                        st.caption(
                            f"Calibration: {ml_pred.get('calibration_method') or cal.get('method')} | "
                            f"segment={ml_pred.get('calibration_segment') or cal.get('segment')} | "
                            f"multiplier={(ml_pred.get('calibration_multiplier') or cal.get('multiplier') or 0):.2f}x"
                        )
                        enrichment = ml_details.get("input_enrichment") or {}
                        if enrichment:
                            match = enrichment.get("match") or {}
                            st.markdown("#### Input Enrichment")
                            e1, e2, e3 = st.columns(3)
                            e1.metric("Mode", enrichment.get("enrichment_mode", "n/a"))
                            e2.metric("Local CB match", "yes" if match.get("matched") else "no")
                            e3.metric("Match score", f"{(match.get('match_score') or 0):.2f}")
                            st.caption(enrichment.get("web_search_note") or "")
                            facts = enrichment.get("facts") or {}
                            fact_rows = [
                                {"field": key, "value": value}
                                for key, value in facts.items()
                                if value not in (None, "", 0, 0.0)
                            ]
                            if fact_rows:
                                st.dataframe(pd.DataFrame(fact_rows), use_container_width=True, hide_index=True)

                        context = explain.get("company_context") or {}
                        context_df = pd.DataFrame(
                            [
                                {"factor": key, "value": value}
                                for key, value in context.items()
                                if value not in (None, "", 0, 0.0)
                            ]
                        )
                        if not context_df.empty:
                            st.markdown("#### Company Context")
                            st.dataframe(context_df, use_container_width=True, hide_index=True)

                        priors = pd.DataFrame(explain.get("historical_priors") or [])
                        st.markdown("#### Historical Priors")
                        if not priors.empty:
                            st.dataframe(priors, use_container_width=True, hide_index=True)
                        else:
                            st.info("Historical priors are not available for this input. For PDF/manual text this is expected unless the company is matched to enriched Crunchbase rows.")

                        investor_strength = explain.get("investor_strength") or {}
                        investor_df = pd.DataFrame(
                            [
                                {"metric": key, "value": value}
                                for key, value in investor_strength.items()
                                if value not in (None, "", 0, 0.0)
                            ]
                        )
                        st.markdown("#### Investor Strength")
                        if not investor_df.empty:
                            st.dataframe(investor_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("Investor historical strength is not available for this input.")

                        signal_rows = []
                        for label, values in [
                            ("top_investors", explain.get("top_investor_flags") or []),
                            ("keywords", explain.get("keyword_flags") or []),
                            ("text_signals", explain.get("text_signals") or []),
                            ("confidence_reasons", explain.get("confidence_reasons") or []),
                            ("risk_reasons", explain.get("risk_reasons") or []),
                        ]:
                            if values:
                                signal_rows.append({"group": label, "signals": ", ".join(map(str, values))})
                        if signal_rows:
                            st.markdown("#### Signals")
                            st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)
                        st.caption(f"Explanation confidence: {explain.get('explanation_confidence', 'n/a')}")
            with st.expander("Multi-method JSON", expanded=False):
                st.code(json.dumps(multi_payload, ensure_ascii=False, indent=2), language="json")
        elif multi_error:
            st.warning(f"Multi-method слой не сработал, основной расчёт не затронут: {multi_error}")
        else:
            st.info("Multi-method слой пока не рассчитан.")

        st.markdown("### Comparable Analogs (DB)")
        analogs = payload.get("analogs") or []
        if analogs:
            for i, a in enumerate(analogs, 1):
                st.markdown(f"**{i}. score={a.get('score', 0):.4f}**")
                st.caption(str((a.get('metadata') or {}).get("source_id", "")))
                st.write(a.get("content", ""))
        else:
            st.info("No analog comparables found by script keyword matcher.")

        st.markdown("### Investment Comps (DB)")
        inv = payload.get("investment_comps") or []
        if inv:
            for i, a in enumerate(inv, 1):
                st.markdown(f"**{i}. score={a.get('score', 0):.4f}**")
                st.caption(str((a.get('metadata') or {}).get("source_id", "")))
                st.write(a.get("content", ""))
        else:
            st.info("No investment comparables found in loaded investment sources.")

        st.markdown("### Math Trace")
        steps = payload.get("calculation_steps") or []
        if steps:
            for i, step in enumerate(steps, 1):
                st.markdown(f"{i}. {step}")
        else:
            st.info("Calculation steps are not available for this run.")

        st.markdown("### AI Audit (Input + Calculation)")
        st.caption("Дополнительный AI-анализ качества входных данных и корректности расчета.")
        if st.button("Сгенерировать AI-аудит расчета"):
            with st.spinner("AI анализирует входные и расчетные данные..."):
                st.session_state.valuation_ai_audit = generate_ai_valuation_diagnostics(
                    llm=llm,
                    project_text=st.session_state.valuation_last_input_text,
                    valuation_payload=payload,
                )
        if st.session_state.valuation_ai_audit:
            st.markdown(st.session_state.valuation_ai_audit)


with tab_damodaran:
    st.subheader("Damodaran Analytics")
    st.caption("Обычная SQL-загрузка (без векторов) + визуальная проверка качества данных.")

    data_dir = APP_DIR.parent / "damodardan_data"
    st.code(str(data_dir), language="text")

    replace_existing = st.checkbox("Перезалить с очисткой текущих SQL-таблиц", value=True)
    include_global = st.checkbox("Загружать globalcompfirms*.xlsx", value=True)
    if st.button("Загрузить Damodaran в SQL БД"):
        if not data_dir.exists():
            st.error(f"Папка не найдена: {data_dir}")
        else:
            with st.spinner("Читаем файлы и загружаем в SQL-таблицы..."):
                stats = load_damodaran_to_sql_tables(
                    supabase=supabase,
                    data_dir=data_dir,
                    replace_existing=replace_existing,
                    include_global_companies=include_global,
                    batch_size=500,
                )
            st.success("Загрузка завершена.")
            st.json(stats)

    try:
        industry_rows = fetch_all_rows(
            supabase,
            table_name="damodaran_industry_metrics",
            columns="year,dataset_family,industry_name,number_of_firms,current_pe,trailing_pe,forward_pe,peg_ratio,price_sales,ev_sales,ev_ebitda,ev_ebit,ev_ebit_after_tax,beta,cost_of_equity,cost_of_debt,tax_rate,cost_of_capital",
            batch_size=1000,
        )
        country_rows = fetch_all_rows(
            supabase,
            table_name="damodaran_country_metrics",
            columns="year,dataset_family,country,moody_rating,default_spread,equity_risk_premium,country_risk_premium,tax_rate",
            batch_size=1000,
        )
        company_rows = fetch_all_rows(
            supabase,
            table_name="damodaran_global_companies",
            columns="year,company_name,industry_group,country,ev_sales,ev_ebitda,pe_ratio,beta,cost_of_capital",
            batch_size=1000,
        )
    except Exception as e:
        st.error(
            "SQL-таблицы Damodaran пока недоступны. "
            "Примените `app/damodaran_tables.sql` в Postgres."
        )
        st.code(str(e))
        industry_rows = []
        country_rows = []
        company_rows = []

    df_ind = pd.DataFrame(industry_rows)
    df_country = pd.DataFrame(country_rows)
    df_comp = pd.DataFrame(company_rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Industry rows", f"{len(df_ind):,}")
    c2.metric("Country rows", f"{len(df_country):,}")
    c3.metric("Global company rows", f"{len(df_comp):,}")
    min_year = int(df_ind["year"].min()) if not df_ind.empty else None
    max_year = int(df_ind["year"].max()) if not df_ind.empty else None
    c4.metric("Year coverage (industry)", f"{min_year}–{max_year}" if min_year is not None else "n/a")

    if df_ind.empty and df_country.empty:
        st.info("Данные еще не загружены. Нажми кнопку выше.")
    else:
        st.markdown("### Покрытие по годам")
        if not df_ind.empty:
            yearly_ind = (
                df_ind.groupby(["year", "dataset_family"], as_index=False)
                .size()
                .rename(columns={"size": "rows"})
                .sort_values(["year", "dataset_family"])
            )
            fig_year_ind = px.line(
                yearly_ind,
                x="year",
                y="rows",
                color="dataset_family",
                markers=True,
                title="Industry datasets: количество записей по годам",
            )
            st.plotly_chart(fig_year_ind, use_container_width=True)

        if not df_country.empty:
            yearly_country = (
                df_country.groupby(["year", "dataset_family"], as_index=False)
                .size()
                .rename(columns={"size": "rows"})
                .sort_values(["year", "dataset_family"])
            )
            fig_year_country = px.line(
                yearly_country,
                x="year",
                y="rows",
                color="dataset_family",
                markers=True,
                title="Country datasets: количество записей по годам",
            )
            st.plotly_chart(fig_year_country, use_container_width=True)

        st.markdown("### Отраслевые мультипликаторы")
        metric_map = {
            "EV/Sales (psdata)": ("ev_sales", "psdata"),
            "Price/Sales (psdata)": ("price_sales", "psdata"),
            "Current PE (pedata)": ("current_pe", "pedata"),
            "EV/EBITDA (vebitda)": ("ev_ebitda", "vebitda"),
            "Cost of Capital (wacc)": ("cost_of_capital", "wacc"),
            "Beta (wacc)": ("beta", "wacc"),
        }
        metric_label = st.selectbox("Метрика", list(metric_map.keys()))
        metric_col, family = metric_map[metric_label]
        family_df = df_ind[df_ind["dataset_family"] == family].copy()
        family_df[metric_col] = pd.to_numeric(family_df[metric_col], errors="coerce")
        family_df = family_df.dropna(subset=[metric_col])
        if family_df.empty:
            st.info("Нет данных для выбранной метрики.")
        else:
            years = sorted(family_df["year"].dropna().astype(int).unique().tolist())
            selected_year = st.select_slider("Год", options=years, value=years[-1])
            year_df = (
                family_df[family_df["year"] == selected_year]
                .sort_values(metric_col, ascending=False)
                .head(20)
            )
            fig_top = px.bar(
                year_df.sort_values(metric_col),
                x=metric_col,
                y="industry_name",
                orientation="h",
                title=f"Top-20 отраслей по {metric_col} ({selected_year})",
            )
            st.plotly_chart(fig_top, use_container_width=True)

            industries = sorted(family_df["industry_name"].dropna().astype(str).unique().tolist())
            selected_industry = st.selectbox("Динамика по отрасли", industries, index=0)
            trend_df = family_df[family_df["industry_name"] == selected_industry].sort_values("year")
            fig_trend = px.line(
                trend_df,
                x="year",
                y=metric_col,
                markers=True,
                title=f"{selected_industry}: динамика {metric_col}",
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            missing_ratio = year_df[metric_col].isna().mean() if len(year_df) else 0.0
            st.caption(f"Проверка качества: доля пропусков в топ-срезе = {missing_ratio:.1%}")

        st.markdown("### Страновые метрики")
        if not df_country.empty:
            merged_country = (
                df_country.groupby(["year", "country"], as_index=False)[
                    ["default_spread", "equity_risk_premium", "country_risk_premium", "tax_rate"]
                ]
                .max()
                .sort_values(["year", "country"])
            )
            for col in ["default_spread", "equity_risk_premium", "country_risk_premium", "tax_rate"]:
                merged_country[col] = pd.to_numeric(merged_country[col], errors="coerce")

            c_metric = st.selectbox(
                "Метрика по странам",
                ["equity_risk_premium", "country_risk_premium", "default_spread", "tax_rate"],
                index=0,
            )
            c_years = sorted(merged_country["year"].dropna().astype(int).unique().tolist())
            if c_years:
                c_year = st.select_slider("Год (страны)", options=c_years, value=c_years[-1])
                c_top = (
                    merged_country[merged_country["year"] == c_year]
                    .dropna(subset=[c_metric])
                    .sort_values(c_metric, ascending=False)
                    .head(20)
                )
                fig_country_top = px.bar(
                    c_top.sort_values(c_metric),
                    x=c_metric,
                    y="country",
                    orientation="h",
                    title=f"Top-20 стран по {c_metric} ({c_year})",
                )
                st.plotly_chart(fig_country_top, use_container_width=True)

                countries = sorted(merged_country["country"].dropna().astype(str).unique().tolist())
                selected_country = st.selectbox("Динамика по стране", countries, index=0)
                c_trend = merged_country[merged_country["country"] == selected_country].sort_values("year")
                fig_country_trend = px.line(
                    c_trend,
                    x="year",
                    y=c_metric,
                    markers=True,
                    title=f"{selected_country}: динамика {c_metric}",
                )
                st.plotly_chart(fig_country_trend, use_container_width=True)

        st.markdown("### Превью данных")
        p1, p2, p3 = st.tabs(["Industry table", "Country table", "Global companies"])
        with p1:
            st.dataframe(df_ind.head(200), use_container_width=True)
        with p2:
            st.dataframe(df_country.head(200), use_container_width=True)
        with p3:
            st.dataframe(df_comp.head(200), use_container_width=True)


with tab_benchmark:
    st.subheader("Benchmark (R²) — deterministic valuation accuracy")
    st.caption(
        "По умолчанию LLM не участвует в расчете бенчмарка: "
        "используется скриптовая математика из `evaluate_project_deterministic(..., llm=None)`."
    )

    with st.expander("ML CSV Benchmark", expanded=True):
        st.caption(
            "Загрузите CSV с признаками компаний и target valuation. "
            "Блок прогоняет строки через CatBoost ML benchmark и считает R²/MAE/error_x."
        )
        ml_benchmark_csv = st.file_uploader("CSV для ML benchmark", type=["csv"], key="benchmark_ml_csv")
        ml_benchmark_target_col = st.text_input(
            "Target column",
            value="valuation_usd",
            key="benchmark_ml_target_col",
        )
        ml_benchmark_max_rows = st.number_input(
            "Max rows (0 = all)",
            min_value=0,
            max_value=100000,
            value=0,
            step=1000,
            key="benchmark_ml_max_rows",
        )
        if st.button("Запустить ML CSV Benchmark", disabled=ml_benchmark_csv is None):
            try:
                ml_df = pd.read_csv(ml_benchmark_csv, low_memory=False)
                with st.spinner("ML модель считает оценки и метрики..."):
                    ml_metrics, ml_predictions_df = evaluate_ml_csv(
                        ml_df,
                        target_col=ml_benchmark_target_col.strip() or None,
                        max_rows=int(ml_benchmark_max_rows) or None,
                    )
                st.success("ML CSV benchmark готов.")
                b1, b2, b3, b4, b5 = st.columns(5)
                b1.metric("R²", f"{ml_metrics['r2']:.3f}")
                b2.metric("MAE log10", f"{ml_metrics['mae_log10']:.3f}")
                b3.metric("Error x", f"±{ml_metrics['error_x']:.2f}x")
                b4.metric("Rows", f"{ml_metrics['rows_predicted']:,}/{ml_metrics['rows_valid_target']:,}")
                b5.metric(
                    "Range hit",
                    f"{ml_metrics.get('range_hit_rate', 0.0) * 100:.1f}%",
                    f"{ml_metrics.get('range_hits', 0)}/{ml_metrics['rows_predicted']}",
                )
                if "median_error_x" in ml_metrics:
                    st.caption(f"Median absolute miss: {ml_metrics['median_error_x']:.2f}x")
                coverage = ml_metrics.get("csv_direct_feature_coverage", 0.0)
                st.caption(
                    "CSV feature coverage: "
                    f"{ml_metrics.get('csv_direct_feature_cols', 0)}/"
                    f"{ml_metrics.get('model_feature_cols', 0)} "
                    f"({coverage * 100:.1f}%)"
                )
                if coverage < 0.25:
                    st.warning(
                        "Этот CSV содержит мало прямых признаков модели. "
                        "Результат показывает качество sparse-input интеграции, "
                        "а не качество модели на enriched v6 feature table."
                    )
                if ml_metrics["rows_failed"]:
                    st.warning(f"Prediction failures: {ml_metrics['rows_failed']}. First errors: {ml_metrics['first_errors']}")

                stage_metrics = pd.DataFrame(ml_metrics.get("stage_metrics") or [])
                st.markdown("#### Stage metrics")
                if not stage_metrics.empty:
                    st.dataframe(stage_metrics, use_container_width=True)
                else:
                    st.info("No stage metrics available.")

                st.markdown("#### Predictions")
                preview_cols = [
                    c
                    for c in [
                        "Organization Name",
                        "Funding Type",
                        "target_usd_eval",
                        "raw_prediction_log10",
                        "prediction_mid_usd",
                        "prediction_min_usd",
                        "prediction_max_usd",
                        "prediction_log10",
                        "prediction_error",
                        "prediction_abs_error_x",
                        "target_in_prediction_range",
                        "calibration_method",
                        "calibration_multiplier",
                        "calibration_segment",
                    ]
                    if c in ml_predictions_df.columns
                ]
                st.dataframe(ml_predictions_df[preview_cols].head(300), use_container_width=True)
                st.download_button(
                    "Скачать ML predictions CSV",
                    data=ml_predictions_df.to_csv(index=False).encode("utf-8"),
                    file_name="ml_benchmark_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"ML CSV benchmark failed: {e}")

    legacy_broken_path = APP_DIR / "legacy_old_system" / "benchmark_dataset.json.broken"
    legacy_path = APP_DIR / "legacy_old_system" / "benchmark_dataset.json"
    baseline_path = APP_DIR / "benchmark_baseline_2010_2020.json"
    dotcom_path = APP_DIR / "benchmark_dotcom_dataset.json"

    legacy_dataset = load_json_dataset(legacy_broken_path)
    if not legacy_dataset:
        legacy_dataset = load_json_dataset(legacy_path)
    baseline_dataset = load_json_dataset(baseline_path)
    dotcom_dataset = load_json_dataset(dotcom_path)

    if not legacy_dataset:
        st.error(f"Не найден benchmark dataset: {legacy_broken_path} или {legacy_path}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline 2010-2020", f"{len(baseline_dataset)}")
    c2.metric("Dot-com dataset", f"{len(dotcom_dataset)}")
    c3.metric("LLM in benchmark", "OFF by default")

    mode = st.selectbox(
        "Набор данных для прогона",
        [
            "Baseline 2010-2020 (default)",
            "Legacy mixed",
            "Legacy AI (2023+)",
            "Dot-com bubble (1998-2001)",
            "Merged (Legacy + Dot-com)",
        ],
        index=0,
    )
    use_llm_stage = st.checkbox(
        "Использовать LLM только для stage-detection (для сравнения, не рекомендуется для чистого теста)",
        value=False,
    )
    max_cases = st.slider("Лимит кейсов в прогоне", min_value=3, max_value=60, value=20, step=1)

    if mode == "Baseline 2010-2020 (default)":
        selected_dataset = baseline_dataset if baseline_dataset else legacy_dataset
    elif mode == "Legacy mixed":
        selected_dataset = legacy_dataset
    elif mode == "Legacy AI (2023+)":
        selected_dataset = []
        for item in legacy_dataset:
            y = parse_year_from_source(str(item.get("source") or ""))
            if (y is not None and y >= 2023) and is_ai_case(item):
                selected_dataset.append(item)
    elif mode == "Dot-com bubble (1998-2001)":
        selected_dataset = dotcom_dataset
    else:
        selected_dataset = legacy_dataset + dotcom_dataset

    selected_dataset = selected_dataset[:max_cases]

    if st.button("Запустить Benchmark"):
        with st.spinner("Считаем предсказания и метрики..."):
            rows, metrics, text_log = run_benchmark(
                dataset=selected_dataset,
                supabase=supabase,
                llm=llm,
                use_llm_stage=use_llm_stage,
            )
        st.markdown("### Текстовый протокол прогона")
        st.code(text_log, language="text")

        st.markdown("### Метрики")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", f"{metrics.r2:.4f}")
        m2.metric("R² (log10)", f"{metrics.r2_log10:.4f}")
        m3.metric("MAE (USD)", f"{metrics.mae_usd:,.0f}")
        m4.metric("MAPE", f"{metrics.mape_pct:.2f}%")
        m5, m6, m7, _ = st.columns(4)
        m5.metric("Median AE (USD)", f"{metrics.median_ae_usd:,.0f}")
        m6.metric("Median Pred/Real", f"{metrics.median_ratio_pred_to_real:.3f}")
        m7.metric("Underpricing share", f"{metrics.underpricing_share_pct:.1f}%")

        if rows:
            df = pd.DataFrame([r.__dict__ for r in rows])
            st.markdown("### Результаты по кейсам")
            st.dataframe(df, use_container_width=True)

            fig = px.scatter(
                df,
                x="actual_usd",
                y="predicted_usd",
                hover_name="name",
                color="stage",
                log_x=True,
                log_y=True,
                title="Actual vs Predicted (log-log)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Нет корректных кейсов для расчета метрик.")

    st.divider()
    st.markdown("### Гипотеза: AI 2023+ vs Dot-com bubble")
    st.caption(
        "Сравнение двух срезов: современные AI-стартапы и bubble-era dot-com кейсы. "
        "Если машинная оценка системно занижает оба сегмента, R² и median ratio будут низкими."
    )
    if st.button("Сравнить срезы"):
        ai_2023 = []
        for item in legacy_dataset:
            y = parse_year_from_source(str(item.get("source") or ""))
            if (y is not None and y >= 2023) and is_ai_case(item):
                ai_2023.append(item)
        dotcom = dotcom_dataset

        with st.spinner("Считаем AI 2023+ cohort..."):
            ai_rows, ai_metrics, ai_log = run_benchmark(
                dataset=ai_2023,
                supabase=supabase,
                llm=llm,
                use_llm_stage=use_llm_stage,
            )
        with st.spinner("Считаем dot-com cohort..."):
            dot_rows, dot_metrics, dot_log = run_benchmark(
                dataset=dotcom,
                supabase=supabase,
                llm=llm,
                use_llm_stage=use_llm_stage,
            )

        cmp_df = pd.DataFrame(
            [
                {
                    "cohort": "AI 2023+",
                    "n": ai_metrics.n,
                    "r2": ai_metrics.r2,
                    "r2_log10": ai_metrics.r2_log10,
                    "mae_usd": ai_metrics.mae_usd,
                    "mape_pct": ai_metrics.mape_pct,
                    "median_ratio_pred_to_real": ai_metrics.median_ratio_pred_to_real,
                    "underpricing_share_pct": ai_metrics.underpricing_share_pct,
                },
                {
                    "cohort": "Dot-com 1998-2001",
                    "n": dot_metrics.n,
                    "r2": dot_metrics.r2,
                    "r2_log10": dot_metrics.r2_log10,
                    "mae_usd": dot_metrics.mae_usd,
                    "mape_pct": dot_metrics.mape_pct,
                    "median_ratio_pred_to_real": dot_metrics.median_ratio_pred_to_real,
                    "underpricing_share_pct": dot_metrics.underpricing_share_pct,
                },
            ]
        )
        st.dataframe(cmp_df, use_container_width=True)

        st.markdown("#### Интерпретация")
        st.markdown(
            "- Низкий `R²` и `R² log10` означает, что модель плохо объясняет разброс реальных оценок.\n"
            "- `median_ratio_pred_to_real < 1.0` означает системное занижение.\n"
            "- Если оба cohort показывают низкий `R²`, это подтверждает гипотезу о слабой объясняющей силе чисто машинной формулы в периодах пузырей/хайпа."
        )

        with st.expander("Текстовый лог: AI 2023+"):
            st.code(ai_log, language="text")
        with st.expander("Текстовый лог: Dot-com"):
            st.code(dot_log, language="text")
