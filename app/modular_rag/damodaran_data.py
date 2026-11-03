from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd
from supabase import Client


FAMILY_PREFIXES = ("pedata", "psdata", "vebitda", "wacc", "ctryprem", "countrytaxrates")


@dataclass
class LoadStats:
    files_seen: int = 0
    industry_rows: int = 0
    country_rows: int = 0
    company_rows: int = 0
    failed_files: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "files_seen": self.files_seen,
            "industry_rows": self.industry_rows,
            "country_rows": self.country_rows,
            "company_rows": self.company_rows,
            "failed_files": self.failed_files,
        }


def _parse_year_from_name(filename: str) -> int | None:
    # `*24.xls` => 2024; `*98.xls` => 1998
    m2 = re.search(r"(\d{2})(?=\.[^.]+$)", filename)
    if m2:
        y = int(m2.group(1))
        return 1900 + y if y >= 98 else 2000 + y

    # Fallback for explicit four-digit years in names (e.g. globalcompfirms2025.xlsx)
    m4 = re.search(r"(19\d{2}|20\d{2})", filename)
    if m4:
        return int(m4.group(1))
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)
    s = str(value).strip().replace("\u00a0", " ")
    if not s or s.lower() in {"nan", "none", "na", "n/a", "-"}:
        return None
    s = s.replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    f = _to_float(value)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None


def _normalize_col(col: Any) -> str:
    return re.sub(r"\s+", " ", str(col).strip().lower())


def _find_header_row(path: Path, sheet_name: str, anchor: str, max_scan_rows: int = 60) -> int:
    preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=max_scan_rows)
    anchor_norm = _normalize_col(anchor)
    for idx, row in preview.iterrows():
        cells = [_normalize_col(v) for v in row.tolist()]
        if any(anchor_norm == c or anchor_norm in c for c in cells):
            return int(idx)
    raise ValueError(f"Header row not found in {path.name}:{sheet_name} for anchor={anchor}")


def _find_header_row_any(path: Path, sheet_name: str, anchors: list[str], max_scan_rows: int = 60) -> int:
    preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=max_scan_rows)
    anchors_norm = [_normalize_col(a) for a in anchors]
    for idx, row in preview.iterrows():
        cells = [_normalize_col(v) for v in row.tolist()]
        for a in anchors_norm:
            if any(a == c for c in cells):
                return int(idx)
    raise ValueError(f"Header row not found in {path.name}:{sheet_name} for anchors={anchors}")


def _sheet_names(path: Path) -> list[str]:
    return list(pd.ExcelFile(path).sheet_names)


def _pick_sheet_name(path: Path, preferred: list[str], fallback_anchor: str | None = None) -> str:
    names = _sheet_names(path)
    by_norm = {_normalize_col(n): n for n in names}
    for p in preferred:
        hit = by_norm.get(_normalize_col(p))
        if hit:
            return hit
    if fallback_anchor:
        for n in names:
            try:
                _find_header_row(path, n, fallback_anchor, max_scan_rows=80)
                return n
            except Exception:
                continue
    return names[0]


def _read_with_detected_header(path: Path, sheet_name: str, anchor: str) -> pd.DataFrame:
    header_row = _find_header_row(path, sheet_name, anchor)
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    keep_cols = []
    for c in df.columns:
        c_norm = _normalize_col(c)
        if c_norm.startswith("unnamed:"):
            continue
        keep_cols.append(c)
    if keep_cols:
        df = df[keep_cols]
    return df


def _pick_col(df: pd.DataFrame, candidates: list[str], avoid_suffix_1: bool = True) -> str | None:
    norm = {_normalize_col(c): str(c) for c in df.columns}
    for want in candidates:
        for c in df.columns:
            cn = _normalize_col(c)
            if want in cn:
                if avoid_suffix_1 and cn.endswith(".1"):
                    continue
                return str(c)
    for want in candidates:
        for cn, orig in norm.items():
            if want in cn:
                return orig
    return None


def _not_blank(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    return bool(s and s.lower() != "nan")


def _parse_industry_file(path: Path, dataset_family: str, year: int) -> list[dict[str, Any]]:
    sheet_name = _pick_sheet_name(
        path,
        preferred=["Industry Averages", "Industry average", "Data", "Sheet1"],
        fallback_anchor="Industry Name",
    )
    try:
        df = _read_with_detected_header(path, sheet_name, "Industry Name")
    except Exception:
        header_row = _find_header_row_any(path, sheet_name, anchors=["industry name", "industry"], max_scan_rows=100)
        df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)

    col_industry = _pick_col(df, ["industry name", "industry"], avoid_suffix_1=False)
    col_firms = _pick_col(df, ["number of firms"], avoid_suffix_1=False)

    col_current_pe = _pick_col(df, ["current pe"])
    col_trailing_pe = _pick_col(df, ["trailing pe"])
    col_forward_pe = _pick_col(df, ["forward pe"])
    col_peg = _pick_col(df, ["peg ratio"])

    col_price_sales = _pick_col(df, ["price/sales"])
    col_ev_sales = _pick_col(df, ["ev/sales", "value/sales"])

    col_ev_ebitda = _pick_col(df, ["ev/ebitda"])
    col_ev_ebit = _pick_col(df, ["ev/ebit"], avoid_suffix_1=True)
    col_ev_ebit_after_tax = _pick_col(df, ["ev/ebit (1-t)"], avoid_suffix_1=True)

    col_beta = _pick_col(df, ["beta"])
    col_cost_equity = _pick_col(df, ["cost of equity"])
    col_cost_debt = _pick_col(df, ["cost of debt"])
    col_tax_rate = _pick_col(df, ["tax rate"])
    col_cost_capital = _pick_col(df, ["cost of capital"], avoid_suffix_1=True)

    if not col_industry:
        return []

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        industry = r.get(col_industry)
        if not _not_blank(industry):
            continue
        industry_name = str(industry).strip()
        if industry_name.lower().startswith("date updated"):
            continue

        rows.append(
            {
                "dataset_family": dataset_family,
                "year": year,
                "industry_name": industry_name,
                "number_of_firms": _to_int(r.get(col_firms)) if col_firms else None,
                "current_pe": _to_float(r.get(col_current_pe)) if col_current_pe else None,
                "trailing_pe": _to_float(r.get(col_trailing_pe)) if col_trailing_pe else None,
                "forward_pe": _to_float(r.get(col_forward_pe)) if col_forward_pe else None,
                "peg_ratio": _to_float(r.get(col_peg)) if col_peg else None,
                "price_sales": _to_float(r.get(col_price_sales)) if col_price_sales else None,
                "ev_sales": _to_float(r.get(col_ev_sales)) if col_ev_sales else None,
                "ev_ebitda": _to_float(r.get(col_ev_ebitda)) if col_ev_ebitda else None,
                "ev_ebit": _to_float(r.get(col_ev_ebit)) if col_ev_ebit else None,
                "ev_ebit_after_tax": _to_float(r.get(col_ev_ebit_after_tax)) if col_ev_ebit_after_tax else None,
                "beta": _to_float(r.get(col_beta)) if col_beta else None,
                "cost_of_equity": _to_float(r.get(col_cost_equity)) if col_cost_equity else None,
                "cost_of_debt": _to_float(r.get(col_cost_debt)) if col_cost_debt else None,
                "tax_rate": _to_float(r.get(col_tax_rate)) if col_tax_rate else None,
                "cost_of_capital": _to_float(r.get(col_cost_capital)) if col_cost_capital else None,
                "source_file": path.name,
            }
        )
    return rows


def _parse_countrytaxrates(path: Path, year: int) -> list[dict[str, Any]]:
    sheet_name = _pick_sheet_name(path, preferred=["Sheet1", "Data"], fallback_anchor="Country")
    try:
        df = _read_with_detected_header(path, sheet_name, "Country")
    except Exception:
        # Older files sometimes use "Location" instead of "Country".
        header_row = _find_header_row_any(path, sheet_name, anchors=["country", "nation", "location"], max_scan_rows=100)
        df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    if "Country" not in df.columns:
        # Some versions have different capitalization.
        col_country = _pick_col(df, ["country", "location"], avoid_suffix_1=False)
        if not col_country:
            return []
        df = df.rename(columns={col_country: "Country"})

    year_cols = []
    for c in df.columns:
        m = re.fullmatch(r"(19\d{2}|20\d{2})", str(c).strip())
        if m:
            year_cols.append(str(c).strip())
        else:
            # Sometimes headers are numeric float (`2023.0`)
            f = _to_float(c)
            if f and int(f) >= 1900:
                year_cols.append(str(int(f)))
                df = df.rename(columns={c: str(int(f))})

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        country = r.get("Country")
        if not _not_blank(country):
            continue
        country_name = str(country).strip()
        for yc in sorted(set(year_cols)):
            tax = _to_float(r.get(yc))
            if tax is None:
                continue
            rows.append(
                {
                    "dataset_family": "countrytaxrates",
                    "year": int(yc),
                    "country": country_name,
                    "moody_rating": None,
                    "default_spread": None,
                    "equity_risk_premium": None,
                    "country_risk_premium": None,
                    "tax_rate": tax,
                    "source_file": path.name,
                }
            )
    return rows


def _parse_ctryprem(path: Path, year: int) -> list[dict[str, Any]]:
    sheet_name = _pick_sheet_name(
        path,
        preferred=["ERPs by country", "Country Equity Risk", "Data", "Sheet1"],
        fallback_anchor="Country",
    )
    df = _read_with_detected_header(path, sheet_name, "Country")

    col_country = _pick_col(df, ["country"], avoid_suffix_1=False)
    col_moody = _pick_col(df, ["moody"])
    col_default = _pick_col(df, ["rating-based default spread"])
    col_erp = _pick_col(df, ["total equity risk premium"], avoid_suffix_1=True)
    col_crp = _pick_col(df, ["country risk premium"], avoid_suffix_1=True)

    if not col_country:
        return []

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        country = r.get(col_country)
        if not _not_blank(country):
            continue
        country_name = str(country).strip()
        if country_name.lower() in {"rating", "country"}:
            continue

        rows.append(
            {
                "dataset_family": "ctryprem",
                "year": year,
                "country": country_name,
                "moody_rating": str(r.get(col_moody)).strip() if col_moody and _not_blank(r.get(col_moody)) else None,
                "default_spread": _to_float(r.get(col_default)) if col_default else None,
                "equity_risk_premium": _to_float(r.get(col_erp)) if col_erp else None,
                "country_risk_premium": _to_float(r.get(col_crp)) if col_crp else None,
                "tax_rate": None,
                "source_file": path.name,
            }
        )
    return rows


def _parse_global_companies(path: Path, year: int | None) -> list[dict[str, Any]]:
    df = pd.read_excel(path, sheet_name=0)

    col_company = _pick_col(df, ["company name"], avoid_suffix_1=False)
    col_ticker = _pick_col(df, ["exchange:ticker"], avoid_suffix_1=False)
    col_industry = _pick_col(df, ["industry group"])
    col_sector = _pick_col(df, ["primary sector"])
    col_country = _pick_col(df, ["country"], avoid_suffix_1=False)
    col_broad = _pick_col(df, ["broad group"])
    col_sub = _pick_col(df, ["sub group"])
    col_mkt = _pick_col(df, ["market cap"])
    col_rev = _pick_col(df, ["revenue"])
    col_ev_sales = _pick_col(df, ["ev/sales"])
    col_ev_ebitda = _pick_col(df, ["ev/ebitda"])
    col_pe = _pick_col(df, ["pe ratio", "trailing pe"])
    col_beta = _pick_col(df, ["levered beta", "beta"])
    col_wacc = _pick_col(df, ["cost of capital"])

    if not col_company:
        return []

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        company = r.get(col_company)
        if not _not_blank(company):
            continue
        rows.append(
            {
                "year": year,
                "company_name": str(company).strip(),
                "exchange_ticker": str(r.get(col_ticker)).strip() if col_ticker and _not_blank(r.get(col_ticker)) else None,
                "industry_group": str(r.get(col_industry)).strip() if col_industry and _not_blank(r.get(col_industry)) else None,
                "primary_sector": str(r.get(col_sector)).strip() if col_sector and _not_blank(r.get(col_sector)) else None,
                "country": str(r.get(col_country)).strip() if col_country and _not_blank(r.get(col_country)) else None,
                "broad_group": str(r.get(col_broad)).strip() if col_broad and _not_blank(r.get(col_broad)) else None,
                "sub_group": str(r.get(col_sub)).strip() if col_sub and _not_blank(r.get(col_sub)) else None,
                "market_cap_usd": _to_float(r.get(col_mkt)) if col_mkt else None,
                "revenue_usd": _to_float(r.get(col_rev)) if col_rev else None,
                "ev_sales": _to_float(r.get(col_ev_sales)) if col_ev_sales else None,
                "ev_ebitda": _to_float(r.get(col_ev_ebitda)) if col_ev_ebitda else None,
                "pe_ratio": _to_float(r.get(col_pe)) if col_pe else None,
                "beta": _to_float(r.get(col_beta)) if col_beta else None,
                "cost_of_capital": _to_float(r.get(col_wacc)) if col_wacc else None,
                "source_file": path.name,
            }
        )
    return rows


def _chunks(rows: list[dict[str, Any]], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _insert_upsert_batches(
    supabase: Client,
    table_name: str,
    rows: list[dict[str, Any]],
    on_conflict: str,
    batch_size: int = 500,
) -> int:
    total = 0
    for batch in _chunks(rows, batch_size):
        supabase.table(table_name).upsert(batch, on_conflict=on_conflict).execute()
        total += len(batch)
    return total


def _dedupe_rows(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in key_fields)
        out[key] = row
    return list(out.values())


def ensure_damodaran_tables_exist(supabase: Client) -> None:
    # Quick check via lightweight select. If table is missing, PostgREST will throw.
    checks = [
        "damodaran_industry_metrics",
        "damodaran_country_metrics",
        "damodaran_global_companies",
    ]
    for t in checks:
        try:
            supabase.table(t).select("*").limit(1).execute()
        except Exception as e:
            raise RuntimeError(
                f"SQL table `{t}` не найдена. "
                "Сначала примени `app/damodaran_tables.sql` в Postgres."
            ) from e


def load_damodaran_to_sql_tables(
    supabase: Client,
    data_dir: Path,
    replace_existing: bool = True,
    include_global_companies: bool = True,
    batch_size: int = 500,
) -> dict[str, int]:
    ensure_damodaran_tables_exist(supabase)
    stats = LoadStats()

    if replace_existing:
        supabase.table("damodaran_industry_metrics").delete().neq("id", 0).execute()
        supabase.table("damodaran_country_metrics").delete().neq("id", 0).execute()
        if include_global_companies:
            supabase.table("damodaran_global_companies").delete().neq("id", 0).execute()

    industry_rows: list[dict[str, Any]] = []
    country_rows: list[dict[str, Any]] = []
    company_rows: list[dict[str, Any]] = []

    files = sorted([p for p in data_dir.glob("*") if p.is_file()])
    for p in files:
        name = p.name.lower()
        if not name.startswith(FAMILY_PREFIXES) and not name.startswith("globalcompfirms"):
            continue
        stats.files_seen += 1

        year = _parse_year_from_name(p.name)
        try:
            if name.startswith(("pedata", "psdata", "vebitda", "wacc")) and year:
                family = name.split(".")[0]
                family = re.match(r"[a-z]+", family).group(0) if re.match(r"[a-z]+", family) else family
                industry_rows.extend(_parse_industry_file(p, family, year))
            elif name.startswith("ctryprem") and year:
                country_rows.extend(_parse_ctryprem(p, year))
            elif name.startswith("countrytaxrates"):
                # `countrytaxrates24.xls` includes multiple years inside.
                country_rows.extend(_parse_countrytaxrates(p, year or 0))
            elif include_global_companies and name.startswith("globalcompfirms"):
                company_rows.extend(_parse_global_companies(p, year))
        except Exception:
            stats.failed_files += 1

    stats.industry_rows = _insert_upsert_batches(
        supabase,
        table_name="damodaran_industry_metrics",
        rows=_dedupe_rows(industry_rows, ["dataset_family", "year", "industry_name"]),
        on_conflict="dataset_family,year,industry_name",
        batch_size=batch_size,
    ) if industry_rows else 0

    stats.country_rows = _insert_upsert_batches(
        supabase,
        table_name="damodaran_country_metrics",
        rows=_dedupe_rows(country_rows, ["dataset_family", "year", "country"]),
        on_conflict="dataset_family,year,country",
        batch_size=batch_size,
    ) if country_rows else 0

    stats.company_rows = _insert_upsert_batches(
        supabase,
        table_name="damodaran_global_companies",
        rows=_dedupe_rows(company_rows, ["source_file", "company_name"]),
        on_conflict="source_file,company_name",
        batch_size=batch_size,
    ) if company_rows else 0

    return stats.to_dict()


def fetch_all_rows(
    supabase: Client,
    table_name: str,
    columns: str = "*",
    batch_size: int = 1000,
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        res = supabase.table(table_name).select(columns).range(offset, offset + batch_size - 1).execute()
        rows = res.data or []
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < batch_size:
            break
        offset += batch_size
    return all_rows
