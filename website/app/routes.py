from __future__ import annotations

import pandas as pd
from flask import Blueprint, redirect, render_template, request, url_for

from .linkingML import fmt_num, fmt_pct, fmt_proba, get_state_row, load_forecast, state_label

bp = Blueprint("main", __name__)


def _format_display_date(value) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "N/A"
    return parsed.strftime("%B %d, %Y")


def _format_mean_pct(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return "N/A"
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return "N/A"
    return f"{values.mean():.1f}%"


def _critical_mask(df: pd.DataFrame) -> pd.Series:
    if "critical_risk_next_week_pred" not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df["critical_risk_next_week_pred"], errors="coerce").fillna(0).astype(int).eq(1)


def _risk_tone(risk_pred, risk_proba) -> str:
    try:
        pred = int(float(risk_pred))
    except Exception:
        pred = 0

    try:
        proba = float(risk_proba)
    except Exception:
        proba = 0.0

    if pred == 1 or proba >= 0.18:
        return "high"
    if proba >= 0.08:
        return "moderate"
    return "low"


def _risk_label(risk_pred, risk_proba) -> str:
    return {
        "high": "High alert",
        "moderate": "Watch closely",
        "low": "Stable",
    }[_risk_tone(risk_pred, risk_proba)]


def _overview(df: pd.DataFrame) -> dict[str, str | int]:
    return {
        "state_count": int(df["state"].dropna().nunique()) if "state" in df.columns else 0,
        "critical_count": int(_critical_mask(df).sum()),
        "current_week": _format_display_date(df["current_week"].max()) if "current_week" in df.columns else "N/A",
        "forecast_week": _format_display_date(df["forecast_week"].max()) if "forecast_week" in df.columns else "N/A",
        "avg_icu": _format_mean_pct(df, "icu_pct_next_week_pred"),
        "avg_inpatient": _format_mean_pct(df, "inpatient_pct_next_week_pred"),
    }


def _spotlight_rows(df: pd.DataFrame, limit: int = 3) -> list[dict[str, str]]:
    if df.empty:
        return []

    sort_columns = [c for c in ["critical_risk_next_week_pred", "critical_risk_proba", "icu_pct_next_week_pred"] if c in df.columns]
    spotlight = df.sort_values(sort_columns, ascending=[False] * len(sort_columns)).head(limit) if sort_columns else df.head(limit)

    items: list[dict[str, str]] = []
    for _, row in spotlight.iterrows():
        neighbor_code = str(row.get("suggested_neighbor_state", "") or "")
        items.append(
            {
                "code": row.get("state", ""),
                "state": state_label(str(row.get("state", ""))),
                "risk_tone": _risk_tone(row.get("critical_risk_next_week_pred", 0), row.get("critical_risk_proba", 0)),
                "risk_label": _risk_label(row.get("critical_risk_next_week_pred", 0), row.get("critical_risk_proba", 0)),
                "risk_proba": fmt_proba(row.get("critical_risk_proba")),
                "icu": fmt_pct(row.get("icu_pct_next_week_pred")),
                "inpatient": fmt_pct(row.get("inpatient_pct_next_week_pred")),
                "neighbor": state_label(neighbor_code) if neighbor_code else "No neighbor recommendation",
                "recommendation": str(row.get("recommendation", "") or "No recommendation available."),
            }
        )
    return items


@bp.route("/", methods=["GET"])
def index():
    df = load_forecast()

    states = sorted(df["state"].dropna().unique().tolist())
    state_options = [(s, state_label(s)) for s in states]

    selected = request.args.get("state")
    if selected:
        return redirect(url_for("main.state_page", state=selected))

    return render_template(
        "index.html",
        title="Care For Hospitals | Overview",
        state_options=state_options,
        summary=_overview(df),
        spotlight=_spotlight_rows(df),
        missing_cols=df.attrs.get("missing_cols", []),
    )


@bp.route("/state/<state>", methods=["GET"])
def state_page(state: str):
    df = load_forecast()
    row = get_state_row(df, state)
    if row is None:
        return render_template(
            "state.html",
            title="Care For Hospitals | State Not Found",
            not_found=True,
            state=state,
            missing_cols=df.attrs.get("missing_cols", []),
        )

    neighbor_code = str(row.get("suggested_neighbor_state", "") or "")
    nrow = get_state_row(df, neighbor_code) if neighbor_code else None

    data = {
        "state": state_label(state),
        "current_week_label": _format_display_date(row.get("current_week")),
        "forecast_week_label": _format_display_date(row.get("forecast_week")),
        "icu": fmt_pct(row.get("icu_pct_next_week_pred")),
        "inpatient": fmt_pct(row.get("inpatient_pct_next_week_pred")),
        "risk_proba": fmt_proba(row.get("critical_risk_proba")),
        "risk_pred": int(row.get("critical_risk_next_week_pred", 0)),
        "risk_tone": _risk_tone(row.get("critical_risk_next_week_pred", 0), row.get("critical_risk_proba", 0)),
        "risk_label": _risk_label(row.get("critical_risk_next_week_pred", 0), row.get("critical_risk_proba", 0)),
        "disease": fmt_num(row.get("disease_burden_next_week_pred")),
        "recommendation": row.get("recommendation", ""),
        "neighbor": state_label(neighbor_code) if neighbor_code else None,
        "neighbor_code": neighbor_code if neighbor_code else None,
    }

    neighbor_data = None
    if nrow is not None:
        neighbor_data = {
            "state": state_label(neighbor_code),
            "icu": fmt_pct(nrow.get("icu_pct_next_week_pred")),
            "inpatient": fmt_pct(nrow.get("inpatient_pct_next_week_pred")),
            "risk_proba": fmt_proba(nrow.get("critical_risk_proba")),
            "risk_pred": int(nrow.get("critical_risk_next_week_pred", 0)),
            "risk_tone": _risk_tone(nrow.get("critical_risk_next_week_pred", 0), nrow.get("critical_risk_proba", 0)),
            "risk_label": _risk_label(nrow.get("critical_risk_next_week_pred", 0), nrow.get("critical_risk_proba", 0)),
            "disease": fmt_num(nrow.get("disease_burden_next_week_pred")),
        }

    return render_template(
        "state.html",
        title=f"Care For Hospitals | {state}",
        data=data,
        neighbor_data=neighbor_data,
        missing_cols=df.attrs.get("missing_cols", []),
    )


@bp.route("/top-risk", methods=["GET"])
def top_risk():
    df = load_forecast()

    n = request.args.get("n", "15")
    try:
        n = max(5, min(50, int(n)))
    except Exception:
        n = 15

    top_risks = (
        df[_critical_mask(df)]
        .sort_values("critical_risk_proba", ascending=False)
        .head(n)
        .copy()
    )

    top_risks["state_label"] = top_risks["state"].apply(state_label)
    top_risks["risk_tone"] = top_risks.apply(
        lambda row: _risk_tone(row.get("critical_risk_next_week_pred", 0), row.get("critical_risk_proba", 0)),
        axis=1,
    )
    top_risks["risk_label"] = top_risks.apply(
        lambda row: _risk_label(row.get("critical_risk_next_week_pred", 0), row.get("critical_risk_proba", 0)),
        axis=1,
    )

    if "suggested_neighbor_state" in top_risks.columns:
        top_risks["neighbor_label"] = top_risks["suggested_neighbor_state"].apply(
            lambda value: state_label(str(value)) if pd.notna(value) and str(value).strip() else ""
        )
    else:
        top_risks["neighbor_label"] = ""

    if "icu_pct_next_week_pred" in top_risks.columns:
        top_risks["icu_pct_next_week_pred"] = top_risks["icu_pct_next_week_pred"].map(lambda x: f"{x:.1f}%")
    if "inpatient_pct_next_week_pred" in top_risks.columns:
        top_risks["inpatient_pct_next_week_pred"] = top_risks["inpatient_pct_next_week_pred"].map(lambda x: f"{x:.1f}%")
    if "critical_risk_proba" in top_risks.columns:
        top_risks["critical_risk_proba"] = top_risks["critical_risk_proba"].map(lambda x: f"{x:.2f}")
    if "disease_burden_next_week_pred" in top_risks.columns:
        top_risks["disease_burden_next_week_pred"] = top_risks["disease_burden_next_week_pred"].map(lambda x: f"{x:,.0f}")

    return render_template(
        "top_risk.html",
        title="Care For Hospitals | Top Risk States",
        top_risks=top_risks.to_dict(orient="records"),
        summary=_overview(df),
        missing_cols=df.attrs.get("missing_cols", []),
    )
