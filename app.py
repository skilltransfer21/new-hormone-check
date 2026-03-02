import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import hashlib
import uuid
import json
import base64
import hmac
import secrets
import time
import calendar

from streamlit_js_eval import get_geolocation

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
    try:
        alt.data_transformers.disable_max_rows()
    except Exception:
        pass
except Exception:
    ALTAIR_AVAILABLE = False

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False

try:
    import pydeck as pdk
    PYDECK_AVAILABLE = True
except Exception:
    PYDECK_AVAILABLE = False


# =========================================================
# Config
# =========================================================
DATA_FILE = "user_records.csv"
ANALYTICS_FILE = "analytics_events.csv"

ADMIN_PASSWORD = "0000"
USER_PASSWORD = "012345"

SESSION_TTL_SEC = 60 * 10  # 10分以内にイベントがあればactive扱い

# JWT settings (HS256). 依存追加なしの自前JWT。
# 本番運用なら st.secrets["JWT_SECRET"] に置くのが推奨。
JWT_SECRET = os.environ.get("JWT_SECRET", "CHANGE_ME_IN_PROD")


# =========================================================
# App settings (admin toggles)
# =========================================================
def init_settings():
    if "settings" not in st.session_state:
        st.session_state["settings"] = {
            "geo_enabled": True,  # 位置情報取得 ON/OFF
        }


# =========================================================
# Model
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("hormone_risk_model.pkl")


model = load_model()


# =========================================================
# Basic helpers
# =========================================================
def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def append_csv(path: str, df: pd.DataFrame):
    if os.path.exists(path):
        old = safe_read_csv(path)
        out = pd.concat([old, df], ignore_index=True)
    else:
        out = df
    out.to_csv(path, index=False)


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def today_date():
    return datetime.date.today()


def ensure_sid():
    if "sid" not in st.session_state:
        st.session_state["sid"] = str(uuid.uuid4())


# =========================================================
# JWT (HS256) - standard library implementation
# =========================================================
def b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("utf-8")


def b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def jwt_sign(payload: dict, secret: str, exp_seconds: int = 3600) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload2 = dict(payload)
    payload2["iat"] = int(time.time())
    payload2["exp"] = int(time.time()) + int(exp_seconds)

    header_b64 = b64url_encode(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    payload_b64 = b64url_encode(json.dumps(payload2, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = b64url_encode(sig)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def jwt_verify(token: str, secret: str) -> dict | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

        expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        expected_b64 = b64url_encode(expected)

        if not secrets.compare_digest(expected_b64, sig_b64):
            return None

        payload = json.loads(b64url_decode(payload_b64).decode("utf-8"))
        exp = int(payload.get("exp", 0))
        if int(time.time()) > exp:
            return None
        return payload
    except Exception:
        return None


def require_admin_jwt():
    token = st.session_state.get("admin_jwt")
    if not token:
        return False
    payload = jwt_verify(token, JWT_SECRET)
    if not payload:
        return False
    if payload.get("role") != "admin":
        return False
    return True


# =========================================================
# Analytics
# =========================================================
def log_event(event_name: str):
    ensure_sid()
    ts = now_iso()

    row = pd.DataFrame([{
        "timestamp": ts,
        "event": event_name,
        "sid": st.session_state["sid"],
        "path": "admin" if st.session_state.get("is_admin_page") else "user"
    }])

    append_csv(ANALYTICS_FILE, row)


def load_analytics():
    df = safe_read_csv(ANALYTICS_FILE)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday  # Mon=0
    return df


def compute_metrics(adf: pd.DataFrame):
    if adf.empty:
        return {
            "pv_total": 0,
            "pv_admin": 0,
            "pv_user": 0,
            "sessions_total": 0,
            "active_sessions": 0,
            "uu_total": 0
        }

    pv = adf[adf["event"] == "page_view"]
    pv_total = int(len(pv))
    pv_admin = int((pv["path"] == "admin").sum())
    pv_user = int((pv["path"] == "user").sum())

    sessions_total = int(adf["sid"].nunique())
    uu_total = sessions_total  # 近似UU＝sidユニーク

    cutoff = datetime.datetime.now() - datetime.timedelta(seconds=SESSION_TTL_SEC)
    recent = adf[adf["timestamp"] >= cutoff]
    active_sessions = int(recent["sid"].nunique()) if not recent.empty else 0

    return {
        "pv_total": pv_total,
        "pv_admin": pv_admin,
        "pv_user": pv_user,
        "sessions_total": sessions_total,
        "active_sessions": active_sessions,
        "uu_total": uu_total
    }


def daily_pv_uu(adf: pd.DataFrame) -> pd.DataFrame:
    if adf.empty:
        return pd.DataFrame(columns=[
            "date", "pv_total", "pv_admin", "pv_user", "uu_total", "uu_admin", "uu_user"
        ])

    pv = adf[adf["event"] == "page_view"].copy()
    if pv.empty:
        return pd.DataFrame(columns=[
            "date", "pv_total", "pv_admin", "pv_user", "uu_total", "uu_admin", "uu_user"
        ])

    pv_total = pv.groupby("date").size().rename("pv_total")
    pv_admin = pv[pv["path"] == "admin"].groupby("date").size().rename("pv_admin")
    pv_user = pv[pv["path"] == "user"].groupby("date").size().rename("pv_user")

    uu_total = pv.groupby("date")["sid"].nunique().rename("uu_total")
    uu_admin = pv[pv["path"] == "admin"].groupby("date")["sid"].nunique().rename("uu_admin")
    uu_user = pv[pv["path"] == "user"].groupby("date")["sid"].nunique().rename("uu_user")

    out = pd.concat([pv_total, pv_admin, pv_user, uu_total, uu_admin, uu_user], axis=1).fillna(0).reset_index()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    for c in ["pv_total", "pv_admin", "pv_user", "uu_total", "uu_admin", "uu_user"]:
        out[c] = out[c].astype(int)
    return out


def hourly_pv(adf: pd.DataFrame) -> pd.DataFrame:
    if adf.empty:
        return pd.DataFrame(columns=["hour", "pv_total", "pv_admin", "pv_user"])

    pv = adf[adf["event"] == "page_view"].copy()
    if pv.empty:
        return pd.DataFrame(columns=["hour", "pv_total", "pv_admin", "pv_user"])

    base = pv.groupby("hour").size().rename("pv_total")
    admin = pv[pv["path"] == "admin"].groupby("hour").size().rename("pv_admin")
    user = pv[pv["path"] == "user"].groupby("hour").size().rename("pv_user")

    out = pd.concat([base, admin, user], axis=1).fillna(0).reset_index()
    out["hour"] = out["hour"].astype(int)
    for c in ["pv_total", "pv_admin", "pv_user"]:
        out[c] = out[c].astype(int)

    all_hours = pd.DataFrame({"hour": list(range(24))})
    out = all_hours.merge(out, on="hour", how="left").fillna(0)
    for c in ["pv_total", "pv_admin", "pv_user"]:
        out[c] = out[c].astype(int)
    return out.sort_values("hour")


def weekday_pv(adf: pd.DataFrame) -> pd.DataFrame:
    if adf.empty:
        return pd.DataFrame(columns=["weekday", "weekday_label", "pv_total", "pv_admin", "pv_user"])

    pv = adf[adf["event"] == "page_view"].copy()
    if pv.empty:
        return pd.DataFrame(columns=["weekday", "weekday_label", "pv_total", "pv_admin", "pv_user"])

    base = pv.groupby("weekday").size().rename("pv_total")
    admin = pv[pv["path"] == "admin"].groupby("weekday").size().rename("pv_admin")
    user = pv[pv["path"] == "user"].groupby("weekday").size().rename("pv_user")

    out = pd.concat([base, admin, user], axis=1).fillna(0).reset_index()
    out["weekday"] = out["weekday"].astype(int)
    for c in ["pv_total", "pv_admin", "pv_user"]:
        out[c] = out[c].astype(int)

    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    out["weekday_label"] = out["weekday"].apply(lambda x: labels[x] if 0 <= x <= 6 else str(x))

    all_wd = pd.DataFrame({"weekday": list(range(7))})
    all_wd["weekday_label"] = all_wd["weekday"].apply(lambda x: labels[x])
    out = all_wd.merge(out.drop(columns=["weekday_label"]), on="weekday", how="left").fillna(0)
    for c in ["pv_total", "pv_admin", "pv_user"]:
        out[c] = out[c].astype(int)

    return out.sort_values("weekday")


def chart_daily_pv_uu_combo(daily_df: pd.DataFrame, days: int = 30, show_admin_user: bool = False):
    if daily_df.empty:
        st.info("まだPVデータがありません。")
        return

    d = daily_df.tail(days).copy()
    d = d.sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).copy()

    if not ALTAIR_AVAILABLE:
        st.bar_chart(d.set_index("date")[["pv_total"]], use_container_width=True)
        st.line_chart(d.set_index("date")[["uu_total"]], use_container_width=True)
        return

    try:
        if show_admin_user:
            pv_long = d.melt(
                id_vars=["date"],
                value_vars=["pv_total", "pv_admin", "pv_user"],
                var_name="pv_kind",
                value_name="pv"
            )
            bars = alt.Chart(pv_long).mark_bar().encode(
                x=alt.X("date:T", title="Date", axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y("pv:Q", title="PV"),
                color=alt.Color("pv_kind:N", title="PV Type"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                    alt.Tooltip("pv_kind:N", title="Type"),
                    alt.Tooltip("pv:Q", title="PV"),
                ],
            )
        else:
            bars = alt.Chart(d).mark_bar().encode(
                x=alt.X("date:T", title="Date", axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y("pv_total:Q", title="PV"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                    alt.Tooltip("pv_total:Q", title="PV"),
                ],
            )

        line = alt.Chart(d).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%Y-%m-%d")),
            y=alt.Y("uu_total:Q", title="UU (sid approx)"),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("uu_total:Q", title="UU"),
            ],
        )

        layered = alt.layer(bars, line).resolve_scale(y="independent").properties(height=320)
        st.altair_chart(layered, use_container_width=True)

    except Exception:
        st.warning("複合チャートの描画に失敗したため、分割チャートに切り替えました。")
        st.bar_chart(d.set_index("date")[["pv_total"]], use_container_width=True)
        st.line_chart(d.set_index("date")[["uu_total"]], use_container_width=True)


# =========================================================
# Geo helpers
# =========================================================
def extract_geo(loc):
    if not isinstance(loc, dict):
        return None, None, None

    coords = loc.get("coords")
    if isinstance(coords, dict):
        lat = coords.get("latitude")
        lon = coords.get("longitude")
        acc = coords.get("accuracy")
    else:
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        acc = loc.get("accuracy")

    try:
        lat = float(lat) if lat is not None else None
        lon = float(lon) if lon is not None else None
        acc = float(acc) if acc is not None else None
    except Exception:
        return None, None, None

    return lat, lon, acc


def normalize_geo_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "緯度 / Latitude" in df.columns:
        df["lat"] = pd.to_numeric(df["緯度 / Latitude"], errors="coerce")
    else:
        df["lat"] = np.nan

    if "経度 / Longitude" in df.columns:
        df["lon"] = pd.to_numeric(df["経度 / Longitude"], errors="coerce")
    else:
        df["lon"] = np.nan

    if "地域情報取得時刻 / Geo Obtained At" in df.columns:
        df["geo_time"] = pd.to_datetime(df["地域情報取得時刻 / Geo Obtained At"], errors="coerce")
    else:
        df["geo_time"] = pd.NaT

    if "日付 / Date" in df.columns:
        df["record_date"] = pd.to_datetime(df["日付 / Date"], errors="coerce")
    else:
        df["record_date"] = pd.NaT

    return df


def get_latest_by_user(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "名前 / Name" not in df.columns:
        return pd.DataFrame()

    df = normalize_geo_columns(df.copy())
    df = df.sort_values(by=["名前 / Name", "geo_time", "record_date"], ascending=[True, False, False])
    return df.groupby("名前 / Name", as_index=False).head(1).copy()


def color_for_user(name: str):
    if not name:
        return [80, 160, 200]
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)

    def lift(x):
        return int(50 + (x / 255) * 180)

    return [lift(r), lift(g), lift(b)]


def build_paths_df(df: pd.DataFrame, max_points_per_user: int = 200) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = normalize_geo_columns(df.copy())
    df = df.dropna(subset=["lat", "lon", "名前 / Name"]).copy()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for name, g in df.groupby("名前 / Name"):
        gg = g.dropna(subset=["lat", "lon"]).copy()
        if gg.empty:
            continue

        if gg["geo_time"].notna().any():
            gg = gg.sort_values("geo_time", ascending=True)
        else:
            gg = gg.sort_values("record_date", ascending=True)

        if len(gg) > max_points_per_user:
            gg = gg.tail(max_points_per_user)

        path = gg[["lon", "lat"]].values.tolist()
        if len(path) < 2:
            continue

        rows.append({"name": name, "path": path, "color": color_for_user(str(name))})

    return pd.DataFrame(rows)


def user_history_view(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    drop_cols = [
        "緯度 / Latitude",
        "経度 / Longitude",
        "精度(m) / Accuracy(m)",
        "地域情報取得時刻 / Geo Obtained At",
    ]
    cols = [c for c in df.columns if c not in drop_cols]
    out = df[cols].copy()

    preferred = [
        "日付 / Date",
        "名前 / Name",
        "地域 / Region",
        "基礎体温 / BBT",
        "前回生理日 / LMP",
        "体重 / Weight",
        "食事 / Meals",
        "運動 / Exercise",
        "運動内容 / Exercise Details",
        "AI判定 / AI Result",
    ]
    ordered = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[ordered]


# =========================================================
# User calendar helpers (High/Low + 📝 + 🏃)
# =========================================================
def _normalize_user_hist_for_calendar(user_hist: pd.DataFrame) -> pd.DataFrame:
    if user_hist.empty:
        return pd.DataFrame(columns=["date", "risk", "bbt", "weight", "exercise", "meals", "exercise_details", "region", "did_exercise_day"])

    df = user_hist.copy()

    if "日付 / Date" in df.columns:
        df["date"] = pd.to_datetime(df["日付 / Date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "risk", "bbt", "weight", "exercise", "meals", "exercise_details", "region", "did_exercise_day"])

    df["date_only"] = df["date"].dt.date

    def col_or_blank(col):
        return df[col] if col in df.columns else ""

    risk = col_or_blank("AI判定 / AI Result")
    bbt = col_or_blank("基礎体温 / BBT")
    weight = col_or_blank("体重 / Weight")
    exercise = col_or_blank("運動 / Exercise")
    meals = col_or_blank("食事 / Meals")
    ex_details = col_or_blank("運動内容 / Exercise Details")
    region = col_or_blank("地域 / Region")

    if "運動 / Exercise" in df.columns:
        did_exercise = df["運動 / Exercise"].astype(str).str.contains("Yes|はい", regex=True).astype(int)
    else:
        did_exercise = 0

    out = pd.DataFrame({
        "date": df["date_only"],
        "risk": risk.astype(str) if hasattr(risk, "astype") else risk,
        "bbt": bbt.astype(str) if hasattr(bbt, "astype") else bbt,
        "weight": weight.astype(str) if hasattr(weight, "astype") else weight,
        "exercise": exercise.astype(str) if hasattr(exercise, "astype") else exercise,
        "meals": meals.astype(str) if hasattr(meals, "astype") else meals,
        "exercise_details": ex_details.astype(str) if hasattr(ex_details, "astype") else ex_details,
        "region": region.astype(str) if hasattr(region, "astype") else region,
        "did_exercise": did_exercise if hasattr(did_exercise, "__len__") else 0,
    })

    out["_idx"] = np.arange(len(out))
    out = out.sort_values(["date", "_idx"], ascending=[True, True])

    day_ex = out.groupby("date", as_index=False)["did_exercise"].max().rename(columns={"did_exercise": "did_exercise_day"})
    out_last = out.groupby("date", as_index=False).tail(1).drop(columns=["_idx", "did_exercise"])
    out = out_last.merge(day_ex, on="date", how="left")

    return out.sort_values("date").reset_index(drop=True)


def _risk_badge(risk_str: str) -> str:
    r = str(risk_str).strip().lower()
    if r == "high risk":
        return "🔴 High"
    if r == "normal/low risk":
        return "🟢 Low"
    if r == "" or r == "nan":
        return ""
    return f"⚪ {risk_str}"


def _day_icons(rec: dict | None) -> str:
    if not rec:
        return ""
    icons = ["📝"]  # 記録がある＝📝
    try:
        if int(rec.get("did_exercise_day", 0)) == 1:
            icons.append("🏃")
    except Exception:
        pass
    return " ".join(icons)


def show_user_calendar(user_hist: pd.DataFrame, user_name: str):
    st.markdown("### 🗓️ 判定カレンダー / Result Calendar")

    if user_hist.empty:
        st.info("まだ記録がありません。記録するとカレンダーに判定が表示されます。")
        return

    cal_df = _normalize_user_hist_for_calendar(user_hist)
    if cal_df.empty:
        st.info("まだ記録がありません。記録するとカレンダーに判定が表示されます。")
        return

    key_sel = f"selected_calendar_date_{user_name}"
    if key_sel not in st.session_state:
        st.session_state[key_sel] = None

    latest_date = cal_df["date"].max()
    default_year = latest_date.year if isinstance(latest_date, datetime.date) else today_date().year
    default_month = latest_date.month if isinstance(latest_date, datetime.date) else today_date().month

    col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
    with col_m1:
        year = st.number_input("Year", min_value=2020, max_value=2100, value=int(default_year), step=1, key=f"cal_year_{user_name}")
    with col_m2:
        month = st.selectbox("Month", list(range(1, 13)), index=int(default_month) - 1, key=f"cal_month_{user_name}")
    with col_m3:
        st.caption("凡例：🔴 High / 🟢 Low / 📝 記録 / 🏃 運動　※日付をクリックすると詳細が見られます。")

    day_map = {d: r for d, r in zip(cal_df["date"].tolist(), cal_df.to_dict("records"))}

    cal = calendar.Calendar(firstweekday=0)  # Mon=0
    weeks = cal.monthdayscalendar(int(year), int(month))  # 0 means padding

    headers = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hcols = st.columns(7)
    for i, h in enumerate(headers):
        hcols[i].markdown(f"**{h}**")

    for w in weeks:
        cols = st.columns(7)
        for i, day in enumerate(w):
            if day == 0:
                cols[i].markdown("&nbsp;", unsafe_allow_html=True)
                continue

            d = datetime.date(int(year), int(month), int(day))
            rec = day_map.get(d)

            badge = _risk_badge(rec["risk"]) if rec else ""
            icons = _day_icons(rec) if rec else ""
            label_parts = [str(day)]
            if badge:
                label_parts.append(badge)
            if icons:
                label_parts.append(icons)
            label = "\n".join(label_parts)

            if cols[i].button(label, key=f"calbtn_{user_name}_{year}_{month}_{day}", use_container_width=True):
                st.session_state[key_sel] = d

    sel = st.session_state.get(key_sel)
    if sel is None:
        st.info("見たい日付をクリックしてください。 / Click a date to see details.")
        return

    st.markdown("---")
    st.markdown(f"#### 📌 {sel.strftime('%Y-%m-%d')} の詳細 / Details")
    rec = day_map.get(sel)
    if not rec:
        st.write("この日は記録がありません。")
        return

    risk_badge = _risk_badge(rec.get("risk", ""))
    bbt = rec.get("bbt", "")
    wt = rec.get("weight", "")
    ex = rec.get("exercise", "")
    meals = rec.get("meals", "")
    exd = rec.get("exercise_details", "")
    region = rec.get("region", "")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("判定 / Result", risk_badge if risk_badge else "—")
    c2.metric("体温 / BBT", bbt if str(bbt).strip().lower() not in ["", "nan"] else "—")
    c3.metric("体重 / Weight", wt if str(wt).strip().lower() not in ["", "nan"] else "—")
    c4.metric("運動 / Exercise", ex if str(ex).strip().lower() not in ["", "nan"] else "—")

    with st.expander("食事・運動メモ / Notes", expanded=True):
        if region and str(region).strip().lower() not in ["nan", ""]:
            st.write(f"**地域 / Region:** {region}")
        st.write(f"**食事 / Meals:** {meals if str(meals).strip().lower() not in ['nan',''] else '—'}")
        st.write(f"**運動内容 / Exercise Details:** {exd if str(exd).strip().lower() not in ['nan',''] else '—'}")


# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="Hormone Diary + Admin Dashboard", layout="wide")
st.title("🌿 女性ホルモン変動日記 / Hormone Fluctuation Diary")

# Init settings early
init_settings()

# Session init
if "admin_jwt" not in st.session_state:
    st.session_state["admin_jwt"] = None

if "save_requested" not in st.session_state:
    st.session_state["save_requested"] = False
if "save_poll_count" not in st.session_state:
    st.session_state["save_poll_count"] = 0
if "pending_save_payload" not in st.session_state:
    st.session_state["pending_save_payload"] = None


# =========================================================
# Sidebar Mode
# =========================================================
mode = st.sidebar.radio("モード / Mode", ["ユーザー画面 / User", "管理画面 / Admin"], index=0)
st.session_state["is_admin_page"] = (mode == "管理画面 / Admin")

# PV log (once per session)
if "pv_logged" not in st.session_state:
    log_event("page_view")
    st.session_state["pv_logged"] = True


# =========================================================
# ADMIN PAGE
# =========================================================
if mode == "管理画面 / Admin":
    st.sidebar.markdown("### 🔐 管理者ログイン / Admin Login")

    authed = require_admin_jwt()

    if not authed:
        admin_pw = st.sidebar.text_input("パスワード / Password", type="password")
        login_btn = st.sidebar.button("ログイン / Login")
        if login_btn:
            if admin_pw == ADMIN_PASSWORD:
                st.session_state["admin_jwt"] = jwt_sign({"role": "admin", "sub": "admin"}, JWT_SECRET, exp_seconds=3600)
                st.success("ログイン成功（JWT発行済み）")
                st.rerun()
            else:
                st.error("パスワードが違います。")

    if not require_admin_jwt():
        st.warning("管理画面に入るにはログインが必要です。")
        st.stop()

    if st.sidebar.button("ログアウト / Logout"):
        st.session_state["admin_jwt"] = None
        st.success("ログアウトしました。")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 運用設定 / Settings")

    geo_enabled = st.sidebar.toggle(
        "位置情報取得を有効化 / Enable geolocation",
        value=bool(st.session_state["settings"].get("geo_enabled", True))
    )
    st.session_state["settings"]["geo_enabled"] = bool(geo_enabled)
    st.sidebar.caption("OFFにすると、ユーザー画面で位置情報の許可を求めず、緯度経度は保存されません。")

    st.sidebar.markdown("---")
    colA, colB = st.sidebar.columns(2)
    with colA:
        refresh_sec = st.number_input("更新間隔(秒)", min_value=1, max_value=60, value=5)
    with colB:
        auto_refresh = st.checkbox("自動更新", value=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("JWT有効期限：1時間（再ログインで更新）")

    if auto_refresh and AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=int(refresh_sec * 1000), key="admin_refresh")

    st.subheader("🧭 管理画面 / Admin Console")
    st.caption("タブで整理：Analytics / Map / Logs / Export")

    adf = load_analytics()
    m = compute_metrics(adf)
    daily = daily_pv_uu(adf)
    hpv = hourly_pv(adf)
    wpv = weekday_pv(adf)

    df_records = safe_read_csv(DATA_FILE)
    if not df_records.empty:
        df_records = normalize_geo_columns(df_records)

    tab_analytics, tab_map, tab_logs, tab_export = st.tabs(["Analytics", "Map", "Logs", "Export"])

    # ---- Analytics ----
    with tab_analytics:
        st.markdown("### 📊 KPI")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("PV（累計）", m["pv_total"])
        c2.metric("UU（近似・累計）", m["uu_total"])
        c3.metric("管理画面PV", m["pv_admin"])
        c4.metric("ユーザー画面PV", m["pv_user"])
        c5.metric("アクティブ（10分以内）", m["active_sessions"])
        st.caption("※ UUは sid（StreamlitセッションID）ベースの近似値です（同一人物でも別端末/別ブラウザは別UU扱い）。")

        st.markdown("### 📈 日別PV/UU（棒＋折れ線）")
        if daily.empty:
            st.info("まだPVデータがありません。")
        else:
            max_days = max(3, min(180, len(daily)))
            view_days = st.slider("表示期間（日）", min_value=3, max_value=max_days, value=min(30, len(daily)))
            show_breakdown = st.checkbox("PVを（合計/管理/ユーザー）で内訳表示", value=False)
            chart_daily_pv_uu_combo(daily, days=view_days, show_admin_user=show_breakdown)

        st.markdown("### 🕒 時間帯別PV")
        if hpv.empty:
            st.info("まだPVデータがありません。")
        else:
            if ALTAIR_AVAILABLE:
                chart = alt.Chart(hpv).mark_bar().encode(
                    x=alt.X("hour:O", title="Hour (0-23)", sort=list(range(24))),
                    y=alt.Y("pv_total:Q", title="PV"),
                    tooltip=["hour", "pv_total", "pv_admin", "pv_user"]
                ).properties(height=260)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.bar_chart(hpv.set_index("hour")[["pv_total"]], use_container_width=True)

        st.markdown("### 📅 曜日別PV")
        if wpv.empty:
            st.info("まだPVデータがありません。")
        else:
            if ALTAIR_AVAILABLE:
                chart = alt.Chart(wpv).mark_bar().encode(
                    x=alt.X("weekday_label:O", title="Weekday", sort=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
                    y=alt.Y("pv_total:Q", title="PV"),
                    tooltip=["weekday_label", "pv_total", "pv_admin", "pv_user"]
                ).properties(height=260)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.bar_chart(wpv.set_index("weekday_label")[["pv_total"]], use_container_width=True)

    # ---- Map ----
    with tab_map:
        st.markdown("### 🗺️ 地域情報マップ / Geo Map")
        if df_records.empty:
            st.info("まだユーザー記録がありません。")
        else:
            latest = get_latest_by_user(df_records)

            user_list = sorted([u for u in latest["名前 / Name"].dropna().astype(str).unique().tolist() if u.strip() != ""])
            sel_mode = st.radio("地図に表示するユーザー", ["全員", "特定ユーザーを選ぶ"], horizontal=True)

            selected_user = None
            if sel_mode == "特定ユーザーを選ぶ":
                if len(user_list) == 0:
                    st.warning("ユーザーが見つかりません。")
                    st.stop()
                selected_user = st.selectbox("表示ユーザー / Selected User", user_list)

            df_base = df_records.copy()
            latest_base = latest.copy()

            if selected_user:
                df_base = df_base[df_base["名前 / Name"].astype(str) == str(selected_user)].copy()
                latest_base = latest_base[latest_base["名前 / Name"].astype(str) == str(selected_user)].copy()

            show_only_with_geo = st.checkbox("地域情報があるユーザーのみ表示", value=True)
            latest_view = latest_base.dropna(subset=["lat", "lon"]).copy() if show_only_with_geo else latest_base.copy()

            max_points_per_user = st.slider("各ユーザーの軌跡点数（最大）", 20, 500, 150, 10)
            paths_df = build_paths_df(df_base, max_points_per_user=max_points_per_user)

            if not PYDECK_AVAILABLE:
                st.warning("pydeckが無いので地図表示が簡易になります。")
                pts = latest_view.dropna(subset=["lat", "lon"])[["lat", "lon"]]
                if not pts.empty:
                    st.map(pts)
            else:
                latest_pts = latest_view.dropna(subset=["lat", "lon"]).copy()
                hist_pts = df_base.dropna(subset=["lat", "lon", "名前 / Name"]).copy()

                show_history_points = st.checkbox("履歴点（小さめの点）も表示", value=True)
                layers = []

                if not latest_pts.empty:
                    layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=latest_pts,
                            get_position="[lon, lat]",
                            get_fill_color=[255, 0, 0],
                            get_line_color=[255, 255, 255],
                            line_width_min_pixels=2,
                            get_radius=180,
                            pickable=True,
                            auto_highlight=True,
                        )
                    )

                if not paths_df.empty:
                    layers.append(
                        pdk.Layer(
                            "PathLayer",
                            data=paths_df,
                            get_path="path",
                            get_color="color",
                            width_scale=25,
                            width_min_pixels=4,
                            pickable=True,
                        )
                    )

                if show_history_points and not hist_pts.empty:
                    hist_pts = hist_pts.copy()
                    hist_pts["color"] = hist_pts["名前 / Name"].astype(str).apply(color_for_user)
                    layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=hist_pts,
                            get_position="[lon, lat]",
                            get_fill_color="color",
                            get_radius=40,
                            pickable=False,
                        )
                    )

                tooltip = {"text": "User: {名前 / Name}\nlat: {lat}\nlon: {lon}\ngeo_time: {地域情報取得時刻 / Geo Obtained At}"}

                map_col1, map_col2 = st.columns(2)
                with map_col1:
                    st.write("**🌍 世界地図**")
                    world_view = pdk.ViewState(latitude=20, longitude=0, zoom=1, pitch=0)
                    st.pydeck_chart(
                        pdk.Deck(map_style=None, initial_view_state=world_view, layers=layers, tooltip=tooltip),
                        use_container_width=True
                    )
                with map_col2:
                    st.write("**🗾 詳細地図**")
                    if not latest_pts.empty:
                        center_lat = float(latest_pts["lat"].mean())
                        center_lon = float(latest_pts["lon"].mean())
                    else:
                        center_lat, center_lon = 35.6812, 139.7671
                    detail_view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12, pitch=0)
                    st.pydeck_chart(
                        pdk.Deck(map_style=None, initial_view_state=detail_view, layers=layers, tooltip=tooltip),
                        use_container_width=True
                    )

            st.markdown("### 📋 最新状態（ユーザー別）")
            st.dataframe(latest_view.reset_index(drop=True), use_container_width=True)

    # ---- Logs ----
    with tab_logs:
        st.markdown("### 🧾 Logs")

        st.write("**アナリティクス（最新200件）**")
        if adf.empty:
            st.info("analytics_events.csv がまだありません。")
        else:
            show_ana = adf.sort_values("timestamp", ascending=False).head(200).copy()
            show_ana["timestamp"] = show_ana["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(show_ana.reset_index(drop=True), use_container_width=True)

        st.markdown("---")
        st.write("**ユーザー記録（最新200件）**")
        if df_records.empty:
            st.info("user_records.csv がまだありません。")
        else:
            df2 = df_records.copy()
            if "geo_time" in df2.columns and df2["geo_time"].notna().any():
                df2 = df2.sort_values("geo_time", ascending=False)
            else:
                df2 = df2.sort_values("record_date", ascending=False)
            st.dataframe(df2.head(200).reset_index(drop=True), use_container_width=True)

    # ---- Export ----
    with tab_export:
        st.markdown("### 📥 Export")

        if df_records.empty:
            st.info("ユーザー記録がないため、記録CSVのダウンロードはまだできません。")
        else:
            latest = get_latest_by_user(df_records)
            user_list = sorted([u for u in latest["名前 / Name"].dropna().astype(str).unique().tolist() if u.strip() != ""])
            exp_mode = st.radio("エクスポート対象", ["全員", "特定ユーザー"], horizontal=True, key="export_mode")
            exp_user = None
            exp_df = df_records.copy()

            if exp_mode == "特定ユーザー":
                if len(user_list) == 0:
                    st.warning("ユーザーが見つかりません。")
                else:
                    exp_user = st.selectbox("対象ユーザー", user_list, key="export_user")
                    exp_df = exp_df[exp_df["名前 / Name"].astype(str) == str(exp_user)].copy()

            csv_records = exp_df.to_csv(index=False).encode("utf-8")
            filename = "all_user_records.csv" if not exp_user else f"{exp_user}_records.csv"
            st.download_button("📥 記録CSVをダウンロード", data=csv_records, file_name=filename, mime="text/csv")

        csv_ana = adf.to_csv(index=False).encode("utf-8") if not adf.empty else "timestamp,event,sid,path\n".encode("utf-8")
        st.download_button("📥 アナリティクスCSVをダウンロード", data=csv_ana, file_name="analytics_events.csv", mime="text/csv")

        st.caption("運用メモ：JWT_SECRET はデフォルト CHANGE_ME_IN_PROD。Streamlit Cloud では st.secrets か環境変数で設定推奨。")

    st.stop()


# =========================================================
# USER PAGE (Gated)
# =========================================================
st.info(
    "👤 **ユーザー確認 / User Verification**\n\n"
    "あなたの記録を保存・表示するために、ユーザー名・地域・パスワードを入力してください。\n"
    "/ Enter User Name, Region, and Password to unlock the diary."
)

col_u1, col_u2, col_u3 = st.columns([1.2, 1.4, 1.0])
with col_u1:
    user_name = st.text_input("ユーザー名 / User Name", placeholder="例: Hanako")
with col_u2:
    region = st.selectbox(
        "お住まいの地域 / Region",
        ["選択してください / Select", "日本 / Japan", "北米 / North America", "ヨーロッパ / Europe", "アジア（日本以外） / Asia", "その他 / Others"],
    )
with col_u3:
    user_pw = st.text_input("ユーザーパスワード / User Password", type="password")

st.markdown("---")

# ===== NEW: Gate all fields before showing the rest =====
gate_ok = True

if not user_name or str(user_name).strip() == "":
    gate_ok = False

if region == "選択してください / Select":
    gate_ok = False

if user_pw != USER_PASSWORD:
    gate_ok = False

if not gate_ok:
    # Only show minimal hints; do NOT render the rest of the page
    if not user_name or str(user_name).strip() == "":
        st.warning("ユーザー名 / User Name を入力してください。")
    if region == "選択してください / Select":
        st.warning("お住まいの地域 / Region を選択してください。")
    if user_pw != USER_PASSWORD:
        st.warning("ユーザーパスワードが正しくありません。")
    st.info("上記3項目が正しく入力されると、記録フォームと過去の記録が表示されます。")
    st.stop()


# =========================================================
# USER FORM (Unlocked)
# =========================================================
def make_record_row(
    user_name, region, record_date, bbt, last_period, weight, meals, exercise_done, exercise_details, risk_result,
    lat, lon, acc, geo_time
):
    return {
        "日付 / Date": record_date,
        "名前 / Name": user_name,
        "地域 / Region": region,
        "基礎体温 / BBT": bbt,
        "前回生理日 / LMP": last_period,
        "体重 / Weight": weight,
        "食事 / Meals": meals,
        "運動 / Exercise": exercise_done,
        "運動内容 / Exercise Details": exercise_details,
        "AI判定 / AI Result": risk_result,
        "緯度 / Latitude": lat,
        "経度 / Longitude": lon,
        "精度(m) / Accuracy(m)": acc,
        "地域情報取得時刻 / Geo Obtained At": geo_time,
    }


st.subheader(f"📝 {user_name} さんの今日の記録 / Today's Record")

col_top1, col_top2 = st.columns(2)
with col_top1:
    record_date = st.date_input("日付 / Date", today_date())
    bbt = st.number_input("基礎体温 / Basal Body Temp (°C)", min_value=35.0, max_value=42.0, value=36.5, step=0.01)
with col_top2:
    last_period = st.date_input("前回の生理開始日 / Last Menstrual Period", today_date() - datetime.timedelta(days=14))

st.markdown("---")
st.subheader("🍽️ 食事と運動 / Diet & Exercise")
meals = st.text_area("今日食べたもの / Meals today", placeholder="例: 朝: トースト, 昼: サラダとパスタ, 夜: 焼き魚定食")
exercise_done = st.radio("運動しましたか？ / Did you exercise today?", options=["はい / Yes", "いいえ / No"], index=1, horizontal=True)

exercise_details = ""
if exercise_done == "はい / Yes":
    exercise_details = st.text_input("どんな運動をしましたか？ / Exercise Details", placeholder="例: ウォーキング30分")

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**基本情報 / Basic Info**")
    age = st.number_input("年齢 / Age", min_value=18, max_value=100, value=40)
    height = st.number_input("身長 / Height (cm)", min_value=100.0, max_value=250.0, value=160.0)
    weight_in = st.number_input("体重 / Weight (kg)", min_value=30.0, max_value=200.0, value=55.0)
    waist = st.number_input("ウエスト周囲径 / Waist Circumference (cm)", min_value=50.0, max_value=150.0, value=80.0)

with col2:
    st.markdown("**ライフスタイル / Lifestyle**")
    menstruation = st.selectbox("月経の有無 / Menstruation Status", options=[1.0, 2.0],
                               format_func=lambda x: "あり / Yes" if x == 1.0 else "なし / No")
    menopause_age = st.number_input("閉経年齢（該当しない場合は0） / Menopause Age (0 if N/A)", min_value=0, max_value=80, value=0)
    smoking = st.selectbox("喫煙状況 / Smoking Status", options=[1.0, 2.0, 3.0],
                          format_func=lambda x: "現在吸っている / Current smoker" if x == 1.0 else "過去に吸っていた / Past smoker" if x == 2.0 else "吸わない / Never smoked")
    alcohol = st.selectbox("飲酒状況（頻度） / Alcohol Consumption", options=[1.0, 2.0, 3.0, 0.0],
                          format_func=lambda x: "よく飲む / Often" if x == 1.0 else "たまに飲む / Sometimes" if x == 2.0 else "めったに飲まない / Rarely" if x == 3.0 else "飲まない / None")

st.markdown("---")

if st.button("記録して判定する / Record & Check Status"):
    bmi = weight_in / ((height / 100) ** 2)
    input_data = pd.DataFrame({
        "RIDAGEYR": [age],
        "BMXBMI": [bmi],
        "BMXWT": [weight_in],
        "BMXHT": [height],
        "BMXWAIST": [waist],
        "RHQ031": [menstruation],
        "RHQ060": [menopause_age if menopause_age > 0 else np.nan],
        "SMQ040": [smoking],
        "ALQ121": [alcohol],
    })

    try:
        prediction = model.predict(input_data)
        risk_result = "High Risk" if prediction[0] == 1 else "Normal/Low Risk"

        st.session_state["save_requested"] = True
        st.session_state["save_poll_count"] = 0
        st.session_state["pending_save_payload"] = {
            "risk_result": risk_result,
            "record_date": record_date,
            "user_name": user_name,
            "region": region,
            "bbt": bbt,
            "last_period": last_period,
            "weight": weight_in,
            "meals": meals,
            "exercise_done": exercise_done,
            "exercise_details": exercise_details,
        }
        st.rerun()

    except Exception as e:
        st.error(f"エラーが発生しました / Error: {e}")
        st.session_state["save_requested"] = False
        st.stop()

# =========================================================
# Save flow (geo ON/OFF)
# =========================================================
geo_enabled = bool(st.session_state.get("settings", {}).get("geo_enabled", True))

if st.session_state["save_requested"]:
    if not geo_enabled:
        p = st.session_state.get("pending_save_payload") or {}
        risk_result = p.get("risk_result", "")

        st.subheader("診断結果 / Result")
        if risk_result == "High Risk":
            st.error("⚠️ **AI判定: ハイリスクの可能性があります / High Risk Potential**")
        else:
            st.success("✅ **AI判定: リスクは低〜通常レベルです / Low to Normal Risk**")

        row = make_record_row(
            user_name=p.get("user_name"),
            region=p.get("region"),
            record_date=p.get("record_date"),
            bbt=p.get("bbt"),
            last_period=p.get("last_period"),
            weight=p.get("weight"),
            meals=p.get("meals"),
            exercise_done=p.get("exercise_done"),
            exercise_details=p.get("exercise_details"),
            risk_result=risk_result,
            lat=np.nan, lon=np.nan, acc=np.nan, geo_time=""
        )
        append_csv(DATA_FILE, pd.DataFrame([row]))

        st.session_state["save_requested"] = False
        st.session_state["pending_save_payload"] = None
        st.success("📝 データを記録しました！ / Data has been recorded!（位置情報OFF）")

    else:
        st.info("情報を計算中…（許可ダイアログが出たら許可してください）")
        loc = get_geolocation()

        if loc is None:
            st.session_state["save_poll_count"] += 1
            if st.session_state["save_poll_count"] <= 20 and AUTOREFRESH_AVAILABLE:
                st_autorefresh(interval=500, key="save_poll")
            elif st.session_state["save_poll_count"] > 20:
                st.session_state["save_requested"] = False
                st.error("地域情報が取得できませんでした。ブラウザの許可設定をご確認ください。")

        elif isinstance(loc, dict) and "error" in loc:
            st.session_state["save_requested"] = False
            err = loc.get("error", {})
            st.error(f"地域情報エラー: {err.get('message')} (code={err.get('code')})")

        else:
            lat, lon, acc = extract_geo(loc)
            if lat is None or lon is None:
                st.session_state["save_requested"] = False
                st.error("地域情報は取得できましたが、緯度経度の取り出しに失敗しました。")
            else:
                geo_time = now_iso()
                p = st.session_state.get("pending_save_payload") or {}
                risk_result = p.get("risk_result", "")

                st.subheader("診断結果 / Result")
                if risk_result == "High Risk":
                    st.error("⚠️ **AI判定: ハイリスクの可能性があります / High Risk Potential**")
                else:
                    st.success("✅ **AI判定: リスクは低〜通常レベルです / Low to Normal Risk**")

                row = make_record_row(
                    user_name=p.get("user_name"),
                    region=p.get("region"),
                    record_date=p.get("record_date"),
                    bbt=p.get("bbt"),
                    last_period=p.get("last_period"),
                    weight=p.get("weight"),
                    meals=p.get("meals"),
                    exercise_done=p.get("exercise_done"),
                    exercise_details=p.get("exercise_details"),
                    risk_result=risk_result,
                    lat=lat, lon=lon, acc=acc, geo_time=geo_time
                )

                append_csv(DATA_FILE, pd.DataFrame([row]))

                st.session_state["save_requested"] = False
                st.session_state["pending_save_payload"] = None
                st.success("📝 データを記録しました！ / Data has been recorded!")


# =========================================================
# Calendar + Past records
# =========================================================
hist = safe_read_csv(DATA_FILE)
if not hist.empty and "名前 / Name" in hist.columns:
    user_hist = hist[hist["名前 / Name"] == user_name].copy()
else:
    user_hist = pd.DataFrame()

st.markdown("---")
show_user_calendar(user_hist, user_name=user_name)

st.markdown("---")
st.subheader(f"📊 {user_name} さんの過去の記録 / Your Past Records")
if hist.empty:
    st.write("まだ記録がありません。 / No records yet.")
else:
    if user_hist.empty:
        st.write("まだあなたの記録はありません。 / No records found for you yet.")
    else:
        st.dataframe(user_history_view(user_hist).tail(20), use_container_width=True)
