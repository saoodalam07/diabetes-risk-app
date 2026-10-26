"""
streamlit_app.py  —  Diabetes Risk Predictor (UI)
-------------------------------------------------
A visual front-end that talks to the FastAPI /predict route over HTTP ONLY.
It never loads model.pkl. diabetes.csv is read just for context
(presets, population comparison, similar real patients).

Run (two terminals):
    uvicorn main:app --reload
    streamlit run streamlit_app.py

Deploying? Set an environment variable or Streamlit secret named API_URL
pointing at your hosted FastAPI, e.g. https://diabetes-api.onrender.com
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── 1. Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"
# Optional override for deployment (env var or .streamlit/secrets.toml)
try:
    API_URL = st.secrets.get("API_URL", os.getenv("API_URL", API_URL))
except Exception:
    API_URL = os.getenv("API_URL", API_URL)
API_URL = API_URL.rstrip("/")

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Blood pressure (mm Hg)",
    "SkinThickness": "Skin thickness (mm)",
    "Insulin": "Insulin (µU/ml)",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Family-history score",
    "Age": "Age (years)",
}
# In the Pima dataset a 0 in these columns means "not measured", not a real 0.
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
DEFAULTS = dict(Pregnancies=2, Glucose=120, BloodPressure=70, SkinThickness=20,
                Insulin=79, BMI=25.0, DiabetesPedigreeFunction=0.5, Age=30)

RED, GREEN, AMBER, INK_SOFT = "#E5484D", "#30A46C", "#F5A524", "#8B8D98"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1250px;}
      .hero-sub {color: #8B8D98; margin-top: -0.6rem; font-size: 1.02rem;}
      .chip {display:inline-block; padding: 4px 11px; margin: 3px 6px 3px 0;
             border-radius: 999px; font-size: 0.83rem; font-weight: 600;
             border: 1px solid rgba(128,128,128,.25);}
      .chip.ok   {background: rgba(48,164,108,.12); color:#30A46C;}
      .chip.warn {background: rgba(245,165,36,.14); color:#C98500;}
      .chip.bad  {background: rgba(229,72,77,.12); color:#E5484D;}
      .chip.na   {background: rgba(139,141,152,.12); color:#8B8D98;}
      div[data-testid="stMetricValue"] {font-size: 2rem;}
      .small-note {color:#8B8D98; font-size:0.82rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data (context only — NOT the model) ──────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


df = load_data()


# ── API helpers ──────────────────────────────────────────────────────────────
@st.cache_resource
def http():
    return requests.Session()


def api_health():
    try:
        r = http().get(f"{API_URL}/", timeout=3)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


def call_predict(payload):
    """Returns (result_dict, None) on success or (None, friendly_message)."""
    try:
        r = http().post(f"{API_URL}/predict", json=payload, timeout=8)
    except requests.exceptions.RequestException:
        return None, "conn"
    if r.status_code != 200:
        return None, f"status {r.status_code}"
    try:
        return r.json(), None
    except ValueError:
        return None, "bad json"


def diabetic_prob(result):
    """API returns confidence of the predicted class → convert to P(diabetic)."""
    c = float(result["confidence"])
    return c if result["prediction"] == 1 else 1 - c


def show_api_error(err):
    st.warning(
        "**Can't reach the prediction service right now.**  \n"
        "Please start the FastAPI server first, then press Predict again:\n\n"
        "```bash\nuvicorn main:app --reload\n```\n"
        f"Expected at `{API_URL}`."
        + ("" if err == "conn" else f"  \n<span class='small-note'>(server replied with {err})</span>"),
        icon="🔌",
    )


# ── Session state for inputs + history ───────────────────────────────────────
for f in FEATURES:
    st.session_state.setdefault(f"in_{f}", DEFAULTS[f])
st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)
st.session_state.setdefault("source", None)


def load_patient(values, source=None):
    for f in FEATURES:
        v = values[f]
        if f in ("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "Age"):
            v = int(v)
        else:
            v = float(v)
        st.session_state[f"in_{f}"] = v
    st.session_state["source"] = source
    st.session_state["last"] = None


def load_random_real():
    row = df.sample(1).iloc[0]
    row = row.copy()
    row["Age"] = int(np.clip(row["Age"], 18, 90))
    load_patient(row, source={"index": int(row.name), "outcome": int(row["Outcome"])})


PRESETS = {
    "🟢 Low-risk profile": dict(Pregnancies=1, Glucose=89, BloodPressure=66, SkinThickness=23,
                               Insulin=94, BMI=28.1, DiabetesPedigreeFunction=0.167, Age=21),
    "🟡 Borderline profile": dict(Pregnancies=4, Glucose=129, BloodPressure=76, SkinThickness=28,
                                 Insulin=122, BMI=33.0, DiabetesPedigreeFunction=0.45, Age=38),
    "🔴 High-risk profile": dict(Pregnancies=8, Glucose=183, BloodPressure=64, SkinThickness=33,
                                Insulin=200, BMI=36.5, DiabetesPedigreeFunction=0.672, Age=52),
}


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 Control panel")
    online = api_health()
    if online:
        st.success("API online · model loaded", icon="✅")
    else:
        st.error("API offline", icon="🔌")
        st.caption("Run `uvicorn main:app --reload` in another terminal.")
    st.caption(f"Endpoint: `{API_URL}/predict`")

    st.markdown("#### Quick-fill a patient")
    for name, vals in PRESETS.items():
        st.button(name, on_click=load_patient, args=(vals,), use_container_width=True)
    if df is not None:
        st.button("🎲 Random real patient from dataset", on_click=load_random_real,
                  use_container_width=True, type="primary")
    st.button("↺ Reset to defaults", on_click=load_patient, args=(DEFAULTS,),
              use_container_width=True)

    st.divider()
    st.markdown("#### About the model")
    st.caption(
        "K-Nearest Neighbours (k = 5) with standard scaling, trained on the Pima "
        "Indians Diabetes dataset (768 patients). Test accuracy ≈ 70%. "
        "Because k = 5, confidence moves in steps of 20% — it is literally how many "
        "of the 5 most similar training patients agreed."
    )
    st.caption("⚠️ Educational demo — not a medical diagnosis.")


# ── Header ───────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.markdown(
    "<p class='hero-sub'>Enter 8 clinical measurements → the FastAPI model returns a "
    "prediction → this page explains <b>why</b> and <b>what would change it</b>.</p>",
    unsafe_allow_html=True,
)

src = st.session_state.get("source")
if src:
    truth = "Diabetic" if src["outcome"] == 1 else "Not Diabetic"
    st.info(f"Loaded real patient **#{src['index']}** from diabetes.csv — "
            f"their recorded outcome is **{truth}**. Let's see if the model agrees.", icon="🎲")

# ── 2. Eight input widgets ───────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("##### Patient measurements")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.slider("Pregnancies", 0, 17, key="in_Pregnancies",
                  help="Number of times pregnant")
        st.slider("Glucose (mg/dL)", 0, 200, key="in_Glucose",
                  help="2-hour plasma glucose in an oral glucose tolerance test")
        st.slider("Blood pressure (mm Hg)", 0, 130, key="in_BloodPressure",
                  help="Diastolic blood pressure")
    with c2:
        st.slider("Skin thickness (mm)", 0, 100, key="in_SkinThickness",
                  help="Triceps skin-fold thickness")
        st.slider("Insulin (µU/ml)", 0, 850, key="in_Insulin",
                  help="2-hour serum insulin")
        st.slider("BMI", 0.0, 70.0, step=0.1, key="in_BMI",
                  help="Body-mass index (kg/m²)")
    with c3:
        st.number_input("Family-history score (Diabetes Pedigree Function)",
                        min_value=0.0, max_value=2.5, step=0.01, format="%.3f",
                        key="in_DiabetesPedigreeFunction",
                        help="Higher = more diabetes in the family tree (typical 0.1–1.0)")
        st.slider("Age (years)", 18, 90, key="in_Age")
        st.write("")
        predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)

payload = {
    "Pregnancies": int(st.session_state["in_Pregnancies"]),
    "Glucose": float(st.session_state["in_Glucose"]),
    "BloodPressure": float(st.session_state["in_BloodPressure"]),
    "SkinThickness": float(st.session_state["in_SkinThickness"]),
    "Insulin": float(st.session_state["in_Insulin"]),
    "BMI": float(st.session_state["in_BMI"]),
    "DiabetesPedigreeFunction": float(st.session_state["in_DiabetesPedigreeFunction"]),
    "Age": int(st.session_state["in_Age"]),
}

missing = [LABELS[f] for f in ZERO_MEANS_MISSING if payload[f] == 0]
if missing:
    st.caption("ℹ️ A value of 0 for " + ", ".join(missing) +
               " means *not measured* in this dataset — the model has seen such rows, "
               "but real values give a better prediction.")

# ── 3. Predict ───────────────────────────────────────────────────────────────
if predict_clicked:
    with st.spinner("Asking the model…"):
        result, err = call_predict(payload)
    if err:
        st.session_state["last"] = None
        show_api_error(err)
    else:
        st.session_state["last"] = {"payload": payload, "result": result}
        st.session_state["history"].insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            **payload,
            "result": result["result_label"],
            "confidence": result["confidence"],
            "P(diabetic)": round(diabetic_prob(result), 3),
        })

last = st.session_state.get("last")
if last and last["payload"] != payload:
    st.caption("✏️ Inputs changed since the last prediction — press **Predict** to refresh.")


# ── Visual helpers ───────────────────────────────────────────────────────────
def gauge(p):
    color = RED if p >= 0.5 else (AMBER if p >= 0.3 else GREEN)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(p * 100),
        number={"suffix": "%", "font": {"size": 44, "color": color}},
        title={"text": "Estimated diabetes risk", "font": {"size": 14, "color": INK_SOFT}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": INK_SOFT,
                     "tickvals": [0, 20, 40, 60, 80, 100]},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(48,164,108,.15)"},
                {"range": [30, 50], "color": "rgba(245,165,36,.15)"},
                {"range": [50, 100], "color": "rgba(229,72,77,.15)"},
            ],
            "threshold": {"line": {"color": INK_SOFT, "width": 2}, "value": 50},
        },
    ))
    fig.update_layout(height=250, margin=dict(l=25, r=25, t=50, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": INK_SOFT})
    return fig


def clinical_flags(p):
    chips = []
    g = p["Glucose"]
    if g == 0: chips.append(("na", "Glucose not measured"))
    elif g < 140: chips.append(("ok", f"Glucose {g:.0f} · normal (<140)"))
    elif g < 200: chips.append(("warn", f"Glucose {g:.0f} · impaired tolerance (140–199)"))
    else: chips.append(("bad", f"Glucose {g:.0f} · diabetic range (≥200)"))

    b = p["BMI"]
    if b == 0: chips.append(("na", "BMI not measured"))
    elif b < 18.5: chips.append(("warn", f"BMI {b:.1f} · underweight"))
    elif b < 25: chips.append(("ok", f"BMI {b:.1f} · healthy"))
    elif b < 30: chips.append(("warn", f"BMI {b:.1f} · overweight"))
    else: chips.append(("bad", f"BMI {b:.1f} · obese (≥30)"))

    bp = p["BloodPressure"]
    if bp == 0: chips.append(("na", "BP not measured"))
    elif bp < 80: chips.append(("ok", f"Diastolic BP {bp:.0f} · normal"))
    elif bp < 90: chips.append(("warn", f"Diastolic BP {bp:.0f} · stage 1 high"))
    else: chips.append(("bad", f"Diastolic BP {bp:.0f} · stage 2 high"))

    a = p["Age"]
    chips.append(("warn" if a >= 45 else "ok", f"Age {a} · {'higher' if a >= 45 else 'lower'}-risk age group"))

    d = p["DiabetesPedigreeFunction"]
    chips.append(("warn" if d >= 0.8 else "ok",
                  f"Family history {d:.2f} · {'strong' if d >= 0.8 else 'typical'}"))
    return "".join(f"<span class='chip {k}'>{t}</span>" for k, t in chips)


def similar_patients(p, k=5):
    """Find the k most similar REAL patients in diabetes.csv (same idea KNN uses).
    Pure numpy on the CSV — the trained model is never loaded here."""
    X = df[FEATURES].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    q = (np.array([p[f] for f in FEATURES], float) - mu) / sd
    d = np.sqrt((((X - mu) / sd - q) ** 2).sum(1))
    idx = np.argsort(d)[:k]
    out = df.iloc[idx].copy()
    out.insert(0, "Similarity", (1 / (1 + d[idx])).round(2))
    out["Outcome"] = out["Outcome"].map({1: "🔴 Diabetic", 0: "🟢 Not diabetic"})
    return out


# ── 4. Results ───────────────────────────────────────────────────────────────
if last:
    res, p = last["result"], last["payload"]
    prob = diabetic_prob(res)
    st.divider()

    left, mid, right = st.columns([1.15, 1, 1.25], gap="large")
    with left:
        st.markdown("#### Result")
        if res["result_label"] == "Diabetic":
            st.error(f"### {res['result_label']}", icon="⚠️")
        else:
            st.success(f"### {res['result_label']}", icon="✅")
        m1, m2 = st.columns(2)
        m1.metric("Model confidence", f"{res['confidence'] * 100:.0f}%")
        votes = round(prob * 5)
        m2.metric("Neighbour vote", f"{votes} / 5", help="How many of the 5 nearest "
                  "training patients were diabetic")
        if src and src.get("index") is not None:
            truth = src["outcome"]
            ok = truth == res["prediction"]
            st.caption(("✅ Matches" if ok else "❌ Differs from") +
                       " this patient's real recorded outcome.")
    with mid:
        st.plotly_chart(gauge(prob), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("#### Clinical context")
        st.markdown(clinical_flags(p), unsafe_allow_html=True)
        st.caption("Reference bands: 2-hour OGTT glucose (ADA), WHO BMI classes, "
                   "ACC/AHA diastolic BP. Shown for context — the model doesn't use them.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🧭 What would change it?", "📈 What-if explorer", "👥 Patients like you", "🕘 History"])

    # ── Counterfactual levers ────────────────────────────────────────────────
    with tab1:
        if df is None:
            st.info("Add diabetes.csv next to this file to enable this view.")
        else:
            healthy = df[df.Outcome == 0]
            levers = ["Glucose", "BMI", "BloodPressure", "Insulin", "SkinThickness"]
            rows = []
            with st.spinner("Testing scenarios against the API…"):
                for f in levers:
                    col = healthy[f]
                    target = float(col[col > 0].median()) if f in ZERO_MEANS_MISSING else float(col.median())
                    if f in ("Glucose", "BloodPressure", "SkinThickness", "Insulin"):
                        target = round(target)
                    if p[f] == target:
                        continue
                    trial = {**p, f: target}
                    r, e = call_predict(trial)
                    if e:
                        show_api_error(e); rows = None; break
                    rows.append((f, p[f], target, diabetic_prob(r)))
            if rows:
                st.markdown(
                    "Each bar answers: *if just this one value moved to the typical level of "
                    "non-diabetic patients in the dataset, what would the model say?* "
                    "(Things you can't change — age, pregnancies, family history — are left out.)")
                rows.sort(key=lambda r: r[3] - prob)
                names = [f"{LABELS[f]}<br><sub>{cur:g} → {tgt:g}</sub>" for f, cur, tgt, _ in rows]
                deltas = [(new - prob) * 100 for *_, new in rows]
                fig = go.Figure(go.Bar(
                    x=deltas, y=names, orientation="h",
                    marker_color=[GREEN if d < 0 else (RED if d > 0 else INK_SOFT) for d in deltas],
                    text=[f"{d:+.0f} pts" for d in deltas], textposition="outside",
                    hovertemplate="%{y}<br>Risk change: %{x:+.0f} points<extra></extra>",
                ))
                fig.update_layout(height=80 + 55 * len(rows), margin=dict(l=10, r=40, t=10, b=30),
                                  xaxis_title="Change in risk (percentage points)",
                                  yaxis=dict(autorange="reversed"),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                fig.add_vline(x=0, line_color=INK_SOFT, line_width=1)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                best = min(rows, key=lambda r: r[3])
                if best[3] < prob:
                    st.success(f"**Biggest lever:** bringing {LABELS[best[0]]} from "
                               f"{best[1]:g} to {best[2]:g} would move estimated risk from "
                               f"**{prob:.0%} → {best[3]:.0%}**.", icon="🧭")
                else:
                    st.info("No single change lowers the estimate — this profile already "
                            "sits among low-risk patients.", icon="👍")

    # ── What-if sweep ────────────────────────────────────────────────────────
    with tab2:
        ranges = {"Glucose": (40, 200, 8), "BMI": (16, 60, 2), "Age": (18, 85, 3),
                  "BloodPressure": (40, 120, 5), "Insulin": (0, 600, 30),
                  "DiabetesPedigreeFunction": (0.05, 2.4, 0.1)}
        fsel = st.selectbox("Sweep one measurement, keep the rest fixed",
                            list(ranges), format_func=lambda f: LABELS[f])
        lo, hi, step = ranges[fsel]
        grid = np.round(np.arange(lo, hi + 1e-9, step), 3)
        xs, ys = [], []
        with st.spinner(f"Calling the API {len(grid)} times…"):
            for v in grid:
                r, e = call_predict({**p, fsel: int(v) if fsel in ("Age",) else float(v)})
                if e:
                    show_api_error(e); xs = None; break
                xs.append(v); ys.append(diabetic_prob(r) * 100)
        if xs:
            fig = go.Figure()
            fig.add_hrect(y0=50, y1=100, fillcolor=RED, opacity=0.06, line_width=0)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line_shape="hv",
                                     line=dict(color="#3E63DD", width=3),
                                     hovertemplate=f"{LABELS[fsel]}: %{{x}}<br>Risk: %{{y:.0f}}%<extra></extra>"))
            fig.add_trace(go.Scatter(x=[p[fsel]], y=[prob * 100], mode="markers",
                                     marker=dict(size=16, color=RED if prob >= .5 else GREEN,
                                                 line=dict(width=3, color="white")),
                                     name="You", hovertemplate="Current: %{x}<extra></extra>"))
            fig.add_hline(y=50, line_dash="dot", line_color=INK_SOFT,
                          annotation_text="decision line", annotation_position="top left")
            fig.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=20, b=40),
                              xaxis_title=LABELS[fsel], yaxis_title="Estimated risk (%)",
                              yaxis_range=[-5, 105],
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            flips = [xs[i] for i in range(1, len(ys)) if (ys[i] >= 50) != (ys[i - 1] >= 50)]
            if flips:
                st.caption(f"The prediction flips around **{LABELS[fsel]} ≈ "
                           f"{', '.join(f'{v:g}' for v in flips[:4])}** for this patient. "
                           "Steps are expected: KNN risk can only be 0, 20, 40, 60, 80 or 100%.")
            else:
                st.caption("The prediction never flips across this range for this patient.")

    # ── Similar real patients ────────────────────────────────────────────────
    with tab3:
        if df is None:
            st.info("Add diabetes.csv next to this file to enable this view.")
        else:
            sim = similar_patients(p)
            n_d = (sim["Outcome"] == "🔴 Diabetic").sum()
            st.markdown(
                f"KNN decides by looking at the most similar patients it has seen. "
                f"Here are the **5 closest real patients** in the dataset — "
                f"**{n_d} of 5** were diabetic.")
            st.dataframe(sim, hide_index=True, use_container_width=True,
                         column_config={"Similarity": st.column_config.ProgressColumn(
                             "Similarity", min_value=0, max_value=1, format="%.2f")})
            st.caption("Searched across all 768 rows, so this can differ slightly from the "
                       "model's own vote (which only saw the 80% training split).")

            st.markdown("##### Where this patient sits vs. the population")
            feats = ["Glucose", "BMI", "Age", "BloodPressure", "Insulin", "DiabetesPedigreeFunction"]
            pct, med0, med1 = [], [], []
            for f in feats:
                col = df[f][df[f] > 0] if f in ZERO_MEANS_MISSING else df[f]
                pct.append((col < p[f]).mean() * 100 if p[f] > 0 or f not in ZERO_MEANS_MISSING else None)
                c0 = df[df.Outcome == 0][f]; c1 = df[df.Outcome == 1][f]
                if f in ZERO_MEANS_MISSING:
                    c0, c1 = c0[c0 > 0], c1[c1 > 0]
                med0.append((col < c0.median()).mean() * 100)
                med1.append((col < c1.median()).mean() * 100)
            names = [LABELS[f] for f in feats]
            fig = go.Figure()
            for i, n in enumerate(names):
                fig.add_shape(type="line", x0=0, x1=100, y0=i, y1=i,
                              line=dict(color="rgba(139,141,152,.25)", width=6))
            fig.add_trace(go.Scatter(x=med0, y=names, mode="markers", name="Typical non-diabetic",
                                     marker=dict(size=13, color=GREEN, symbol="line-ns-open",
                                                 line=dict(width=3, color=GREEN))))
            fig.add_trace(go.Scatter(x=med1, y=names, mode="markers", name="Typical diabetic",
                                     marker=dict(size=13, color=RED, symbol="line-ns-open",
                                                 line=dict(width=3, color=RED))))
            fig.add_trace(go.Scatter(x=pct, y=names, mode="markers", name="This patient",
                                     marker=dict(size=15, color="#3E63DD",
                                                 line=dict(width=2, color="white")),
                                     hovertemplate="%{y}: higher than %{x:.0f}% of patients<extra></extra>"))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=40),
                              xaxis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100], title="Percentile in dataset", ticksuffix="%"),
                              yaxis=dict(autorange="reversed"),
                              legend=dict(orientation="h", y=-0.22),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── History ──────────────────────────────────────────────────────────────
    with tab4:
        hist = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist, hide_index=True, use_container_width=True)
        c_a, c_b = st.columns([1, 4])
        c_a.download_button("⬇️ Download CSV", hist.to_csv(index=False),
                            "predictions.csv", "text/csv", use_container_width=True)
        if c_b.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()

elif not predict_clicked:
    st.markdown(
        "<p class='small-note' style='text-align:center;margin-top:2rem'>"
        "Adjust the measurements or pick a profile in the sidebar, then press <b>Predict</b>.</p>",
        unsafe_allow_html=True)

st.markdown(
    "<p class='small-note' style='text-align:center;margin-top:3rem'>"
    "Educational project · KNN on the Pima Indians Diabetes dataset · "
    "Not a substitute for medical advice.</p>", unsafe_allow_html=True)
