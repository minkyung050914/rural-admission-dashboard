"""
농어촌 전형 실효성 진단 대시보드
실행: streamlit run dashboard.py
필요 패키지: pip install streamlit pandas numpy plotly scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

# ─────────────────────────────────────────────────────────────
# 0. 페이지 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="농어촌 전형 실효성 진단",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
.tag-rural   { background:#fee2e2; color:#b91c1c; padding:2px 9px;
               border-radius:4px; font-size:12px; font-weight:600; }
.tag-urban   { background:#dbeafe; color:#1d4ed8; padding:2px 9px;
               border-radius:4px; font-size:12px; font-weight:600; }
.tag-false   { background:#fef3c7; color:#b45309; padding:2px 9px;
               border-radius:4px; font-size:12px; font-weight:600; }
.tag-blind   { background:#fde8d0; color:#c2410c; padding:2px 9px;
               border-radius:4px; font-size:12px; font-weight:600; }
h1 { font-size:1.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 1. 데이터 & 상수
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("final_master.csv", encoding="utf-8-sig")

df_raw = load_data()

NORM_COLS = [
    "α1_학원수_norm",
    "β1_최단광역접근시간_norm",
    "β2_광역접근가능시설수_norm",
    "γ1_학령인구10대_norm",
    "δ1_사교육비_norm",
    "δ3_참여율_norm",
]
RAW_COLS = [
    "α1_학원수",
    "β1_최단광역접근시간",
    "β2_광역접근가능시설수",
    "γ1_학령인구10대",
    "δ1_사교육비",
    "δ3_참여율",
]
VAR_LABELS = ["α₁ 학원수", "β₁ 교통접근성", "β₂ 광역교통시설",
              "γ₁ 학령인구", "δ₁ 사교육비", "δ₃ 참여율"]
VAR_COLORS = ["#378ADD", "#1D9E75", "#5DCAA5", "#7F77DD", "#EF9F27", "#D85A30"]

# ML 역산 가중치 (Step 7 결과)
ML_W  = [0.097, 0.421, 0.072, 0.047, 0.354, 0.008]
# 균등 가중치
EQ_W  = [1/6] * 6
# ML RF 중요도
RF_W  = [0.197, 0.273, 0.120, 0.204, 0.165, 0.041]


# ─────────────────────────────────────────────────────────────
# 2. 유틸 함수
# ─────────────────────────────────────────────────────────────
def calc_es(df, weights):
    """교육소외지수 E(s) 계산. weights: 6개 변수 가중치 리스트"""
    w = np.array(weights)
    w = w / w.sum() if w.sum() > 0 else w
    X = df[NORM_COLS].values
    return (X * w).sum(axis=1)


def classify(es_vals, threshold):
    """E(s) < threshold → 농어촌(1), 이상 → 비농어촌(0)"""
    return (es_vals < threshold).astype(int)


def label_type(current, proposed):
    """4가지 케이스 분류"""
    if current == 1 and proposed == 1:
        return "정합 농어촌"
    elif current == 1 and proposed == 0:
        return "허위 농어촌"
    elif current == 0 and proposed == 1:
        return "사각지대"
    else:
        return "정합 비농어촌"


TYPE_COLOR = {
    "정합 농어촌":  "#16a34a",
    "허위 농어촌":  "#dc2626",
    "사각지대":     "#d97706",
    "정합 비농어촌": "#2563eb",
}
TYPE_ORDER = ["정합 농어촌", "허위 농어촌", "사각지대", "정합 비농어촌"]


# ─────────────────────────────────────────────────────────────
# 3. 사이드바 — 가중치 & 필터
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 가중치 설정")

    col_a, col_b = st.columns(2)
    if col_a.button("ML 최적값", use_container_width=True):
        st.session_state["preset"] = "ml"
    if col_b.button("균등 가중치", use_container_width=True):
        st.session_state["preset"] = "eq"

    preset = st.session_state.get("preset", "eq")
    defaults = ML_W if preset == "ml" else EQ_W

    weights = []
    for i, (label, default, color) in enumerate(zip(VAR_LABELS, defaults, VAR_COLORS)):
        w = st.slider(
            label,
            min_value=0.0, max_value=1.0,
            value=float(round(default, 3)),
            step=0.01,
            key=f"w_{i}",
        )
        weights.append(w)

    st.divider()
    st.markdown("## 🎯 농어촌 판정 임계값")
    threshold = st.slider(
        "E(s) < 임계값 → 농어촌",
        min_value=0.10, max_value=0.70,
        value=0.35, step=0.01,
    )
    st.caption("낮출수록 농어촌 지정 범위 축소 / 높일수록 확대")

    st.divider()
    st.markdown("## 🔍 필터")
    all_sido = ["전체"] + sorted(df_raw["시도명"].unique().tolist())
    sido_sel = st.selectbox("시도", all_sido)
    level_sel = st.multiselect(
        "학교급",
        ["중학교", "고등학교"],
        default=["중학교", "고등학교"],
    )


# ─────────────────────────────────────────────────────────────
# 4. 데이터 가공 (가중치 적용)
# ─────────────────────────────────────────────────────────────
df = df_raw.copy()
df["E_s_new"]  = calc_es(df, weights)
df["제안_농어촌"] = classify(df["E_s_new"], threshold)
df["분류유형"]   = df.apply(
    lambda r: label_type(int(r["현행_농어촌"]), int(r["제안_농어촌"])), axis=1
)

# 필터 적용
if sido_sel != "전체":
    df = df[df["시도명"] == sido_sel]
if level_sel:
    df = df[df["학교급구분"].isin(level_sel)]


# ─────────────────────────────────────────────────────────────
# 5. 헤더
# ─────────────────────────────────────────────────────────────
st.title("🏫 농어촌 대입 전형 실효성 진단 대시보드")
st.caption(
    "읍·면 행정구역 이분법 vs 실질 교육소외지수(E_s) 기반 분류 비교 · "
    f"분석 대상 {len(df):,}개교"
)

# 핵심 지표 카드
c1, c2, c3, c4, c5 = st.columns(5)
n_total       = len(df)
n_cur_rural   = df["현행_농어촌"].sum()
n_prop_rural  = df["제안_농어촌"].sum()
n_false       = (df["분류유형"] == "허위 농어촌").sum()
n_blind       = (df["분류유형"] == "사각지대").sum()

c1.metric("전체 학교",    f"{n_total:,}개교")
c2.metric("현행 농어촌",  f"{n_cur_rural:,}개교",
          delta=f"{n_cur_rural/n_total*100:.1f}%")
c3.metric("제안 농어촌",  f"{n_prop_rural:,}개교",
          delta=f"{int(n_prop_rural)-int(n_cur_rural):+,}",
          delta_color="normal")
c4.metric("허위 농어촌",  f"{n_false:,}개교",
          delta="읍·면이지만 실질 도시형",
          delta_color="inverse")
c5.metric("사각지대",     f"{n_blind:,}개교",
          delta="동이지만 실질 교육소외",
          delta_color="inverse")

st.divider()

# ─────────────────────────────────────────────────────────────
# 6. 탭
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 개요·ML 분석", "🗺️ 지도", "📋 학교 목록", "🔬 학교 상세 진단", "📖 데이터 정보"]
)


# ═══════════════════════════════════════════════════════
# TAB 1 — 개요 & ML 역산 분석
# ═══════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([1, 1])

    # ── 현행 vs 제안 분류 도넛 차트 ──────────────────────
    with col_l:
        st.markdown("#### 현행 vs 제안 기준 분류 현황")
        type_counts = df["분류유형"].value_counts().reindex(TYPE_ORDER, fill_value=0)
        fig_pie = go.Figure(go.Pie(
            labels=type_counts.index.tolist(),
            values=type_counts.values.tolist(),
            hole=0.55,
            marker_colors=[TYPE_COLOR[t] for t in type_counts.index],
            textinfo="label+value",
            textfont_size=12,
        ))
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── 변수별 ML 가중치 비교 바 차트 ───────────────────
    with col_r:
        st.markdown("#### ML 역산 — 현행 기준의 암묵적 가중치")
        ml_df = pd.DataFrame({
            "변수":       VAR_LABELS,
            "로지스틱":   ML_W,
            "랜덤포레스트": RF_W,
        })
        fig_ml = go.Figure()
        fig_ml.add_trace(go.Bar(
            name="로지스틱 회귀",
            x=ml_df["변수"], y=ml_df["로지스틱"],
            marker_color="#378ADD", opacity=0.85,
        ))
        fig_ml.add_trace(go.Bar(
            name="랜덤 포레스트",
            x=ml_df["변수"], y=ml_df["랜덤포레스트"],
            marker_color="#D85A30", opacity=0.85,
        ))
        fig_ml.update_layout(
            barmode="group",
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_title="정규화 가중치",
            xaxis_tickangle=-15,
        )
        st.plotly_chart(fig_ml, use_container_width=True)
        st.caption(
            "δ₃ 참여율(0.008)과 γ₁ 학령인구(0.047)는 ML 역산 결과 "
            "현행 기준에서 사실상 무시되고 있음 → 읍·면 행정구역이 이 변수들을 반영하지 못하는 구조적 한계"
        )

    st.divider()

    # ── E_s 분포 히스토그램 ──────────────────────────────
    st.markdown("#### 교육소외지수 E(s) 분포 (현 가중치 기준)")
    fig_hist = go.Figure()
    for label, grp in df.groupby("현행_농어촌"):
        name = "현행 농어촌" if label == 1 else "현행 비농어촌"
        color = "#ef4444" if label == 1 else "#3b82f6"
        fig_hist.add_trace(go.Histogram(
            x=grp["E_s_new"], name=name,
            marker_color=color, opacity=0.6,
            nbinsx=50,
        ))
    fig_hist.add_vline(
        x=threshold, line_dash="dash", line_color="#1f2937",
        annotation_text=f"임계값 {threshold:.2f}",
        annotation_position="top right",
    )
    fig_hist.update_layout(
        barmode="overlay",
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="E(s) — 0:농촌형, 1:도시형",
        yaxis_title="학교 수",
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── 시도별 괴리도 박스플롯 ────────────────────────────
    st.markdown("#### 시도별 괴리도 분포 (높을수록 현행 기준과 실제 환경 차이 큼)")
    sido_order = (
        df.groupby("시도명")["G_괴리도"].median()
        .sort_values(ascending=False).index.tolist()
    )
    fig_box = px.box(
        df, x="시도명", y="G_괴리도",
        color="시도명",
        category_orders={"시도명": sido_order},
        labels={"G_괴리도": "괴리도 G"},
        height=350,
    )
    fig_box.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=60),
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ═══════════════════════════════════════════════════════
# TAB 2 — 지도
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 전국 학교 분류 지도")
    st.caption(
        "🟢 정합 농어촌 &nbsp; 🔴 허위 농어촌 &nbsp; 🟠 사각지대 &nbsp; 🔵 정합 비농어촌 &nbsp;"
        "— 마우스를 올려 학교 정보 확인"
    )

    map_df = df.dropna(subset=["위도", "경도"]).copy()
    map_df["color"] = map_df["분류유형"].map(TYPE_COLOR)
    map_df["hover"] = (
        map_df["학교명"] + "<br>" +
        map_df["시도명"] + " " + map_df["시군구명"] + " " + map_df["읍면동명"] +
        "<br>E_s: " + map_df["E_s_new"].round(3).astype(str) +
        " | 분류: " + map_df["분류유형"]
    )

    fig_map = px.scatter_mapbox(
        map_df,
        lat="위도", lon="경도",
        color="분류유형",
        color_discrete_map=TYPE_COLOR,
        category_orders={"분류유형": TYPE_ORDER},
        hover_name="학교명",
        hover_data={
            "위도": False, "경도": False,
            "시도명": True, "시군구명": True,
            "학교급구분": True,
            "E_s_new": ":.3f",
            "현행_농어촌": True,
            "제안_농어촌": True,
        },
        zoom=6,
        center={"lat": 36.5, "lon": 127.5},
        opacity=0.75,
        size_max=8,
        height=600,
    )
    fig_map.update_traces(marker=dict(size=6))
    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="v", yanchor="top", y=0.98,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ═══════════════════════════════════════════════════════
# TAB 3 — 학교 목록
# ═══════════════════════════════════════════════════════
with tab3:
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    type_filter = col_f1.multiselect(
        "분류유형 필터",
        TYPE_ORDER,
        default=["허위 농어촌", "사각지대"],
    )
    sort_col = col_f2.selectbox(
        "정렬 기준",
        ["G_괴리도", "E_s_new", "α1_학원수", "β1_최단광역접근시간"],
        index=0,
    )
    sort_asc = col_f3.radio("정렬 방향", ["내림차순", "오름차순"], horizontal=True) == "오름차순"

    show_df = df.copy()
    if type_filter:
        show_df = show_df[show_df["분류유형"].isin(type_filter)]

    show_df = show_df.sort_values(sort_col, ascending=sort_asc)

    display_cols = {
        "학교명": "학교명",
        "학교급구분": "학교급",
        "시도명": "시도",
        "시군구명": "시군구",
        "읍면동명": "읍면동",
        "분류유형": "분류유형",
        "E_s_new": "E(s)",
        "G_괴리도": "괴리도",
        "α1_학원수": "학원수",
        "β1_최단광역접근시간": "교통시간(분)",
        "γ1_학령인구10대": "학령인구",
        "δ1_사교육비": "사교육비(만원)",
    }
    tbl = show_df[list(display_cols.keys())].rename(columns=display_cols)
    tbl["E(s)"] = tbl["E(s)"].round(3)
    tbl["괴리도"] = tbl["괴리도"].round(3)

    st.markdown(f"**{len(tbl):,}개교** 표시 중")

    def highlight_type(row):
        color_map = {
            "허위 농어촌":  "background-color:#fee2e2",
            "사각지대":     "background-color:#fff7ed",
            "정합 농어촌":  "background-color:#f0fdf4",
            "정합 비농어촌": "",
        }
        c = color_map.get(row["분류유형"], "")
        return [c] * len(row)

    styled = tbl.style.apply(highlight_type, axis=1).format({
        "E(s)": "{:.3f}", "괴리도": "{:.3f}",
        "학원수": "{:,.0f}", "학령인구": "{:,.0f}",
        "사교육비(만원)": "{:.1f}", "교통시간(분)": "{:.1f}",
    })
    st.dataframe(styled, use_container_width=True, height=450)

    csv = tbl.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "📥 CSV 다운로드",
        data=csv.encode("utf-8-sig"),
        file_name="농어촌전형_진단결과.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════
# TAB 4 — 학교 상세 진단
# ═══════════════════════════════════════════════════════
with tab4:
    col_search, col_pick = st.columns([2, 3])

    search_q = col_search.text_input("학교명 검색", placeholder="예: 봉담중학교")
    if search_q:
        candidates = df[df["학교명"].str.contains(search_q, na=False)]
    else:
        candidates = df

    if len(candidates) == 0:
        st.warning("검색 결과가 없습니다.")
    else:
        sel_name = col_pick.selectbox(
            "학교 선택",
            candidates["학교명"].tolist(),
        )
        row = candidates[candidates["학교명"] == sel_name].iloc[0]

        st.divider()
        h1, h2, h3 = st.columns([2, 1, 1])
        h1.markdown(f"### {row['학교명']}")
        h2.markdown(
            f"<span class='tag-rural'>현행 {'농어촌' if row['현행_농어촌']==1 else '비농어촌'}</span>",
            unsafe_allow_html=True,
        )
        ltype = row["분류유형"]
        tag_class = {
            "허위 농어촌": "tag-false",
            "사각지대":    "tag-blind",
            "정합 농어촌": "tag-rural",
            "정합 비농어촌": "tag-urban",
        }.get(ltype, "tag-urban")
        h3.markdown(f"<span class='{tag_class}'>제안: {ltype}</span>", unsafe_allow_html=True)

        st.caption(f"{row['시도명']} {row['시군구명']} {row['읍면동명']} · {row['학교급구분']}")

        m1, m2, m3 = st.columns(3)
        m1.metric("교육소외지수 E(s)", f"{row['E_s_new']:.3f}",
                  delta=f"임계값 {threshold:.2f} {'미만 → 농어촌' if row['E_s_new']<threshold else '이상 → 도시형'}")
        m2.metric("괴리도 G",        f"{row['G_괴리도']:.3f}")
        m3.metric("학원수 (시군구)",  f"{int(row['α1_학원수']):,}개")

        st.divider()

        col_bar, col_radar = st.columns(2)

        # 변수별 막대 (학교 vs 전국 평균)
        with col_bar:
            st.markdown("##### 변수별 정규화 값 (전국 평균 비교)")
            school_vals = [row[c] for c in NORM_COLS]
            nat_avg     = [df_raw[c].mean() for c in NORM_COLS]
            rural_avg   = [df_raw[df_raw["현행_농어촌"]==1][c].mean() for c in NORM_COLS]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="이 학교",
                x=VAR_LABELS, y=[round(v, 3) for v in school_vals],
                marker_color=VAR_COLORS, opacity=0.85,
            ))
            fig_bar.add_trace(go.Scatter(
                name="전국 평균",
                x=VAR_LABELS, y=[round(v, 3) for v in nat_avg],
                mode="markers", marker=dict(size=10, color="#1f2937", symbol="diamond"),
            ))
            fig_bar.add_trace(go.Scatter(
                name="농어촌 평균",
                x=VAR_LABELS, y=[round(v, 3) for v in rural_avg],
                mode="markers", marker=dict(size=10, color="#dc2626", symbol="x"),
            ))
            fig_bar.update_layout(
                height=320,
                margin=dict(l=5, r=5, t=10, b=5),
                yaxis=dict(range=[0, 1], title="정규화값 (0~1)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.0),
                xaxis_tickangle=-15,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # 레이더 차트
        with col_radar:
            st.markdown("##### 레이더 차트")
            categories = VAR_LABELS + [VAR_LABELS[0]]
            school_r   = school_vals + [school_vals[0]]
            nat_r      = nat_avg + [nat_avg[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[round(v, 3) for v in school_r],
                theta=categories,
                fill="toself",
                name="이 학교",
                line_color="#378ADD",
                fillcolor="rgba(55,138,221,0.15)",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[round(v, 3) for v in nat_r],
                theta=categories,
                fill="toself",
                name="전국 평균",
                line_color="#9ca3af",
                fillcolor="rgba(156,163,175,0.1)",
                line_dash="dot",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=320,
                margin=dict(l=30, r=30, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.05),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # 원본 수치
        with st.expander("원본 수치 보기"):
            detail_data = {
                "변수":   VAR_LABELS,
                "원본값": [
                    f"{int(row['α1_학원수']):,}개",
                    f"{row['β1_최단광역접근시간']:.1f}분",
                    f"{row['β2_광역접근가능시설수']:.2f}개",
                    f"{int(row['γ1_학령인구10대']):,}명",
                    f"{row['δ1_사교육비']:.1f}만원",
                    f"{row['δ3_참여율']:.1f}%",
                ],
                "정규화값": [round(row[c], 3) for c in NORM_COLS],
                "가중치":   [round(w, 3) for w in
                            (np.array(weights) / (sum(weights) or 1))],
            }
            st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# TAB 5 — 데이터 정보
# ═══════════════════════════════════════════════════════
with tab5:
    st.markdown("""
### 데이터 출처 및 변수 설명

| 변수 | 설명 | 출처 | 집계 단위 |
|------|------|------|----------|
| α₁ 학원수 | 시군구 내 개원 학원·교습소 수 | 교육부 학원교습소 현황 (공공데이터포털) | 시군구 |
| β₁ 교통접근성 | 대중교통 기준 광역교통시설(버스터미널·철도역) 최단 접근시간 | 한국교통연구원 교통접근성지표 (KOTI, 승인번호 444001) | 읍면동 |
| β₂ 광역교통시설 | 30분 이내 도달 가능한 광역교통시설 수 | 동일 | 읍면동 |
| γ₁ 학령인구 | 행정동 단위 10~19세 주민등록인구 | 행정안전부 주민등록인구통계 | 행정동 |
| δ₁ 사교육비 | 시도·학교급별 학생 1인당 월평균 사교육비 | 교육부·통계청 사교육비조사 | 시도 |
| δ₃ 참여율 | 시도·학교급별 사교육 참여율 | 동일 | 시도 |

### 현행 기준 (농어촌 전형 라벨)
읍·면 소재 중고등학교 = 농어촌 전형 대상 (현행 기준)
도로명주소 파싱 후 읍·면 여부로 자동 분류

### 교육소외지수 E(s) 산출 공식

```
E(s) = Σ (wᵢ × normᵢ) / Σwᵢ

normᵢ: Min-Max 정규화 (0~1), 모든 변수가 높을수록 도시형
β₁ 교통접근성: 접근시간이 짧을수록 도시형이므로 1-정규화 적용
```

### ML 역산 분석 결과 (AUC 0.894)

| 변수 | 로지스틱 회귀 | 랜덤 포레스트 | 평균 |
|------|------------|------------|------|
| β₁ 교통시간 | 0.421 | 0.273 | **0.347** |
| δ₁ 사교육비 | 0.354 | 0.165 | **0.259** |
| α₁ 학원수 | 0.097 | 0.197 | 0.147 |
| γ₁ 학령인구 | 0.047 | 0.204 | 0.125 |
| β₂ 교통시설 | 0.072 | 0.120 | 0.096 |
| **δ₃ 참여율** | **0.008** | **0.041** | **0.025** |

→ 현행 읍·면 기준은 사교육 참여율(δ₃)과 학령인구(γ₁)를 사실상 반영하지 않음

### 라이선스
공공누리 1유형 데이터 활용 · 교육 공공데이터 AI 활용대회 제출작
""")

    with st.expander("전체 데이터 미리보기 (상위 100행)"):
        st.dataframe(df_raw.head(100), use_container_width=True)
