import os, json, joblib, warnings, base64
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="WC 2026 | AI Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Flag emoji (100% reliable, no CDN) ───────────────────────────────────────
FLAG_EMOJI = {
    'Brazil':'🇧🇷','France':'🇫🇷','Argentina':'🇦🇷','England':'🏴󠁧󠁢󠁥󠁮󠁧󠁿',
    'Spain':'🇪🇸','Belgium':'🇧🇪','Portugal':'🇵🇹','Netherlands':'🇳🇱',
    'Germany':'🇩🇪','Croatia':'🇭🇷','Morocco':'🇲🇦','Uruguay':'🇺🇾',
    'Senegal':'🇸🇳','Mexico':'🇲🇽','Switzerland':'🇨🇭','South Korea':'🇰🇷',
    'South Africa':'🇿🇦','Czechia':'🇨🇿','Canada':'🇨🇦','Qatar':'🇶🇦',
    'Bosnia and Herzegovina':'🇧🇦','Scotland':'🏴󠁧󠁢󠁳󠁣󠁴󠁿','Haiti':'🇭🇹',
    'USA':'🇺🇸','Australia':'🇦🇺','Paraguay':'🇵🇾','Turkiye':'🇹🇷',
    'Ecuador':'🇪🇨','Ivory Coast':'🇨🇮','Curacao':'🇨🇼','Japan':'🇯🇵',
    'Sweden':'🇸🇪','Tunisia':'🇹🇳','Iran':'🇮🇷','Egypt':'🇪🇬',
    'New Zealand':'🇳🇿','Saudi Arabia':'🇸🇦','Cape Verde':'🇨🇻',
    'Iraq':'🇮🇶','Norway':'🇳🇴','Algeria':'🇩🇿','Austria':'🇦🇹',
    'Jordan':'🇯🇴','Colombia':'🇨🇴','Uzbekistan':'🇺🇿','DR Congo':'🇨🇩',
    'Ghana':'🇬🇭','Panama':'🇵🇦'
}

# ── Flag image CDN (for large displays only) ──────────────────────────────────
FLAG_CODES = {
    'Brazil':'br','France':'fr','Argentina':'ar','England':'gb-eng',
    'Spain':'es','Belgium':'be','Portugal':'pt','Netherlands':'nl',
    'Germany':'de','Croatia':'hr','Morocco':'ma','Uruguay':'uy',
    'Senegal':'sn','Mexico':'mx','Switzerland':'ch','South Korea':'kr',
    'South Africa':'za','Czechia':'cz','Canada':'ca','Qatar':'qa',
    'Bosnia and Herzegovina':'ba','Scotland':'gb-sct','Haiti':'ht',
    'USA':'us','Australia':'au','Paraguay':'py','Turkiye':'tr',
    'Ecuador':'ec','Ivory Coast':'ci','Curacao':'cw','Japan':'jp',
    'Sweden':'se','Tunisia':'tn','Iran':'ir','Egypt':'eg',
    'New Zealand':'nz','Saudi Arabia':'sa','Cape Verde':'cv',
    'Iraq':'iq','Norway':'no','Algeria':'dz','Austria':'at',
    'Jordan':'jo','Colombia':'co','Uzbekistan':'uz','DR Congo':'cd',
    'Ghana':'gh','Panama':'pa'
}

@st.cache_data(show_spinner=False)
def flag_img_src(team: str, width: int = 80) -> str | None:
    """Fetch a flag image server-side and return a base64 data URI.

    Bypasses browser CSP restrictions entirely — works on localhost and
    Streamlit Cloud without any inline-JS onerror hacks.
    flagcdn.com format: /{width}x{height}/{code}.png  (NOT /w{size}/...)
    """
    code = FLAG_CODES.get(team)
    if code is None:
        return None
    height = (width * 3) // 4   # flagcdn requires exact 4:3 ratio (e.g. 80x60, 128x96)
    try:
        r = requests.get(
            f"https://flagcdn.com/{width}x{height}/{code}.png",
            timeout=5,
        )
        if r.status_code == 200:
            return "data:image/png;base64," + base64.b64encode(r.content).decode()
    except requests.RequestException:
        pass
    return None

def flag_emoji(team: str) -> str:
    return FLAG_EMOJI.get(team, '🌍')

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080e1e !important;
    color: #cdd8e8 !important;
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060b18 0%, #080e20 100%) !important;
    border-right: 1px solid rgba(0, 192, 122, 0.15) !important;
}

.sb-brand {
    text-align: center;
    padding: 1.8rem 1rem 1.4rem;
    border-bottom: 1px solid rgba(0,192,122,0.18);
    margin-bottom: 1.5rem;
}
.sb-logo {
    font-size: 2.8rem;
    filter: drop-shadow(0 0 12px rgba(0,192,122,0.5));
}
.sb-title {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem;
    background: linear-gradient(135deg, #00c07a, #00e8b0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 5px;
    display: block;
    margin: .5rem 0 .15rem;
}
.sb-sub {
    font-size: .6rem;
    color: #2a5040;
    letter-spacing: 3px;
    text-transform: uppercase;
}

[data-testid="stSidebar"] .stRadio > label {
    color: #2a5040 !important;
    font-size: .62rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,.018) !important;
    border: 1px solid rgba(255,255,255,.04) !important;
    border-radius: 10px !important;
    padding: .6rem 1rem !important;
    margin: .2rem 0 !important;
    transition: all .22s ease !important;
    color: #4a7060 !important;
    font-size: .86rem !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(0,192,122,.08) !important;
    border-color: rgba(0,192,122,.3) !important;
    color: #00c07a !important;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero-badge {
    display: inline-block;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #00c07a;
    border: 1px solid rgba(0,192,122,.3);
    border-radius: 20px;
    padding: .3rem .9rem;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    letter-spacing: 8px;
    line-height: 1;
    margin: 0 0 1.2rem;
    background: linear-gradient(135deg, #ffffff 0%, #00c07a 50%, #ff5c1a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: .9rem;
    color: #7090a0;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.8;
    font-weight: 400;
}

/* ── Podium cards ── */
.pod {
    background: linear-gradient(145deg, rgba(0,192,122,.06), rgba(255,92,26,.03));
    border: 1px solid rgba(0,192,122,.2);
    border-radius: 20px;
    padding: 2rem 1rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all .35s cubic-bezier(.4,0,.2,1);
    cursor: default;
}
.pod::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00c07a, #ff5c1a);
}
.pod:hover {
    transform: translateY(-6px);
    border-color: rgba(0,192,122,.5);
    box-shadow: 0 24px 60px rgba(0,192,122,.08);
}
.pod-medal { font-size: 1.6rem; margin-bottom: .6rem; }
.pod-flag {
    width: 80px; height: 54px;
    object-fit: cover;
    border-radius: 8px;
    box-shadow: 0 6px 24px rgba(0,0,0,.7);
    margin: .5rem 0;
}
.pod-team {
    font-size: 1rem;
    font-weight: 700;
    color: #e0eaf5;
    letter-spacing: .5px;
    margin: .6rem 0 .2rem;
}
.pod-prob {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.5rem;
    background: linear-gradient(135deg, #00c07a, #00e8b0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2px;
}

/* ── Section heading ── */
.sec {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    color: #e0eaf5;
    letter-spacing: 5px;
    margin: 2.5rem 0 1.2rem;
    padding-bottom: .6rem;
    border-bottom: 1px solid rgba(0,192,122,.2);
}

/* ── Gold divider ── */
.gdiv {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,192,122,.25), transparent);
    margin: 2rem 0;
}

/* ── All 48 teams rows ── */
.t-row {
    display: flex;
    align-items: center;
    gap: .8rem;
    padding: .6rem .8rem;
    border-radius: 10px;
    transition: all .18s;
    border: 1px solid transparent;
}
.t-row:hover {
    background: rgba(0,192,122,.04);
    border-color: rgba(0,192,122,.12);
}
.t-num {
    width: 26px; height: 26px;
    border-radius: 50%;
    background: rgba(0,192,122,.1);
    display: flex; align-items: center; justify-content: center;
    font-size: .68rem; font-weight: 700;
    color: #00c07a; flex-shrink: 0;
}
.t-flag { font-size: 1.2rem; flex-shrink: 0; }
.t-name { flex: 1; font-size: .9rem; font-weight: 500; color: #8090a8; }
.t-bar-wrap {
    width: 120px; height: 5px;
    background: rgba(255,255,255,.06);
    border-radius: 3px; overflow: hidden;
}
.t-bar {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, #00c07a, #00e8b0);
}
.t-pct {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    color: #00c07a;
    min-width: 48px;
    text-align: right;
}

/* ── Match predictor ── */
.vs-wrap { text-align: center; padding: 2rem 0; }
.vs-text {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    color: #ff5c1a;
    text-shadow: 0 0 40px rgba(255,92,26,.4);
    letter-spacing: 6px;
}

.mc {
    text-align: center;
    padding: 2rem 1.25rem;
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 20px;
    transition: all .3s;
}
.mc:hover {
    border-color: rgba(0,192,122,.2);
    background: rgba(0,192,122,.03);
}
.mc-flag-img {
    width: 128px; height: 86px;
    object-fit: cover;
    border-radius: 10px;
    box-shadow: 0 8px 28px rgba(0,0,0,.6);
}
.mc-team {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 3px;
    color: #e0eaf5;
    margin-top: .75rem;
}
.mc-prob {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    background: linear-gradient(135deg, #00c07a, #00e8b0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.mc-lbl {
    font-size: .62rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #2a5040;
    margin-top: .2rem;
}

.draw-mid { text-align: center; padding: 3rem 0; }
.draw-lbl {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    color: rgba(0,192,122,.2);
    letter-spacing: 4px;
}
.draw-prob {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    color: #00c07a;
}

.prob-bar {
    border-radius: 12px;
    overflow: hidden;
    height: 52px;
    display: flex;
    margin: 1.5rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,.4);
}
.pba {
    background: linear-gradient(90deg, #1460b0, #1a90e8);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: .9rem; color: #fff;
}
.pbd {
    background: rgba(50,70,90,.8);
    display: flex; align-items: center; justify-content: center;
    font-size: .82rem; font-weight: 600;
    color: rgba(255,255,255,.4);
}
.pbb {
    background: linear-gradient(90deg, #8b2010, #d02808);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: .9rem; color: #fff;
}

/* ── Comparison table ── */
.ct { width: 100%; border-collapse: collapse; margin: 1rem 0; }
.ct th {
    font-size: .62rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #2a5040;
    padding: .75rem 1rem;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,.05);
}
.ct td {
    padding: .7rem 1rem;
    font-size: .88rem;
    color: #5a7080;
    border-bottom: 1px solid rgba(255,255,255,.04);
    transition: color .2s;
}
.ct tr:hover td { color: #b0c8d8; }
.ct td.w { color: #00c07a !important; font-weight: 600; }

/* ── Group cards ── */
.gc {
    background: rgba(255,255,255,.02);
    border: 1px solid rgba(255,255,255,.055);
    border-radius: 14px;
    padding: 1.1rem;
    margin-bottom: .9rem;
    transition: all .3s;
}
.gc:hover {
    border-color: rgba(0,192,122,.2);
    background: rgba(0,192,122,.025);
}
.gc-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: .72rem;
    letter-spacing: 3px;
    color: #00c07a;
    margin-bottom: .75rem;
}
.gc-team {
    display: flex;
    align-items: center;
    gap: .5rem;
    padding: .4rem 0;
    border-bottom: 1px solid rgba(255,255,255,.035);
    font-size: .85rem;
    color: #6080a0;
    transition: color .2s;
}
.gc-team:last-child { border-bottom: none; }
.gc-team:hover { color: #c0d8e8; }
.gc-emoji { font-size: 1rem; flex-shrink: 0; }

/* ── Glass cards (About) ── */
.glass {
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(255,255,255,.065);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all .3s;
}
.glass:hover {
    border-color: rgba(0,192,122,.2);
    background: rgba(0,192,122,.025);
}
.glass-lbl {
    font-size: .62rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #00c07a;
    margin-bottom: 1rem;
    display: block;
}
.glass-body {
    color: #6888a0;
    line-height: 2;
    font-size: .88rem;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 10px !important;
    color: #c0d8e8 !important;
    transition: all .2s !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(0,192,122,.4) !important;
}

/* ── Footer ── */
.foot {
    text-align: center;
    padding: 2.5rem 0 1rem;
    font-size: .62rem;
    color: #0d2018;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

/* ── FIFA 2026 header banner ── */
.wc-banner {
    background: linear-gradient(135deg, #00301e 0%, #080e1e 40%, #1a0a2e 100%);
    border: 1px solid rgba(0,192,122,.15);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.wc-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,192,122,.08) 0%, transparent 70%);
    pointer-events: none;
}
.wc-banner-year {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 7rem;
    line-height: 1;
    background: linear-gradient(180deg, #00e8b0 0%, #00c07a 50%, #004a30 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 10px;
    display: block;
}
.wc-banner-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    color: rgba(255,255,255,.3);
    letter-spacing: 8px;
    display: block;
    margin-top: .2rem;
}
.wc-banner-sub {
    font-size: .8rem;
    color: #3a6050;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: .75rem;
    display: block;
}
</style>
"""

# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_files():
    m = joblib.load(os.path.join(ROOT, 'models', 'best_model.pkl'))
    s = joblib.load(os.path.join(ROOT, 'models', 'scaler.pkl'))
    f = joblib.load(os.path.join(ROOT, 'models', 'features.pkl'))
    return m, s, f

@st.cache_data
def load_data():
    with open(os.path.join(ROOT, 'data', 'processed', 'team_features.json')) as f:
        tf = json.load(f)
    preds = pd.read_csv(
        os.path.join(ROOT, 'data', 'processed', 'wc2026_predictions.csv')
    )
    return tf, preds.sort_values('probability', ascending=False).reset_index(drop=True)

model, scaler, FEATURES = load_model_files()
team_features, predictions = load_data()
WC_TEAMS = sorted(team_features.keys())

GROUPS = {
    'A': ['Mexico', 'South Korea', 'South Africa', 'Czechia'],
    'B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'],
    'C': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
    'D': ['USA', 'Australia', 'Paraguay', 'Turkiye'],
    'E': ['Germany', 'Ecuador', 'Ivory Coast', 'Curacao'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Iran', 'Egypt', 'New Zealand'],
    'H': ['Spain', 'Uruguay', 'Saudi Arabia', 'Cape Verde'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'Colombia', 'Uzbekistan', 'DR Congo'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

def predict_match(a, b):
    ta, tb = team_features[a], team_features[b]
    row = pd.DataFrame([{
        'rank_diff':        ta['rank']      - tb['rank'],
        'points_diff':      ta['points']    - tb['points'],
        'is_neutral':       1,
        'home_form':        ta['form'],
        'away_form':        tb['form'],
        'form_diff':        ta['form']      - tb['form'],
        'home_rank':        ta['rank'],
        'away_rank':        tb['rank'],
        'squad_avg_diff':   ta['squad_avg'] - tb['squad_avg'],
        'top11_diff':       ta['top11']     - tb['top11'],
        'star_player_diff': ta['star']      - tb['star'],
        'depth_diff':       ta['depth']     - tb['depth'],
        'home_squad_avg':   ta['squad_avg'],
        'away_squad_avg':   tb['squad_avg'],
        'home_star_rating': ta['star'],
        'away_star_rating': tb['star'],
    }])
    p = model.predict_proba(scaler.transform(row[FEATURES]))[0]
    return p[2], p[1], p[0]

# ── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-logo">⚽</div>
        <span class="sb-title">WC 2026</span>
        <span class="sb-sub">AI Predictor</span>
    </div>""", unsafe_allow_html=True)

    page = st.radio("", [
        "🏆  Tournament Predictions",
        "🆚  Match Predictor",
        "👥  Groups",
        "ℹ️  About",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        '<p style="font-size:.68rem;color:#2a5040;letter-spacing:1px;'
        'text-align:center;line-height:1.8;">'
        '10,000 Monte Carlo Simulations<br>'
        'Logistic Regression · 60.4% Accuracy</p>',
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — TOURNAMENT PREDICTIONS
# ═══════════════════════════════════════════════════════════════
if "Tournament" in page:

    # FIFA 2026 styled banner
    st.markdown("""
    <div class="wc-banner">
        <span class="wc-banner-year">2026</span>
        <span class="wc-banner-label">FIFA WORLD CUP</span>
        <span class="wc-banner-sub">
            ⚽ United States · Canada · Mexico &nbsp;·&nbsp;
            AI-Powered Win Probabilities
        </span>
    </div>""", unsafe_allow_html=True)

    # Top 4 podium
    medals = ['🥇', '🥈', '🥉', '4️⃣']
    cols = st.columns(4)
    for i, (_, row) in enumerate(predictions.head(4).iterrows()):
        team = row['team']
        prob = row['probability']
        src  = flag_img_src(team, 80)
        img  = f'<img class="pod-flag" src="{src}" alt="{team}">' if src else ''
        cols[i].markdown(f"""
        <div class="pod">
            <div class="pod-medal">{medals[i]}</div>
            {img}
            <div style="font-size:2rem;margin:.3rem 0">{flag_emoji(team)}</div>
            <div class="pod-team">{team}</div>
            <div class="pod-prob">{prob:.1%}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">FULL STANDINGS</div>', unsafe_allow_html=True)

    # Plotly chart — fixed titlefont → title_font
    top20 = predictions.head(20)
    fig = go.Figure(go.Bar(
        x=top20['probability'] * 100,
        y=top20['team'],
        orientation='h',
        marker=dict(
            color=list(range(len(top20))),
            colorscale=[
                [0,   '#00e8b0'],
                [0.4, '#00c07a'],
                [0.8, '#1a6fc4'],
                [1,   '#0d2040'],
            ],
            reversescale=True,
            line=dict(width=0),
        ),
        text=[f"{p:.1%}" for p in top20['probability']],
        textposition='outside',
        textfont=dict(color='#3a6050', size=11, family='Inter'),
        hovertemplate='<b>%{y}</b><br>Win probability: %{text}<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#4a7060'),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,.04)',
            title='Win Probability (%)',
            title_font=dict(color='#2a5040', size=11),
            tickfont=dict(color='#2a5040'),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color='#8090a8', size=12),
            autorange='reversed',
        ),
        margin=dict(l=10, r=80, t=10, b=40),
        height=580,
        bargap=0.28,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec">ALL 48 TEAMS</div>', unsafe_allow_html=True)

    # All 48 — one HTML block, emoji flags (no CDN)
    max_p = predictions['probability'].max()
    rows_html = ''
    for i, row in predictions.iterrows():
        team = row['team']
        prob = row['probability']
        fill = (prob / max_p) * 100
        emoji = flag_emoji(team)
        rows_html += (
            f'<div class="t-row">'
            f'<div class="t-num">{i + 1}</div>'
            f'<span class="t-flag">{emoji}</span>'
            f'<span class="t-name">{team}</span>'
            f'<div class="t-bar-wrap">'
            f'<div class="t-bar" style="width:{fill:.1f}%"></div>'
            f'</div>'
            f'<span class="t-pct">{prob:.1%}</span>'
            f'</div>'
        )
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown(
        '<div class="foot">Built by Abbas Amir · BSBI Berlin · 2026</div>',
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════
elif "Match" in page:

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">⚡ Live AI Prediction</div>
        <div class="hero-title">HEAD TO HEAD</div>
        <div class="hero-sub">
            Select any two World Cup 2026 teams
            for an instant match prediction.
        </div>
    </div>""", unsafe_allow_html=True)

    c1, cm, c2 = st.columns([5, 1, 5])
    with c1:
        team_a = st.selectbox("TEAM A", WC_TEAMS,
                              index=WC_TEAMS.index('Brazil'), key='ta')
    with cm:
        st.markdown(
            '<div class="vs-wrap"><div class="vs-text">VS</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        team_b = st.selectbox("TEAM B", WC_TEAMS,
                              index=WC_TEAMS.index('France'), key='tb')

    if team_a == team_b:
        st.warning("Please select two different teams.")
    else:
        p_a, p_d, p_b = predict_match(team_a, team_b)

        # Fetch flag images server-side so they work regardless of browser CSP
        src_a = flag_img_src(team_a, 128)
        src_b = flag_img_src(team_b, 128)
        flag_a = (f'<img class="mc-flag-img" src="{src_a}" alt="{team_a}">'
                  if src_a else
                  f'<div style="font-size:4rem;margin:1rem 0">{flag_emoji(team_a)}</div>')
        flag_b = (f'<img class="mc-flag-img" src="{src_b}" alt="{team_b}">'
                  if src_b else
                  f'<div style="font-size:4rem;margin:1rem 0">{flag_emoji(team_b)}</div>')

        st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)

        ca, cd, cb = st.columns([4, 2, 4])
        with ca:
            st.markdown(f"""
            <div class="mc">
                {flag_a}
                <div class="mc-team">{team_a}</div>
                <div class="mc-lbl">Win Probability</div>
                <div class="mc-prob">{p_a:.1%}</div>
            </div>""", unsafe_allow_html=True)

        with cd:
            st.markdown(f"""
            <div class="draw-mid">
                <div class="draw-lbl">DRAW</div>
                <div class="draw-prob">{p_d:.1%}</div>
            </div>""", unsafe_allow_html=True)

        with cb:
            st.markdown(f"""
            <div class="mc">
                {flag_b}
                <div class="mc-team">{team_b}</div>
                <div class="mc-lbl">Win Probability</div>
                <div class="mc-prob">{p_b:.1%}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prob-bar">
            <div class="pba" style="width:{p_a * 100:.1f}%">{p_a:.0%}</div>
            <div class="pbd" style="width:{p_d * 100:.1f}%">{p_d:.0%}</div>
            <div class="pbb" style="width:{p_b * 100:.1f}%">{p_b:.0%}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec">TEAM COMPARISON</div>',
            unsafe_allow_html=True
        )

        ta = team_features[team_a]
        tb = team_features[team_b]

        def css_winner(va, vb, lower_is_better=False):
            """Return CSS class tuple ('w', '') where 'w' marks the better value."""
            if lower_is_better:
                return ('w', '') if va < vb else ('', 'w') if vb < va else ('', '')
            return ('w', '') if va > vb else ('', 'w') if vb > va else ('', '')

        r1 = css_winner(ta['rank'],      tb['rank'],      lower_is_better=True)
        r2 = css_winner(ta['points'],    tb['points'])
        r3 = css_winner(ta['squad_avg'], tb['squad_avg'])
        r4 = css_winner(ta['top11'],     tb['top11'])
        r5 = css_winner(ta['star'],      tb['star'])
        r6 = css_winner(ta['form'],      tb['form'])

        ea = flag_emoji(team_a)
        eb = flag_emoji(team_b)

        st.markdown(f"""
        <table class="ct">
          <tr>
            <th>METRIC</th>
            <th>{ea} {team_a.upper()}</th>
            <th>{eb} {team_b.upper()}</th>
          </tr>
          <tr>
            <td>FIFA Rank</td>
            <td class="{r1[0]}">#{int(ta['rank'])}</td>
            <td class="{r1[1]}">#{int(tb['rank'])}</td>
          </tr>
          <tr>
            <td>FIFA Points</td>
            <td class="{r2[0]}">{int(ta['points'])}</td>
            <td class="{r2[1]}">{int(tb['points'])}</td>
          </tr>
          <tr>
            <td>Squad Avg Rating</td>
            <td class="{r3[0]}">{ta['squad_avg']:.1f}</td>
            <td class="{r3[1]}">{tb['squad_avg']:.1f}</td>
          </tr>
          <tr>
            <td>Top 11 Rating</td>
            <td class="{r4[0]}">{ta['top11']:.1f}</td>
            <td class="{r4[1]}">{tb['top11']:.1f}</td>
          </tr>
          <tr>
            <td>Star Player Rating</td>
            <td class="{r5[0]}">{ta['star']:.0f}</td>
            <td class="{r5[1]}">{tb['star']:.0f}</td>
          </tr>
          <tr>
            <td>Recent Form</td>
            <td class="{r6[0]}">{ta['form']:.0%}</td>
            <td class="{r6[1]}">{tb['form']:.0%}</td>
          </tr>
        </table>""", unsafe_allow_html=True)

        st.markdown(
            '<div class="foot">All World Cup matches played on neutral ground</div>',
            unsafe_allow_html=True
        )

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — GROUPS
# ═══════════════════════════════════════════════════════════════
elif "Groups" in page:

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">📋 Group Stage Draw</div>
        <div class="hero-title">ALL 12 GROUPS</div>
        <div class="hero-sub">
            48 teams · 12 groups ·
            Top 2 + 8 best third-place teams advance.
        </div>
    </div>""", unsafe_allow_html=True)

    group_list = list(GROUPS.items())
    for row_start in range(0, 12, 3):
        cols = st.columns(3)
        for col_idx in range(3):
            gi = row_start + col_idx
            if gi >= len(group_list):
                break
            gname, teams = group_list[gi]
            team_rows = ''
            for team in teams:
                match_row = predictions[predictions['team'] == team]
                prob = match_row['probability'].values[0] if len(match_row) else 0.0
                rank = int(team_features[team]['rank'])
                emoji = flag_emoji(team)
                team_rows += (
                    f'<div class="gc-team">'
                    f'<span class="gc-emoji">{emoji}</span>'
                    f'<span style="flex:1">{team}</span>'
                    f'<span style="color:#1a4030;font-size:.72rem;">#{rank}</span>'
                    f'<span style="color:#00c07a;font-size:.78rem;'
                    f'font-weight:600;min-width:40px;text-align:right;">'
                    f'{prob:.1%}</span>'
                    f'</div>'
                )
            with cols[col_idx]:
                st.markdown(
                    f'<div class="gc">'
                    f'<div class="gc-label">GROUP {gname}</div>'
                    f'{team_rows}'
                    f'</div>',
                    unsafe_allow_html=True
                )

    st.markdown(
        '<div class="foot">'
        'Win % = tournament win probability from 10,000 Monte Carlo simulations'
        '</div>',
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ═══════════════════════════════════════════════════════════════
elif "About" in page:

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🧠 Methodology</div>
        <div class="hero-title">ABOUT THE MODEL</div>
        <div class="hero-sub">
            Built by Abbas Amir · BSc Computer Science · BSBI Berlin
        </div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass">
            <span class="glass-lbl">📊 Data Sources</span>
            <div class="glass-body">
                49,477 international results (1872–2026)<br>
                FIFA World Rankings (1992–2024)<br>
                EA Sports FC 24 · 18,350 players<br>
                160 nations · filtered to 2020+
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass">
            <span class="glass-lbl">🤖 Model Performance</span>
            <div class="glass-body">
                🥇 Logistic Regression —
                <span style="color:#00c07a;font-weight:700;">60.4%</span><br>
                🥈 XGBoost — 58.5%<br>
                🥉 Random Forest — 55.8%<br>
                Baseline (random) — 33.3%
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <span class="glass-lbl">⚡ Features Used</span>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);
                    gap:.5rem;" class="glass-body">
            <div>
                FIFA ranking difference<br>
                FIFA points difference<br>
                Squad average rating (top 23)<br>
                Top 11 starting lineup rating
            </div>
            <div>
                Star player rating<br>
                Squad depth rating<br>
                Recent form (last 20 matches)<br>
                Neutral ground indicator
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <span class="glass-lbl">🎲 Simulation Method</span>
        <div class="glass-body">
            The model outputs win / draw / loss probabilities for any matchup.
            10,000 Monte Carlo simulations each run the full 48-team bracket
            — group stage round-robin, best-thirds selection, and all five
            knockout rounds — then aggregate results into tournament
            win probabilities shown throughout this app.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="foot">
        Abbas Aamir · BSc Computer Science · BSBI Berlin · 2026
    </div>""", unsafe_allow_html=True)