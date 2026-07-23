#!/usr/bin/env python3
"""
FIFA World Cup 2026 ML Predictor — Training Script
Outputs:
  models/best_model.pkl  — trained classifier
  models/scaler.pkl      — fitted StandardScaler
  models/features.pkl    — ordered feature list
  data/processed/team_features.json      — per-team stats for all 48 teams
  data/processed/wc2026_predictions.csv  — tournament win probabilities
"""

import os, json, random, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(ROOT, 'data', 'raw')
PROC = os.path.join(ROOT, 'data', 'processed')
MDL  = os.path.join(ROOT, 'models')
os.makedirs(PROC, exist_ok=True)
os.makedirs(MDL,  exist_ok=True)

# ── WC 2026 groups (must match app.py exactly) ───────────────────────────────
GROUPS = {
    'A': ['Mexico',    'South Korea',  'South Africa', 'Czechia'],
    'B': ['Canada',    'Switzerland',  'Qatar',         'Bosnia and Herzegovina'],
    'C': ['Brazil',    'Morocco',      'Scotland',      'Haiti'],
    'D': ['USA',       'Australia',    'Paraguay',      'Turkiye'],
    'E': ['Germany',   'Ecuador',      'Ivory Coast',   'Curacao'],
    'F': ['Netherlands','Japan',       'Sweden',        'Tunisia'],
    'G': ['Belgium',   'Iran',         'Egypt',         'New Zealand'],
    'H': ['Spain',     'Uruguay',      'Saudi Arabia',  'Cape Verde'],
    'I': ['France',    'Senegal',      'Iraq',          'Norway'],
    'J': ['Argentina', 'Algeria',      'Austria',       'Jordan'],
    'K': ['Portugal',  'Colombia',     'Uzbekistan',    'DR Congo'],
    'L': ['England',   'Croatia',      'Ghana',         'Panama'],
}
WC_TEAMS = [t for teams in GROUPS.values() for t in teams]

# ── Name mappings ─────────────────────────────────────────────────────────────
# App name → FIFA ranking 'country_full'
TO_RANKING = {
    'USA':                   'USA',
    'South Korea':           'Korea Republic',
    'Ivory Coast':           "Côte d'Ivoire",
    'Turkiye':               'Turkey',
    'Iran':                  'IR Iran',
    'DR Congo':              'Congo DR',
    'Cape Verde':            'Cabo Verde',
    'Curacao':               'Curacao',
}

# App name → results.csv team name
TO_RESULTS = {
    'USA':       'United States',
    'Turkiye':   'Turkey',
    'Curacao':   'Curaçao',
    'South Korea': 'South Korea',
    'Ivory Coast': 'Ivory Coast',
    'DR Congo':    'DR Congo',
    'Cape Verde':  'Cape Verde',
    'Iran':        'Iran',
}

# App name → male_players.csv 'nationality_name'
TO_PLAYERS = {
    'USA':         'United States',
    'Turkiye':     'Turkey',
    'Curacao':     'Curaçao',
    'South Korea': 'South Korea',
    'Ivory Coast': 'Ivory Coast',
    'DR Congo':    'DR Congo',
    'Cape Verde':  'Cape Verde',
    'Iran':        'Iran',
}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — FIFA Rankings (rank + points)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading FIFA rankings")
rank_df  = pd.read_csv(os.path.join(RAW, 'fifa_ranking-2024-06-20.csv'))
latest   = rank_df[rank_df['rank_date'] == rank_df['rank_date'].max()].copy()
rank_map = latest.set_index('country_full')[['rank', 'total_points']].to_dict('index')

def get_rank_points(app_team):
    rname = TO_RANKING.get(app_team, app_team)
    if rname in rank_map:
        r = rank_map[rname]
        return float(r['rank']), float(r['total_points'])
    # fallback: search case-insensitive
    for k, v in rank_map.items():
        if k.lower() == rname.lower():
            return float(v['rank']), float(v['total_points'])
    return 100.0, 1000.0


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Squad ratings (FIFA 24 player data)
# ═══════════════════════════════════════════════════════════════════════════
print("STEP 2: Loading FIFA 24 player data")
players = pd.read_csv(os.path.join(RAW, 'male_players.csv'), low_memory=False)
p24     = players[players['fifa_version'] == 24.0][['nationality_name', 'overall']].dropna()

# Pre-index by nationality for fast lookups
nation_ratings = {}
for nation, grp in p24.groupby('nationality_name'):
    nation_ratings[nation] = np.sort(grp['overall'].values)[::-1]  # desc

def get_squad_stats(app_team):
    pname = TO_PLAYERS.get(app_team, app_team)
    ratings = nation_ratings.get(pname)
    if ratings is None or len(ratings) < 5:
        return 70.0, 68.0, 75.0, 65.0
    top23 = ratings[:23]
    top11 = ratings[:11]
    bench = ratings[11:23]
    return (
        float(top23.mean()),
        float(top11.mean()),
        float(ratings[0]),
        float(bench.mean()) if len(bench) > 0 else float(top11.mean()) - 2.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Recent form (results 2020+)
# ═══════════════════════════════════════════════════════════════════════════
print("STEP 3: Computing recent form from match results")
res = pd.read_csv(os.path.join(RAW, 'results.csv'))
res['date'] = pd.to_datetime(res['date'])
recent = res[res['date'] >= '2020-01-01'].copy()

def compute_form(rteam_name):
    home = recent[recent['home_team'] == rteam_name].copy()
    home['won'] = home['home_score'] > home['away_score']
    away = recent[recent['away_team'] == rteam_name].copy()
    away['won'] = away['away_score'] > away['home_score']
    all_m = pd.concat([home[['date', 'won']], away[['date', 'won']]])
    all_m = all_m.sort_values('date', ascending=False).head(20)
    return float(all_m['won'].mean()) if len(all_m) > 0 else 0.33

def get_form(app_team):
    return compute_form(TO_RESULTS.get(app_team, app_team))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Build team_features.json for all 48 WC teams
# ═══════════════════════════════════════════════════════════════════════════
print("STEP 4: Building team_features.json")
team_features = {}
for team in WC_TEAMS:
    rank, pts          = get_rank_points(team)
    sq_avg, top11, star, depth = get_squad_stats(team)
    form               = get_form(team)
    team_features[team] = {
        'rank':      rank,
        'points':    pts,
        'form':      round(form, 4),
        'squad_avg': round(sq_avg, 2),
        'top11':     round(top11, 2),
        'star':      round(star, 2),
        'depth':     round(depth, 2),
    }
    print(f"  {team:<30} rank={rank:>3.0f}  pts={pts:>7.0f}  "
          f"form={form:.0%}  star={star:.0f}")

with open(os.path.join(PROC, 'team_features.json'), 'w') as fh:
    json.dump(team_features, fh, indent=2)
print(f"\nSaved team_features.json ({len(team_features)} teams)")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Build training dataset (all recent matches)
# ═══════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Building training dataset")

# Pre-compute FIFA rank lookup keyed by results.csv team name
results_to_ranking = {v: TO_RANKING.get(k, k) for k, v in TO_RESULTS.items()}
results_to_players = {v: TO_PLAYERS.get(k, k) for k, v in TO_RESULTS.items()}

# Form cache for every team that appears in recent results
print("  Pre-computing form for all teams in dataset...")
all_rteams = set(recent['home_team']) | set(recent['away_team'])
form_cache = {t: compute_form(t) for t in all_rteams}

def feature_row(rteam):
    """Return feature dict for a team referenced by its results.csv name."""
    rk_name = results_to_ranking.get(rteam, rteam)
    if rk_name in rank_map:
        rank = float(rank_map[rk_name]['rank'])
        pts  = float(rank_map[rk_name]['total_points'])
    elif rteam in rank_map:
        rank = float(rank_map[rteam]['rank'])
        pts  = float(rank_map[rteam]['total_points'])
    else:
        rank, pts = 100.0, 1000.0

    pl_name = results_to_players.get(rteam, rteam)
    ratings = nation_ratings.get(pl_name)
    if ratings is None or len(ratings) < 5:
        sq_avg, top11, star, depth = 70.0, 68.0, 75.0, 65.0
    else:
        top23 = ratings[:23]
        bench = ratings[11:23]
        sq_avg = float(top23.mean())
        top11  = float(ratings[:11].mean())
        star   = float(ratings[0])
        depth  = float(bench.mean()) if len(bench) > 0 else top11 - 2.0

    form = form_cache.get(rteam, 0.33)
    return {'rank': rank, 'pts': pts, 'form': form,
            'sq_avg': sq_avg, 'top11': top11, 'star': star, 'depth': depth}

FEATURES = [
    'rank_diff', 'points_diff', 'is_neutral',
    'home_form', 'away_form', 'form_diff',
    'home_rank', 'away_rank',
    'squad_avg_diff', 'top11_diff', 'star_player_diff', 'depth_diff',
    'home_squad_avg', 'away_squad_avg', 'home_star_rating', 'away_star_rating',
]

rows, labels = [], []
for _, m in recent.iterrows():
    h = feature_row(m['home_team'])
    a = feature_row(m['away_team'])
    if m['home_score'] > m['away_score']:
        label = 2
    elif m['home_score'] == m['away_score']:
        label = 1
    else:
        label = 0
    rows.append({
        'rank_diff':        h['rank']   - a['rank'],
        'points_diff':      h['pts']    - a['pts'],
        'is_neutral':       1 if m['neutral'] else 0,
        'home_form':        h['form'],
        'away_form':        a['form'],
        'form_diff':        h['form']   - a['form'],
        'home_rank':        h['rank'],
        'away_rank':        a['rank'],
        'squad_avg_diff':   h['sq_avg'] - a['sq_avg'],
        'top11_diff':       h['top11']  - a['top11'],
        'star_player_diff': h['star']   - a['star'],
        'depth_diff':       h['depth']  - a['depth'],
        'home_squad_avg':   h['sq_avg'],
        'away_squad_avg':   a['sq_avg'],
        'home_star_rating': h['star'],
        'away_star_rating': a['star'],
    })
    labels.append(label)

X = pd.DataFrame(rows)[FEATURES]
y = np.array(labels)
counts = np.bincount(y)
print(f"  Rows: {len(X)}  "
      f"away_wins={counts[0]}  draws={counts[1]}  home_wins={counts[2]}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — Train and evaluate models
# ═══════════════════════════════════════════════════════════════════════════
print("\nSTEP 6: Training models (5-fold CV)")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

candidates = {
    'Logistic Regression': LogisticRegression(max_iter=2000, C=1.0, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                   random_state=42, n_jobs=-1),
    'XGBoost':             XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8,
                                          random_state=42, eval_metric='mlogloss',
                                          verbosity=0),
}

best_score, best_name, best_clf = 0.0, None, None
for name, clf in candidates.items():
    cv = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"  {name:<25}  {cv.mean():.4f} ± {cv.std():.4f}")
    if cv.mean() > best_score:
        best_score, best_name, best_clf = cv.mean(), name, clf

print(f"\n  Winner: {best_name}  ({best_score:.4f})")
best_clf.fit(X_scaled, y)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — Save model artefacts
# ═══════════════════════════════════════════════════════════════════════════
print("\nSTEP 7: Saving model artefacts")
joblib.dump(best_clf, os.path.join(MDL, 'best_model.pkl'))
joblib.dump(scaler,   os.path.join(MDL, 'scaler.pkl'))
joblib.dump(FEATURES, os.path.join(MDL, 'features.pkl'))
print("  models/best_model.pkl")
print("  models/scaler.pkl")
print("  models/features.pkl")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — Monte Carlo tournament simulation (10,000 runs)
# ═══════════════════════════════════════════════════════════════════════════
print("\nSTEP 8: Monte Carlo simulation (10,000 runs)")

def match_proba(team_a, team_b):
    """Return (p_a_wins, p_draw, p_b_wins) for a neutral-ground match."""
    ta, tb = team_features[team_a], team_features[team_b]
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
    p = best_clf.predict_proba(scaler.transform(row[FEATURES]))[0]
    return p[2], p[1], p[0]  # win_a, draw, win_b


def sim_group(teams):
    """Round-robin group stage. Returns ordered list (best → worst) with points."""
    pts = {t: 0 for t in teams}
    gd  = {t: 0 for t in teams}
    gf  = {t: 0 for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            pw, pd_, pl = match_proba(a, b)
            r = random.random()
            if r < pw:            # A wins
                pts[a] += 3
                gd[a]  += 1; gd[b] -= 1
                gf[a]  += 1
            elif r < pw + pd_:    # Draw
                pts[a] += 1; pts[b] += 1
            else:                 # B wins
                pts[b] += 3
                gd[b]  += 1; gd[a] -= 1
                gf[b]  += 1
    ranked = sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
    return ranked, {t: pts[t] for t in teams}


def sim_knockout(team_a, team_b):
    """Knockout match — no draws; tiebreaker via 50/50 on draw probability."""
    pw, pd_, _ = match_proba(team_a, team_b)
    pa = pw + pd_ / 2
    return team_a if random.random() < pa else team_b


def sim_tournament():
    third_place_records = []

    # Group stage
    r32_pool = []
    for gname, teams in GROUPS.items():
        ranked, group_pts = sim_group(teams)
        r32_pool.append(ranked[0])      # 1st
        r32_pool.append(ranked[1])      # 2nd
        if len(ranked) >= 3:
            third = ranked[2]
            third_place_records.append((group_pts[third], third))

    # Best 8 of 12 third-place finishers (by points)
    third_place_records.sort(key=lambda x: x[0], reverse=True)
    best_thirds = [t for _, t in third_place_records[:8]]
    r32_pool.extend(best_thirds)        # 32 teams total

    random.shuffle(r32_pool)

    # Five knockout rounds: R32 → R16 → QF → SF → Final
    bracket = r32_pool
    while len(bracket) > 1:
        next_round = []
        for i in range(0, len(bracket), 2):
            winner = sim_knockout(bracket[i], bracket[i + 1])
            next_round.append(winner)
        bracket = next_round

    return bracket[0]


N_SIMS = 10_000
win_counts = {t: 0 for t in WC_TEAMS}

for sim_i in range(N_SIMS):
    if (sim_i + 1) % 2000 == 0:
        print(f"  Run {sim_i + 1:,} / {N_SIMS:,} ...")
    champion = sim_tournament()
    if champion in win_counts:
        win_counts[champion] += 1

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9 — Save predictions CSV
# ═══════════════════════════════════════════════════════════════════════════
print("\nSTEP 9: Saving predictions")
preds = pd.DataFrame([
    {'team': t, 'probability': win_counts[t] / N_SIMS}
    for t in WC_TEAMS
]).sort_values('probability', ascending=False).reset_index(drop=True)

preds.to_csv(os.path.join(PROC, 'wc2026_predictions.csv'), index=False)
print("  data/processed/wc2026_predictions.csv")

print("\n── Top 10 predicted champions ──")
for _, row in preds.head(10).iterrows():
    bar = '█' * int(row['probability'] * 300)
    print(f"  {row['team']:<30} {row['probability']:.2%}  {bar}")

print(f"\nAll done. Run the app with:")
print(f"  venv/bin/streamlit run app/app.py")
