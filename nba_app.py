"""
NBA Sim — nba_app.py
Simulate every NBA game, predict and track bets, and watch an animation of the
most common outcome. Same layout as the NFL / NHL / WHL apps.

Run on your PC (PowerShell, in the folder with this file):
    python -m streamlit run nba_app.py
Daily robot (what the GitHub Action runs):
    python nba_app.py --robot
"""
import sys, os, io, json, math, time, base64, random, datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

ET = ZoneInfo("America/New_York")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ROBOT_MODE = "--robot" in sys.argv

# ---------------------------------------------------------------------------
# Team info: abbr, name, conference, colours, arena lat/lon, fallback rating
# ---------------------------------------------------------------------------
TEAM_ROWS = [
    ("ATL", "Atlanta Hawks", "E", "#E03A3E", "#C1D32F", 33.757, -84.396, 0),
    ("BOS", "Boston Celtics", "E", "#007A33", "#BA9653", 42.366, -71.062, 4),
    ("BKN", "Brooklyn Nets", "E", "#000000", "#FFFFFF", 40.683, -73.975, -7),
    ("CHA", "Charlotte Hornets", "E", "#1D1160", "#00788C", 35.225, -80.839, -5),
    ("CHI", "Chicago Bulls", "E", "#CE1141", "#000000", 41.881, -87.674, -1),
    ("CLE", "Cleveland Cavaliers", "E", "#860038", "#FDBB30", 41.496, -81.688, 6),
    ("DAL", "Dallas Mavericks", "W", "#00538C", "#B8C4CA", 32.790, -96.810, -1),
    ("DEN", "Denver Nuggets", "W", "#0E2240", "#FEC524", 39.749, -105.008, 4),
    ("DET", "Detroit Pistons", "E", "#C8102E", "#1D42BA", 42.341, -83.055, 3),
    ("GS", "Golden State Warriors", "W", "#1D428A", "#FFC72C", 37.768, -122.388, 2),
    ("HOU", "Houston Rockets", "W", "#CE1141", "#C4CED4", 29.751, -95.362, 4),
    ("IND", "Indiana Pacers", "E", "#002D62", "#FDBB30", 39.764, -86.155, 0),
    ("LAC", "LA Clippers", "W", "#C8102E", "#1D428A", 33.945, -118.341, 3),
    ("LAL", "Los Angeles Lakers", "W", "#552583", "#FDB927", 34.043, -118.267, 2),
    ("MEM", "Memphis Grizzlies", "W", "#5D76A9", "#12173F", 35.138, -90.051, 0),
    ("MIA", "Miami Heat", "E", "#98002E", "#F9A01B", 25.781, -80.188, 0),
    ("MIL", "Milwaukee Bucks", "E", "#00471B", "#EEE1C6", 43.045, -87.917, 0),
    ("MIN", "Minnesota Timberwolves", "W", "#0C2340", "#78BE20", 44.979, -93.276, 3),
    ("NO", "New Orleans Pelicans", "W", "#0C2340", "#C8102E", 29.949, -90.082, -5),
    ("NY", "New York Knicks", "E", "#006BB6", "#F58426", 40.751, -73.993, 4),
    ("OKC", "Oklahoma City Thunder", "W", "#007AC1", "#EF3B24", 35.463, -97.515, 9),
    ("ORL", "Orlando Magic", "E", "#0077C0", "#C4CED4", 28.539, -81.384, 2),
    ("PHI", "Philadelphia 76ers", "E", "#006BB6", "#ED174C", 39.901, -75.172, -2),
    ("PHX", "Phoenix Suns", "W", "#1D1160", "#E56020", 33.446, -112.071, -1),
    ("POR", "Portland Trail Blazers", "W", "#E03A3E", "#000000", 45.532, -122.667, -2),
    ("SAC", "Sacramento Kings", "W", "#5A2D81", "#63727A", 38.580, -121.500, -2),
    ("SA", "San Antonio Spurs", "W", "#C4CED4", "#000000", 29.427, -98.437, 1),
    ("TOR", "Toronto Raptors", "E", "#CE1141", "#000000", 43.643, -79.379, -2),
    ("UTAH", "Utah Jazz", "W", "#002B5C", "#F9A01B", 40.768, -111.901, -8),
    ("WSH", "Washington Wizards", "E", "#002B5C", "#E31837", 38.898, -77.021, -9),
]
TEAMS = {r[0]: dict(abbr=r[0], name=r[1], conf=r[2], c1=r[3], c2=r[4], lat=r[5], lon=r[6], prior=r[7])
         for r in TEAM_ROWS}
ABBRS = list(TEAMS)
NICK = {v["name"].split()[-1].lower(): k for k, v in TEAMS.items()}   # "celtics" -> BOS
ALT_ABBR = {"GSW": "GS", "NYK": "NY", "NOP": "NO", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH",
            "PHO": "PHX", "BRK": "BKN", "CHO": "CHA", "NJN": "BKN", "SEA": "OKC", "NOH": "NO"}

LEAGUE_ORTG = 114.5      # points per 100 possessions (fallback)
LEAGUE_PACE = 99.5       # possessions per team per 48 min (fallback)
HCA_PTS = 2.4            # home-court edge, points
ALTITUDE = {"DEN": 0.8, "UTAH": 0.6}
MARGIN_SD = 12.5         # used for quick win-probability maths


def team_abbr(raw_abbr=None, display_name=None):
    """Turn whatever ESPN / The Odds API gives us into our team code."""
    if raw_abbr:
        a = str(raw_abbr).upper()
        if a in TEAMS:
            return a
        if a in ALT_ABBR:
            return ALT_ABBR[a]
    if display_name:
        nick = str(display_name).split()[-1].lower()
        if nick in NICK:
            return NICK[nick]
    return None


def logo_url(abbr):
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{abbr.lower()}.png"


def today_et():
    return dt.datetime.now(ET).date()


def season_for(d):
    """ESPN names a season by the year it ends: 2026-27 season = 2027."""
    return d.year + 1 if d.month >= 8 else d.year


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def american_to_prob(odds):
    odds = float(odds)
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def prob_to_american(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def american_to_decimal(odds):
    odds = float(odds)
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def fmt_odds(o):
    o = int(round(o))
    return f"+{o}" if o > 0 else str(o)


def secret(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Data: ESPN feeds (with a demo league if ESPN can't be reached)
# ---------------------------------------------------------------------------
def http_json(url, params=None, timeout=15):
    r = requests.get(url, params=params, timeout=timeout,
                     headers={"User-Agent": "Mozilla/5.0 (nba-sim app)"})
    r.raise_for_status()
    return r.json()


def _num(x):
    if x is None:
        return np.nan
    if isinstance(x, dict):
        x = x.get("value", x.get("displayValue"))
    try:
        return float(str(x).replace("+", ""))
    except Exception:
        return np.nan


def _parse_odds(comp, home, away):
    out = dict(mkt_spread=np.nan, mkt_total=np.nan, ml_home=np.nan, ml_away=np.nan)
    odds = comp.get("odds") or []
    if not odds:
        return out
    o = odds[0]
    det = str(o.get("details") or "").strip()
    if det.upper() in ("EVEN", "PK", "PICK"):
        out["mkt_spread"] = 0.0
    elif det:
        parts = det.split()
        if len(parts) >= 2:
            fav = team_abbr(parts[0])
            val = _num(parts[-1])
            if fav and not np.isnan(val):
                out["mkt_spread"] = val if fav == home else -val   # home-team spread (neg = home favoured)
    if np.isnan(out["mkt_spread"]) and o.get("spread") is not None:
        out["mkt_spread"] = _num(o.get("spread"))
    out["mkt_total"] = _num(o.get("overUnder"))
    ho, ao = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}
    out["ml_home"], out["ml_away"] = _num(ho.get("moneyLine")), _num(ao.get("moneyLine"))
    ml = o.get("moneyline") or {}
    if np.isnan(out["ml_home"]) and ml:
        for side, key in (("home", "ml_home"), ("away", "ml_away")):
            node = (ml.get(side) or {})
            node = node.get("close") or node.get("open") or {}
            out[key] = _num(node.get("odds"))
    return out


STAT_KEYS = {"fieldGoalsAttempted": "fga", "fieldGoalsMade": "fgm", "threePointFieldGoalsAttempted": "fg3a",
             "threePointFieldGoalsMade": "fg3m", "freeThrowsAttempted": "fta", "freeThrowsMade": "ftm",
             "rebounds": "reb", "assists": "ast"}


def parse_events(payload):
    rows = []
    for ev in payload.get("events", []):
        try:
            comp = ev["competitions"][0]
            cs = comp["competitors"]
            h = next(c for c in cs if c.get("homeAway") == "home")
            a = next(c for c in cs if c.get("homeAway") == "away")
            ht = team_abbr(h["team"].get("abbreviation"), h["team"].get("displayName"))
            at = team_abbr(a["team"].get("abbreviation"), a["team"].get("displayName"))
            if not ht or not at:
                continue
            stype_ = comp.get("status", {}).get("type", {}) or ev.get("status", {}).get("type", {})
            state = stype_.get("state")
            if any(w in str(stype_.get("description", "")) + str(stype_.get("name", ""))
                   for w in ("Postponed", "Canceled", "Cancelled", "POSTPONED", "CANCELED")):
                state = "postponed"
            when = pd.Timestamp(ev["date"]).tz_convert(ET)
            row = dict(event_id=str(ev["id"]), start=when, date=when.date(),
                       season_type=int((ev.get("season") or {}).get("type", 2)),
                       state=state, home=ht, away=at,
                       home_pts=_num(h.get("score")) if state != "pre" else np.nan,
                       away_pts=_num(a.get("score")) if state != "pre" else np.nan,
                       neutral=bool(comp.get("neutralSite", False)),
                       status_text=comp.get("status", {}).get("type", {}).get("shortDetail", ""))
            for side, c in (("h", h), ("a", a)):
                stats = {s.get("name"): s.get("displayValue") for s in (c.get("statistics") or [])}
                for k, short in STAT_KEYS.items():
                    row[f"{side}_{short}"] = _num(stats.get(k))
                for rec in c.get("records") or []:
                    if rec.get("type") in ("total", None) or rec.get("name") == "overall":
                        row[f"{side}_record"] = rec.get("summary")
                        break
            row.update(_parse_odds(comp, ht, at))
            rows.append(row)
        except Exception:
            continue
    return rows


def _fetch_range(a, b, limit=True):
    params = {"dates": f"{a:%Y%m%d}-{b:%Y%m%d}"}
    if limit:
        params["limit"] = 1000
    return parse_events(http_json(f"{ESPN}/scoreboard", params))


def _fetch_days(days):
    """One request per day, several at a time (always works, just slower the first time)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(d):
        for _ in range(3):
            try:
                return parse_events(http_json(f"{ESPN}/scoreboard", {"dates": f"{d:%Y%m%d}"}))
            except Exception:
                time.sleep(0.5)
        return []
    with ThreadPoolExecutor(max_workers=12) as ex:
        return [r for rows in ex.map(one, days) for r in rows]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_season_events(season):
    """Every game of a season (played and still to come) from ESPN's scoreboard."""
    start, end = dt.date(season - 1, 10, 1), dt.date(season, 6, 30)
    months = []
    a = start
    while a <= end:
        b = (pd.Timestamp(a) + pd.DateOffset(months=1) - pd.Timedelta(days=1)).date()
        months.append((a, min(b, end)))
        a = b + dt.timedelta(days=1)
    rows = []
    # ESPN sometimes refuses date ranges (HTTP 400); try ranges first, then fall back to day-by-day.
    for use_limit in (True, False):
        try:
            rows = [r for a, b in months for r in _fetch_range(a, b, use_limit)]
            if rows:
                break
        except Exception:
            rows = []
    if not rows:
        days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
        rows = _fetch_days(days)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("ESPN returned no games")
    df = df.drop_duplicates("event_id").sort_values("start").reset_index(drop=True)
    return df[df.season_type.isin([2, 3, 5])].reset_index(drop=True)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_day(d):
    """Fresh scoreboard for one day (live scores + latest lines)."""
    return pd.DataFrame(parse_events(http_json(f"{ESPN}/scoreboard", {"dates": f"{d:%Y%m%d}"})))


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_team_ids():
    out = {}
    js = http_json(f"{ESPN}/teams")
    for sp in js.get("sports", []):
        for lg in sp.get("leagues", []):
            for t in lg.get("teams", []):
                t = t.get("team", t)
                ab = team_abbr(t.get("abbreviation"), t.get("displayName"))
                if ab:
                    out[ab] = str(t.get("id"))
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_roster(abbr):
    ids = fetch_team_ids()
    js = http_json(f"{ESPN}/teams/{ids[abbr]}/roster")
    out = []
    aths = js.get("athletes", [])
    if aths and isinstance(aths[0], dict) and "items" in aths[0]:
        aths = [x for grp in aths for x in grp.get("items", [])]
    for p in aths:
        out.append(dict(player_id=str(p.get("id")), name=p.get("displayName"),
                        jersey=str(p.get("jersey") or ""),
                        pos=(p.get("position") or {}).get("abbreviation", "")))
    return pd.DataFrame(out)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_injuries():
    js = http_json(f"{ESPN}/injuries")
    rows = []
    for t in js.get("injuries", []):
        ab = team_abbr(t.get("abbreviation"), t.get("displayName"))
        for inj in t.get("injuries", []):
            ath = inj.get("athlete") or {}
            rows.append(dict(team=ab, player_id=str(ath.get("id", "")), name=ath.get("displayName"),
                             status=inj.get("status") or (inj.get("type") or {}).get("description", ""),
                             note=inj.get("shortComment") or inj.get("longComment") or ""))
    return pd.DataFrame(rows, columns=["team", "player_id", "name", "status", "note"])


def _split_made(s):
    try:
        m, a = str(s).split("-")
        return float(m), float(a)
    except Exception:
        return np.nan, np.nan


@st.cache_data(ttl=86400 * 3, show_spinner=False)
def fetch_box(event_id):
    """Player box score for one finished game."""
    js = http_json(f"{ESPN}/summary", {"event": event_id})
    rows = []
    for tb in (js.get("boxscore") or {}).get("players", []):
        ab = team_abbr((tb.get("team") or {}).get("abbreviation"), (tb.get("team") or {}).get("displayName"))
        for grp in tb.get("statistics", []):
            labels = [l.upper() for l in grp.get("labels", [])]
            for ath in grp.get("athletes", []):
                st_ = ath.get("stats") or []
                if not st_ or ath.get("didNotPlay"):
                    continue
                d = dict(zip(labels, st_))
                info = ath.get("athlete") or {}
                fg3m, fg3a = _split_made(d.get("3PT"))
                fgm, fga = _split_made(d.get("FG"))
                ftm, fta = _split_made(d.get("FT"))
                rows.append(dict(event_id=str(event_id), team=ab, player_id=str(info.get("id")),
                                 name=info.get("displayName"), jersey=str(info.get("jersey") or ""),
                                 starter=bool(ath.get("starter")), min=_num(d.get("MIN")),
                                 pts=_num(d.get("PTS")), reb=_num(d.get("REB")), ast=_num(d.get("AST")),
                                 stl=_num(d.get("STL")), blk=_num(d.get("BLK")), tov=_num(d.get("TO")),
                                 fg3m=fg3m, fg3a=fg3a, fgm=fgm, fga=fga, ftm=ftm, fta=fta))
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_team_schedule(abbr, season, stype):
    ids = fetch_team_ids()
    js = http_json(f"{ESPN}/teams/{ids[abbr]}/schedule", {"season": season, "seasontype": stype})
    return parse_events(js)


# ---------- demo league (used only when ESPN can't be reached) --------------
FIRST = ["Jalen", "Marcus", "Tyrese", "Darius", "Jaylen", "Devin", "Anthony", "Cade", "Evan", "Scottie",
         "Paolo", "Franz", "Amen", "Keegan", "Josh", "Desmond", "Mikal", "Derrick", "Coby", "Brandon"]
LAST = ["Walker", "Brooks", "Hayes", "Morris", "Carter", "Bridges", "Mitchell", "Grant", "Freeman", "Bell",
        "Ward", "Owens", "Reed", "Price", "Foster", "Murray", "Banks", "Holmes", "Sutton", "Wells"]


def demo_events(season):
    rng = np.random.default_rng(season)
    start, end = dt.date(season - 1, 10, 21), dt.date(season, 4, 12)
    today = today_et()
    rows, eid = [], 900000
    d = start
    while d <= end:
        teams = list(rng.permutation(ABBRS))
        for k in range(int(rng.integers(4, 9))):
            h, a = teams[2 * k], teams[2 * k + 1]
            eid += 1
            when = pd.Timestamp(dt.datetime(d.year, d.month, d.day, 19, 30), tz=ET)
            mu = (TEAMS[h]["prior"] - TEAMS[a]["prior"]) + HCA_PTS
            row = dict(event_id=str(eid), start=when, date=d, season_type=2, home=h, away=a, neutral=False,
                       mkt_spread=-round(mu * 2) / 2, mkt_total=round(rng.normal(228, 6) * 2) / 2,
                       ml_home=prob_to_american(phi(mu / MARGIN_SD)), ml_away=prob_to_american(1 - phi(mu / MARGIN_SD)))
            if d < today:
                tot = rng.normal(228, 15)
                mg = rng.normal(mu, 12.5)
                hp, ap = int(round((tot + mg) / 2)), int(round((tot - mg) / 2))
                if hp == ap:
                    hp += 1
                row.update(state="post", home_pts=hp, away_pts=ap, status_text="Final")
                for side, p in (("h", hp), ("a", ap)):
                    fga = rng.normal(88, 4); f3a = fga * rng.normal(0.42, 0.04); fta = rng.normal(22, 4)
                    row.update({f"{side}_fga": fga, f"{side}_fg3a": f3a, f"{side}_fta": fta,
                                f"{side}_fg3m": f3a * 0.36, f"{side}_ftm": fta * 0.78,
                                f"{side}_fgm": (p - fta * 0.78 - f3a * 0.36) / 2 + f3a * 0.36,
                                f"{side}_reb": rng.normal(44, 4), f"{side}_ast": rng.normal(26, 3)})
            else:
                row.update(state="pre", home_pts=np.nan, away_pts=np.nan, status_text="7:30 PM ET")
            rows.append(row)
        d += dt.timedelta(days=1)
    return pd.DataFrame(rows)


def demo_players(abbr):
    rng = np.random.default_rng(int.from_bytes(abbr.encode(), "little"))
    mins = np.array([35, 33, 32, 30, 28, 24, 20, 18, 12, 8], float)
    ppm = np.array([0.80, 0.62, 0.55, 0.45, 0.40, 0.42, 0.38, 0.35, 0.35, 0.33])
    rows = []
    used = set()
    for i in range(10):
        while True:
            nm = f"{FIRST[rng.integers(20)]} {LAST[rng.integers(20)]}"
            if nm not in used:
                used.add(nm); break
        rows.append(dict(player_id=f"{abbr}{i}", name=nm, jersey=str(int(rng.integers(0, 55))), team=abbr,
                         gp=12, min=mins[i], pts=mins[i] * ppm[i], reb=mins[i] * rng.uniform(0.12, 0.3),
                         ast=mins[i] * rng.uniform(0.05, 0.25), fg3m=mins[i] * rng.uniform(0.02, 0.09),
                         stl=mins[i] * 0.03, blk=mins[i] * rng.uniform(0.01, 0.05), tov=mins[i] * 0.05,
                         starter=i < 5, avail=1.0))
    return pd.DataFrame(rows)


def load_events(season, allow_demo=True):
    """Returns (events, is_demo)."""
    try:
        return fetch_season_events(season), False
    except Exception as e:
        if not allow_demo:
            raise
        st.session_state["espn_error"] = f"{e.__class__.__name__}: {e}"[:300]
        return demo_events(season), True


# ---------------------------------------------------------------------------
# Team ratings (opponent-adjusted offense/defense per 100 possessions + pace)
# ---------------------------------------------------------------------------
def _possessions(g):
    est = []
    for s in ("h", "a"):
        fga, fta = g.get(f"{s}_fga"), g.get(f"{s}_fta")
        if fga is None or fta is None:
            est.append(pd.Series(np.nan, index=g.index))
        else:
            est.append(fga + 0.44 * fta + 3.0)
    p = pd.concat(est, axis=1).mean(axis=1)
    fallback = (g.home_pts + g.away_pts) / 2 / 1.145
    p = p.where(p.between(80, 125), fallback)
    return p.fillna(fallback)


def compute_ratings(ev, prior_net, asof=None, half_life=45.0, K=8.0):
    """Opponent-adjusted ratings from games played before `asof` (blended with a prior early in the season)."""
    # only real, finished games (skips postponed/cancelled games that ESPN lists as "final" 0-0)
    g = ev[(ev.state == "post") & ev.home_pts.notna() & ev.away_pts.notna()
           & (ev.home_pts + ev.away_pts > 100)].copy()
    if asof is not None:
        g = g[g.date < asof]
    asof = asof or (g.date.max() + dt.timedelta(days=1) if len(g) else today_et())
    idx = {t: i for i, t in enumerate(ABBRS)}
    prior = np.array([prior_net.get(t, 0.0) for t in ABBRS], float)
    p_off, p_def = prior / 2, -prior / 2
    if len(g) == 0:
        r = pd.DataFrame(dict(off=p_off, de=p_def, net=prior, pace=LEAGUE_PACE, gp=0, w=0, l=0), index=ABBRS)
        return r, LEAGUE_ORTG, LEAGUE_PACE
    g["poss"] = _possessions(g)
    hi = g.home.map(idx).to_numpy(); ai = g.away.map(idx).to_numpy()
    poss = g.poss.to_numpy()
    oh = 100 * g.home_pts.to_numpy() / poss
    oa = 100 * g.away_pts.to_numpy() / poss
    L = float(np.average(np.r_[oh, oa]))
    P = float(poss.mean())
    days = np.array([(asof - d).days for d in g.date])
    w = 0.5 ** (days / half_life)
    hca = (HCA_PTS / 2) * 100 / P * (1.0 - g.neutral.fillna(False).astype(bool).to_numpy().astype(float))
    off, de = p_off.copy(), p_def.copy()
    den = np.bincount(hi, w, 30) + np.bincount(ai, w, 30) + K
    for _ in range(40):
        r_oh = oh - L - de[ai] - hca
        r_oa = oa - L - de[hi] + hca
        off = (np.bincount(hi, w * r_oh, 30) + np.bincount(ai, w * r_oa, 30) + K * p_off) / den
        r_dh = oa - L - off[ai] + hca
        r_da = oh - L - off[hi] - hca
        de = (np.bincount(hi, w * r_dh, 30) + np.bincount(ai, w * r_da, 30) + K * p_def) / den
        off -= off.mean(); de -= de.mean()
    pace = (np.bincount(hi, w * poss, 30) + np.bincount(ai, w * poss, 30) + K * P) / den
    reg = g[g.season_type == 2]
    hw = (reg.home_pts > reg.away_pts)
    wins = reg[hw].home.value_counts().add(reg[~hw].away.value_counts(), fill_value=0)
    losses = reg[~hw].home.value_counts().add(reg[hw].away.value_counts(), fill_value=0)
    r = pd.DataFrame(dict(off=off, de=de, net=off - de, pace=pace), index=ABBRS)
    r["w"] = wins.reindex(ABBRS).fillna(0).astype(int)
    r["l"] = losses.reindex(ABBRS).fillna(0).astype(int)
    r["gp"] = r.w + r.l
    return r, L, P


def team_styles(ev):
    g = ev[(ev.state == "post")]
    rows = []
    for t in ABBRS:
        parts = []
        for s, col in (("h", "home"), ("a", "away")):
            sub = g[g[col] == t]
            if f"{s}_fga" in sub:
                parts.append(pd.DataFrame({k: sub.get(f"{s}_{k}") for k in ("fga", "fg3a", "fg3m", "fta", "ftm", "reb", "ast")}))
        df = pd.concat(parts) if parts else pd.DataFrame()
        df = df.dropna() if len(df) else df
        if len(df) >= 3:
            rows.append(dict(team=t, three_rate=df.fg3a.sum() / df.fga.sum(), ft_rate=df.fta.sum() / df.fga.sum(),
                             fg3_pct=df.fg3m.sum() / max(df.fg3a.sum(), 1), ft_pct=df.ftm.sum() / max(df.fta.sum(), 1),
                             reb=df.reb.mean(), ast=df.ast.mean()))
        else:
            rows.append(dict(team=t, three_rate=0.42, ft_rate=0.25, fg3_pct=0.36, ft_pct=0.78, reb=44.0, ast=26.5))
    return pd.DataFrame(rows).set_index("team")


@st.cache_data(ttl=86400, show_spinner=False)
def last_season_prior(season):
    """Last season's ratings, pulled 1/3 back toward average, used as this season's starting point."""
    try:
        ev = fetch_season_events(season - 1)
        r, _, _ = compute_ratings(ev, {t: 0.0 for t in ABBRS}, asof=dt.date(season - 1, 7, 1), half_life=9999, K=2)
        return {t: float(0.67 * r.loc[t, "net"]) for t in ABBRS}, True
    except Exception:
        return {t: float(TEAMS[t]["prior"]) for t in ABBRS}, False


@st.cache_data(ttl=1800, show_spinner="Loading the NBA season…")
def build_model(season):
    ev, demo = load_events(season)
    prior, _ = (last_season_prior(season) if not demo else ({t: float(TEAMS[t]["prior"]) for t in ABBRS}, False))
    prior = {t: (float(v) if np.isfinite(v) else float(TEAMS[t]["prior"])) for t, v in prior.items()}
    ratings, L, P = compute_ratings(ev, prior)
    if not (np.isfinite(L) and np.isfinite(P) and np.isfinite(ratings[["off", "de", "pace"]].to_numpy()).all()):
        ratings, L, P = compute_ratings(ev.iloc[0:0], prior)      # safety net: start-of-season ratings
    return dict(ev=ev, demo=demo, ratings=ratings, L=L, P=P, styles=team_styles(ev), prior=prior, season=season)


# ---------------------------------------------------------------------------
# Rest, travel & fatigue
# ---------------------------------------------------------------------------
def fatigue(ev, team, event_id):
    g = ev[(ev.home == team) | (ev.away == team)].sort_values("start").reset_index(drop=True)
    hit = g.index[g.event_id == str(event_id)]
    out = dict(rest_days=3, b2b=False, three_in_four=False, travel_km=0.0, road_trip=0, pts=0.0, notes=[])
    if len(hit) == 0:
        return out
    i = int(hit[0]); cur = g.loc[i]
    if i > 0:
        prev = g.loc[i - 1]
        out["rest_days"] = max((cur.date - prev.date).days - 1, 0)
        out["b2b"] = out["rest_days"] == 0
        loc_prev, loc_cur = TEAMS[prev.home], TEAMS[cur.home]
        out["travel_km"] = haversine_km(loc_prev["lat"], loc_prev["lon"], loc_cur["lat"], loc_cur["lon"])
        recent = g[(g.date > cur.date - dt.timedelta(days=4)) & (g.date <= cur.date)]
        out["three_in_four"] = len(recent) >= 3
    k = i
    while k >= 0 and g.loc[k, "away"] == team:
        out["road_trip"] += 1; k -= 1
    p = 0.0
    if out["b2b"]:
        p -= 1.8; out["notes"].append("back-to-back")
    if out["three_in_four"]:
        p -= 0.8; out["notes"].append("3 games in 4 nights")
    if out["rest_days"] >= 2:
        p += 0.4; out["notes"].append(f"{out['rest_days']} days rest")
    if out["travel_km"] > 400:
        p -= min(out["travel_km"] / 1000 * 0.35, 1.2); out["notes"].append(f"{out['travel_km']:,.0f} km travel")
    if out["road_trip"] >= 4:
        p -= 0.5; out["notes"].append(f"game {out['road_trip']} of road trip")
    out["pts"] = round(p, 2)
    return out


# ---------------------------------------------------------------------------
# Injuries -> rating hit (uses season player stats when we have them)
# ---------------------------------------------------------------------------
def player_value_per_game(row):
    return row.pts + 0.5 * row.reb + 0.8 * row.ast + row.stl + row.blk - row.tov


def injury_penalty(pool, out_ids):
    """Points of margin a team loses when these players sit (beyond what the ratings already know)."""
    if pool is None or len(pool) == 0 or not out_ids:
        return 0.0
    pen = 0.0
    for _, r in pool[pool.player_id.isin(out_ids)].iterrows():
        if r["min"] <= 0:
            continue
        per_min = player_value_per_game(r) / r["min"]
        pen += max(per_min - 0.45, 0) * r["min"] * 0.28 * min(r.get("avail", 1.0), 1.0)
    return round(pen, 2)


# ---------------------------------------------------------------------------
# Game projection + fast Monte Carlo sim
# ---------------------------------------------------------------------------
def project(model, h, a, fat_h=0.0, fat_a=0.0, neutral=False, inj_h=0.0, inj_a=0.0, boost_h=0.0, boost_a=0.0,
            pace_adj=0.0):
    r, L, P = model["ratings"], model["L"], model["P"]
    pace = r.loc[h, "pace"] + r.loc[a, "pace"] - P + pace_adj
    hca = 0.0 if neutral else HCA_PTS + ALTITUDE.get(h, 0.0)
    edge_h = hca + (fat_h - inj_h) - (fat_a - inj_a)       # points to home team
    h_ortg = L + r.loc[h, "off"] + r.loc[a, "de"] + (edge_h / 2) * 100 / pace + boost_h
    a_ortg = L + r.loc[a, "off"] + r.loc[h, "de"] - (edge_h / 2) * 100 / pace + boost_a
    mu = (h_ortg - a_ortg) * pace / 100
    tot = (h_ortg + a_ortg) * pace / 100
    return dict(h_ortg=h_ortg, a_ortg=a_ortg, pace=pace, margin=mu, total=tot,
                win=phi(mu / MARGIN_SD), h_pts=(tot + mu) / 2, a_pts=(tot - mu) / 2)


VAR_PP = 1.05     # scoring variance per possession (tuned so margins/totals spread like real NBA games)
CATCH_UP = 0.16   # trailing teams play harder / leaders ease off (per 100 poss, per point of lead)
PACE_SD = 3.5


def sim_game(h_ortg, a_ortg, pace, n=10000, seed=None):
    rng = np.random.default_rng(seed)
    gp = np.clip(pace + rng.normal(0, PACE_SD, n), 80, 125)
    hq, aq = np.zeros((n, 4), int), np.zeros((n, 4), int)
    lead = np.zeros(n)
    for q in range(4):
        poss = gp / 4
        k = CATCH_UP * (1.6 if q == 3 else 1.0)
        exp_lead = (h_ortg - a_ortg) * gp / 100 * q / 4
        adj = -k * np.clip(lead - exp_lead, -30, 30) if q > 0 else 0.0
        mh = poss * (h_ortg + adj) / 100
        ma = poss * (a_ortg - adj) / 100
        sd = np.sqrt(poss * VAR_PP)
        hq[:, q] = np.clip(np.rint(rng.normal(mh, sd)), 4, None)
        aq[:, q] = np.clip(np.rint(rng.normal(ma, sd)), 4, None)
        lead = hq[:, :q + 1].sum(1) - aq[:, :q + 1].sum(1)
    hs, as_ = hq.sum(1), aq.sum(1)
    ot_h, ot_a, n_ot = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    tied = hs + ot_h == as_ + ot_a
    while tied.any():
        m = tied.sum()
        poss = gp[tied] / 9.6
        ot_h[tied] += np.clip(np.rint(rng.normal(poss * h_ortg / 100, np.sqrt(poss * VAR_PP))), 0, None).astype(int)
        ot_a[tied] += np.clip(np.rint(rng.normal(poss * a_ortg / 100, np.sqrt(poss * VAR_PP))), 0, None).astype(int)
        n_ot[tied] += 1
        tied = hs + ot_h == as_ + ot_a
    return dict(hq=hq, aq=aq, ot_h=ot_h, ot_a=ot_a, n_ot=n_ot, h=hs + ot_h, a=as_ + ot_a, pace=gp)


def most_common_score(sim):
    """Most frequent exact final score; ties go to the one nearest the average margin."""
    pairs = pd.DataFrame(dict(h=sim["h"], a=sim["a"]))
    counts = pairs.value_counts()
    top = counts[counts == counts.iloc[0]].index
    mean_m = (sim["h"] - sim["a"]).mean(); mean_t = (sim["h"] + sim["a"]).mean()
    best = min(top, key=lambda p: abs((p[0] - p[1]) - mean_m) + 0.3 * abs((p[0] + p[1]) - mean_t))
    idx = int(np.flatnonzero((sim["h"] == best[0]) & (sim["a"] == best[1]))[0])
    return best, idx, int(counts.iloc[0])


# ---------------------------------------------------------------------------
# Players: recent box scores -> minutes & per-minute rates -> simulated stats
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def player_pool(season, team, ev_ids):
    """Recency-weighted player rates from the team's last 10 finished games, limited to today's roster."""
    frames = []
    for k, eid in enumerate(ev_ids):
        try:
            b = fetch_box(eid)
        except Exception:
            continue
        b = b[b.team == team].copy()
        b["wt"] = 0.9 ** k
        frames.append(b)
    if not frames:
        return pd.DataFrame()
    box = pd.concat(frames)
    box = box[box["min"].fillna(0) > 0]
    n_games = len(frames)
    stats = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]
    rows = []
    for pid, g in box.groupby("player_id"):
        w = g.wt.to_numpy(); mins = g["min"].fillna(0).to_numpy()
        row = dict(player_id=pid, name=g.name.iloc[0], jersey=g.jersey.iloc[0], team=team,
                   gp=len(g), min=float(np.average(mins, weights=w)), avail=len(g) / n_games,
                   starter=g.starter.mean() >= 0.5)
        for s in stats:
            row[s] = float(np.average(g[s].fillna(0), weights=w))
        rows.append(row)
    pool = pd.DataFrame(rows)
    try:
        roster = fetch_roster(team)
        if len(roster):
            pool = pool[pool.player_id.isin(roster.player_id)]
            jer = roster.set_index("player_id").jersey
            pool["jersey"] = pool.player_id.map(jer).fillna(pool.jersey)
    except Exception:
        pass
    return pool.sort_values("min", ascending=False).reset_index(drop=True)


def get_pool(model, team):
    if model["demo"]:
        return demo_players(team)
    ev = model["ev"]
    done = ev[((ev.home == team) | (ev.away == team)) & (ev.state == "post")].sort_values("start", ascending=False)
    ids = tuple(done.event_id.head(10))
    return player_pool(model["season"], team, ids)


def allot_minutes(pool, out_ids):
    p = pool.copy()
    base = p["min"] * np.where(p.avail < 0.35, p.avail, 1.0)
    base[p.player_id.isin(out_ids)] = 0.0
    m = base.to_numpy(float)
    if m.sum() <= 0:
        p["exp_min"] = 0.0
        return p
    for _ in range(12):
        m = m * 240 / m.sum()
        m = np.minimum(m, np.maximum(base.to_numpy() * 1.25 + 4, 0) * (base.to_numpy() > 0))
        m = np.minimum(m, 40)
    m = m * 240 / m.sum()
    p["exp_min"] = m
    return p.sort_values("exp_min", ascending=False).reset_index(drop=True)


def _dirichlet(rng, alpha, n, conc):
    a = np.maximum(alpha * conc, 1e-3)
    g = rng.gamma(a, size=(n, len(alpha)))
    return g / g.sum(1, keepdims=True)


def sim_players(p, team_pts, seed=None):
    """Split each simulated team score into player stat lines."""
    rng = np.random.default_rng(seed)
    p = p[p.exp_min > 0.5].reset_index(drop=True)
    n = len(team_pts)
    m = p.exp_min.to_numpy()
    rate = {s: (p[s] / p["min"].clip(lower=1)).to_numpy() for s in ["pts", "reb", "ast", "fg3m", "stl", "blk"]}
    exp = {s: m * rate[s] for s in rate}
    out = {}
    tp = np.asarray(team_pts, float)
    scale = tp / max(tp.mean(), 1)
    sh = exp["pts"] / max(exp["pts"].sum(), 1e-6)
    out["pts"] = np.rint(tp[:, None] * _dirichlet(rng, sh, n, 45))
    treb = exp["reb"].sum() * rng.normal(1, 0.09, n)
    out["reb"] = np.rint(treb[:, None] * _dirichlet(rng, exp["reb"] / max(exp["reb"].sum(), 1e-6), n, 35))
    tast = exp["ast"].sum() * scale ** 0.9 * rng.normal(1, 0.08, n)
    out["ast"] = np.rint(tast[:, None] * _dirichlet(rng, exp["ast"] / max(exp["ast"].sum(), 1e-6), n, 30))
    t3 = exp["fg3m"].sum() * scale ** 0.8 * rng.normal(1, 0.15, n)
    out["fg3m"] = np.rint(np.clip(t3, 0, None)[:, None] * _dirichlet(rng, exp["fg3m"] / max(exp["fg3m"].sum(), 1e-6), n, 20))
    out["stl"] = rng.poisson(np.tile(exp["stl"], (n, 1)))
    out["blk"] = rng.poisson(np.tile(exp["blk"], (n, 1)))
    out["pra"] = out["pts"] + out["reb"] + out["ast"]
    return p, out


def projected_box(p, stats):
    df = p[["name", "jersey", "exp_min"]].copy()
    for s in ["pts", "reb", "ast", "fg3m", "stl", "blk"]:
        df[s] = stats[s].mean(0)
    return df.rename(columns=dict(exp_min="MIN", pts="PTS", reb="REB", ast="AST", fg3m="3PM", stl="STL", blk="BLK",
                                  name="Player", jersey="#"))


# ---------------------------------------------------------------------------
# Play-by-play for the animation (built to land exactly on the chosen score)
# ---------------------------------------------------------------------------
OUTCOMES = {"tov": 0, "miss2": 0, "miss3": 0, "ft1": 1, "ft2": 2, "make2": 2, "make3": 3, "and1": 3}
UP = [("miss2", "make2"), ("miss3", "make3"), ("miss2", "ft2"), ("miss2", "ft1"), ("ft1", "ft2"),
      ("make2", "and1"), ("make2", "make3"), ("tov", "make2"), ("tov", "ft1")]
DOWN = [(b, a) for a, b in UP]


def _outcome_probs(three_rate):
    s = three_rate / 0.42
    base = dict(tov=0.13, miss2=0.26 * (2 - s) / 1, miss3=0.19 * s, ft1=0.03, ft2=0.07,
                make2=0.20 * (2 - s) / 1, make3=0.12 * s, and1=0.015)
    tot = sum(base.values())
    return {k: v / tot for k, v in base.items()}


def _quarter_outcomes(rng, n, target, probs):
    keys = list(probs)
    n = max(n, math.ceil(target / 3) + 1)
    res = list(rng.choice(keys, size=n, p=[probs[k] for k in keys]))
    for _ in range(500):
        diff = target - sum(OUTCOMES[x] for x in res)
        if diff == 0:
            break
        moves = UP if diff > 0 else DOWN
        ok = [(i, b) for i, x in enumerate(res) for a, b in moves
              if x == a and 0 < abs(OUTCOMES[b] - OUTCOMES[a]) <= abs(diff)]
        if not ok:
            if diff > 0:
                res.append("make2" if diff >= 2 else "ft1")
            else:
                i = next((i for i, x in enumerate(res) if OUTCOMES[x] > 0), None)
                if i is None:
                    break
                res[i] = "miss2"
            continue
        big = max(abs(OUTCOMES[b] - OUTCOMES[res[i]]) for i, b in ok)
        pick = [(i, b) for i, b in ok if abs(OUTCOMES[b] - OUTCOMES[res[i]]) == big] if abs(diff) >= 2 else ok
        i, b = pick[rng.integers(len(pick))]
        res[i] = b
    return res


SHOT_TEXT = {
    "make3": ["{s} drills a three from the wing!", "{s} pulls up from deep… SPLASH!", "Corner three by {s} — good!",
              "{s} with the step-back three!"],
    "make2": ["{s} drives and lays it in.", "{s} THROWS IT DOWN!", "Pull-up jumper by {s}, good.",
              "{s} spins in the paint and scores.", "{s} with the floater — good!"],
    "and1": ["{s} scores through the contact — AND ONE!"],
    "miss3": ["{s} fires from three… off the rim.", "{s} misses the long three."],
    "miss2": ["{s} misses in the lane.", "{s}'s jumper rims out.", "{s} can't finish at the rim."],
    "ft2": ["{s} gets fouled and hits both free throws."], "ft1": ["{s} splits a pair at the line."],
    "tov": ["Turnover by {s}!", "{s} throws it away.", "Stolen from {s}!"],
}


def build_pbp(sim, idx, pool_h, pool_a, style_h, style_a, mu_margin, seed=7):
    """A full possession-by-possession game whose quarter scores match simulation #idx."""
    rng = np.random.default_rng(seed)
    q_h = list(sim["hq"][idx]); q_a = list(sim["aq"][idx])
    n_ot = int(sim["n_ot"][idx])
    if n_ot:
        # spread OT points across OT periods
        oh, oa = int(sim["ot_h"][idx]), int(sim["ot_a"][idx])
        for k in range(n_ot):
            last = k == n_ot - 1
            if last:
                q_h.append(oh); q_a.append(oa)
            else:
                v = int(rng.integers(8, 14)); q_h.append(v); q_a.append(v); oh -= v; oa -= v
    pace = float(sim["pace"][idx])
    teams = {"H": pool_h[pool_h.exp_min > 0].reset_index(drop=True),
             "A": pool_a[pool_a.exp_min > 0].reset_index(drop=True)}
    probs = {"H": _outcome_probs(style_h["three_rate"]), "A": _outcome_probs(style_a["three_rate"])}

    def five(side, q, frac, margin):
        t = teams[side]; ids = list(range(min(len(t), 10)))
        st5, bench = ids[:5], ids[5:]
        if q == 3 and frac > 0.55 and abs(margin) >= 22 and len(bench) >= 5:
            return bench[:5]
        if len(bench) >= 3 and (0.5 < frac < 0.85 and q in (0, 2) or q == 1 and frac < 0.5):
            return st5[:3] + bench[:2] if q != 1 else st5[:2] + bench[:3]
        return st5

    def pick(side, on, col, excl=None):
        t = teams[side]
        w = np.array([max(t.loc[i, col] / max(t.loc[i, "min"], 1), 1e-3) for i in on])
        if excl is not None and excl in on:
            w[on.index(excl)] = 0
        w = w / w.sum()
        return on[int(rng.choice(len(on), p=w))]

    events = []
    sh = sa = 0
    tip_winner = "H" if rng.random() < 0.5 else "A"
    other = {"H": "A", "A": "H"}
    total_q = len(q_h)
    for q in range(total_q):
        qlen = 720 if q < 4 else 300
        n_base = max(int(round(pace / (4 if q < 4 else 9.6))), 6)
        first = tip_winner if q in (0, 3) else other[tip_winner]
        if q >= 4:
            first = "H" if rng.random() < 0.5 else "A"
        n_first = n_base; n_second = n_base - int(rng.random() < 0.5)
        outs = {first: _quarter_outcomes(rng, n_first, int((q_h if first == "H" else q_a)[q]), probs[first])}
        sec = other[first]
        outs[sec] = _quarter_outcomes(rng, n_second, int((q_h if sec == "H" else q_a)[q]), probs[sec])
        order = []
        a_, b_ = list(outs[first]), list(outs[sec])
        while a_ or b_:
            if a_:
                order.append((first, a_.pop(0)))
            if b_:
                order.append((sec, b_.pop(0)))
        dur = rng.dirichlet(np.full(len(order), 6.0)) * qlen
        clock = float(qlen)
        for k, (side, oc) in enumerate(order):
            clock = max(clock - dur[k], 0.0)
            frac = 1 - clock / qlen
            on_h, on_a = five("H", q, frac, sh - sa), five("A", q, frac, sh - sa)
            on = on_h if side == "H" else on_a
            don = on_a if side == "H" else on_h
            dside = other[side]
            shooter = pick(side, on, "pts" if oc != "tov" else "tov")
            ast = pick(side, on, "ast", excl=shooter) if oc in ("make2", "make3", "and1") and rng.random() < 0.62 and len(on) > 1 else None
            reb = None; reb_side = None; stl = None; blk = None
            if oc in ("miss2", "miss3"):
                if rng.random() < 0.24:
                    reb_side = side; reb = pick(side, on, "reb")
                else:
                    reb_side = dside; reb = pick(dside, don, "reb")
                if oc == "miss2" and rng.random() < 0.12:
                    blk = pick(dside, don, "blk")
            if oc == "tov" and rng.random() < 0.55:
                stl = pick(dside, don, "stl")
            pts = OUTCOMES[oc]
            lead_before = sh - sa
            if side == "H":
                sh += pts
            else:
                sa += pts
            t = teams[side]; s_name = t.loc[shooter, "name"]
            text = random.Random(int(rng.integers(1e9))).choice(SHOT_TEXT[oc]).format(s=s_name)
            if ast is not None:
                text += f" (assist {t.loc[ast, 'name'].split()[-1]})"
            if blk is not None:
                text = f"{s_name} gets STUFFED by {teams[dside].loc[blk, 'name']}!"
            if stl is not None:
                text = f"{s_name} loses it — {teams[dside].loc[stl, 'name']} with the steal!"
            dunk = oc in ("make2", "and1") and "DOWN" in text or "AND ONE" in text
            remaining = sum(720 if j < 4 else 300 for j in range(q + 1, total_q)) + clock
            frac_left = remaining / 2880
            if q == total_q - 1 and k == len(order) - 1:
                wp = 1.0 if sh > sa else 0.0
            else:
                wp = phi(((sh - sa) + mu_margin * frac_left) / (MARGIN_SD * math.sqrt(max(frac_left, 1e-4)) + 0.5))
            lead_after = sh - sa
            big = (oc == "make3" and q >= 3 and np.sign(lead_after) != np.sign(lead_before) and lead_after != 0) or \
                  (q >= 3 and clock < 30 and pts > 0 and abs(lead_after) <= 3)
            events.append(dict(q=q + 1, clock=round(clock, 1), side=side, type=oc, pts=pts,
                               shooter=int(shooter), ast=None if ast is None else int(ast),
                               reb=None if reb is None else int(reb), reb_side=reb_side,
                               stl=None if stl is None else int(stl), blk=None if blk is None else int(blk),
                               sh=sh, sa=sa, wp=round(wp, 4), text=text,
                               on_h=[int(x) for x in on_h], on_a=[int(x) for x in on_a],
                               dunk=bool(dunk), big=bool(big)))
    return events, teams


def pbp_box(events, teams):
    rows = []
    for side in ("H", "A"):
        t = teams[side]
        for i in range(len(t)):
            rows.append(dict(side=side, idx=i, Player=t.loc[i, "name"], **{"#": t.loc[i, "jersey"]},
                             PTS=0, REB=0, AST=0, **{"3PM": 0}, STL=0, BLK=0))
    box = pd.DataFrame(rows).set_index(["side", "idx"])
    for e in events:
        box.loc[(e["side"], e["shooter"]), "PTS"] += e["pts"]
        if e["type"] == "make3":
            box.loc[(e["side"], e["shooter"]), "3PM"] += 1
        if e["ast"] is not None:
            box.loc[(e["side"], e["ast"]), "AST"] += 1
        if e["reb"] is not None:
            box.loc[(e["reb_side"], e["reb"]), "REB"] += 1
        d = "A" if e["side"] == "H" else "H"
        if e["stl"] is not None:
            box.loc[(d, e["stl"]), "STL"] += 1
        if e["blk"] is not None:
            box.loc[(d, e["blk"]), "BLK"] += 1
    box["GS"] = box.PTS + 0.7 * box.REB + 0.9 * box.AST + 1.2 * box.STL + 1.0 * box.BLK
    return box.reset_index()


# ---------------------------------------------------------------------------
# Season + playoff simulator (with the play-in)
# ---------------------------------------------------------------------------
def game_prob(model, h, a, neutral=False):
    r = model["ratings"]
    pace = r.loc[h, "pace"] + r.loc[a, "pace"] - model["P"]
    mu = (r.loc[h, "net"] - r.loc[a, "net"]) * pace / 100 + (0 if neutral else HCA_PTS + ALTITUDE.get(h, 0))
    return phi(mu / MARGIN_SD)


def series_prob(p_home, p_away, need=4):
    """Higher seed has home court in games 1,2,5,7 (2-2-1-1-1)."""
    home_games = [1, 1, 0, 0, 1, 0, 1]
    from functools import lru_cache

    @lru_cache(None)
    def f(w, l):
        if w == need:
            return 1.0
        if l == need:
            return 0.0
        g = w + l
        p = p_home if home_games[g] else p_away
        return p * f(w + 1, l) + (1 - p) * f(w, l + 1)
    return f(0, 0)


@st.cache_data(ttl=1800, show_spinner="Simulating the rest of the season…")
def season_sim(_model, n=2000, seed=11, key=None):
    model = _model
    ev = model["ev"]; r = model["ratings"]
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(ABBRS)}
    rem = ev[(ev.season_type == 2) & (ev.state != "post")]
    wins0 = r.w.reindex(ABBRS).to_numpy(float)
    if len(rem):
        p = np.array([game_prob(model, h, a) for h, a in zip(rem.home, rem.away)])
        hw = rng.random((n, len(rem))) < p
        H = np.zeros((len(rem), 30)); A = np.zeros((len(rem), 30))
        H[np.arange(len(rem)), rem.home.map(idx)] = 1
        A[np.arange(len(rem)), rem.away.map(idx)] = 1
        wins = wins0 + hw @ H + (~hw) @ A
    else:
        wins = np.tile(wins0, (n, 1))
    wins = wins + rng.random(wins.shape) * 0.01
    confs = {c: [t for t in ABBRS if TEAMS[t]["conf"] == c] for c in ("E", "W")}
    cnt = {k: np.zeros(30) for k in ("playoffs", "playin", "top_seed", "r2", "cf", "finals", "title", "top6")}
    seed_sum = np.zeros(30)
    sp_cache = {}

    def sp(hi_t, lo_t):
        key_ = (hi_t, lo_t)
        if key_ not in sp_cache:
            sp_cache[key_] = series_prob(game_prob(model, hi_t, lo_t), 1 - game_prob(model, lo_t, hi_t))
        return sp_cache[key_]

    for s in range(n):
        finalists = []
        for c, teams in confs.items():
            ci = [idx[t] for t in teams]
            order = [teams[j] for j in np.argsort(-wins[s, ci])]
            for k, t in enumerate(order):
                seed_sum[idx[t]] += k + 1
            cnt["top_seed"][idx[order[0]]] += 1
            for t in order[:6]:
                cnt["top6"][idx[t]] += 1
            for t in order[6:10]:
                cnt["playin"][idx[t]] += 1
            s7, s8, s9, s10 = order[6:10]
            w78 = s7 if rng.random() < game_prob(model, s7, s8) else s8
            l78 = s8 if w78 == s7 else s7
            w910 = s9 if rng.random() < game_prob(model, s9, s10) else s10
            seed8 = l78 if rng.random() < game_prob(model, l78, w910) else w910
            seeds = order[:6] + [w78, seed8]
            for t in seeds:
                cnt["playoffs"][idx[t]] += 1
            rank = {t: k for k, t in enumerate(seeds)}

            def play(x, y):
                hi_t, lo_t = (x, y) if rank[x] < rank[y] else (y, x)
                return hi_t if rng.random() < sp(hi_t, lo_t) else lo_t
            r1 = [play(seeds[0], seeds[7]), play(seeds[3], seeds[4]), play(seeds[1], seeds[6]), play(seeds[2], seeds[5])]
            for t in r1:
                cnt["r2"][idx[t]] += 1
            r2 = [play(r1[0], r1[1]), play(r1[2], r1[3])]
            for t in r2:
                cnt["cf"][idx[t]] += 1
            champ = play(r2[0], r2[1])
            cnt["finals"][idx[champ]] += 1
            finalists.append(champ)
        a, b = finalists
        hi_t, lo_t = (a, b) if wins[s, idx[a]] >= wins[s, idx[b]] else (b, a)
        cnt["title"][idx[hi_t if rng.random() < sp(hi_t, lo_t) else lo_t]] += 1
    out = pd.DataFrame({k: v / n * 100 for k, v in cnt.items()}, index=ABBRS)
    out["proj_w"] = wins.mean(0).round(1)
    out["proj_l"] = (82 - wins.mean(0)).round(1)
    out["avg_seed"] = seed_sum / n
    out["w"] = r.w; out["l"] = r.l
    out["conf"] = [TEAMS[t]["conf"] for t in ABBRS]
    out["net"] = r.net.round(1)
    return out


# ---------------------------------------------------------------------------
# Saving data (predictions, bets, pick'em): GitHub repo if secrets set, else local files
# ---------------------------------------------------------------------------
TABLES = {
    "predictions.csv": ["date", "event_id", "home", "away", "home_win", "proj_home", "proj_away", "model_margin",
                        "model_total", "mkt_spread", "mkt_total", "pick", "saved_at"],
    "bets.csv": ["id", "placed", "date", "event_id", "game", "kind", "pick", "line", "odds", "stake", "model_prob", "detail",
                 "status", "profit"],
    "pickem.csv": ["date", "event_id", "name", "pick", "time"],
    "odds_history.csv": ["date", "team", "playoffs", "top_seed", "title"],
    "player_games.csv": ["event_id", "date", "team", "player_id", "name", "min", "pts", "reb", "ast", "fg3m", "stl",
                         "blk", "tov"],
}
STR_COLS = {"event_id": str, "player_id": str, "id": str, "date": str, "detail": str}


def _gh():
    tok, repo = secret("GH_TOKEN"), secret("GH_REPO")
    return (tok, repo) if tok and repo and not ROBOT_MODE else None


def _gh_headers(tok):
    return {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}


def load_table(name):
    cfg = _gh()
    if cfg:
        try:
            tok, repo = cfg
            r = requests.get(f"https://api.github.com/repos/{repo}/contents/data/{name}",
                             headers={**_gh_headers(tok), "Accept": "application/vnd.github.raw"}, timeout=10)
            if r.status_code == 200:
                return pd.read_csv(io.StringIO(r.text), dtype=STR_COLS)
            if r.status_code == 404:
                return pd.DataFrame(columns=TABLES[name])
        except Exception:
            pass
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        try:
            return pd.read_csv(path, dtype=STR_COLS)
        except Exception:
            pass
    return pd.DataFrame(columns=TABLES[name])


def save_table(name, df, msg="update data"):
    os.makedirs(DATA_DIR, exist_ok=True)
    text = df.to_csv(index=False)
    with open(os.path.join(DATA_DIR, name), "w", encoding="utf-8", newline="") as f:
        f.write(text)
    cfg = _gh()
    if not cfg:
        return "local"
    tok, repo = cfg
    url = f"https://api.github.com/repos/{repo}/contents/data/{name}"
    try:
        r = requests.get(url, headers=_gh_headers(tok), timeout=10)
        sha = r.json().get("sha") if r.status_code == 200 else None
        body = {"message": msg, "content": base64.b64encode(text.encode()).decode()}
        if sha:
            body["sha"] = sha
        r = requests.put(url, headers=_gh_headers(tok), json=body, timeout=15)
        return "github" if r.status_code in (200, 201) else f"local (GitHub said {r.status_code})"
    except Exception as e:
        return f"local ({e.__class__.__name__})"


# ---------------------------------------------------------------------------
# Season player stats (built by the robot from box scores; used for injuries + races)
# ---------------------------------------------------------------------------
def update_player_games(ev, max_new=400, progress=None):
    pg = load_table("player_games.csv")
    have = set(pg.event_id.astype(str)) if len(pg) else set()
    todo = ev[(ev.state == "post") & (ev.season_type == 2) & (~ev.event_id.isin(have))]
    todo = todo.sort_values("start").head(max_new)
    frames = []
    for k, (eid, d) in enumerate(zip(todo.event_id, todo.date)):
        try:
            b = fetch_box(eid)
            if len(b):
                b["date"] = str(d)
                frames.append(b[TABLES["player_games.csv"]])
        except Exception:
            pass
        if progress:
            progress((k + 1) / max(len(todo), 1))
    if frames:
        pg = pd.concat([pg] + frames, ignore_index=True)
        save_table("player_games.csv", pg, "player box scores")
    return pg, len(frames)


def season_player_table(pg, team=None):
    if pg is None or len(pg) == 0:
        return pd.DataFrame()
    d = pg.copy()
    for c in ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    if team:
        d = d[d.team == team]
    g = d.groupby(["player_id", "name", "team"])
    out = g[["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]].mean()
    out["gp"] = g.size()
    out = out.reset_index()
    tg = d.groupby("team").event_id.nunique()
    out["avail"] = out.gp / out.team.map(tg).clip(lower=1)
    return out


def slate_injury_penalties(model, inj, pg):
    """Margin hit per team from players listed OUT (needs season player stats)."""
    pen = {t: 0.0 for t in ABBRS}
    if inj is None or len(inj) == 0 or pg is None or len(pg) == 0:
        return pen
    table = season_player_table(pg)
    outs = inj[inj.status.str.lower().str.contains("out", na=False)]
    for t in ABBRS:
        ids = list(outs[outs.team == t].player_id)
        pen[t] = injury_penalty(table[table.team == t], ids)
    return pen


# ---------------------------------------------------------------------------
# Game predictions for a day's slate
# ---------------------------------------------------------------------------
def predict_slate(model, games, inj_pen=None):
    rows = []
    for _, g in games.iterrows():
        fh, fa = fatigue(model["ev"], g.home, g.event_id), fatigue(model["ev"], g.away, g.event_id)
        ih = (inj_pen or {}).get(g.home, 0.0); ia = (inj_pen or {}).get(g.away, 0.0)
        pr = project(model, g.home, g.away, fh["pts"], fa["pts"], bool(g.get("neutral", False)), ih, ia)
        row = dict(g)
        row.update(home_win=pr["win"], proj_home=pr["h_pts"], proj_away=pr["a_pts"], model_margin=pr["margin"],
                   model_total=pr["total"], fat_h=fh, fat_a=fa, inj_h=ih, inj_a=ia,
                   pick=g.home if pr["win"] >= 0.5 else g.away)
        rows.append(row)
    return pd.DataFrame(rows)


def picks_of_day(pred):
    """Lock, upset alert and coin flip."""
    if pred is None or len(pred) == 0:
        return {}
    p = pred[pred.home_win.notna()].copy()
    if len(p) == 0:
        return {}
    p["fav_prob"] = np.maximum(p.home_win, 1 - p.home_win)
    out = {"lock": p.loc[p.fav_prob.idxmax()]}
    p["mkt_home"] = p.apply(lambda r: american_to_prob(r.ml_home) / (american_to_prob(r.ml_home) + american_to_prob(r.ml_away))
                            if pd.notna(r.get("ml_home")) and pd.notna(r.get("ml_away"))
                            else (phi(-r.mkt_spread / MARGIN_SD) if pd.notna(r.get("mkt_spread")) else np.nan), axis=1)
    dog = []
    for i, r in p.iterrows():
        if pd.isna(r.mkt_home):
            continue
        if r.mkt_home < 0.5 and r.home_win > r.mkt_home:
            dog.append((r.home_win - r.mkt_home, i, r.home, r.home_win))
        if r.mkt_home > 0.5 and (1 - r.home_win) > (1 - r.mkt_home):
            dog.append(((1 - r.home_win) - (1 - r.mkt_home), i, r.away, 1 - r.home_win))
    if dog:
        d = max(dog)
        if d[3] >= 0.35:
            out["upset"] = (p.loc[d[1]], d[2], d[3])
    out["coin"] = p.loc[(p.home_win - 0.5).abs().idxmin()]
    return out


def grade_predictions(preds, ev):
    if len(preds) == 0:
        return preds
    res = ev[ev.state == "post"].set_index("event_id")[["home_pts", "away_pts"]]
    p = preds.copy()
    p["event_id"] = p.event_id.astype(str)
    p = p.join(res, on="event_id")
    for c in ["home_win", "model_margin", "model_total", "mkt_spread", "mkt_total"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["final"] = p.home_pts.notna()
    p["winner"] = np.where(p.home_pts > p.away_pts, p.home, p.away)
    p["correct"] = np.where(p.final, p.pick == p.winner, np.nan)
    margin = p.home_pts - p.away_pts
    edge = p.model_margin + p.mkt_spread          # >0 -> model likes home vs the spread
    cover = margin + p.mkt_spread
    p["ats"] = np.where(~p.final | p.mkt_spread.isna(), None,
                        np.where(cover == 0, "push", np.where((edge > 0) == (cover > 0), "win", "loss")))
    p["brier"] = (p.home_win - (margin > 0).astype(float)) ** 2
    p.loc[~p.final, "brier"] = np.nan
    return p


# ---------------------------------------------------------------------------
# The Odds API (optional key) – best lines + player props
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def odds_api_games(key):
    js = http_json("https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
                   {"apiKey": key, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"})
    rows = []
    for g in js:
        h, a = team_abbr(None, g["home_team"]), team_abbr(None, g["away_team"])
        for bk in g.get("bookmakers", []):
            row = dict(odds_id=g["id"], home=h, away=a, book=bk["title"],
                       date=pd.Timestamp(g["commence_time"]).tz_convert(ET).date())
            for m in bk.get("markets", []):
                for o in m.get("outcomes", []):
                    side = "h" if o["name"] == g["home_team"] else ("a" if o["name"] == g["away_team"] else o["name"].lower())
                    if m["key"] == "h2h":
                        row[f"ml_{side}"] = o["price"]
                    elif m["key"] == "spreads":
                        row[f"sp_{side}"] = o.get("point"); row[f"sp_{side}_price"] = o["price"]
                    elif m["key"] == "totals":
                        row["total"] = o.get("point"); row[f"{side}_price"] = o["price"]
            rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(ttl=900, show_spinner=False)
def odds_api_props(key, odds_id):
    js = http_json(f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{odds_id}/odds",
                   {"apiKey": key, "regions": "us", "oddsFormat": "american",
                    "markets": "player_points,player_rebounds,player_assists,player_threes"})
    rows = []
    for bk in js.get("bookmakers", []):
        for m in bk.get("markets", []):
            for o in m.get("outcomes", []):
                rows.append(dict(book=bk["title"], market=m["key"], player=o.get("description"), side=o["name"],
                                 line=o.get("point"), price=o["price"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Game animation (HTML canvas)
# ---------------------------------------------------------------------------
ANIM_HTML = r"""
<div id="wrap" style="font-family:Inter,system-ui,sans-serif;color:#e8eaf0;background:#0b0f17;border-radius:14px;padding:8px;">
  <canvas id="cv" width="1000" height="600" style="width:100%;height:auto;border-radius:10px;display:block;"></canvas>
  <div style="display:flex;flex-wrap:wrap;gap:6px;margin:8px 0;align-items:center;">
    <button id="play" class="b">⏸ Pause</button><button id="restart" class="b">⟲ Restart</button>
    <span style="margin-left:4px">Speed</span>
    <select id="speed" class="b"><option>0.5</option><option selected>1</option><option>2</option><option>4</option><option>8</option></select>
    <span id="qbtns"></span><button id="end" class="b">⏭ Final</button>
  </div>
  <div id="log" style="height:118px;overflow:auto;font-size:14px;line-height:1.5;background:#121826;border-radius:8px;padding:6px 10px;"></div>
</div>
<style>.b{background:#1b2333;color:#e8eaf0;border:1px solid #2c3850;border-radius:8px;padding:5px 10px;font-size:13px;cursor:pointer}
.b:hover{background:#26314a}</style>
<script>
const D = __DATA__;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const W = 1000, CT = 70, CB = 550, CL = 50, CR = 950, PPF = 900/94, MIDY = (CT+CB)/2;
const BASK = {L: CL + 5.25*PPF, R: CR - 5.25*PPF};
const imgs = {};
for (const s of ['H','A']) { const im = new Image(); im.src = D[s].logo; imgs[s] = im; }
let idx = 0, t = 0, playing = true, speed = 1, last = performance.now(), pause = 0, banner = '';
let score = {H:0, A:0}, conf = [], floaters = [], flash = 0;
const BASE = 2300;
function dirOf(side, q){ const home = (q<=2 || q>=5) ? 'R' : 'L'; return side==='H' ? home : (home==='R'?'L':'R'); }
const OFF = [[27,0],[21,-15],[21,15],[3,-21],[8,7]];
function spot(basket, dx, dy){ const bx = BASK[basket]; const s = basket==='R'?-1:1; return [bx + s*dx*PPF, MIDY + dy*PPF]; }
function forms(e){
  const b = dirOf(e.side, e.q); const pos = {H:[],A:[]};
  const off = e.side, def = off==='H'?'A':'H';
  const onOff = off==='H'?e.on_h:e.on_a, onDef = off==='H'?e.on_a:e.on_h;
  onOff.forEach((p,i)=>{ const [dx,dy]=OFF[i%5]; pos[off][i]=spot(b,dx+Math.sin(e.clock+i)*1.5,dy+Math.cos(e.clock*1.3+i)*1.5); });
  onDef.forEach((p,i)=>{ const [dx,dy]=OFF[i%5]; pos[def][i]=spot(b,dx*0.72,dy*0.8); });
  return pos;
}
function holderPos(e, pos, side, pid){ const on = side==='H'?e.on_h:e.on_a; let i = on.indexOf(pid); if(i<0) i=0; return pos[side][i]; }
function shotSpot(e){
  const b = dirOf(e.side, e.q); const ang = ((e.clock*37)%140-70)*Math.PI/180;
  if (e.type==='make3'||e.type==='miss3') return spot(b, 24*Math.cos(ang)+1, 24*Math.sin(ang));
  if (e.type==='ft1'||e.type==='ft2') return spot(b, 13.75, 0);
  if (e.dunk) return spot(b, 2, (e.clock%3)-1.5);
  return spot(b, 6 + (e.clock%9), ((e.clock*7)%20)-10);
}
function lerp(a,b,k){ return [a[0]+(b[0]-a[0])*k, a[1]+(b[1]-a[1])*k]; }
function ease(k){ return k<.5 ? 2*k*k : 1-Math.pow(-2*k+2,2)/2; }
let prevPos = null;
function fmtClock(s){ s=Math.max(0,s); const m=Math.floor(s/60), r=Math.floor(s%60); return m+':'+(r<10?'0':'')+r; }
function qname(q){ return q<=4 ? 'Q'+q : (q===5?'OT':'OT'+(q-4)); }
function drawCourt(){
  const g = ctx.createLinearGradient(0,CT,0,CB); g.addColorStop(0,'#c8995c'); g.addColorStop(1,'#b8874b');
  ctx.fillStyle = g; ctx.fillRect(CL,CT,CR-CL,CB-CT);
  ctx.strokeStyle='rgba(255,255,255,.85)'; ctx.lineWidth=2; ctx.strokeRect(CL,CT,CR-CL,CB-CT);
  ctx.beginPath(); ctx.moveTo(W/2,CT); ctx.lineTo(W/2,CB); ctx.stroke();
  ctx.beginPath(); ctx.arc(W/2,MIDY,6*PPF,0,2*Math.PI); ctx.stroke();
  if (imgs.H.complete && imgs.H.naturalWidth){ ctx.globalAlpha=.35; ctx.drawImage(imgs.H, W/2-50, MIDY-50, 100, 100); ctx.globalAlpha=1; }
  for (const b of ['L','R']){
    const s = b==='L'?1:-1, x0 = b==='L'?CL:CR, bx = BASK[b];
    ctx.fillStyle = b==='L' ? hexA(D.H.c1,.35) : hexA(D.H.c1,.35);
    ctx.fillRect(b==='L'?CL:CR-19*PPF, MIDY-8*PPF, 19*PPF, 16*PPF);
    ctx.strokeRect(b==='L'?CL:CR-19*PPF, MIDY-8*PPF, 19*PPF, 16*PPF);
    ctx.beginPath(); ctx.arc(x0+s*19*PPF, MIDY, 6*PPF, b==='L'?-Math.PI/2:Math.PI/2, b==='L'?Math.PI/2:3*Math.PI/2); ctx.stroke();
    const ang = Math.asin(22/23.75);
    ctx.beginPath(); ctx.moveTo(x0, MIDY-22*PPF); ctx.lineTo(bx+s*8.95*PPF, MIDY-22*PPF);
    if (b==='L') ctx.arc(bx, MIDY, 23.75*PPF, -ang, ang, false); else ctx.arc(bx, MIDY, 23.75*PPF, Math.PI+ang, Math.PI-ang, true);
    ctx.lineTo(x0, MIDY+22*PPF); ctx.stroke();
    ctx.strokeStyle='#ff6a00'; ctx.lineWidth=3; ctx.beginPath(); ctx.arc(bx, MIDY, 0.75*PPF+2, 0, 2*Math.PI); ctx.stroke();
    ctx.strokeStyle='#fff'; ctx.beginPath(); ctx.moveTo(x0+s*4*PPF, MIDY-3*PPF); ctx.lineTo(x0+s*4*PPF, MIDY+3*PPF); ctx.stroke();
    ctx.lineWidth=2; ctx.strokeStyle='rgba(255,255,255,.85)';
  }
}
function hexA(h,a){ h=h.replace('#',''); if(h.length===3) h=h.split('').map(c=>c+c).join(''); const n=parseInt(h,16); return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`; }
function drawPlayer(p, side, pid, hasBall){
  const T = D[side];
  ctx.beginPath(); ctx.arc(p[0],p[1],14,0,2*Math.PI); ctx.fillStyle=T.c1; ctx.fill();
  ctx.lineWidth=3; ctx.strokeStyle= hasBall ? '#ffd34d' : T.c2; ctx.stroke(); ctx.lineWidth=2;
  ctx.fillStyle = lum(T.c1) > 0.6 ? '#111' : '#fff'; ctx.font='bold 12px system-ui'; ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText(T.players[pid] ? T.players[pid].j : '', p[0], p[1]+1);
}
function lum(h){ h=h.replace('#',''); if(h.length===3) h=h.split('').map(c=>c+c).join(''); const n=parseInt(h,16); return (0.299*(n>>16&255)+0.587*(n>>8&255)+0.114*(n&255))/255; }
function drawBoard(e){
  ctx.fillStyle='#0f1522'; ctx.fillRect(0,0,W,62);
  for (const [s,x,al] of [['A',20,'left'],['H',W-20,'right']]){
    const T = D[s]; const lx = s==='A'? x : x-44;
    if (imgs[s].complete && imgs[s].naturalWidth) ctx.drawImage(imgs[s], lx, 9, 44, 44);
    ctx.fillStyle='#fff'; ctx.font='bold 22px system-ui'; ctx.textAlign=al; ctx.textBaseline='middle';
    ctx.fillText(T.abbr, s==='A'? x+54 : x-54, 22);
    ctx.font='12px system-ui'; ctx.fillStyle='#9aa6bf'; ctx.fillText(s==='A'?'AWAY':'HOME', s==='A'? x+54 : x-54, 44);
    ctx.font='bold 38px system-ui'; ctx.fillStyle = T.c1==='#000000'? '#fff' : '#fff';
    ctx.fillText(score[s], s==='A'? x+150 : x-150, 32);
  }
  ctx.textAlign='center'; ctx.fillStyle='#ffd34d'; ctx.font='bold 22px system-ui';
  ctx.fillText(e ? qname(e.q)+'  '+fmtClock(e.clock) : 'TIP-OFF', W/2, 24);
  ctx.font='13px system-ui'; ctx.fillStyle='#9aa6bf'; ctx.fillText('Most common outcome of '+D.nsims+' sims', W/2, 48);
}
function drawWP(e){
  const y=562, h=22, wp = e ? e.wp : D.wp0;
  ctx.fillStyle='#0f1522'; ctx.fillRect(0,552,W,48);
  ctx.fillStyle=D.A.c1; ctx.fillRect(CL,y,(CR-CL)*(1-wp),h);
  ctx.fillStyle=D.H.c1; ctx.fillRect(CL+(CR-CL)*(1-wp),y,(CR-CL)*wp,h);
  ctx.strokeStyle='#2c3850'; ctx.strokeRect(CL,y,CR-CL,h);
  ctx.fillStyle='#fff'; ctx.font='bold 13px system-ui'; ctx.textBaseline='middle';
  ctx.textAlign='left'; ctx.fillText(D.A.abbr+' '+Math.round((1-wp)*100)+'%', CL+6, y+h/2);
  ctx.textAlign='right'; ctx.fillText(Math.round(wp*100)+'% '+D.H.abbr, CR-6, y+h/2);
  ctx.textAlign='center'; ctx.fillStyle='#9aa6bf'; ctx.font='11px system-ui'; ctx.fillText('WIN PROBABILITY', W/2, y+h/2);
}
function burst(x,y,n,cols){ for(let i=0;i<n;i++) conf.push({x,y,vx:(Math.random()-.5)*9,vy:-Math.random()*9-2,c:cols[i%cols.length],life:90}); }
function logLine(e){
  const el = document.getElementById('log'); const T = D[e.side];
  const d = document.createElement('div');
  d.innerHTML = `<span style="color:#9aa6bf">${qname(e.q)} ${fmtClock(e.clock)}</span> <b style="color:${lum(T.c1)<0.15?T.c2:T.c1}">${T.abbr}</b> ${e.text} <span style="float:right;color:#9aa6bf">${D.A.abbr} ${e.sa} – ${e.sh} ${D.H.abbr}</span>`;
  el.prepend(d);
}
function setTo(i){
  idx = i; t = 0; conf=[]; floaters=[]; banner='';
  const p = D.events[i-1]; score = {H: p?p.sh:0, A: p?p.sa:0};
  document.getElementById('log').innerHTML=''; prevPos=null;
  for (let k=Math.max(0,i-6); k<i; k++) logLine(D.events[k]);
}
function frame(now){
  let dtm = Math.min(now-last, 60); last = now;
  if (playing) { if (pause>0) pause -= dtm*speed; else t += dtm*speed; }
  ctx.clearRect(0,0,W,600); drawCourt();
  const e = D.events[Math.min(idx, D.events.length-1)];
  const done = idx >= D.events.length;
  const pos = forms(e);
  if (!prevPos) prevPos = pos;
  const k = Math.min(t/BASE, 1);
  const move = ease(Math.min(k/0.3,1));
  const cur = {H: pos.H.map((p,i)=> lerp(prevPos.H[i]||p, p, move)), A: pos.A.map((p,i)=> lerp(prevPos.A[i]||p, p, move))};
  const off = e.side, def = off==='H'?'A':'H';
  const holders = [];
  const onOff = off==='H'?e.on_h:e.on_a;
  const starter = onOff[(Math.floor(e.clock)%onOff.length)];
  holders.push(starter); if (e.ast!==null && e.ast!==starter) holders.push(e.ast); if (holders[holders.length-1]!==e.shooter) holders.push(e.shooter);
  const sIdx = onOff.indexOf(e.shooter); const sSpot = shotSpot(e);
  if (k>0.42 && sIdx>=0) cur[off][sIdx] = lerp(cur[off][sIdx], sSpot, ease(Math.min((k-0.42)/0.12,1)));
  let ball; const rim = [BASK[dirOf(e.side,e.q)], MIDY];
  if (done) { ball = [W/2, MIDY]; }
  else if (k < 0.42) {
    const seg = (k/0.42)*(holders.length-1); const j = Math.min(Math.floor(seg), holders.length-2);
    if (holders.length===1) ball = holderPos(e,cur,off,holders[0]);
    else ball = lerp(holderPos(e,cur,off,holders[Math.max(j,0)]), holderPos(e,cur,off,holders[Math.max(j,0)+1]), ease(seg-Math.max(j,0)));
  } else if (k < 0.56) { ball = holderPos(e,cur,off,e.shooter); }
  else if (e.type==='tov') {
    const thief = e.stl!==null ? holderPos(e,cur,def,e.stl) : [W/2, CT+10];
    ball = lerp(holderPos(e,cur,off,e.shooter), thief, ease(Math.min((k-0.56)/0.3,1)));
  } else if (k < 0.84) {
    const kk = (k-0.56)/0.28; const from = holderPos(e,cur,off,e.shooter);
    const p = lerp(from, rim, kk); const h = Math.hypot(rim[0]-from[0], rim[1]-from[1]);
    ball = [p[0], p[1] - Math.sin(kk*Math.PI)*Math.max(40, h*0.45)];
  } else {
    const kk = Math.min((k-0.84)/0.16,1);
    if (e.pts>0) ball = [rim[0], rim[1]+kk*18];
    else if (e.reb!==null) ball = lerp(rim, holderPos(e,cur,e.reb_side,e.reb), ease(kk));
    else ball = rim;
  }
  if (!done && k>=0.84 && !e._scored) {
    e._scored = true; score = {H:e.sh, A:e.sa};
    if (e.pts>0) { floaters.push({x:rim[0], y:rim[1]-30, txt:'+'+e.pts, c:D[e.side].c2==='#000000'?'#fff':D[e.side].c2, life:60}); flash = 12; }
    if (e.dunk) burst(rim[0], rim[1], 25, [D[e.side].c1, D[e.side].c2, '#fff']);
    if (e.big) burst(rim[0], rim[1], 70, [D[e.side].c1, D[e.side].c2, '#ffd34d']);
  }
  for (const s of ['H','A']) { const on = s==='H'?e.on_h:e.on_a; cur[s].forEach((p,i)=> drawPlayer(p, s, on[i], !done && s===off && on[i]===(k<0.56? holders[Math.min(Math.floor((k/0.42)*(holders.length-1)), holders.length-1)] : -1))); }
  if (flash>0) { ctx.strokeStyle='rgba(255,211,77,'+(flash/12)+')'; ctx.lineWidth=6; ctx.beginPath(); ctx.arc(rim[0],rim[1],18,0,2*Math.PI); ctx.stroke(); ctx.lineWidth=2; flash--; }
  ctx.beginPath(); ctx.arc(ball[0],ball[1],7,0,2*Math.PI); ctx.fillStyle='#f28a1e'; ctx.fill(); ctx.strokeStyle='#5a2c00'; ctx.lineWidth=1.5; ctx.stroke(); ctx.lineWidth=2;
  floaters = floaters.filter(f=>f.life>0); for (const f of floaters){ ctx.globalAlpha=f.life/60; ctx.fillStyle=f.c; ctx.font='bold 28px system-ui'; ctx.textAlign='center'; ctx.fillText(f.txt,f.x,f.y-(60-f.life)); f.life--; } ctx.globalAlpha=1;
  conf = conf.filter(c=>c.life>0); for (const c of conf){ c.x+=c.vx; c.y+=c.vy; c.vy+=0.25; c.life--; ctx.fillStyle=c.c; ctx.fillRect(c.x,c.y,6,6); }
  drawBoard(done ? D.events[D.events.length-1] : e); drawWP(done ? D.events[D.events.length-1] : e);
  if (banner) { ctx.fillStyle='rgba(10,14,22,.8)'; ctx.fillRect(W/2-260, MIDY-60, 520, 120); ctx.fillStyle='#ffd34d'; ctx.font='bold 34px system-ui'; ctx.textAlign='center'; ctx.fillText(banner.split('|')[0], W/2, MIDY-18); ctx.fillStyle='#e8eaf0'; ctx.font='16px system-ui'; ctx.fillText(banner.split('|')[1]||'', W/2, MIDY+22); }
  if (!done && k >= 1) {
    prevPos = cur; idx++;
    if (idx >= D.events.length) { finalBanner(); } else {
      const n = D.events[idx]; t = 0; banner = '';
      if (n.q !== e.q) { banner = 'End of '+qname(e.q)+'|'+D.A.abbr+' '+e.sa+' – '+e.sh+' '+D.H.abbr; pause = 1600; prevPos = null; }
      logLine(n);
    }
  }
  requestAnimationFrame(frame);
}
function finalBanner(){
  const L = D.events[D.events.length-1]; const win = L.sh > L.sa ? 'H':'A';
  banner = 'FINAL: ' + D.A.abbr+' '+L.sa+' – '+L.sh+' '+D.H.abbr + '|⭐ ' + D.stars.join('   ⭐ ');
  burst(W/2, MIDY, 160, [D[win].c1, D[win].c2, '#ffd34d', '#fff']);
}
document.getElementById('play').onclick = ()=>{ playing=!playing; document.getElementById('play').textContent = playing?'⏸ Pause':'▶ Play'; };
document.getElementById('restart').onclick = ()=>{ D.events.forEach(e=>e._scored=false); setTo(0); logLine(D.events[0]); };
document.getElementById('speed').onchange = (ev)=>{ speed = parseFloat(ev.target.value); };
document.getElementById('end').onclick = ()=>{ D.events.forEach(e=>e._scored=true); setTo(D.events.length); finalBanner(); };
const qs = [...new Set(D.events.map(e=>e.q))]; const qb = document.getElementById('qbtns');
qs.forEach(q=>{ const b=document.createElement('button'); b.className='b'; b.textContent=qname(q); b.onclick=()=>{ const i=D.events.findIndex(e=>e.q===q); D.events.forEach((e,j)=>e._scored=j<i); setTo(i); logLine(D.events[i]); }; qb.appendChild(b); });
logLine(D.events[0]);
requestAnimationFrame(frame);
</script>
"""


def render_animation(events, teams, h, a, wp0, stars, nsims):
    def side(abbr, s):
        t = teams[s]
        return dict(abbr=abbr, c1=TEAMS[abbr]["c1"], c2=TEAMS[abbr]["c2"], logo=logo_url(abbr),
                    players={int(i): dict(j=str(t.loc[i, "jersey"]), n=str(t.loc[i, "name"])) for i in range(len(t))})
    data = dict(events=events, H=side(h, "H"), A=side(a, "A"), wp0=wp0, stars=stars, nsims=f"{nsims:,}")
    blob = json.dumps(data, default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("</", "<\\/")
    html = ANIM_HTML.replace("__DATA__", blob)
    if hasattr(st, "iframe"):
        st.iframe(html, height=790)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=790, scrolling=False)


# ===========================================================================
#                                   THE APP
# ===========================================================================
import plotly.graph_objects as go
import plotly.express as px

CSS = """
<style>
.block-container{padding-top:1.2rem;max-width:1250px}
.card{background:#141b2a;border:1px solid #232d42;border-radius:14px;padding:12px 14px;margin-bottom:10px}
.gcard{border-radius:14px;padding:12px 14px;margin-bottom:6px;border:1px solid #232d42;background:#121826}
.trow{display:flex;align-items:center;gap:10px;margin:3px 0}
.trow img{width:34px;height:34px}
.tname{font-weight:700;font-size:1.05rem}
.muted{color:#93a0ba;font-size:.85rem}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.75rem;margin:2px 4px 2px 0;background:#232d42;color:#cfd6e6}
.pill.good{background:#123d2a;color:#7ff0b0}.pill.bad{background:#40171c;color:#ff9aa5}.pill.gold{background:#3d3112;color:#ffd34d}
.big{font-size:1.6rem;font-weight:800}
.bar{height:8px;border-radius:6px;background:#232d42;overflow:hidden;display:flex}
.pick{border-left:5px solid #ffd34d}
@media (max-width:640px){.big{font-size:1.25rem}.trow img{width:28px;height:28px}}
</style>
"""

PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10))


def model_now():
    return build_model(season_for(today_et()))


def demo_banner(model):
    if model["demo"]:
        st.warning("⚠️ Couldn't load ESPN data, so this is a **demo league** with made-up results and players.", icon="🛰️")
        err = st.session_state.get("espn_error")
        if err:
            st.caption(f"What went wrong: `{err}`")
        if st.button("🔄 Try ESPN again"):
            st.cache_data.clear()
            st.session_state.pop("espn_error", None)
            st.rerun()


def tlogo(abbr, size=34):
    return f'<img src="{logo_url(abbr)}" width="{size}" height="{size}" style="vertical-align:middle">'


def team_line(abbr, extra=""):
    return f'<div class="trow">{tlogo(abbr)}<span class="tname">{TEAMS[abbr]["name"]}</span>{extra}</div>'


def prob_bar(p_away, a, h):
    return (f'<div class="bar"><div style="width:{p_away*100:.1f}%;background:{TEAMS[a]["c1"]}"></div>'
            f'<div style="width:{(1-p_away)*100:.1f}%;background:{TEAMS[h]["c1"]}"></div></div>')


def games_on(model, d):
    ev = model["ev"]
    g = ev[ev.date == d].copy()
    if not model["demo"] and abs((d - today_et()).days) <= 1:
        try:
            fresh = fetch_day(d)
            if len(fresh):
                fresh = fresh.set_index("event_id")
                for c in ["state", "home_pts", "away_pts", "status_text", "mkt_spread", "mkt_total", "ml_home",
                          "ml_away", "h_record", "a_record"]:
                    if c in fresh:
                        upd = g.event_id.map(fresh[c])
                        g[c] = upd.where(upd.notna(), g[c] if c in g else None)
        except Exception:
            pass
    return g.sort_values("start").reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def _injuries_safe(demo):
    if demo:
        return pd.DataFrame(columns=["team", "player_id", "name", "status", "note"])
    try:
        return fetch_injuries()
    except Exception:
        return pd.DataFrame(columns=["team", "player_id", "name", "status", "note"])


def demo_player_games(model):
    rows = []
    ev = model["ev"]
    rng = np.random.default_rng(5)
    for _, g in ev[ev.state == "post"].iterrows():
        for t in (g.home, g.away):
            for _, p in demo_players(t).iterrows():
                f = rng.normal(1, 0.25)
                rows.append(dict(event_id=g.event_id, date=str(g.date), team=t, player_id=p.player_id, name=p["name"],
                                 min=p["min"], pts=max(0, round(p.pts * f)), reb=max(0, round(p.reb * rng.normal(1, .3))),
                                 ast=max(0, round(p.ast * rng.normal(1, .3))), fg3m=max(0, round(p.fg3m * rng.normal(1, .5))),
                                 stl=rng.poisson(p.stl), blk=rng.poisson(p.blk), tov=rng.poisson(p.tov)))
    return pd.DataFrame(rows)


def get_player_games(model):
    if model["demo"]:
        if "demo_pg" not in st.session_state:
            st.session_state.demo_pg = demo_player_games(model)
        return st.session_state.demo_pg
    if "pg" not in st.session_state:
        st.session_state.pg = load_table("player_games.csv")
    return st.session_state.pg


def slate_predictions(model, d):
    games = games_on(model, d)
    if len(games) == 0:
        return games
    inj = _injuries_safe(model["demo"]) if d >= today_et() else None
    pen = slate_injury_penalties(model, inj, get_player_games(model)) if inj is not None else None
    return predict_slate(model, games, pen)


def date_picker(key, label="Date"):
    return st.date_input(label, value=st.session_state.get(key + "_v", today_et()), key=key)


# ------------------------------- Tonight's slate ---------------------------
def page_slate():
    model = model_now(); demo_banner(model)
    st.title("🏀 Tonight's Slate")
    d = date_picker("slate_date")
    pred = slate_predictions(model, d)
    if len(pred) == 0:
        st.info("No NBA games on this date. Try another day.")
        return

    pk = picks_of_day(pred[pred.state == "pre"] if (pred.state == "pre").any() else pred)
    st.subheader("⭐ Picks of the day")
    c1, c2, c3 = st.columns(3)
    if "lock" in pk:
        r = pk["lock"]; t = r.pick; p = max(r.home_win, 1 - r.home_win)
        c1.markdown(f'<div class="card pick"><div class="muted">🔒 LOCK</div>{team_line(t)}'
                    f'<div class="big">{p:.0%}</div><div class="muted">{r.away} @ {r.home}</div></div>', unsafe_allow_html=True)
    if "upset" in pk:
        r, t, p = pk["upset"]
        c2.markdown(f'<div class="card pick"><div class="muted">🚨 UPSET ALERT</div>{team_line(t)}'
                    f'<div class="big">{p:.0%}</div><div class="muted">Model likes the underdog more than the market</div></div>',
                    unsafe_allow_html=True)
    else:
        c2.markdown('<div class="card"><div class="muted">🚨 UPSET ALERT</div>No underdog the model really likes today.</div>',
                    unsafe_allow_html=True)
    if "coin" in pk:
        r = pk["coin"]
        c3.markdown(f'<div class="card pick"><div class="muted">🪙 COIN FLIP</div>'
                    f'<div class="trow">{tlogo(r.away)} vs {tlogo(r.home)}</div>'
                    f'<div class="big">{r.home_win:.0%} {r.home}</div><div class="muted">Closest game on the board</div></div>',
                    unsafe_allow_html=True)

    preds = grade_predictions(load_table("predictions.csv"), model["ev"])
    if len(preds) and preds.final.any():
        f = preds[preds.final]
        ats = f.ats.value_counts()
        st.caption(f"📈 Model record: **{int(f.correct.sum())}-{int((~f.correct.astype(bool)).sum())}** straight up "
                   f"({f.correct.mean():.0%}) · ATS **{ats.get('win', 0)}-{ats.get('loss', 0)}-{ats.get('push', 0)}**")

    st.subheader(f"Games · {d:%a %b %d}")
    cols = st.columns(2)
    for i, r in pred.iterrows():
        h, a = r.home, r.away
        status = r.status_text or ""
        if r.state != "pre" and pd.notna(r.home_pts):
            score = f" <b>{int(r.away_pts)}</b>", f" <b>{int(r.home_pts)}</b>"
        else:
            score = ("", "")
        mk = []
        if pd.notna(r.mkt_spread):
            mk.append(f"Line {h} {r.mkt_spread:+.1f}")
        if pd.notna(r.mkt_total):
            mk.append(f"O/U {r.mkt_total:.1f}")
        edge = ""
        if pd.notna(r.mkt_spread):
            e = r.model_margin + r.mkt_spread
            if abs(e) >= 2:
                edge = f'<span class="pill good">Value: {(h if e > 0 else a)} ATS ({abs(e):.1f} pts)</span>'
        notes = "".join(f'<span class="pill bad">{a}: {n}</span>' for n in r.fat_a["notes"] if "back" in n or "3 games" in n or "road" in n)
        notes += "".join(f'<span class="pill bad">{h}: {n}</span>' for n in r.fat_h["notes"] if "back" in n or "3 games" in n or "road" in n)
        for t, v in ((a, r.inj_a), (h, r.inj_h)):
            if v >= 1:
                notes += f'<span class="pill bad">{t} injuries −{v:.1f}</span>'
        rec_a = f' <span class="muted">{r.get("a_record") or ""}</span>' if isinstance(r.get("a_record"), str) else ""
        rec_h = f' <span class="muted">{r.get("h_record") or ""}</span>' if isinstance(r.get("h_record"), str) else ""
        html = (f'<div class="gcard" style="border-top:4px solid {TEAMS[h]["c1"]}">'
                f'<div class="muted">{status}</div>'
                f'{team_line(a, rec_a + score[0] + f"<span style=margin-left:auto class=big>{1-r.home_win:.0%}</span>")}'
                f'{team_line(h, rec_h + score[1] + f"<span style=margin-left:auto class=big>{r.home_win:.0%}</span>")}'
                f'{prob_bar(1-r.home_win, a, h)}'
                f'<div class="muted" style="margin-top:6px">Sim: {a} {r.proj_away:.0f} – {r.proj_home:.0f} {h} · '
                f'Model {h} {-r.model_margin:+.1f}, total {r.model_total:.1f}'
                f'{" · " + " · ".join(mk) if mk else ""}</div>{edge}{notes}</div>')
        with cols[i % 2]:
            st.markdown(html, unsafe_allow_html=True)
            if st.button("Open in Game Center →", key=f"gc_{r.event_id}", width="stretch"):
                st.session_state.gc_event = r.event_id
                st.session_state.gc_date_v = d
                st.switch_page(PAGES["gc"])

    if (pred.state == "pre").any() and st.button("💾 Save today's predictions to the tracker"):
        n = save_predictions(pred)
        st.success(f"Saved {n} predictions.")


def save_predictions(pred):
    pre = pred[pred.state == "pre"]
    if len(pre) == 0:
        return 0
    old = load_table("predictions.csv")
    new = pd.DataFrame(dict(date=pre.date.astype(str), event_id=pre.event_id.astype(str), home=pre.home, away=pre.away,
                            home_win=pre.home_win.round(4), proj_home=pre.proj_home.round(1), proj_away=pre.proj_away.round(1),
                            model_margin=pre.model_margin.round(2), model_total=pre.model_total.round(1),
                            mkt_spread=pre.mkt_spread, mkt_total=pre.mkt_total, pick=pre.pick,
                            saved_at=dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M")))
    old = old[~old.event_id.astype(str).isin(new.event_id)] if len(old) else old
    save_table("predictions.csv", pd.concat([old, new], ignore_index=True), "save predictions")
    return len(new)


# ------------------------------- Game Center --------------------------------
def run_game(model, g, out_h, out_a, lab, n):
    h, a = g.home, g.away
    fh, fa = fatigue(model["ev"], h, g.event_id), fatigue(model["ev"], a, g.event_id)
    for f, key in ((fh, "rest_h"), (fa, "rest_a")):
        if lab[key] == "Back-to-back":
            f["pts"] = -1.8
        elif lab[key] == "Fully rested":
            f["pts"] = 0.4
    pool_h, pool_a = get_pool(model, h), get_pool(model, a)
    if len(pool_h) == 0:
        pool_h = demo_players(h)
    if len(pool_a) == 0:
        pool_a = demo_players(a)
    ih, ia = injury_penalty(pool_h, out_h), injury_penalty(pool_a, out_a)
    pr = project(model, h, a, fh["pts"], fa["pts"], lab["neutral"] or bool(g.get("neutral", False)), ih, ia,
                 lab["boost_h"], lab["boost_a"], lab["pace"])
    seed = int(str(g.event_id)[-6:]) if str(g.event_id)[-6:].isdigit() else 1
    sim = sim_game(pr["h_ortg"], pr["a_ortg"], pr["pace"], n=n, seed=seed)
    ph, stats_h = sim_players(allot_minutes(pool_h, out_h), sim["h"], seed + 1)
    pa, stats_a = sim_players(allot_minutes(pool_a, out_a), sim["a"], seed + 2)
    return dict(pr=pr, sim=sim, fh=fh, fa=fa, ih=ih, ia=ia, ph=ph, pa=pa, sh=stats_h, sa=stats_a, pool_h=pool_h, pool_a=pool_a)


def page_game_center():
    model = model_now(); demo_banner(model)
    st.title("🎮 Game Center")
    c1, c2 = st.columns([1, 2])
    with c1:
        d = date_picker("gc_date")
    games = games_on(model, d)
    if len(games) == 0:
        st.info("No games on this date.")
        return
    labels = {r.event_id: f"{r.away} @ {r.home} · {r.status_text}" for _, r in games.iterrows()}
    ids = list(labels)
    default = ids.index(st.session_state.get("gc_event")) if st.session_state.get("gc_event") in ids else 0
    with c2:
        eid = st.selectbox("Game", ids, index=default, format_func=lambda x: labels[x])
    st.session_state.gc_event = eid
    g = games[games.event_id == eid].iloc[0]
    h, a = g.home, g.away

    inj = _injuries_safe(model["demo"])
    pool_h, pool_a = get_pool(model, h), get_pool(model, a)
    with st.expander("🧪 What-if lab · lineups, injuries, rest & more", expanded=False):
        cc = st.columns(2)
        outs = {}
        for col, t, pool in ((cc[0], a, pool_a), (cc[1], h, pool_h)):
            with col:
                st.markdown(team_line(t), unsafe_allow_html=True)
                ti = inj[inj.team == t]
                if len(ti):
                    st.caption(" · ".join(f"**{r['name']}** ({r.status})" for _, r in ti.iterrows()))
                names = dict(zip(pool.player_id, pool.name)) if len(pool) else {}
                default_out = [pid for pid in names if pid in set(ti[ti.status.str.lower().str.contains("out", na=False)].player_id)]
                outs[t] = st.multiselect(f"{t} players OUT", list(names), default=default_out,
                                         format_func=lambda x, n=names: n.get(x, x), key=f"out_{eid}_{t}")
        l1, l2, l3 = st.columns(3)
        lab = dict(
            neutral=l1.checkbox("Neutral court", False, key=f"neu_{eid}"),
            rest_a=l1.selectbox(f"{a} rest", ["Auto", "Back-to-back", "Fully rested"], key=f"ra_{eid}"),
            rest_h=l1.selectbox(f"{h} rest", ["Auto", "Back-to-back", "Fully rested"], key=f"rh_{eid}"),
            boost_a=l2.slider(f"{a} hot/cold (pts per 100)", -8.0, 8.0, 0.0, 0.5, key=f"ba_{eid}"),
            boost_h=l2.slider(f"{h} hot/cold (pts per 100)", -8.0, 8.0, 0.0, 0.5, key=f"bh_{eid}"),
            pace=l3.slider("Pace change (possessions)", -8.0, 8.0, 0.0, 0.5, key=f"pc_{eid}"))
        n = l3.select_slider("Simulations", [2000, 5000, 10000, 20000], value=10000, key=f"n_{eid}")
    res = run_game(model, g, outs[h], outs[a], lab, n)
    pr, sim = res["pr"], res["sim"]
    hw = float((sim["h"] > sim["a"]).mean())
    best, idx, cnt = most_common_score(sim)

    st.markdown(
        f'<div class="card" style="border-top:4px solid {TEAMS[h]["c1"]}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">'
        f'<div style="text-align:center">{tlogo(a,64)}<div class="tname">{a}</div><div class="big">{1-hw:.0%}</div></div>'
        f'<div style="text-align:center"><div class="muted">{g.status_text}</div>'
        f'<div class="big">{a} {best[1]} – {best[0]} {h}</div><div class="muted">most common final score</div></div>'
        f'<div style="text-align:center">{tlogo(h,64)}<div class="tname">{h}</div><div class="big">{hw:.0%}</div></div>'
        f'</div><div style="margin-top:8px">{prob_bar(1-hw, a, h)}</div></div>', unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["📊 Prediction", "🏀 Animation", "📋 Box score", "🎯 Props & Parlay"])
    with t1:
        m = sim["h"] - sim["a"]; tot = sim["h"] + sim["a"]
        k = st.columns(4)
        k[0].metric("Model spread", f"{h} {-m.mean():+.1f}", None if pd.isna(g.mkt_spread) else f"market {g.mkt_spread:+.1f}", delta_color="off")
        k[1].metric("Model total", f"{tot.mean():.1f}", None if pd.isna(g.mkt_total) else f"market {g.mkt_total:.1f}", delta_color="off")
        k[2].metric("Avg score", f"{sim['a'].mean():.0f}–{sim['h'].mean():.0f}")
        k[3].metric("Overtime chance", f"{(sim['n_ot'] > 0).mean():.1%}")
        vc = pd.Series(m).value_counts().sort_index()
        fig = go.Figure(go.Bar(x=vc.index, y=vc.values / len(m) * 100,
                               marker_color=[TEAMS[h]["c1"] if x > 0 else TEAMS[a]["c1"] for x in vc.index]))
        if pd.notna(g.mkt_spread):
            fig.add_vline(x=-g.mkt_spread, line_dash="dash", line_color="#ffd34d", annotation_text="market line")
        fig.update_layout(title=f"Final margin ({h} minus {a})", xaxis_title="points", yaxis_title="% of sims", **PLOT, height=320)
        st.plotly_chart(fig, width="stretch")
        qd = pd.DataFrame({"Q1": [sim["aq"][:, 0].mean(), sim["hq"][:, 0].mean()], "Q2": [sim["aq"][:, 1].mean(), sim["hq"][:, 1].mean()],
                           "Q3": [sim["aq"][:, 2].mean(), sim["hq"][:, 2].mean()], "Q4": [sim["aq"][:, 3].mean(), sim["hq"][:, 3].mean()],
                           "Final": [sim["a"].mean(), sim["h"].mean()]}, index=[a, h]).round(1)
        st.markdown("**Average points by quarter**")
        st.dataframe(qd, width="stretch")
        c = st.columns(2)
        for col, t, f, inj_pts in ((c[0], a, res["fa"], res["ia"]), (c[1], h, res["fh"], res["ih"])):
            col.markdown(f"**{t} schedule spot:** " + (", ".join(f["notes"]) or "normal rest") +
                         f" → {f['pts']:+.1f} pts" + (f" · injuries −{inj_pts:.1f} pts" if inj_pts else ""))
        r = model["ratings"]
        st.caption(f"Ratings — {a}: off {r.loc[a,'off']:+.1f}, def {r.loc[a,'de']:+.1f}, pace {r.loc[a,'pace']:.1f} · "
                   f"{h}: off {r.loc[h,'off']:+.1f}, def {r.loc[h,'de']:+.1f}, pace {r.loc[h,'pace']:.1f} "
                   f"(per 100 possessions vs league average; lower defense is better)")
    with t2:
        st.caption(f"This is the single most common final score across {n:,} sims (it came up {cnt} times). "
                   "Press a quarter button to jump ahead.")
        events, teams = build_pbp(sim, idx, res["ph"], res["pa"], model["styles"].loc[h], model["styles"].loc[a], pr["margin"],
                                  seed=idx + 3)
        box = pbp_box(events, teams)
        top = box.sort_values("GS", ascending=False).head(3)
        stars = [f"{(h if s == 'H' else a)} {p}: {pt} pts, {rb} reb, {asst} ast" for s, p, pt, rb, asst in
                 zip(top.side, top.Player, top.PTS, top.REB, top.AST)]
        render_animation(events, teams, h, a, pr["win"], stars, n)
        wp = pd.DataFrame([dict(t=(min(e["q"], 4) - 1) * 12 + (12 - e["clock"] / 60) if e["q"] <= 4 else 48 + (e["q"] - 4) * 5 - e["clock"] / 60,
                                wp=e["wp"]) for e in events])
        fig = go.Figure(go.Scatter(x=wp.t, y=wp.wp * 100, mode="lines", line=dict(color=TEAMS[h]["c1"] if TEAMS[h]["c1"] != "#000000" else "#ccc", width=3),
                                   fill="tozeroy"))
        fig.update_layout(title=f"{h} win probability during the game", xaxis_title="minute", yaxis=dict(range=[0, 100], title="%"),
                          **PLOT, height=280)
        st.plotly_chart(fig, width="stretch")
        c = st.columns(2)
        for col, s, t in ((c[0], "A", a), (c[1], "H", h)):
            b = box[(box.side == s) & ((box.PTS + box.REB + box.AST) > 0)].drop(columns=["side", "idx", "GS"])
            col.markdown(f"**{t} box score (animated game)**")
            col.dataframe(b.sort_values("PTS", ascending=False), hide_index=True, width="stretch")
    with t3:
        st.caption("Average stats across every simulation (minutes adjust for who's out).")
        c = st.columns(2)
        for col, t, p, stt in ((c[0], a, res["pa"], res["sa"]), (c[1], h, res["ph"], res["sh"])):
            col.markdown(team_line(t), unsafe_allow_html=True)
            col.dataframe(projected_box(p, stt).round(1), hide_index=True, width="stretch")
    with t4:
        props_and_parlay(model, g, res)


STAT_NAMES = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "3-pointers", "pra": "Pts+Reb+Ast"}
ODDS_MARKET = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m"}


def props_and_parlay(model, g, res):
    h, a = g.home, g.away
    players = [("H", i, r["name"]) for i, r in res["ph"].iterrows()] + [("A", i, r["name"]) for i, r in res["pa"].iterrows()]

    def arr(side, i, stat):
        return (res["sh"] if side == "H" else res["sa"])[stat][:, i]

    st.markdown("#### 🎯 Player props")
    st.caption("Tick **Log** on any prop you like, change the side, line or odds if your sportsbook is different, then press the button.")
    pid = {("H", i): r.player_id for i, r in res["ph"].iterrows()}
    pid.update({("A", i): r.player_id for i, r in res["pa"].iterrows()})
    rows = []
    for side, i, nm in players:
        for s_ in ["pts", "reb", "ast", "fg3m"]:
            v = arr(side, i, s_)
            line = math.floor(np.median(v)) + 0.5
            po = (v > line).mean()
            rows.append(dict(Log=False, Player=nm, Team=h if side == "H" else a, Stat=STAT_NAMES[s_], Avg=round(v.mean(), 1),
                             Side="Over" if po >= 0.5 else "Under", Line=line, Odds=-110,
                             **{"Model %": round(max(po, 1 - po) * 100)}, _side=side, _i=i, _stat=s_))
    pt = pd.DataFrame(rows)
    stat_f = st.segmented_control("Stat", list(STAT_NAMES.values())[:4], default="Points", key="propstat") or "Points"
    view = pt[pt.Stat == stat_f].sort_values("Avg", ascending=False)
    ed = st.data_editor(view, hide_index=True, width="stretch", key=f"props_{g.event_id}_{stat_f}",
                        column_order=["Log", "Player", "Team", "Avg", "Side", "Line", "Odds", "Model %"],
                        disabled=["Player", "Team", "Avg", "Model %"],
                        column_config={"Log": st.column_config.CheckboxColumn(width="small"),
                                       "Side": st.column_config.SelectboxColumn(options=["Over", "Under"]),
                                       "Line": st.column_config.NumberColumn(step=0.5),
                                       "Odds": st.column_config.NumberColumn(step=5, format="%d"),
                                       "Model %": st.column_config.NumberColumn(format="%d%%")})
    pick_props = []
    for _, r in ed[ed.Log].iterrows():
        v = arr(r._side, r._i, r._stat)
        pr_ = float((v > r.Line).mean() if r.Side == "Over" else (v < r.Line).mean())
        pick_props.append(dict(date=str(g.date), event_id=str(g.event_id), game=f"{a} @ {h}", kind="Player prop",
                               pick=f"{r.Player} {r.Side} {r.Line:g} {STAT_NAMES[r._stat].lower()}", line=r.Line,
                               odds=float(r.Odds), model_prob=round(pr_, 4),
                               detail=f"{pid[(r._side, r._i)]}|{r._stat}|{r.Side}"))
    quick_log_button(pick_props, "props")

    key = st.session_state.get("odds_key") or secret("ODDS_API_KEY")
    if key and not model["demo"]:
        skey = f"book_props_{g.event_id}"
        if st.button("📥 Load sportsbook prop lines (uses Odds API credits)"):
            try:
                og = odds_api_games(key)
                og = og[(og.home == h) & (og.away == a)]
                st.session_state[skey] = odds_api_props(key, og.odds_id.iloc[0]) if len(og) else pd.DataFrame()
                if len(og) == 0:
                    st.warning("The Odds API doesn't list this game yet.")
            except Exception as e:
                st.error(f"Couldn't load props: {e}")
        pp = st.session_state.get(skey)
        if pp is not None and len(pp):
            name_map = {nm.lower(): (side, i) for side, i, nm in players}
            out = []
            for _, r in pp.iterrows():
                hit = name_map.get(str(r.player).lower())
                stat = ODDS_MARKET.get(r.market)
                if not hit or not stat or pd.isna(r.line):
                    continue
                v = arr(hit[0], hit[1], stat)
                p = float((v > r.line).mean() if r.side == "Over" else (v < r.line).mean())
                ev_ = p * american_to_decimal(r.price) - 1
                out.append(dict(Log=False, Player=r.player, Stat=STAT_NAMES[stat], Side=r.side, Line=r.line, Book=r.book,
                                Odds=int(r.price), **{"Model %": round(p * 100), "EV %": round(ev_ * 100, 1)},
                                _pid=pid[hit], _stat=stat, _p=p))
            if out:
                st.markdown("**Sportsbook props vs the model** (best value first)")
                bdf = pd.DataFrame(out).sort_values("EV %", ascending=False)
                bed = st.data_editor(bdf, hide_index=True, width="stretch", key=f"bookprops_{g.event_id}",
                                     column_order=["Log", "Player", "Stat", "Side", "Line", "Book", "Odds", "Model %", "EV %"],
                                     disabled=["Player", "Stat", "Side", "Line", "Book", "Odds", "Model %", "EV %"],
                                     column_config={"Log": st.column_config.CheckboxColumn(width="small"),
                                                    "Model %": st.column_config.NumberColumn(format="%d%%"),
                                                    "EV %": st.column_config.NumberColumn(format="%+.1f%%")})
                quick_log_button([dict(date=str(g.date), event_id=str(g.event_id), game=f"{a} @ {h}", kind="Player prop",
                                       pick=f"{r.Player} {r.Side} {r.Line:g} {r.Stat.lower()} ({r.Book})", line=r.Line,
                                       odds=float(r.Odds), model_prob=round(r._p, 4), detail=f"{r._pid}|{r._stat}|{r.Side}")
                                  for _, r in bed[bed.Log].iterrows()], "bookprops")
            else:
                st.info("No prop lines matched these players yet.")
    else:
        st.caption("Add a free The Odds API key on the Betting page to pull real sportsbook prop lines here.")

    st.markdown("#### 🧩 Same-game parlay builder")
    st.caption("Every leg is checked against the **same** simulated games, so legs that go together (a star going off + his team covering) are priced correctly.")
    sim = res["sim"]
    n_legs = st.number_input("Number of legs", 1, 6, 2, key="sgp_n")
    mask = np.ones(len(sim["h"]), bool)
    indep = 1.0
    desc = []
    for k in range(int(n_legs)):
        c = st.columns([1.2, 1.6, 1, 1])
        kind = c[0].selectbox("Leg", ["Moneyline", "Spread", "Total", "Player prop"], key=f"sgp_k{k}")
        if kind == "Moneyline":
            t = c[1].selectbox("Team", [a, h], key=f"sgp_t{k}")
            leg = sim["h"] > sim["a"] if t == h else sim["a"] > sim["h"]
            desc.append(f"{t} ML")
        elif kind == "Spread":
            t = c[1].selectbox("Team", [a, h], key=f"sgp_t{k}")
            dflt = float(g.mkt_spread if t == h else -g.mkt_spread) if pd.notna(g.mkt_spread) else -3.5
            line = c[2].number_input("Line", value=dflt, step=0.5, key=f"sgp_l{k}")
            mg = (sim["h"] - sim["a"]) if t == h else (sim["a"] - sim["h"])
            leg = mg + line > 0
            desc.append(f"{t} {line:+.1f}")
        elif kind == "Total":
            side = c[1].selectbox("Side", ["Over", "Under"], key=f"sgp_t{k}")
            line = c[2].number_input("Line", value=float(g.mkt_total) if pd.notna(g.mkt_total) else 225.5, step=0.5, key=f"sgp_l{k}")
            tot = sim["h"] + sim["a"]
            leg = tot > line if side == "Over" else tot < line
            desc.append(f"{side} {line}")
        else:
            pi = c[1].selectbox("Player", range(len(players)), format_func=lambda j: players[j][2], key=f"sgp_p{k}")
            stat = c[2].selectbox("Stat", list(STAT_NAMES), format_func=STAT_NAMES.get, key=f"sgp_s{k}")
            side, i, nm = players[pi]
            v = arr(side, i, stat)
            line = c[3].number_input("Over line", value=float(math.floor(np.median(v)) + 0.5), step=1.0, key=f"sgp_l{k}")
            leg = v > line
            desc.append(f"{nm} {line}+ {STAT_NAMES[stat].lower()}")
        indep *= leg.mean()
        mask &= leg
    p = mask.mean()
    c = st.columns(4)
    c[0].metric("Chance all legs hit", f"{p:.1%}")
    c[1].metric("Fair odds", fmt_odds(prob_to_american(p)) if p > 0 else "—")
    c[2].metric("If legs were unrelated", f"{indep:.1%}")
    book = c[3].text_input("Book's parlay odds", "", placeholder="+450", key="sgp_book")
    if book.strip():
        try:
            ev_ = p * american_to_decimal(book) - 1
            (st.success if ev_ > 0 else st.error)(f"Expected value: {ev_:+.1%} per $1 bet")
            if st.button("➕ Add this parlay to my bet tracker"):
                add_bet(dict(date=str(g.date), event_id=str(g.event_id), game=f"{a} @ {h}", kind="Parlay",
                             pick=" + ".join(desc), line="", odds=float(book), stake=10.0, model_prob=round(p, 4)))
                st.success("Added (stake $10 — change it on the Betting page).")
        except Exception:
            st.warning("Type odds like +450 or -120.")


# ------------------------------- Betting -----------------------------------
def add_bets(new_bets, stake=None):
    """Log several bets in one save. Skips any bet that's already logged. Returns (added, skipped)."""
    bets = load_table("bets.csv")
    def norm(x):
        try:
            x = float(x)
            return f"{x:g}" if np.isfinite(x) else ""
        except (TypeError, ValueError):
            return str(x or "").strip()
    have = set(zip(bets.event_id.astype(str), bets.kind.astype(str), bets.pick.astype(str), bets.line.map(norm))) if len(bets) else set()
    now = dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M")
    rows, skipped = [], 0
    for k, b in enumerate(new_bets):
        b = dict(b)
        sig = (str(b["event_id"]), str(b["kind"]), str(b["pick"]), norm(b.get("line", "")))
        if sig in have:
            skipped += 1
            continue
        have.add(sig)
        if stake is not None:
            b["stake"] = stake
        b.update(id=f"{int(time.time() * 1000)}{k}", placed=now, status="open", profit=0.0)
        rows.append(b)
    if rows:
        save_table("bets.csv", pd.concat([bets, pd.DataFrame(rows)], ignore_index=True), f"log {len(rows)} bets")
    return len(rows), skipped


def add_bet(b):
    add_bets([b])


def quick_log_button(selected, key):
    """Stake box + one button that logs every ticked bet."""
    if not selected:
        return
    c = st.columns([1, 2])
    stake = c[0].number_input("Stake per bet ($)", 1.0, 100000.0, float(st.session_state.get("default_stake", 10.0)),
                              step=5.0, key=f"stake_{key}")
    st.session_state.default_stake = stake
    ev_total = sum(stake * (b["model_prob"] * american_to_decimal(b["odds"]) - 1) for b in selected if b.get("model_prob") not in ("", None))
    if c[1].button(f"➕ Log {len(selected)} selected bet{'s' if len(selected) > 1 else ''} · ${stake * len(selected):,.0f} total · "
                   f"model expects {ev_total:+,.2f}", type="primary", key=f"log_{key}", width="stretch"):
        added, skipped = add_bets(selected, stake)
        st.toast(f"✅ Logged {added} bet{'s' if added != 1 else ''}" + (f" ({skipped} already logged)" if skipped else ""))
        st.session_state.bet_log_msg = f"Logged {added} bet(s)" + (f", skipped {skipped} already in your tracker." if skipped else ".")


def grade_bets(bets, ev):
    if len(bets) == 0:
        return bets, False
    res = ev[ev.state == "post"].set_index("event_id")
    changed = False
    b = bets.copy()
    for i, r in b.iterrows():
        if r.status != "open" or str(r.event_id) not in res.index:
            continue
        if r.kind == "Player prop" and isinstance(r.get("detail"), str) and r.detail.count("|") == 2:
            try:
                pid_, stat_, side_ = r.detail.split("|")
                box = fetch_box(str(r.event_id))
                row = box[box.player_id.astype(str) == pid_]
                if len(row) == 0:
                    out = "void"                      # player didn't play
                else:
                    row = row.iloc[0]
                    val = row.pts + row.reb + row.ast if stat_ == "pra" else row[stat_]
                    line = float(r.line)
                    out = "push" if val == line else ("won" if (val > line) == (side_ == "Over") else "lost")
                stake = float(r.stake)
                b.at[i, "status"] = out
                b.at[i, "profit"] = round(stake * (american_to_decimal(r.odds) - 1), 2) if out == "won" else (-stake if out == "lost" else 0.0)
                changed = True
            except Exception:
                pass
            continue
        if r.kind not in ("Moneyline", "Spread", "Total"):
            continue
        g = res.loc[str(r.event_id)]
        hp, ap = g.home_pts, g.away_pts
        try:
            line = float(r.line) if str(r.line) not in ("", "nan") else 0.0
        except Exception:
            line = 0.0
        if r.kind == "Moneyline":
            won = (hp > ap) == (r.pick == g.home); out = "won" if won else "lost"
        elif r.kind == "Spread":
            mg = (hp - ap) if r.pick == g.home else (ap - hp)
            out = "push" if mg + line == 0 else ("won" if mg + line > 0 else "lost")
        else:
            tot = hp + ap
            out = "push" if tot == line else ("won" if (tot > line) == (r.pick == "Over") else "lost")
        stake = float(r.stake)
        b.at[i, "status"] = out
        b.at[i, "profit"] = round(stake * (american_to_decimal(r.odds) - 1), 2) if out == "won" else (-stake if out == "lost" else 0.0)
        changed = True
    return b, changed


TOTAL_SD = 17.5


def suggested_bets(pred, show_all=False):
    """Spread, total and moneyline bets the model likes for games that haven't started."""
    rows = []
    for _, r in pred[pred.state == "pre"].iterrows():
        game = f"{r.away} @ {r.home}"
        base = dict(Log=False, Game=game, _eid=str(r.event_id), _date=r.date)
        if pd.notna(r.mkt_spread):
            e = r.model_margin + r.mkt_spread                    # >0: model likes home vs the spread
            team, line = (r.home, r.mkt_spread) if e > 0 else (r.away, -r.mkt_spread)
            p = phi(abs(e) / MARGIN_SD)
            if show_all or abs(e) >= 2:
                rows.append(dict(base, Bet="Spread", Pick=f"{team} {line:+g}", Odds=-110, _pick=team, _line=line, _p=p,
                                 Why=f"Model has {team} {abs(e):.1f} pts better than the line"))
        if pd.notna(r.mkt_total):
            diff = r.model_total - r.mkt_total
            side = "Over" if diff > 0 else "Under"
            p = phi(abs(diff) / TOTAL_SD)
            if show_all or abs(diff) >= 4:
                rows.append(dict(base, Bet="Total", Pick=f"{side} {r.mkt_total:g}", Odds=-110, _pick=side, _line=r.mkt_total, _p=p,
                                 Why=f"Model total {r.model_total:.1f} vs {r.mkt_total:g}"))
        if pd.notna(r.ml_home) and pd.notna(r.ml_away):
            ih, ia = american_to_prob(r.ml_home), american_to_prob(r.ml_away)
            nh = ih / (ih + ia)
            for team, p, ml, mk in ((r.home, r.home_win, r.ml_home, nh), (r.away, 1 - r.home_win, r.ml_away, 1 - nh)):
                if (show_all and p >= 0.5) or p - mk >= 0.03:
                    rows.append(dict(base, Bet="Moneyline", Pick=f"{team} {fmt_odds(ml)}", Odds=int(ml), _pick=team, _line=np.nan, _p=p,
                                     Why=f"Model {p:.0%} vs market {mk:.0%}"))
    df = pd.DataFrame(rows)
    if len(df):
        df["Model %"] = (df._p * 100).round()
        df["EV %"] = [round((p * american_to_decimal(o) - 1) * 100, 1) for p, o in zip(df._p, df.Odds)]
        df = df.sort_values("EV %", ascending=False).reset_index(drop=True)
    return df


def page_betting():
    model = model_now(); demo_banner(model)
    st.title("💰 Betting")
    with st.expander("🔑 Sportsbook odds (optional)"):
        st.caption("Lines from ESPN load automatically. Add a free key from the-odds-api.com to compare every sportsbook "
                   "and pull player props. Best place for it: Streamlit secrets as ODDS_API_KEY.")
        k = st.text_input("The Odds API key", type="password", value=st.session_state.get("odds_key", ""))
        if k:
            st.session_state.odds_key = k
    d = date_picker("bet_date")
    pred = slate_predictions(model, d)
    st.markdown("#### ⚡ Model bets — tick the ones you like, then log them with one click")
    if st.session_state.get("bet_log_msg"):
        st.success(st.session_state.pop("bet_log_msg"))
    show_all = st.toggle("Show every model pick (not just value bets)", False)
    sug = suggested_bets(pred, show_all) if len(pred) else pd.DataFrame()
    if len(sug) == 0:
        st.info("No value bets right now (or no games left to bet on this date). Flip the toggle to see every model pick.")
    else:
        ed = st.data_editor(sug, hide_index=True, width="stretch", key=f"sug_{d}_{show_all}",
                            column_order=["Log", "Game", "Bet", "Pick", "Odds", "Model %", "EV %", "Why"],
                            disabled=["Game", "Bet", "Pick", "Model %", "EV %", "Why"],
                            column_config={"Log": st.column_config.CheckboxColumn(width="small"),
                                           "Odds": st.column_config.NumberColumn(step=5, format="%d", help="Change if your sportsbook's price is different"),
                                           "Model %": st.column_config.NumberColumn(format="%d%%"),
                                           "EV %": st.column_config.NumberColumn(format="%+.1f%%")})
        chosen = [dict(date=str(r._date), event_id=r._eid, game=r.Game, kind=r.Bet, pick=r._pick, line=r._line,
                       odds=float(r.Odds), model_prob=round(r._p, 4), detail="")
                  for _, r in ed[ed.Log].iterrows()]
        quick_log_button(chosen, "games")
        st.caption("Value = model beats the market by 2+ pts on the spread, 4+ on the total, or 3%+ on the moneyline. "
                   "Spreads and totals assume -110 unless you change the odds. Player props can be logged the same way in Game Center → Props.")

    key = st.session_state.get("odds_key") or secret("ODDS_API_KEY")
    if key and not model["demo"] and len(pred):
        try:
            og = odds_api_games(key)
            st.markdown("#### 📚 Best moneyline across sportsbooks")
            out = []
            for _, r in pred.iterrows():
                o = og[(og.home == r.home) & (og.away == r.away)]
                for side, t, p in (("h", r.home, r.home_win), ("a", r.away, 1 - r.home_win)):
                    col = f"ml_{side}"
                    if col in o and o[col].notna().any():
                        best = o.loc[o[col].astype(float).idxmax()]
                        out.append(dict(Team=t, Book=best.book, Odds=fmt_odds(best[col]), Model=f"{p:.0%}",
                                        EV=f"{p * american_to_decimal(best[col]) - 1:+.1%}"))
            if out:
                st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")
        except Exception as e:
            st.caption(f"Couldn't reach The Odds API ({e}).")

    st.markdown("#### 🧮 Odds value check")
    c = st.columns(3)
    p = c[0].number_input("Your/model win chance (%)", 1.0, 99.0, 55.0, 0.5) / 100
    o = c[1].text_input("Sportsbook odds", "-110")
    try:
        dec = american_to_decimal(o)
        ev_ = p * dec - 1
        kelly = max((p * dec - 1) / (dec - 1), 0)
        c[2].metric("Expected value", f"{ev_:+.1%}", f"Kelly stake {kelly / 4:.1%} of bankroll (¼ Kelly)", delta_color="off")
        st.caption(f"Break-even win chance at {o}: {american_to_prob(o):.1%} · fair odds for {p:.0%}: {fmt_odds(prob_to_american(p))}")
    except Exception:
        c[2].write("Type odds like -110 or +150")

    st.markdown("#### 📒 My bet tracker")
    bets = load_table("bets.csv")
    bets, changed = grade_bets(bets, model["ev"])
    if changed:
        save_table("bets.csv", bets, "grade bets")
    with st.expander("✍️ Log a bet by hand"), st.form("addbet", clear_on_submit=True):
        allg = games_on(model, d)
        opts = {r.event_id: f"{r.away} @ {r.home}" for _, r in allg.iterrows()}
        c = st.columns([2, 1.2, 1.3, 1, 1, 1])
        eid = c[0].selectbox("Game", list(opts) or [""], format_func=lambda x: opts.get(x, "—"))
        kind = c[1].selectbox("Bet", ["Moneyline", "Spread", "Total", "Player prop", "Parlay"])
        pick = c[2].text_input("Pick", placeholder="BOS / Over / Tatum 25+")
        line = c[3].text_input("Line", placeholder="-4.5")
        odds = c[4].text_input("Odds", "-110")
        stake = c[5].number_input("Stake $", 1.0, 100000.0, 10.0)
        if st.form_submit_button("➕ Log bet") and eid:
            gg = allg[allg.event_id == eid].iloc[0]
            pk_ = team_abbr(pick) or pick.strip().title()
            try:
                add_bet(dict(date=str(d), event_id=eid, game=opts[eid], kind=kind, pick=pk_, line=line, odds=float(odds),
                             stake=stake, model_prob=""))
                st.success("Bet logged.")
                bets = load_table("bets.csv")
            except ValueError:
                st.error("Odds must look like -110 or +150.")
    if len(bets):
        b = bets.copy()
        b["profit"] = pd.to_numeric(b.profit, errors="coerce").fillna(0)
        b["stake"] = pd.to_numeric(b.stake, errors="coerce").fillna(0)
        settled = b[b.status.isin(["won", "lost", "push"])]
        k = st.columns(4)
        k[0].metric("Record", f"{(settled.status == 'won').sum()}-{(settled.status == 'lost').sum()}-{(settled.status == 'push').sum()}")
        k[1].metric("Profit", f"${settled.profit.sum():,.2f}")
        k[2].metric("ROI", f"{settled.profit.sum() / max(settled.stake.sum(), 1):+.1%}")
        k[3].metric("Open bets", int((b.status == "open").sum()))
        if len(settled):
            s = settled.sort_values("placed")
            fig = go.Figure(go.Scatter(y=s.profit.cumsum(), mode="lines+markers", line=dict(color="#2ecc71", width=3)))
            fig.update_layout(title="Profit over time ($)", **PLOT, height=250)
            st.plotly_chart(fig, width="stretch")
        edited = st.data_editor(b[["id", "date", "game", "kind", "pick", "line", "odds", "stake", "status", "profit"]],
                                hide_index=True, width="stretch", key="bet_editor",
                                column_config={"status": st.column_config.SelectboxColumn(options=["open", "won", "lost", "push", "void"]),
                                               "id": None})
        cbtn = st.columns(2)
        if cbtn[0].button("💾 Save changes (props & parlays are graded by hand here)"):
            e = edited.copy()
            for i, r in e.iterrows():
                st_, stake = r.status, float(r.stake)
                if st_ == "won":
                    e.at[i, "profit"] = round(stake * (american_to_decimal(r.odds) - 1), 2)
                elif st_ == "lost":
                    e.at[i, "profit"] = -stake
                elif st_ in ("push", "void", "open"):
                    e.at[i, "profit"] = 0.0
            merged = bets.set_index("id")
            merged.update(e.set_index("id"))
            save_table("bets.csv", merged.reset_index(), "edit bets")
            st.success("Saved.")
        if cbtn[1].button("🗑️ Delete open bets with stake 0"):
            keep = bets[~((bets.status == "open") & (pd.to_numeric(bets.stake, errors="coerce") == 0))]
            save_table("bets.csv", keep, "delete bets")
            st.rerun()
    else:
        st.caption("No bets logged yet.")


# ------------------------------- Season & playoffs -------------------------
def page_season():
    model = model_now(); demo_banner(model)
    st.title("🏆 Season & Playoffs")
    n = st.select_slider("Season simulations", [500, 1000, 2000, 5000], value=2000)
    ss = season_sim(model, n=n, key=(model["season"], n, round(float(model["ratings"].net.sum()), 4), len(model["ev"])))
    t = ss.sort_values("title", ascending=False).head(12)
    fig = go.Figure(go.Bar(x=t.index, y=t.title, marker_color=[TEAMS[x]["c1"] for x in t.index],
                           text=[f"{v:.1f}%" for v in t.title], textposition="outside"))
    fig.update_layout(title="Championship odds", yaxis_title="%", **PLOT, height=330)
    st.plotly_chart(fig, width="stretch")
    for conf, name in (("E", "Eastern Conference"), ("W", "Western Conference")):
        st.subheader(name)
        c = ss[ss.conf == conf].sort_values("avg_seed")
        df = pd.DataFrame({"Team": [TEAMS[x]["name"] for x in c.index], "W-L": [f"{w}-{l}" for w, l in zip(c.w, c.l)],
                           "Proj W-L": [f"{w:.0f}-{l:.0f}" for w, l in zip(c.proj_w, c.proj_l)], "Net rtg": c.net,
                           "Top-6 %": c.top6.round(0), "Play-in %": c.playin.round(0), "Playoffs %": c.playoffs.round(0),
                           "#1 seed %": c.top_seed.round(1), "Conf finals %": c.cf.round(1), "Finals %": c.finals.round(1),
                           "Title %": c.title.round(1)})
        df.insert(0, "Seed", range(1, len(df) + 1))
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"Playoffs %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
                                    "Title %": st.column_config.ProgressColumn(min_value=0, max_value=max(ss.title.max(), 1), format="%.1f%%")})
    st.subheader("📏 Power ratings")
    r = model["ratings"].sort_values("net", ascending=False)
    st.dataframe(pd.DataFrame({"Team": [TEAMS[x]["name"] for x in r.index], "Net": r.net.round(1), "Offense": r.off.round(1),
                               "Defense (lower = better)": r.de.round(1), "Pace": r.pace.round(1), "W": r.w, "L": r.l}),
                 hide_index=True, width="stretch")


# ------------------------------- Races --------------------------------------
def page_races():
    model = model_now(); demo_banner(model)
    st.title("🏃 Races")
    pg = get_player_games(model)
    if len(pg) == 0:
        st.info("Season player stats haven't been built yet. The daily robot builds them automatically, or you can do it now (about 1–3 minutes).")
        if st.button("Build player stats now"):
            bar = st.progress(0.0)
            pg, k = update_player_games(model["ev"], max_new=1300, progress=bar.progress)
            st.session_state.pg = pg
            st.success(f"Added {k} games.")
            st.rerun()
    else:
        d = pg.copy()
        for c in ["min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
        d["date"] = pd.to_datetime(d.date)
        stat = st.segmented_control("Race", ["pts", "reb", "ast", "fg3m"], default="pts", format_func=STAT_NAMES.get, key="race_stat") or "pts"
        tab = season_player_table(d)
        team_gp = d.groupby("team").event_id.nunique()
        tab["team_gp"] = tab.team.map(team_gp)
        tab = tab[tab.gp >= tab.team_gp * 0.5]
        lead = tab.sort_values(stat, ascending=False).head(15).reset_index(drop=True)
        rng = np.random.default_rng(3)
        n = 3000
        g_left = (82 - lead.team_gp).clip(lower=0).to_numpy()
        avail = (lead.gp / lead.team_gp.clip(lower=1)).clip(upper=1).to_numpy()
        fut = rng.binomial(np.tile(g_left.astype(int), (n, 1)), np.tile(avail, (n, 1)))
        tot_now = (lead[stat] * lead.gp).to_numpy()
        sd_game = {"pts": 7.0, "reb": 3.0, "ast": 2.5, "fg3m": 1.4}[stat]
        talent = rng.normal(lead[stat].to_numpy() * 0.95 + tab[stat].mean() * 0.05, sd_game / np.sqrt(lead.gp.to_numpy() + 10), (n, len(lead)))
        fut_tot = talent * fut + rng.normal(0, sd_game, (n, len(lead))) * np.sqrt(fut)
        final_gp = lead.gp.to_numpy() + fut
        final = (tot_now + fut_tot) / np.maximum(final_gp, 1)
        final[final_gp < 58] = -1
        win = np.bincount(final.argmax(1), minlength=len(lead)) / n
        out = pd.DataFrame({"Player": lead.name, "Team": lead.team, "GP": lead.gp, STAT_NAMES[stat] + "/game": lead[stat].round(1),
                            "Proj. final": np.where((final > 0).mean(0) > 0, np.nanmean(np.where(final > 0, final, np.nan), 0), np.nan).round(1),
                            "Title odds %": (win * 100).round(1)})
        st.dataframe(out, hide_index=True, width="stretch",
                     column_config={"Title odds %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")})
        st.caption("Stat titles need 58+ games (70% of the season); the sim drops anyone who falls short.")

        top = lead.player_id.head(10).tolist()
        weeks = pd.date_range(d.date.min(), d.date.max() + pd.Timedelta(days=6), freq="W")
        frames = []
        for wk in weeks:
            sub = d[(d.date <= wk) & d.player_id.isin(top)]
            if len(sub) == 0:
                continue
            avg = sub.groupby(["player_id", "name", "team"])[stat].mean().reset_index()
            avg["week"] = wk.strftime("%b %d")
            frames.append(avg)
        if frames:
            race = pd.concat(frames)
            race["label"] = race.name.str.split().str[-1] + " (" + race.team + ")"
            fig = px.bar(race, x=stat, y="label", animation_frame="week", orientation="h", color="team",
                         color_discrete_map={t: TEAMS[t]["c1"] if TEAMS[t]["c1"] != "#000000" else "#888" for t in ABBRS},
                         range_x=[0, race[stat].max() * 1.1])
            fig.update_layout(title=f"{STAT_NAMES[stat]} per game race — press ▶", yaxis=dict(categoryorder="total ascending", title=""),
                              showlegend=False, **PLOT, height=460)
            st.plotly_chart(fig, width="stretch")

        st.subheader("🏅 MVP ladder (just for fun)")
        r = model["ratings"]
        tab2 = season_player_table(d)
        tab2 = tab2[tab2.gp >= 5]
        tab2["win%"] = tab2.team.map((r.w / r.gp.clip(lower=1)))
        tab2["score"] = (tab2.pts + 1.1 * tab2.reb + 1.5 * tab2.ast + 2 * (tab2.stl + tab2.blk) - tab2.tov) * (0.6 + 0.8 * tab2["win%"])
        lad = tab2.sort_values("score", ascending=False).head(10)
        st.dataframe(pd.DataFrame({"Player": lad.name, "Team": lad.team, "PTS": lad.pts.round(1), "REB": lad.reb.round(1),
                                   "AST": lad.ast.round(1), "Team win %": (lad["win%"] * 100).round(0), "Ladder score": lad.score.round(1)}),
                     hide_index=True, width="stretch")
        st.caption("Ladder score = all-around production × team success. Not an official MVP model.")

    st.subheader("📈 Title & #1 seed odds over time")
    hist = load_table("odds_history.csv")
    if len(hist) == 0:
        st.caption("The daily robot saves a snapshot every day, so this chart fills in over time.")
    else:
        hist["title"] = pd.to_numeric(hist.title, errors="coerce")
        hist["top_seed"] = pd.to_numeric(hist.top_seed, errors="coerce")
        which = st.radio("Show", ["title", "top_seed"], horizontal=True, format_func={"title": "Title odds", "top_seed": "#1 seed odds"}.get)
        latest = hist[hist.date == hist.date.max()].sort_values(which, ascending=False).team.head(8)
        hh = hist[hist.team.isin(latest)]
        fig = px.line(hh, x="date", y=which, color="team", markers=True,
                      color_discrete_map={t: TEAMS[t]["c1"] if TEAMS[t]["c1"] != "#000000" else "#888" for t in ABBRS})
        fig.update_layout(yaxis_title="%", **PLOT, height=360)
        st.plotly_chart(fig, width="stretch")


# ------------------------------- Rivalries ----------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def meetings(a, b, seasons):
    rows = []
    for s in seasons:
        for stype in (2, 3):
            try:
                rows += fetch_team_schedule(a, s, stype)
            except Exception:
                pass
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df.drop_duplicates("event_id")
    df = df[(df.state == "post") & (((df.home == a) & (df.away == b)) | ((df.home == b) & (df.away == a)))]
    return df.sort_values("start")


def page_rivalries():
    model = model_now(); demo_banner(model)
    st.title("⚔️ Rivalries")
    c = st.columns(3)
    a = c[0].selectbox("Team 1", ABBRS, index=ABBRS.index("LAL"), format_func=lambda t: TEAMS[t]["name"])
    b = c[1].selectbox("Team 2", ABBRS, index=ABBRS.index("BOS"), format_func=lambda t: TEAMS[t]["name"])
    yrs = c[2].slider("Seasons back", 1, 10, 10)
    if a == b:
        st.warning("Pick two different teams.")
        return
    season = model["season"]
    if model["demo"]:
        m = model["ev"]
        m = m[(m.state == "post") & (((m.home == a) & (m.away == b)) | ((m.home == b) & (m.away == a)))]
    else:
        with st.spinner("Pulling every meeting…"):
            m = meetings(a, b, tuple(range(season - yrs + 1, season + 1)))
    if len(m) == 0:
        st.info("No meetings found in that span.")
    else:
        m = m.copy()
        m["a_pts"] = np.where(m.home == a, m.home_pts, m.away_pts)
        m["b_pts"] = np.where(m.home == a, m.away_pts, m.home_pts)
        m["margin"] = m.a_pts - m.b_pts
        m["playoffs"] = m.season_type == 3
        aw = int((m.margin > 0).sum()); bw = len(m) - aw
        streak_team = a if m.margin.iloc[-1] > 0 else b
        sgn = np.sign(m.margin.to_numpy())[::-1]
        streak = int(np.argmax(sgn != sgn[0])) if (sgn != sgn[0]).any() else len(sgn)
        k = st.columns(4)
        k[0].markdown(f'<div class="card">{tlogo(a,48)}<div class="big">{aw}</div><div class="muted">{a} wins</div></div>', unsafe_allow_html=True)
        k[1].markdown(f'<div class="card">{tlogo(b,48)}<div class="big">{bw}</div><div class="muted">{b} wins</div></div>', unsafe_allow_html=True)
        k[2].metric("Average score", f"{m.a_pts.mean():.0f}–{m.b_pts.mean():.0f}", f"{a} margin {m.margin.mean():+.1f}", delta_color="off")
        k[3].metric("Current streak", f"{streak_team} {streak}", f"{int(m.playoffs.sum())} playoff games", delta_color="off")
        fig = go.Figure(go.Bar(x=m.date.astype(str), y=m.margin,
                               marker_color=[TEAMS[a]["c1"] if x > 0 else TEAMS[b]["c1"] for x in m.margin]))
        fig.update_layout(title=f"Every meeting (bars up = {a} won)", **PLOT, height=300, yaxis_title=f"{a} margin")
        st.plotly_chart(fig, width="stretch")
        big = m.loc[m.margin.abs().sort_values(ascending=False).index].head(3)
        st.markdown("**Biggest blowouts:** " + " · ".join(
            f"{r.date}: {a} {int(r.a_pts)}–{int(r.b_pts)} {b}" for _, r in big.iterrows()))
        show = m.sort_values("start", ascending=False).head(15)
        st.dataframe(pd.DataFrame({"Date": show.date, "Home": show.home, "Score": [f"{a} {int(x)}–{int(y)} {b}" for x, y in zip(show.a_pts, show.b_pts)],
                                   "Playoffs": np.where(show.playoffs, "🏆", "")}), hide_index=True, width="stretch")
    nxt = model["ev"]
    nxt = nxt[(nxt.state == "pre") & (((nxt.home == a) & (nxt.away == b)) | ((nxt.home == b) & (nxt.away == a)))].head(1)
    if len(nxt):
        g = nxt.iloc[0]
        pr = project(model, g.home, g.away)
        st.markdown(f"**Next meeting:** {g.date:%b %d} — {g.away} @ {g.home}. Model: {g.home} {pr['win']:.0%} to win, "
                    f"projected {g.away} {pr['a_pts']:.0f}–{pr['h_pts']:.0f} {g.home}.")


# ------------------------------- Beat the Bot -------------------------------
BOT = "🤖 Bot"


def pickem_board(pk, ev):
    res = ev[ev.state == "post"].set_index("event_id")
    p = pk.copy()
    p = p[p.event_id.isin(res.index)]
    if len(p) == 0:
        return pd.DataFrame(), p
    p["winner"] = p.event_id.map(lambda e: res.loc[e, "home"] if res.loc[e, "home_pts"] > res.loc[e, "away_pts"] else res.loc[e, "away"])
    p["right"] = p.pick == p.winner
    bot = p[p.name == BOT].set_index("event_id").pick
    p["bot_pick"] = p.event_id.map(bot)
    rows = []
    for name, g in p.groupby("name"):
        g = g.sort_values(["date", "event_id"])
        r = g.right.to_numpy()
        streak = 0
        for x in r[::-1]:
            if x == r[-1]:
                streak += 1
            else:
                break
        badges = []
        if r[-1] and streak >= 5:
            badges.append("🔥 Hot streak")
        if len(g) >= 20 and g.right.mean() >= 0.65:
            badges.append("🎯 Sharpshooter")
        if ((g.pick != g.bot_pick) & g.right & g.bot_pick.notna()).sum() >= 5:
            badges.append("🐶 Bot fader")
        if len(g) >= 50:
            badges.append("🏅 Grinder")
        rows.append(dict(Player=name, W=int(g.right.sum()), L=int((~g.right).sum()), Pct=g.right.mean(),
                         Streak=("W" if r[-1] else "L") + str(streak), Badges=" ".join(badges)))
    board = pd.DataFrame(rows).sort_values(["W", "Pct"], ascending=False).reset_index(drop=True)
    if (board.Player == BOT).any():
        bw = board.loc[board.Player == BOT, "W"].iloc[0]
        board.loc[(board.Player != BOT) & (board.W > bw), "Badges"] += " 🤖 Bot beater"
    board.insert(0, "#", range(1, len(board) + 1))
    return board, p


def page_pickem():
    model = model_now(); demo_banner(model)
    st.title("🤖 Beat the Bot")
    st.caption("Pick the winner of every game. The Bot picks too. Climb the leaderboard with your friends.")
    pk = load_table("pickem.csv")
    c = st.columns([2, 1])
    name = c[0].text_input("Your name", value=st.session_state.get("pick_name", ""), max_chars=20)
    code_needed = secret("PICKEM_CODE")
    code = c[1].text_input("Group code", type="password") if code_needed else ""
    if name:
        st.session_state.pick_name = name.strip()
    d = date_picker("pick_date", "Games on")
    pred = slate_predictions(model, d)
    open_g = pred[pred.state == "pre"] if len(pred) else pred
    if len(open_g) == 0:
        st.info("No games left to pick on this date.")
    else:
        mine = pk[(pk.name == name.strip()) & pk.event_id.isin(open_g.event_id)].set_index("event_id").pick if name else pd.Series(dtype=str)
        bot_picks = {r.event_id: r.pick for _, r in open_g.iterrows()}
        with st.form("picks"):
            choices = {}
            for _, r in open_g.iterrows():
                opts = [r.away, r.home]
                cur = mine.get(r.event_id)
                choices[r.event_id] = st.radio(f"{r.away} @ {r.home} · {r.status_text} · Bot: {bot_picks[r.event_id]} "
                                               f"({max(r.home_win, 1 - r.home_win):.0%})", opts,
                                               index=opts.index(cur) if cur in opts else None, horizontal=True,
                                               format_func=lambda t: TEAMS[t]["name"], key=f"pk_{r.event_id}")
            go_ = st.form_submit_button("✅ Lock in my picks", width="stretch")
        if go_:
            if not name.strip():
                st.error("Type your name first.")
            elif name.strip() == BOT:
                st.error("Nice try 🤖")
            elif code_needed and code != code_needed:
                st.error("Wrong group code.")
            else:
                new = pd.DataFrame([dict(date=str(d), event_id=e, name=name.strip(), pick=p,
                                         time=dt.datetime.now(ET).strftime("%H:%M")) for e, p in choices.items() if p])
                keep = pk[~((pk.name == name.strip()) & pk.event_id.isin(new.event_id))] if len(pk) else pk
                botrows = pd.DataFrame([dict(date=str(d), event_id=e, name=BOT, pick=p, time="") for e, p in bot_picks.items()
                                        if not ((pk.name == BOT) & (pk.event_id == e)).any()]) if len(new) else pd.DataFrame()
                save_table("pickem.csv", pd.concat([keep, new, botrows], ignore_index=True), f"picks by {name.strip()}")
                st.success(f"Locked in {len(new)} picks. Good luck!")
                st.balloons()
                pk = load_table("pickem.csv")
        today_p = pk[pk.event_id.isin(open_g.event_id) & (pk.name != BOT)]
        if len(today_p):
            st.markdown("**How everyone picked**")
            share = today_p.groupby(["event_id", "pick"]).size().unstack(fill_value=0)
            for e, row in share.iterrows():
                gm = open_g[open_g.event_id == e].iloc[0]
                tot = row.sum()
                st.markdown(f'{gm.away} {row.get(gm.away, 0)} {prob_bar(row.get(gm.away, 0) / tot, gm.away, gm.home)} {row.get(gm.home, 0)} {gm.home}',
                            unsafe_allow_html=True)

    st.subheader("🏆 Leaderboard")
    board, graded = pickem_board(pk, model["ev"])
    if len(board) == 0:
        st.caption("No graded picks yet — check back after tonight's games.")
    else:
        st.dataframe(board, hide_index=True, width="stretch",
                     column_config={"Pct": st.column_config.NumberColumn(format="percent")})
        if name and (board.Player == name.strip()).any() and board.loc[board.Player == name.strip(), "#"].iloc[0] == 1:
            st.success(f"👑 {name.strip()} is #1!")
            if st.session_state.get("crowned") != name.strip():
                st.session_state.crowned = name.strip()
                st.snow()


# ------------------------------- Tracker & backtest -------------------------
@st.cache_data(ttl=86400, show_spinner="Running the backtest (re-rating every team before every game day)…")
def backtest(season):
    ev, demo = load_events(season)
    if demo:
        prior = {t: float(TEAMS[t]["prior"]) for t in ABBRS}
    else:
        prior, _ = last_season_prior(season)
    played = ev[(ev.state == "post") & (ev.season_type == 2)]
    rows = []
    for d in sorted(played.date.unique()):
        r, L, P = compute_ratings(ev, prior, asof=d)
        m = dict(ratings=r, L=L, P=P, ev=ev)
        for _, g in played[played.date == d].iterrows():
            fh, fa = fatigue(ev, g.home, g.event_id), fatigue(ev, g.away, g.event_id)
            pr = project(m, g.home, g.away, fh["pts"], fa["pts"], bool(g.neutral))
            rows.append(dict(date=d, home=g.home, away=g.away, p=pr["win"], mm=pr["margin"], mt=pr["total"],
                             margin=g.home_pts - g.away_pts, total=g.home_pts + g.away_pts,
                             spread=g.mkt_spread, ou=g.mkt_total))
    return pd.DataFrame(rows)


def report(df, label):
    hit = ((df.p >= 0.5) == (df.margin > 0)).mean()
    brier = ((df.p - (df.margin > 0)) ** 2).mean()
    mae = (df.mm - df.margin).abs().mean()
    k = st.columns(4)
    k[0].metric(f"{label}: picks right", f"{hit:.1%}", f"{len(df)} games", delta_color="off")
    k[1].metric("Brier score", f"{brier:.3f}", "lower is better (0.25 = coin flip)", delta_color="off")
    k[2].metric("Avg miss on margin", f"{mae:.1f} pts")
    s = df.dropna(subset=["spread"])
    s = s[(s.mm + s.spread).abs() >= 2]
    if len(s):
        cover = s.margin + s.spread
        win = ((s.mm + s.spread > 0) == (cover > 0)) & (cover != 0)
        loss = ((s.mm + s.spread > 0) != (cover > 0)) & (cover != 0)
        roi = (win.sum() * 0.909 - loss.sum()) / max(win.sum() + loss.sum(), 1)
        k[3].metric("ATS (edge ≥ 2 pts)", f"{win.sum()}-{loss.sum()}", f"ROI {roi:+.1%} at -110", delta_color="off")
    df = df.copy()
    df["bin"] = (df.p * 10).clip(0, 9.99).astype(int)
    cal = df.groupby("bin").agg(pred=("p", "mean"), actual=("margin", lambda x: (x > 0).mean()), n=("p", "size")).reset_index()
    fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#666"), name="perfect"),
                     go.Scatter(x=cal.pred, y=cal.actual, mode="markers+lines", marker=dict(size=np.sqrt(cal.n) * 3 + 4, color="#ffd34d"),
                                name="model")])
    fig.update_layout(title="Calibration: when the model says X%, does the home team win X%?", xaxis_title="model home win %",
                      yaxis_title="actual", **PLOT, height=330)
    st.plotly_chart(fig, width="stretch")


def page_tracker():
    model = model_now(); demo_banner(model)
    st.title("📈 Tracker & Backtest")
    preds = grade_predictions(load_table("predictions.csv"), model["ev"])
    st.subheader("Saved predictions")
    if len(preds) == 0 or not preds.final.any():
        st.caption("Nothing graded yet. The daily robot saves predictions before tip-off (or use the button on Tonight's Slate).")
    else:
        f = preds[preds.final].copy()
        df = pd.DataFrame(dict(p=f.home_win, mm=f.model_margin, mt=f.model_total, margin=f.home_pts - f.away_pts,
                               total=f.home_pts + f.away_pts, spread=f.mkt_spread, ou=f.mkt_total))
        report(df, "Live")
        show = f.sort_values("date", ascending=False).head(40)
        st.dataframe(pd.DataFrame({"Date": show.date, "Game": show.away + " @ " + show.home, "Pick": show.pick,
                                   "Win %": (np.maximum(show.home_win, 1 - show.home_win) * 100).round(0),
                                   "Final": show.away_pts.astype(int).astype(str) + "–" + show.home_pts.astype(int).astype(str),
                                   "Right?": np.where(show.correct.astype(bool), "✅", "❌"), "ATS": show.ats}),
                     hide_index=True, width="stretch")
    st.subheader("🔬 Backtest")
    st.caption("Replays a season game by game, using only what the model would have known that morning.")
    which = st.radio("Season", ["This season so far", "Last season"], horizontal=True)
    s = model["season"] if which.startswith("This") else model["season"] - 1
    if st.button("Run backtest"):
        st.session_state.bt = s
    if st.session_state.get("bt") == s:
        try:
            bt = backtest(s)
            if len(bt):
                report(bt, "Backtest")
            else:
                st.info("No finished games to test yet.")
        except Exception as e:
            st.error(f"Backtest failed: {e}")


# ------------------------------- Daily robot ---------------------------------
def run_robot():
    """What the GitHub Action runs each day: predict, bot picks, title-odds snapshot, player stats, wake the app."""
    season = season_for(today_et())
    model = build_model(season)
    if model["demo"]:
        print("Couldn't reach ESPN — nothing saved.")
        return 1
    d = today_et()
    print(f"Season {season}: {len(model['ev'])} games loaded")
    pg, k = update_player_games(model["ev"], max_new=1300)
    print(f"Player box scores added: {k}")
    try:
        inj = fetch_injuries()
    except Exception:
        inj = None
    games = model["ev"][(model["ev"].date == d)].copy()
    try:
        fresh = fetch_day(d).set_index("event_id")
        for c in ["state", "mkt_spread", "mkt_total", "ml_home", "ml_away"]:
            upd = games.event_id.map(fresh[c])
            games[c] = upd.where(upd.notna(), games[c])
    except Exception:
        pass
    if len(games):
        pen = slate_injury_penalties(model, inj, pg)
        pred = predict_slate(model, games, pen)
        n = save_predictions(pred)
        print(f"Predictions saved/updated: {n}")
        pk = load_table("pickem.csv")
        pre = pred[pred.state == "pre"]
        bot = pd.DataFrame([dict(date=str(d), event_id=r.event_id, name=BOT, pick=r.pick, time="") for _, r in pre.iterrows()])
        if len(bot):
            pk = pk[~((pk.name == BOT) & pk.event_id.isin(bot.event_id))] if len(pk) else pk
            save_table("pickem.csv", pd.concat([pk, bot], ignore_index=True), "bot picks")
            print(f"Bot picks: {len(bot)}")
    hist = load_table("odds_history.csv")
    if not (hist.date == str(d)).any():
        ss = season_sim(model, n=1500, key=("robot", str(d)))
        snap = pd.DataFrame(dict(date=str(d), team=ss.index, playoffs=ss.playoffs.round(1), top_seed=ss.top_seed.round(1),
                                 title=ss.title.round(1)))
        save_table("odds_history.csv", pd.concat([hist, snap], ignore_index=True), "odds snapshot")
        print("Title-odds snapshot saved")
    url = os.environ.get("APP_URL")
    if url:
        try:
            requests.get(url, timeout=30)
            print("Pinged the app")
        except Exception as e:
            print(f"Ping failed: {e}")
    return 0


# ------------------------------- Start ----------------------------------------
if ROBOT_MODE:
    sys.exit(run_robot())

st.set_page_config(page_title="NBA Sim", page_icon="🏀", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)
PAGES = {
    "slate": st.Page(page_slate, title="Slate", icon="🏀", url_path="slate", default=True),
    "gc": st.Page(page_game_center, title="Game Center", icon="🎮", url_path="game"),
    "bet": st.Page(page_betting, title="Betting", icon="💰", url_path="betting"),
    "season": st.Page(page_season, title="Season", icon="🏆", url_path="season"),
    "races": st.Page(page_races, title="Races", icon="🏃", url_path="races"),
    "riv": st.Page(page_rivalries, title="Rivalries", icon="⚔️", url_path="rivalries"),
    "pick": st.Page(page_pickem, title="Beat the Bot", icon="🤖", url_path="pickem"),
    "track": st.Page(page_tracker, title="Tracker", icon="📈", url_path="tracker"),
}
st.navigation(list(PAGES.values()), position="top").run()