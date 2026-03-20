# ─────────────────────────────────────────────────────────────────
#  AirSight — server.py  (Render backend)
#  Flask REST API — frontend lives separately on Vercel
#
#  Deploy on Render:
#    Build:  pip install -r requirements.txt
#    Start:  gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
# ─────────────────────────────────────────────────────────────────
import os, re, json
from datetime import date
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)   # Allow requests from Vercel frontend

BASE = Path(__file__).parent

MODEL_PATH   = BASE / "airfare_price_model.joblib"
MEDIANS_PATH = BASE / "route_medians.json"
META_PATH    = BASE / "model_meta.json"
DATA_PATH    = BASE / "Flight_Data.xlsx"

CITY_LATLON = {
    "Delhi":     (28.5562, 77.1000), "Mumbai":    (19.0896, 72.8656),
    "Chennai":   (12.9941, 80.1709), "Banglore":  (13.1986, 77.7066),
    "Hyderabad": (17.2403, 78.4294), "Kolkata":   (22.6547, 88.4467),
    "Cochin":    (10.1520, 76.4019),
}

ROUTE_DURATION = {
    "Banglore|Delhi":155,"Delhi|Banglore":155,"Delhi|Mumbai":130,"Mumbai|Delhi":130,
    "Delhi|Kolkata":130,"Kolkata|Delhi":130,"Delhi|Chennai":165,"Chennai|Delhi":165,
    "Delhi|Cochin":210,"Cochin|Delhi":210,"Delhi|Hyderabad":120,"Hyderabad|Delhi":120,
    "Mumbai|Banglore":90,"Banglore|Mumbai":90,"Mumbai|Kolkata":150,"Kolkata|Mumbai":150,
    "Mumbai|Chennai":110,"Chennai|Mumbai":110,"Mumbai|Hyderabad":80,"Hyderabad|Mumbai":80,
    "Kolkata|Banglore":175,"Banglore|Kolkata":175,"Chennai|Banglore":60,"Banglore|Chennai":60,
    "Chennai|Kolkata":155,"Kolkata|Chennai":155,"Chennai|Hyderabad":90,"Hyderabad|Chennai":90,
    "Chennai|Cochin":80,"Cochin|Chennai":80,"Kolkata|Hyderabad":120,"Hyderabad|Kolkata":120,
    "Mumbai|Cochin":100,"Cochin|Mumbai":100,
}

def haversine_km(a, b):
    if not (a and b): return 1000.0
    lat1,lon1=a; lat2,lon2=b; R=6371.0
    h=np.sin(np.radians(lat2-lat1)/2)**2+np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(np.radians(lon2-lon1)/2)**2
    return float(2*R*np.arcsin(np.sqrt(h)))

def train_model():
    print("🔧 Training model from Flight_Data.xlsx …")
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    df = pd.read_excel(DATA_PATH)
    df = df.dropna(subset=["Airline","Source","Destination","Price","Date_of_Journey","Dep_Time","Duration"]).copy()
    df["Destination"] = df["Destination"].replace("New Delhi","Delhi")
    df["Source"]      = df["Source"].replace("New Delhi","Delhi")
    df["TravelDate"]  = pd.to_datetime(df["Date_of_Journey"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["TravelDate"]).copy()

    def ph(t):
        try: return int(str(t).split(":")[0])
        except: return 10
    def pd2(d):
        d=str(d); h=re.search(r'(\d+)h',d); m2=re.search(r'(\d+)m',d)
        return int(h.group(1))*60+(int(m2.group(1)) if m2 else 0) if h else 60
    def ps(s):
        s=str(s).lower()
        if "non" in s: return 0
        m2=re.search(r'(\d+)',s); return int(m2.group(1)) if m2 else 1

    df["DepHour"]=df["Dep_Time"].apply(ph)
    df["DurationMin"]=df["Duration"].apply(pd2)
    df["NumStops"]=df["Total_Stops"].apply(ps)
    df["MonthFeat"]=df["TravelDate"].dt.month
    df["DowFeat"]=df["TravelDate"].dt.dayofweek
    df["IsWeekend"]=df["DowFeat"].isin([5,6]).astype(int)
    rng=np.random.default_rng(42)
    df["DaysToDepart"]=rng.integers(0,61,size=len(df))
    df["DistanceKM"]=[haversine_km(CITY_LATLON.get(s),CITY_LATLON.get(d)) for s,d in zip(df["Source"],df["Destination"])]

    route_med=df.groupby(["Source","Destination"])["Price"].median().reset_index()
    route_med.columns=["Source","Destination","RouteMedian"]
    df=df.merge(route_med,on=["Source","Destination"],how="left")
    df["RouteMedian"]=df["RouteMedian"].fillna(df["Price"].median())
    q1,q99=df["Price"].quantile([0.01,0.99])
    df=df[df["Price"].between(q1,q99)].copy()

    FEATS=["Source","Destination","Airline","MonthFeat","DowFeat","IsWeekend",
           "DepHour","DaysToDepart","DurationMin","NumStops","DistanceKM","RouteMedian"]
    CAT=["Source","Destination","Airline"]
    NUM=[c for c in FEATS if c not in CAT]

    df=df.sort_values("TravelDate").reset_index(drop=True)
    split=int(len(df)*0.8); tr,te=df.iloc[:split],df.iloc[split:]
    pre=ColumnTransformer([
        ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),CAT),
        ("num","passthrough",NUM)
    ])
    pipe=Pipeline([
        ("prep",pre),
        ("model",RandomForestRegressor(n_estimators=150,max_depth=20,min_samples_leaf=3,n_jobs=-1,random_state=42))
    ])
    pipe.fit(tr[FEATS],tr["Price"])
    pred=pipe.predict(te[FEATS])
    print(f"✅ R²={r2_score(te['Price'],pred):.3f}  MAE=₹{mean_absolute_error(te['Price'],pred):,.0f}")
    joblib.dump(pipe, MODEL_PATH)

    rm={f"{r['Source']}|{r['Destination']}":round(float(r['RouteMedian'])) for _,r in route_med.iterrows()}
    with open(MEDIANS_PATH,"w") as f: json.dump(rm,f)

    airlines=sorted([a for a in df["Airline"].unique()
                     if "multiple" not in a.lower() and "trujet" not in a.lower()])
    meta={"sources":sorted(df["Source"].unique().tolist()),
          "destinations":sorted(df["Destination"].unique().tolist()),
          "airlines":airlines}
    with open(META_PATH,"w") as f: json.dump(meta,f)
    print("✅ Saved: model, route_medians.json, model_meta.json")
    return pipe, rm, meta

# ── Load or auto-train ────────────────────────────────────────
try:
    price_pipe     = joblib.load(MODEL_PATH)
    with open(MEDIANS_PATH) as f: ROUTE_MEDIANS = json.load(f)
    with open(META_PATH)    as f: META          = json.load(f)
    print("✅ Model loaded from disk")
except Exception:
    price_pipe, ROUTE_MEDIANS, META = train_model()

REAL_AIRLINES = META.get("airlines", [])

# ── API endpoints ─────────────────────────────────────────────
@app.route("/")
def health():
    return jsonify({"status":"ok","message":"AirSight API running","model_loaded": price_pipe is not None})

@app.route("/options")
def options():
    return jsonify({"status":"success","sources":META["sources"],"destinations":META["destinations"]})

@app.route("/predict", methods=["POST"])
def predict():
    if price_pipe is None:
        return jsonify({"status":"error","message":"Model not loaded"}), 500

    data        = request.get_json(force=True) or {}
    src         = str(data.get("Source","")).strip()
    dst         = str(data.get("Destination","")).strip()
    travel_date = data.get("TravelDate","")
    hour        = int(data.get("Hour", 10))
    stops       = int(data.get("NumStops", 0))

    if not src or not dst or not travel_date:
        return jsonify({"status":"error","message":"Provide Source, Destination, TravelDate"}), 400
    if src == dst:
        return jsonify({"status":"error","message":"Source and Destination must differ"}), 400
    try:
        d_travel = pd.to_datetime(travel_date)
    except Exception:
        return jsonify({"status":"error","message":"Invalid date format"}), 400

    today          = pd.Timestamp(date.today())
    days_to_depart = max(0, (d_travel - today).days)
    month          = int(d_travel.month)
    dow            = int(d_travel.dayofweek)
    is_weekend     = 1 if dow in (5, 6) else 0
    distance_km    = haversine_km(CITY_LATLON.get(src), CITY_LATLON.get(dst))
    route_median   = float(ROUTE_MEDIANS.get(f"{src}|{dst}", 6000))
    duration       = ROUTE_DURATION.get(f"{src}|{dst}", 120) + stops * 90

    rows = [{"Source":src,"Destination":dst,"Airline":a,
             "MonthFeat":month,"DowFeat":dow,"IsWeekend":is_weekend,
             "DepHour":hour,"DaysToDepart":days_to_depart,
             "DurationMin":duration,"NumStops":stops,
             "DistanceKM":float(distance_km),"RouteMedian":route_median}
            for a in REAL_AIRLINES]

    prices = price_pipe.predict(pd.DataFrame(rows))

    def tier(name):
        n = name.lower()
        if "business" in n or "premium" in n: return "premium"
        if "vistara" in n or "air india" in n or "jet" in n: return "full-service"
        return "budget"

    ranked = sorted(
        [{"airline":a,"fare":round(float(p),2),"tier":tier(a)}
         for a,p in zip(REAL_AIRLINES,prices)],
        key=lambda x: x["fare"]
    )
    best      = ranked[0]
    fare_conf = int(np.clip(92 - 0.5*days_to_depart, 50, 92))

    return jsonify({
        "status":            "success",
        "fare":              best["fare"],
        "fare_low":          round(best["fare"] * 0.90),
        "fare_high":         round(best["fare"] * 1.10),
        "fare_confidence":   fare_conf,
        "airline":           best["airline"],
        "airline_tier":      best["tier"],
        "all_airlines":      ranked,
        "days_to_departure": days_to_depart,
        "distance_km":       round(distance_km),
        "route_median":      round(route_median),
        "duration_min":      duration,
        "num_stops":         stops,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀  AirSight API →  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
