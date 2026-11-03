# =============================================================================
# TRAVEL BUSINESS OPERATIONS PREDICTION SYSTEM  (Horizon Global Travel Management)
# A Streamlit app that uses the five saved models from the notebook.
#
#   Page                      Business problem                        Model
#   Booking Cancellations     Which bookings are likely to cancel?    Logistic Regression
#   Trip Cost Forecast        What will this trip cost?               Hist Gradient Boosting
#   Policy Compliance         Is this booking likely out of policy?   Logistic Regression
#   Customer Satisfaction     Will this ticket end dissatisfied?      Logistic Regression
#   Booking Amount Anomalies  Is this booking amount unusual?         Isolation Forest
#
# HOW TO RUN (VS Code terminal, inside the project folder):
#   pip install streamlit pandas numpy scikit-learn joblib
#   python -m streamlit run app.py
#
# FOLDER LAYOUT (the app finds the model files by itself):
#   app.py
#   models/business_objectives/01_booking_cancellations/model.joblib   (and the other 4 folders)
# =============================================================================

import itertools
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Travel Business Operations Prediction System"
APP_SUBTITLE = "Horizon Global Travel Management"

st.set_page_config(page_title=APP_TITLE, layout="wide")


# -----------------------------------------------------------------------------
# 1. SETTINGS YOU MAY WANT TO CHANGE
# -----------------------------------------------------------------------------
# Risk tiers: below the first number = Low, between = Medium, above the second = High.
# The numbers come from checking each model on its held-out test rows.
TIER_CUTS = (0.40, 0.60)

# Typical size of the cost model's error on test data (WAPE 18.4%). Used for the planning range.
COST_TYPICAL_ERROR = 0.184

# Which travel classes belong to which service (taken from the booking data).
CLASS_BY_SERVICE = {
    "Flight": ["Economy", "Premium Economy", "Business", "First"],
    "Hotel": ["Standard", "Superior", "Suite"],
    "Car Rental": ["Economy", "SUV", "Luxury"],
}

# City -> (country, region). Same map the notebook used; codes and spellings are accepted too.
CITY_INFO = {
    "lagos": ("Nigeria", "Africa"), "los": ("Nigeria", "Africa"),
    "london": ("United Kingdom", "Europe"), "lon": ("United Kingdom", "Europe"),
    "new york": ("United States", "North America"), "nyc": ("United States", "North America"),
    "dubai": ("United Arab Emirates", "Middle East"), "dxb": ("United Arab Emirates", "Middle East"),
    "frankfurt": ("Germany", "Europe"), "fra": ("Germany", "Europe"),
    "johannesburg": ("South Africa", "Africa"), "joburg": ("South Africa", "Africa"),
    "jnb": ("South Africa", "Africa"),
    "mumbai": ("India", "South Asia"), "bombay": ("India", "South Asia"),
    "bom": ("India", "South Asia"),
    "sao paulo": ("Brazil", "South America"), "gru": ("Brazil", "South America"),
    "singapore": ("Singapore", "Southeast Asia"), "sin": ("Singapore", "Southeast Asia"),
    "sydney": ("Australia", "Oceania"), "syd": ("Australia", "Oceania"),
}
CITY_CHOICES = ["Lagos", "London", "New York", "Dubai", "Frankfurt", "Johannesburg",
                "Mumbai", "Sao Paulo", "Singapore", "Sydney"]

YES_NO = ["Yes", "No"]


# -----------------------------------------------------------------------------
# 2. FINDING AND LOADING THE SAVED MODELS
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent


def find_model_dir():
    """Look for the models folder next to app.py."""
    for candidate in (ROOT / "models" / "business_objectives", ROOT / "business_objectives", ROOT / "models"):
        if (candidate / "01_booking_cancellations").is_dir():
            return candidate
    return None


MODEL_DIR = find_model_dir()


@st.cache_resource
def load_bundle(folder):
    """model.joblib holds a dictionary: the full pipeline, the scaler and the feature details."""
    return joblib.load(MODEL_DIR / folder / "model.joblib")


# -----------------------------------------------------------------------------
# 3. SMALL HELPERS
# -----------------------------------------------------------------------------
SYNONYMS = {"y": "yes", "true": "yes", "1": "yes", "n": "no", "false": "no", "0": "no"}


def key(value):
    """Match spellings loosely: 'Semi-Flexible', 'semi flexible' and 'SEMI FLEXIBLE' all give 'semiflexible'."""
    text = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return SYNONYMS.get(text, text)


def normalize_text(value):
    """Trim, collapse spaces and lower-case, like the notebook did."""
    if pd.isna(value):
        return "missing"
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def normalize_city(value):
    if pd.isna(value):
        return "missing"
    text = str(value).strip().casefold().replace("\ufffd", "a")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


def parse_money(value):
    """Turn '$1,250.50', 'USD 1,250.50' or '(300)' into a number."""
    text = str(value).strip()
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"(?i)\bUSD\b", "", text)
    text = re.sub(r"[$,()]", "", text)
    text = re.sub(r"[^0-9.\-]", "", text)
    number = pd.to_numeric(text, errors="coerce")
    return -number if negative and pd.notna(number) else number


def days_between(start, end):
    """Whole days from start to end. Negative or unreadable dates become missing."""
    start = pd.to_datetime(start, format="mixed", errors="coerce")
    end = pd.to_datetime(end, format="mixed", errors="coerce")
    days = (end - start).dt.days.astype("float64")
    return days.where(days >= 0)


def tier_of(probability):
    low, high = TIER_CUTS
    return "High" if probability >= high else "Medium" if probability >= low else "Low"


def categories_of(pipeline):
    """The exact category spellings the model saw in training, per column."""
    pre = pipeline.named_steps["preprocessor"]
    encoder = pre.named_transformers_["categorical"].named_steps["encoder"]
    columns = pre.transformers_[1][2]
    return dict(zip(columns, encoder.categories_))


def options_for(bundle, column):
    """Clean drop-down choices for a column, built from the model's own categories."""
    groups = {}
    for category in categories_of(bundle["pipeline"])[column]:
        groups.setdefault(key(category), []).append(str(category))
    labels = []
    for variants in groups.values():
        variants = sorted(variants)
        mixed = [v for v in variants if v != v.upper() and v != v.lower()]
        label = mixed[0] if mixed else variants[0].title()
        labels.append("SUV" if label == "Suv" else label)
    return sorted(labels)


def expand(frame, categories):
    """
    The training data held several spellings of the same value (for example 'Phone',
    'PHONE', 'phone'). The model treats each spelling as separate, so we score every
    spelling and average the results. For clean columns this changes nothing.
    """
    spellings = {
        column: {k: [c for c in cats if key(c) == k] for k in {key(c) for c in cats}}
        for column, cats in categories.items()
    }
    rows, ids = [], []
    for position, record in enumerate(frame.to_dict("records")):
        choices = []
        for column in frame.columns:
            if column in spellings:
                choices.append(spellings[column].get(key(record[column]), [record[column]]))
            else:
                choices.append([record[column]])
        for combo in itertools.product(*choices):
            rows.append(combo)
            ids.append(position)
    return pd.DataFrame(rows, columns=frame.columns), np.array(ids)


def predict_probability(pipeline, features):
    """Chance of the 'yes' outcome (cancelled, out of policy, dissatisfied) for every row."""
    big, ids = expand(features, categories_of(pipeline))
    probability = pipeline.predict_proba(big)[:, 1]
    return pd.Series(probability).groupby(ids).mean().to_numpy()


def main_factors(pipeline, features, top=3):
    """For the linear models: which inputs push this one case up or down."""
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]
    big, _ = expand(features.iloc[[0]], categories_of(pipeline))
    contribution = pre.transform(big) * model.coef_[0]
    contribution = np.asarray(contribution).mean(axis=0)
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    row = features.iloc[0]

    merged = {}
    for name, value in zip(names, contribution):
        if name in row.index:                                   # numeric input
            label = f"{name.replace('_', ' ')} = {row[name]}"
        else:                                                   # one-hot category
            column = next(c for c in row.index if name.startswith(c + "_"))
            label = f"{column.replace('_', ' ')} = {row[column]}"
        merged[label] = merged.get(label, 0) + value

    ranked = sorted(merged.items(), key=lambda item: item[1])
    up = [(n, v) for n, v in reversed(ranked) if v > 0.02][:top]
    down = [(n, v) for n, v in ranked if v < -0.02][:top]
    return up, down


# -----------------------------------------------------------------------------
# 4. PAGE 1 - BOOKING CANCELLATIONS
# -----------------------------------------------------------------------------
def form_cancellations(bundle):
    c1, c2 = st.columns(2)
    booking_date = c1.date_input("Booking date", pd.Timestamp.today(), key="c_bd")
    travel_date = c2.date_input("Travel date", pd.Timestamp.today() + pd.Timedelta(days=18), key="c_td")
    service = c1.selectbox("Service type", options_for(bundle, "service_type"), key="c_sv")
    tier = c2.selectbox("Client tier", options_for(bundle, "client_tier"), key="c_tier")
    rating = c1.slider("Supplier service rating (1 to 5)", 1.0, 5.0, 3.7, 0.1, key="c_rt")
    flexibility = c2.selectbox("Cancellation flexibility", options_for(bundle, "cancellation_flexibility"), key="c_fx")
    channel = c1.selectbox("Booking channel", options_for(bundle, "booking_channel"), key="c_ch")
    policy = c2.selectbox("Booking within travel policy?", YES_NO, key="c_pol")
    return pd.DataFrame([{
        "booking_date": str(booking_date), "travel_date": str(travel_date),
        "service_type": service, "client_tier": tier, "service_rating": rating,
        "cancellation_flexibility": flexibility, "booking_channel": channel,
        "policy_compliant": policy,
    }])


def features_cancellations(raw, bundle):
    days = days_between(raw["booking_date"], raw["travel_date"])
    return pd.DataFrame({
        "advance_days": days,
        "advance_days_missing": days.isna().astype(int),
        "service_rating": pd.to_numeric(raw["service_rating"], errors="coerce"),
        "cancellation_flexibility": raw["cancellation_flexibility"].astype(object),
        "service_type": raw["service_type"].astype(object),
        "client_tier": raw["client_tier"].astype(object),
        "booking_channel": raw["booking_channel"].astype(object),
        "policy_compliant": raw["policy_compliant"].astype(object),
    })


# -----------------------------------------------------------------------------
# 5. PAGE 2 - TRIP COST FORECAST
# -----------------------------------------------------------------------------
def form_cost(bundle):
    c1, c2 = st.columns(2)
    service = c1.selectbox("Service type", list(CLASS_BY_SERVICE), key="k_sv")
    travel_class = c2.selectbox("Travel class", CLASS_BY_SERVICE[service], key="k_cl")
    origin = c1.selectbox("Origin city", CITY_CHOICES, index=0, key="k_or")
    destination = c2.selectbox("Destination city", CITY_CHOICES, index=1, key="k_de")
    booking_date = c1.date_input("Booking date", pd.Timestamp.today(), key="k_bd")
    travel_date = c2.date_input("Travel date", pd.Timestamp.today() + pd.Timedelta(days=18), key="k_td")
    return_date = c1.date_input("Return date", pd.Timestamp.today() + pd.Timedelta(days=23), key="k_rd")
    return pd.DataFrame([{
        "service_type": service, "travel_class": travel_class,
        "origin_city": origin, "destination_city": destination,
        "booking_date": str(booking_date), "travel_date": str(travel_date),
        "return_date": str(return_date),
    }])


def features_cost(raw, bundle):
    origin = raw["origin_city"].map(lambda c: CITY_INFO.get(normalize_city(c), (np.nan, np.nan)))
    destination = raw["destination_city"].map(lambda c: CITY_INFO.get(normalize_city(c), (np.nan, np.nan)))
    origin_country = origin.map(lambda x: x[0])
    destination_country = destination.map(lambda x: x[0])
    region = destination.map(lambda x: x[1])
    international = (origin_country != destination_country).astype(float)
    international = international.where(origin_country.notna() & destination_country.notna())

    booking = pd.to_datetime(raw["booking_date"], format="mixed", errors="coerce")
    travel = pd.to_datetime(raw["travel_date"], format="mixed", errors="coerce")
    back = pd.to_datetime(raw["return_date"], format="mixed", errors="coerce")
    swapped = travel.notna() & back.notna() & (back < travel)          # fix swapped dates
    travel, back = travel.where(~swapped, back), back.where(~swapped, travel)

    advance = (travel - booking).dt.days.astype("float64")
    length = (back - travel).dt.days.astype("float64")
    peak = travel.dt.month.isin([6, 7, 8, 12]).astype(float).where(travel.notna())
    return pd.DataFrame({
        "service_type": raw["service_type"].astype(object),
        "travel_class": raw["travel_class"].astype(object),
        "destination_region": region.astype(object),
        "international": international,
        "advance_days": advance.where(advance >= 0),
        "trip_length_days": length.where(length >= 0),
        "peak_season": peak,
    })


# -----------------------------------------------------------------------------
# 6. PAGE 3 - POLICY COMPLIANCE
# -----------------------------------------------------------------------------
def form_policy(bundle):
    c1, c2 = st.columns(2)
    service = c1.selectbox("Service type", list(CLASS_BY_SERVICE), key="p_sv")
    travel_class = c2.selectbox("Travel class", CLASS_BY_SERVICE[service], key="p_cl")
    grade = c1.selectbox("Traveller grade", options_for(bundle, "traveler_grade"), key="p_gr")
    channel = c2.selectbox("Booking channel", options_for(bundle, "booking_channel"), key="p_ch")
    partner = c1.selectbox("Preferred partner supplier?", YES_NO, key="p_pp")
    has_policy = c2.selectbox("Client has a travel policy?", YES_NO, key="p_hp")
    booking_date = c1.date_input("Booking date", pd.Timestamp.today(), key="p_bd")
    travel_date = c2.date_input("Travel date", pd.Timestamp.today() + pd.Timedelta(days=18), key="p_td")
    return pd.DataFrame([{
        "travel_class": travel_class, "traveler_grade": grade, "preferred_partner": partner,
        "booking_channel": channel, "has_travel_policy": has_policy,
        "booking_date": str(booking_date), "travel_date": str(travel_date),
    }])


def features_policy(raw, bundle):
    days = days_between(raw["booking_date"], raw["travel_date"])
    return pd.DataFrame({
        "travel_class": raw["travel_class"].astype(object),
        "traveler_grade": raw["traveler_grade"].astype(object),
        "preferred_partner": raw["preferred_partner"].astype(object),
        "advance_days": days,
        "advance_days_missing": days.isna().astype(int),
        "booking_channel": raw["booking_channel"].astype(object),
        "has_travel_policy": raw["has_travel_policy"].astype(object),
    })


# -----------------------------------------------------------------------------
# 7. PAGE 4 - CUSTOMER SATISFACTION
# -----------------------------------------------------------------------------
def form_satisfaction(bundle):
    c1, c2 = st.columns(2)
    category = c1.selectbox("Ticket category", options_for(bundle, "category"), key="s_ca")
    channel = c2.selectbox("Contact channel", options_for(bundle, "contact_channel"), key="s_ch")
    tier = c1.selectbox("Client tier", options_for(bundle, "client_tier"), key="s_ti")
    escalated = c2.selectbox("Ticket escalated?", YES_NO, index=1, key="s_es")
    first = c1.number_input("First response time (hours)", 0.0, 500.0, 2.0, 0.5, key="s_fr")
    resolution = c2.number_input("Resolution time (hours)", 0.0, 2000.0, 18.0, 0.5, key="s_re")
    return pd.DataFrame([{
        "category": category, "contact_channel": channel, "client_tier": tier,
        "escalated": escalated, "first_response_hours": first, "resolution_hours": resolution,
    }])


def features_satisfaction(raw, bundle):
    return pd.DataFrame({
        "resolution_hours": pd.to_numeric(raw["resolution_hours"], errors="coerce"),
        "first_response_hours": pd.to_numeric(raw["first_response_hours"], errors="coerce"),
        "escalated": raw["escalated"].astype(object),
        "category": raw["category"].astype(object),
        "contact_channel": raw["contact_channel"].astype(object),
        "client_tier": raw["client_tier"].astype(object),
    })


# -----------------------------------------------------------------------------
# 8. PAGE 5 - BOOKING AMOUNT ANOMALIES
# -----------------------------------------------------------------------------
def form_anomaly(bundle):
    c1, c2 = st.columns(2)
    service = c1.selectbox("Service type", list(CLASS_BY_SERVICE), key="a_sv")
    travel_class = c2.selectbox("Travel class", CLASS_BY_SERVICE[service], key="a_cl")
    city = c1.selectbox("Destination city", CITY_CHOICES, key="a_ci")
    amount = c2.number_input("Booking amount (USD)", 0.0, 1_000_000.0, 850.0, 10.0, key="a_am")
    return pd.DataFrame([{
        "service_type": service, "travel_class": travel_class,
        "destination_city": city, "amount_usd": amount,
    }])


def features_anomaly(raw, bundle):
    """Same five features the Isolation Forest was trained on."""
    ref = bundle["feature_reference"]
    lookup = ref["group_stats_lookup"]
    amount = raw["amount_usd"].map(parse_money).astype(float)
    log_amount = np.log(amount.where(amount > 0))

    service = raw["service_type"].map(normalize_text)
    travel_class = raw["travel_class"].map(normalize_text)
    region = raw["destination_city"].map(lambda c: CITY_INFO.get(normalize_city(c), (None, None))[1])

    medians, mads = [], []
    for group in zip(service, travel_class, region):
        stats = lookup.get(group)
        if stats is None or stats["group_count"] < ref["minimum_group_count"]:
            medians.append(ref["global_log_median"])                 # unseen or small group
            mads.append(ref["global_log_mad"])
        else:
            medians.append(stats["group_median"])
            mad = stats["group_mad"]
            mads.append(ref["global_log_mad"] if pd.isna(mad) or mad <= 0 else mad)

    deviation = log_amount.to_numpy() - np.asarray(medians, dtype=float)
    scale = np.maximum(1.4826 * np.asarray(mads, dtype=float), 0.05)
    features = pd.DataFrame({
        "log_amount_abs": np.log1p(amount.abs()).to_numpy(),
        "log_amount_deviation": deviation,
        "robust_peer_deviation": np.abs(deviation) / scale,
        "amount_nonpositive": (amount.notna() & amount.le(0)).astype(int).to_numpy(),
        "amount_missing": amount.isna().astype(int).to_numpy(),
    })
    features.attrs["typical_usd"] = np.exp(np.asarray(medians, dtype=float))
    return features


# -----------------------------------------------------------------------------
# 9. PAGE SETTINGS: ONE ENTRY PER PAGE
# -----------------------------------------------------------------------------
PAGES = {
    "Booking Cancellations": {
        "folder": "01_booking_cancellations", "kind": "risk",
        "question": "How likely is this booking to be cancelled?",
        "form": form_cancellations, "features": features_cancellations,
        "note": "Use the score to decide whom to contact first. About 1 in 6 bookings cancel, "
                "and many flagged bookings will not, so keep any follow-up low-cost.",
        "actions": {
            "High": "Contact the traveller early, confirm the plans and offer a flexible change.",
            "Medium": "Send a reminder and keep the booking on a watch list.",
            "Low": "No extra action needed.",
        },
        "template": pd.DataFrame([{
            "booking_date": "2026-03-01", "travel_date": "2026-03-20", "service_type": "Flight",
            "client_tier": "Gold", "service_rating": 3.8, "cancellation_flexibility": "Semi Flexible",
            "booking_channel": "Online Tool", "policy_compliant": "Yes"}]),
    },
    "Trip Cost Forecast": {
        "folder": "02_trip_cost_forecasting", "kind": "cost",
        "question": "What is this trip likely to cost?",
        "form": form_cost, "features": features_cost,
        "note": "Treat the result as a planning guide, not a quote. Typical error on test data was "
                "about 18%, and a few large bookings are harder to predict.",
        "template": pd.DataFrame([{
            "service_type": "Flight", "travel_class": "Business", "origin_city": "Lagos",
            "destination_city": "London", "booking_date": "2026-03-01",
            "travel_date": "2026-03-20", "return_date": "2026-03-27"}]),
    },
    "Policy Compliance": {
        "folder": "03_policy_compliance", "kind": "risk",
        "question": "How likely is this booking to be out of policy?",
        "form": form_policy, "features": features_policy,
        "note": "Use the score to decide which bookings to review first. It is a review signal, "
                "not a final compliance decision.",
        "actions": {
            "High": "Review before ticketing and ask for approver sign-off or an in-policy alternative.",
            "Medium": "Check the booking against the client's travel policy.",
            "Low": "Likely in policy. Normal approval flow.",
        },
        "template": pd.DataFrame([{
            "travel_class": "Business", "traveler_grade": "Manager", "preferred_partner": "Yes",
            "booking_channel": "Phone", "has_travel_policy": "Yes",
            "booking_date": "2026-03-01", "travel_date": "2026-03-20"}]),
    },
    "Customer Satisfaction": {
        "folder": "04_customer_satisfaction", "kind": "risk",
        "question": "How likely is this support ticket to end with a dissatisfied customer?",
        "form": form_satisfaction, "features": features_satisfaction,
        "note": "Resolution time and escalation are only known near the end of a ticket, so this "
                "page works best for closing or reviewing tickets. Use it to prioritise follow-up, never to reduce support.",
        "actions": {
            "High": "Follow up personally the same day and offer a fix or goodwill gesture.",
            "Medium": "Check progress and send the customer an update.",
            "Low": "Standard handling.",
        },
        "template": pd.DataFrame([{
            "category": "Complaint", "contact_channel": "Phone", "client_tier": "Gold",
            "escalated": "Yes", "first_response_hours": 3.5, "resolution_hours": 40}]),
    },
    "Booking Amount Anomalies": {
        "folder": "05_booking_amount_anomalies", "kind": "anomaly",
        "question": "Is this booking amount unusual compared with similar bookings?",
        "form": form_anomaly, "features": features_anomaly,
        "note": "A flag means 'unusual', not 'wrong'. Check each flagged booking against the supplier "
                "invoice before taking action.",
        "actions": {
            "High": "Flagged. Check the invoice and booking record against the supplier before payment.",
            "Medium": "Borderline. Spot-check if time allows.",
            "Low": "Amount looks normal for similar bookings.",
        },
        "template": pd.DataFrame([{
            "service_type": "Flight", "travel_class": "Economy",
            "destination_city": "Dubai", "amount_usd": 14500}]),
    },
}


# -----------------------------------------------------------------------------
# 10. SCORING: TURN RAW INPUTS INTO A RESULTS TABLE
# -----------------------------------------------------------------------------
def score(cfg, raw, bundle):
    raw = raw.reset_index(drop=True)
    features = cfg["features"](raw, bundle)
    out = raw.copy()
    pipeline = bundle["pipeline"]

    if cfg["kind"] == "risk":
        probability = predict_probability(pipeline, features)
        out["risk_percent"] = (probability * 100).round(1)
        out["risk_tier"] = [tier_of(p) for p in probability]

    elif cfg["kind"] == "cost":
        big, ids = expand(features, categories_of(pipeline))
        log_cost = pd.Series(pipeline.predict(big)).groupby(ids).mean().to_numpy()
        cost = np.exp(log_cost)
        out["predicted_cost_usd"] = cost.round(2)
        out["planning_low_usd"] = (cost * (1 - COST_TYPICAL_ERROR)).round(2)
        out["planning_high_usd"] = (cost * (1 + COST_TYPICAL_ERROR)).round(2)

    else:  # anomaly
        flagged = pipeline.predict(features) == -1
        anomaly_score = -pipeline.decision_function(features)
        out["anomaly_score"] = np.round(anomaly_score, 3)
        out["typical_amount_usd"] = np.round(features.attrs["typical_usd"], 2)
        out["flag"] = np.where(flagged, "Anomaly - review", "Normal")
        out["risk_tier"] = ["High" if f else "Medium" if s > -0.08 else "Low"
                            for f, s in zip(flagged, anomaly_score)]
    return out, features


def show_tier(tier, text):
    box = {"High": st.error, "Medium": st.warning, "Low": st.success}[tier]
    box(f"**{tier}**  -  {text}")


# -----------------------------------------------------------------------------
# 11. SHOWING A PAGE
# -----------------------------------------------------------------------------
def show_single_result(name, cfg, results, features, bundle):
    row = results.iloc[0]
    if cfg["kind"] == "risk":
        st.metric("Chance of this outcome", f"{row['risk_percent']:.1f}%")
        show_tier(row["risk_tier"], cfg["actions"][row["risk_tier"]])
        up, down = main_factors(bundle["pipeline"], features)
        left, right = st.columns(2)
        left.markdown("**Pushing the risk up**")
        for label, _ in up or [("Nothing stands out", 0)]:
            left.write(f"- {label}")
        right.markdown("**Pulling the risk down**")
        for label, _ in down or [("Nothing stands out", 0)]:
            right.write(f"- {label}")

    elif cfg["kind"] == "cost":
        st.metric("Estimated trip cost", f"${row['predicted_cost_usd']:,.0f}")
        st.info(f"Planning range: ${row['planning_low_usd']:,.0f} to ${row['planning_high_usd']:,.0f} "
                f"(typical error about {COST_TYPICAL_ERROR:.0%}).")

    else:
        c1, c2 = st.columns(2)
        c1.metric("Entered amount", f"${float(parse_money(row['amount_usd'])):,.0f}")
        c2.metric("Typical for similar bookings", f"${row['typical_amount_usd']:,.0f}")
        show_tier(row["risk_tier"], cfg["actions"][row["risk_tier"]])


def show_batch(name, cfg, bundle):
    st.write("1. Download the template, 2. fill in your bookings, 3. upload it to score every row.")
    template = cfg["template"]
    st.download_button("Download CSV template", template.to_csv(index=False),
                       file_name=f"{name.lower().replace(' ', '_')}_template.csv", key=f"tpl_{name}")
    upload = st.file_uploader("Upload your CSV", type="csv", key=f"up_{name}")
    if upload is None:
        return
    data = pd.read_csv(upload)
    missing = [c for c in template.columns if c not in data.columns]
    if missing:
        st.error("Your file is missing these columns: " + ", ".join(missing))
        return
    results, _ = score(cfg, data[list(template.columns)], bundle)
    st.dataframe(results)
    if "risk_tier" in results:
        st.bar_chart(results["risk_tier"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0))
    st.download_button("Download scored results", results.to_csv(index=False),
                       file_name=f"{name.lower().replace(' ', '_')}_scored.csv", key=f"dl_{name}")


def show_page(name):
    cfg = PAGES[name]
    bundle = load_bundle(cfg["folder"])
    st.header(name)
    st.write(cfg["question"])
    st.caption(cfg["note"])

    single, batch = st.tabs(["Single prediction", "Many bookings (CSV)"])
    with single:
        raw = cfg["form"](bundle)
        if st.button("Predict", key=f"go_{name}"):
            results, features = score(cfg, raw, bundle)
            show_single_result(name, cfg, results, features, bundle)
    with batch:
        show_batch(name, cfg, bundle)


# -----------------------------------------------------------------------------
# 12. HOME PAGE
# -----------------------------------------------------------------------------
def show_home():
    st.title(APP_TITLE)
    st.subheader(APP_SUBTITLE)
    st.write("Five models help the operations team decide where to look first. "
             "Pick a page on the left, enter a booking or ticket, and press Predict. "
             "Or upload a CSV to score many at once.")

    summary = pd.DataFrame([
        ["Booking Cancellations", "Which bookings may cancel?", "Logistic Regression",
         "ROC AUC 0.63. Catches 59% of cancellations"],
        ["Trip Cost Forecast", "What will a trip cost?", "Hist Gradient Boosting",
         "Average error about $232 per booking (18%)"],
        ["Policy Compliance", "Which bookings may break policy?", "Logistic Regression",
         "ROC AUC 0.68. Catches 62% of out-of-policy bookings"],
        ["Customer Satisfaction", "Which tickets may end badly?", "Logistic Regression",
         "ROC AUC 0.95. Catches 85% of dissatisfied customers"],
        ["Booking Amount Anomalies", "Which amounts look unusual?", "Isolation Forest",
         "ROC AUC 0.95. Catches 72% of anomalies"],
    ], columns=["Page", "Business question", "Model", "Result on unseen data"])
    st.table(summary)

    st.info("These models support decisions; they do not replace people. Use the result to decide "
            "where to look first, and keep a person in the loop for every flagged case.")
    st.caption(f"Model files loaded from: {MODEL_DIR}")


# -----------------------------------------------------------------------------
# 13. MAIN: SIDEBAR MENU
# -----------------------------------------------------------------------------
try:
    if MODEL_DIR is None:
        st.error("Model files not found. Put the 'models' folder (with business_objectives inside) "
                 "in the same folder as app.py.")
        st.stop()

    st.sidebar.title(APP_TITLE)
    choice = st.sidebar.radio("Go to", ["Home"] + list(PAGES))

    if choice == "Home":
        show_home()
    else:
        show_page(choice)
except Exception as error:      # show any problem on the page instead of a blank screen
    st.error("Something went wrong. The details are below.")
    st.exception(error)