# Horizon Global Travel Management: Predicting Cancellations, Cost, Policy Compliance and Service Quality

*An end-to-end Python machine learning project on dirty multi-table corporate travel data.*

![Python](https://img.shields.io/badge/Python-3.x-blue) ![pandas](https://img.shields.io/badge/pandas-data%20cleaning-informational) ![scikit-learn](https://img.shields.io/badge/scikit--learn-modelling-orange) ![Status](https://img.shields.io/badge/status-complete-brightgreen)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Quality Assessment](#3-data-quality-assessment)
4. [Data Cleaning & Error Correction](#4-data-cleaning--error-correction)
5. [Feature Engineering](#5-feature-engineering)
6. [Data Modeling & Relational Schema](#6-data-modeling--relational-schema)
7. [Business Analysis](#7-business-analysis)
8. [Exploratory Data Analysis](#8-exploratory-data-analysis)
9. [Machine Learning](#9-machine-learning)
10. [Data Visualization & Interpretation](#10-data-visualization--interpretation)
11. [Validation and Quality Assurance](#11-validation-and-quality-assurance)
12. [Key Findings & Insights](#12-key-findings--insights)
13. [Recommendations](#13-recommendations)
14. [Conclusion](#14-conclusion)

---

## 1. Project Overview

A travel management company (TMC) books flights, hotels and cars for corporate clients around the world. Its data lives in five separate tables, and it arrived messy: duplicate rows, dates in six different formats, money stored as text, spelling variants for the same country or city, and bookings pointing at clients that don't exist.

This project takes that raw data from first load to working models. I cleaned it, engineered features, organised it into a star schema, analysed the business, and trained machine learning models against five objectives, each judged against a simple baseline.

| # | Business objective | Question it answers | Approach |
|---|---|---|---|
| 1 | Reduce cancellations | Which bookings are likely to be cancelled? | Classification |
| 2 | Forecast trip cost | What will a booking cost, for client budgeting? | Regression |
| 3 | Understand policy compliance | Which factors drive out-of-policy bookings? | Classification + permutation importance |
| 4 | Prioritise support | Which tickets will escalate? | Classification (ROC / AUC) |
| 5 | Audit billing | Which bookings look unusual? | Unsupervised anomaly detection |

**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook.

---

---

## 2. Dataset Overview

The data is synthetic, generated with a seeded script so anyone can reproduce it. Five tables feed the analysis:

| Table | Type | What it holds | Raw rows | Clean rows |
|---|---|---|---|---|
| `dim_clients` | Dimension | Corporate clients, tier, budget | 64 | 58 |
| `dim_travelers` | Dimension | Employees who travel, grade, loyalty | 1,225 | 1,150 |
| `dim_suppliers` | Dimension | Airlines, hotels, car rental firms | 53 | 50 |
| `fact_bookings` | Fact | Every booking with amount and dates | 6,060 | 5,528 |
| `fact_support_tickets` | Fact | Support tickets with response times and CSAT | 2,530 | 2,263 |

A `data_dictionary.csv` file documents every column.

The cleaned data became a star schema of 3 dimensions and 2 facts. Two flattened views were built for modelling: `bookings_full` (5,528 × 64) and `tickets_full` (2,263 × 52).

---

## 3. Data Quality Assessment

Before changing anything, I audited every raw table with five checks: structure, missing values, duplicates, sample rows and numeric ranges.

### 3.1 Structure: columns and data types

```python
for label, df in tables.items():
    print('=' * 70)
    print(f'{label}  shape={df.shape}')
    print('-' * 70)
    for col in df.columns:
        print(f'  {col:<28} {df[col].dtype}')
```

```text
======================================================================
dim_clients  shape=(64, 8)
----------------------------------------------------------------------
  client_id                    str
  client_name                  str
  country                      str
  industry                     str
  client_tier                  str
  contract_start               str
  annual_travel_budget_usd     str
  has_travel_policy            str
======================================================================
dim_travelers  shape=(1225, 7)
----------------------------------------------------------------------
  traveler_id                  str
  client_id                    str
  grade                        str
  home_country                 str
  years_with_employer          float64
  loyalty_member               str
  trips_last_year              float64
======================================================================
dim_suppliers  shape=(53, 7)
----------------------------------------------------------------------
  supplier_id                  str
  supplier_name                str
  supplier_type                str
  country                      str
  preferred_partner            str
  service_rating               str
  cancellation_flexibility     str
======================================================================
fact_bookings  shape=(6060, 16)
----------------------------------------------------------------------
  booking_id                   str
  client_id                    str
  traveler_id                  str
  supplier_id                  str
  service_type                 str
  travel_class                 str
  booking_channel              str
  origin_city                  str
  destination_city             str
  booking_date                 str
  travel_date                  str
  return_date                  str
  amount_usd                   str
  policy_compliant             str
  booking_status               str
  refund_usd                   str
======================================================================
fact_support_tickets  shape=(2530, 12)
----------------------------------------------------------------------
  ticket_id                    str
  booking_id                   str
  client_id                    str
  ticket_date                  str
  category                     str
  priority                     str
  contact_channel              str
  first_response_hours         str
  resolution_hours             float64
  escalated                    str
  csat_score                   str
  agent_notes_length           int64
```

Money, ratings and budgets arrive as text (`str`), which is the first sign that they need converting.

### 3.2 Missing values

```python
missing_summary = []
for label, df in tables.items():
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    for col in df.columns:
        missing_summary.append({
            'table': label,
            'column': col,
            'missing_count': int(miss[col]),
            'missing_pct': float(miss_pct[col]),
        })

missing_df = pd.DataFrame(missing_summary)
missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(
    ['table', 'missing_count'], ascending=[True, False]
)
print('Columns with missing values:')
print(missing_df.to_string(index=False))
if missing_df.empty:
    print('No missing values found.')
```

```text
Columns with missing values:
               table               column  missing_count  missing_pct
       dim_suppliers       service_rating              1         1.89
       dim_travelers      trips_last_year             50         4.08
       dim_travelers  years_with_employer              6         0.49
       fact_bookings          supplier_id             42         0.69
fact_support_tickets            client_id             20         0.79
fact_support_tickets first_response_hours              8         0.32
```

Gaps are small (under 5% in every column), but a missing `supplier_id` or `client_id` breaks a join, so those matter most.

### 3.3 Duplicate rows

```python
for label, df in tables.items():
    dup_count = int(df.duplicated().sum())
    print(f'{label}: {dup_count} fully-duplicated rows')
```

```text
dim_clients: 4 fully-duplicated rows
dim_travelers: 25 fully-duplicated rows
dim_suppliers: 3 fully-duplicated rows
fact_bookings: 60 fully-duplicated rows
fact_support_tickets: 30 fully-duplicated rows
```

That is 122 exact duplicates in total, which would double-count bookings and tickets if left in.

### 3.4 Sample rows

```python
for label, df in tables.items():
    print('=' * 70)
    print(f'{label}')
    print(df.head().to_string(index=False))
```

This prints the first five rows of each table. Reading them is where the formatting problems show up: budgets such as `1.25M` and `$1,250,000`, ratings such as `4/5`, amounts such as `USD 1 250.00`, `2 hrs` for durations, and dates written in different layouts.

### 3.5 Numeric and categorical sanity checks

```python
for label, df in tables.items():
    print('=' * 70)
    print(f'{label}')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f'  Numeric cols ({len(num_cols)}): {num_cols}')
    print(f'  Object  cols ({len(obj_cols)}): {obj_cols}')
    for c in num_cols:
        print(f'    {c:<28} min={df[c].min()}, max={df[c].max()}, mean={df[c].mean():.2f}')
```

```text
======================================================================
dim_clients
  Numeric cols (0): []
  Object  cols (8): ['client_id', 'client_name', 'country', 'industry', 'client_tier', 'contract_start', 'annual_travel_budget_usd', 'has_travel_policy']
======================================================================
dim_travelers
  Numeric cols (2): ['years_with_employer', 'trips_last_year']
  Object  cols (5): ['traveler_id', 'client_id', 'grade', 'home_country', 'loyalty_member']
    years_with_employer          min=-5.0, max=200.0, mean=14.75
    trips_last_year              min=0.0, max=15.0, mean=5.11
======================================================================
dim_suppliers
  Numeric cols (0): []
  Object  cols (7): ['supplier_id', 'supplier_name', 'supplier_type', 'country', 'preferred_partner', 'service_rating', 'cancellation_flexibility']
======================================================================
fact_bookings
  Numeric cols (0): []
  Object  cols (16): ['booking_id', 'client_id', 'traveler_id', 'supplier_id', 'service_type', 'travel_class', 'booking_channel', 'origin_city', 'destination_city', 'booking_date', 'travel_date', 'return_date', 'amount_usd', 'policy_compliant', 'booking_status', 'refund_usd']
======================================================================
fact_support_tickets
  Numeric cols (2): ['resolution_hours', 'agent_notes_length']
  Object  cols (10): ['ticket_id', 'booking_id', 'client_id', 'ticket_date', 'category', 'priority', 'contact_channel', 'first_response_hours', 'escalated', 'csat_score']
    resolution_hours             min=-10.0, max=9999.0, mean=40.47
    agent_notes_length           min=10, max=799, mean=399.04
```

The ranges expose impossible values: negative or extreme tenure in `years_with_employer`, and extreme `resolution_hours` and `first_response_hours` in the support tickets.

### 3.6 Summary of issues found

| Issue | Where | Fixed in |
|---|---|---|
| Exact duplicate rows (122 in total) | Every table | Section 4 |
| Numbers stored as text | Budgets, ratings, amounts, hours | Section 4 |
| Six mixed date formats | Contract, booking, travel, return and ticket dates | Section 4 |
| Yes/No written many ways (`Yes`, `Y`, `TRUE`, `1`) | Policy, loyalty, partner and escalation flags | Section 4 |
| Spelling variants for countries, cities and tiers | `country`, `city`, `client_tier` | Section 4 |
| Impossible values | Tenure, resolution time, amounts, CSAT | Section 4 |
| Missing keys | `supplier_id`, `client_id` | Sections 4 and 6 |
| Orphan foreign keys | Bookings and tickets | Section 6 |

---

## 4. Data Cleaning & Error Correction

The first version of my cleaning pass fixed the obvious problems but left some behind: only some date formats parsed, country and city variants were merely case-normalised, and a blanket `dropna()` threw away rows that could have been repaired. The version below replaces it. It repairs wherever the right answer is knowable and drops a row only when a key field is missing.

**What the cleaning does, in order**

1. Removes exact duplicates, then duplicate primary keys
2. Converts text numbers to real numbers (`$1,250`, `USD 1 250.00`, `1.25M`, `2 hrs`, `3 days`)
3. Parses dates against all six known formats (slash dates are read day-first)
4. Merges spelling variants into one label: countries (`UK`, `U.K.`, `Great Britain`), cities (`NYC`, `New York`) and client tiers (`G` becomes Gold, `S` becomes Standard, `P` becomes Platinum)
5. Standardises Yes/No style columns to True/False
6. Treats impossible values as missing, then fills them with a sensible median: zero or negative amounts (median by service type), CSAT outside 1 to 5, resolution time outside 0 to 720 hours, tenure outside 0 to 60 years
7. Swaps travel and return dates that arrive in the wrong order
8. Recovers a missing ticket `client_id` from its linked booking
9. Drops a row only if a required key or date is still missing

```python
import numpy as np
import pandas as pd

DATA_DIR = "."
FILES = {
    "dim_clients.csv": "dim_clients",
    "dim_travelers.csv": "dim_travelers",
    "dim_suppliers.csv": "dim_suppliers",
    "fact_bookings (1).csv": "fact_bookings",
    "fact_support_tickets.csv": "fact_support_tickets",
}
raw = {label: pd.read_csv(f"{DATA_DIR}/{fname}") for fname, label in FILES.items()}
report = []   # (table, step, rows_removed)

# ---------- helpers ----------
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d %b %Y", "%Y/%m/%d", "%B %d, %Y"]   # day-first for slashes
MISSING = ["", "N/A", "n/a", "NA", "null", "None", "nan", "-", "TBD", "pending", "??"]

def clean_text(s):
    return s.astype("string").str.strip().replace(MISSING, pd.NA)

def to_num(s):
    """'$1,200.50', 'USD 1 200.50', '1.25M', '2 hrs', '3 days' -> float; anything else -> NaN"""
    t = clean_text(s).str.upper().str.replace(r"[,$\s]|USD|HRS?|H$|DAYS?|D$", "", regex=True)
    mult = t.str.endswith("M").fillna(False).map({True: 1e6, False: 1.0})
    return pd.to_numeric(t.str.replace("M", "", regex=False), errors="coerce") * mult

def to_date(s):
    s = clean_text(s)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:                      # try every known format on whatever is still unparsed
        todo = out.isna() & s.notna()
        out[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    return out

def to_bool(s):
    t = clean_text(s).str.lower()
    return t.map({"yes": True, "y": True, "true": True, "1": True, "no": False, "n": False, "false": False, "0": False})

def canon(s, valid, extra=None):
    """Map any spelling/case of a category to its standard label; unknown values become missing."""
    lookup = {v.lower(): v for v in valid}
    lookup.update({k.lower(): v for k, v in (extra or {}).items()})
    return clean_text(s).str.lower().map(lookup)

COUNTRIES = {
    "usa": "United States", "u.s.a": "United States", "us": "United States", "united states": "United States",
    "uk": "United Kingdom", "u.k.": "United Kingdom", "great britain": "United Kingdom", "united kingdom": "United Kingdom",
    "uae": "UAE", "u.a.e": "UAE", "united arab emirates": "UAE",
    "ng": "Nigeria", "nigeria": "Nigeria", "rsa": "South Africa", "s. africa": "South Africa", "south africa": "South Africa",
    "in": "India", "india": "India", "deutschland": "Germany", "germany": "Germany",
    "sg": "Singapore", "singapore": "Singapore", "brasil": "Brazil", "brazil": "Brazil", "aus": "Australia", "australia": "Australia",
}
CITIES = {
    "new york": "New York", "nyc": "New York", "london": "London", "lon": "London", "dubai": "Dubai", "dxb": "Dubai",
    "lagos": "Lagos", "los": "Lagos", "johannesburg": "Johannesburg", "jnb": "Johannesburg", "joburg": "Johannesburg",
    "mumbai": "Mumbai", "bombay": "Mumbai", "bom": "Mumbai", "frankfurt": "Frankfurt", "fra": "Frankfurt",
    "singapore": "Singapore", "sin": "Singapore", "sao paulo": "Sao Paulo", "são paulo": "Sao Paulo", "gru": "Sao Paulo",
    "sydney": "Sydney", "syd": "Sydney",
}
TIERS = ["Standard", "Gold", "Platinum"]
TIER_EXTRA = {"s": "Standard", "g": "Gold", "p": "Platinum"}

def finish(df, required, name):
    """Drop rows missing a REQUIRED field only; fill every other gap (median / 'Unknown' / most common)."""
    before = len(df)
    df = df.dropna(subset=required).copy()
    report.append((name, "rows missing a required field", before - len(df)))
    for c in df.columns:
        if df[c].isna().any():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            elif df[c].dtype == bool or set(df[c].dropna().unique()) <= {True, False}:
                df[c] = df[c].fillna(df[c].mode().iloc[0])
            elif not pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].fillna("Unknown")
    return df

tables = {}

# ---------- dim_clients ----------
d = raw["dim_clients"]; n0 = len(d); d = d.drop_duplicates().copy(); report.append(("dim_clients", "exact duplicates", n0 - len(d)))
d["client_id"] = clean_text(d["client_id"]).str.upper()
d["client_name"] = clean_text(d["client_name"]).str.title()
d["country"] = clean_text(d["country"]).str.lower().map(COUNTRIES)
d["industry"] = clean_text(d["industry"]).str.title()
d["client_tier"] = canon(d["client_tier"], TIERS, TIER_EXTRA)
d["contract_start"] = to_date(d["contract_start"])
d["annual_travel_budget_usd"] = to_num(d["annual_travel_budget_usd"]).where(lambda x: x > 0)
d["has_travel_policy"] = to_bool(d["has_travel_policy"])
tables["dim_clients"] = finish(d.drop_duplicates("client_id"), ["client_id", "contract_start"], "dim_clients")

# ---------- dim_suppliers ----------
d = raw["dim_suppliers"]; n0 = len(d); d = d.drop_duplicates().copy(); report.append(("dim_suppliers", "exact duplicates", n0 - len(d)))
d["supplier_id"] = clean_text(d["supplier_id"]).str.upper()
d["supplier_name"] = clean_text(d["supplier_name"]).str.title()
d["supplier_type"] = canon(d["supplier_type"], ["Airline", "Hotel", "Car Rental"])
d["country"] = clean_text(d["country"]).str.lower().map(COUNTRIES)
d["preferred_partner"] = to_bool(d["preferred_partner"])
d["service_rating"] = pd.to_numeric(clean_text(d["service_rating"]).str.extract(r"^(\d+(?:\.\d+)?)")[0], errors="coerce").where(lambda x: x.between(1, 5))
d["cancellation_flexibility"] = canon(clean_text(d["cancellation_flexibility"]).str.replace("-", " "), ["Flexible"],
                                      {"semi flexible": "Semi-Flexible", "non refundable": "Non-Refundable"})
tables["dim_suppliers"] = finish(d.drop_duplicates("supplier_id"), ["supplier_id"], "dim_suppliers")

# ---------- dim_travelers ----------
d = raw["dim_travelers"]; n0 = len(d); d = d.drop_duplicates().copy(); report.append(("dim_travelers", "exact duplicates", n0 - len(d)))
d["traveler_id"] = clean_text(d["traveler_id"]).str.upper()
d["client_id"] = clean_text(d["client_id"]).str.upper()
d["grade"] = canon(d["grade"], ["Staff", "Manager", "Director", "Executive"])
d["home_country"] = clean_text(d["home_country"]).str.lower().map(COUNTRIES)
d["loyalty_member"] = to_bool(d["loyalty_member"]).map({True: "Yes", False: "No"})
yrs = to_num(d["years_with_employer"])
d["years_with_employer"] = yrs.where(yrs.between(0, 60))                      # impossible tenure -> missing -> median
d["trips_last_year"] = to_num(d["trips_last_year"]).where(lambda x: x.between(0, 200))
tables["dim_travelers"] = finish(d.drop_duplicates("traveler_id"), ["traveler_id", "client_id"], "dim_travelers")

# ---------- fact_bookings ----------
b = raw["fact_bookings"]; n0 = len(b); b = b.drop_duplicates().copy(); report.append(("fact_bookings", "exact duplicates", n0 - len(b)))
for c in ["booking_id", "client_id", "traveler_id", "supplier_id"]:
    b[c] = clean_text(b[c]).str.upper()
for c in ["booking_date", "travel_date", "return_date"]:
    b[c] = to_date(b[c])
swap = b["return_date"] < b["travel_date"]                                     # return before travel: swap them back
b.loc[swap, ["travel_date", "return_date"]] = b.loc[swap, ["return_date", "travel_date"]].values
report.append(("fact_bookings", "travel/return dates swapped back (kept)", int(swap.sum())))
b["service_type"] = canon(b["service_type"], ["Flight", "Hotel", "Car Rental"])
b["travel_class"] = canon(b["travel_class"], ["Economy", "Premium Economy", "Business", "First", "Standard", "Superior", "Suite", "SUV", "Luxury"])
b["booking_channel"] = canon(clean_text(b["booking_channel"]).str.replace(" ", ""), [],
                             {"onlinetool": "Online Tool", "mobileapp": "Mobile App", "travelagent": "Travel Agent", "phone": "Phone"})
b["origin_city"] = clean_text(b["origin_city"]).str.lower().map(CITIES)
b["destination_city"] = clean_text(b["destination_city"]).str.lower().map(CITIES)
b["amount_usd"] = to_num(b["amount_usd"]).where(lambda x: x > 0)              # zero / negative amounts are errors
b["amount_usd"] = b["amount_usd"].fillna(b.groupby("service_type")["amount_usd"].transform("median")).fillna(b["amount_usd"].median())
b["refund_usd"] = to_num(b["refund_usd"]).where(lambda x: x >= 0).fillna(0)
b["policy_compliant"] = to_bool(b["policy_compliant"])
b["booking_status"] = canon(b["booking_status"], ["Completed", "Cancelled", "No-show"], {"canceled": "Cancelled", "no show": "No-show"})
b = b.drop_duplicates("booking_id")
tables["fact_bookings"] = finish(b, ["booking_id", "client_id", "traveler_id", "supplier_id", "booking_date", "travel_date", "return_date"], "fact_bookings")

# ---------- fact_support_tickets ----------
t = raw["fact_support_tickets"]; n0 = len(t); t = t.drop_duplicates().copy(); report.append(("fact_support_tickets", "exact duplicates", n0 - len(t)))
for c in ["ticket_id", "booking_id", "client_id"]:
    t[c] = clean_text(t[c]).str.upper()
recover = t["client_id"].isna() & t["booking_id"].isin(tables["fact_bookings"]["booking_id"])
t.loc[recover, "client_id"] = t.loc[recover, "booking_id"].map(tables["fact_bookings"].set_index("booking_id")["client_id"])
report.append(("fact_support_tickets", "missing client_id recovered from the booking (kept)", int(recover.sum())))
t["ticket_date"] = to_date(t["ticket_date"])
t["category"] = canon(t["category"], ["Refund Request", "Itinerary Change", "Visa or Document Query", "Billing Dispute", "Complaint", "Emergency Rebooking"])
t["priority"] = canon(t["priority"], ["Low", "Medium", "High", "Critical"])
t["contact_channel"] = canon(t["contact_channel"], ["Email", "Phone", "Chat", "Portal"])
t["escalated"] = to_bool(t["escalated"])
t["first_response_hours"] = to_num(t["first_response_hours"]).where(lambda x: x >= 0)
t["resolution_hours"] = to_num(t["resolution_hours"]).where(lambda x: (x > 0) & (x <= 720))
t["csat_score"] = to_num(t["csat_score"]).where(lambda x: x.between(1, 5))
t = t.drop_duplicates("ticket_id")
tables["fact_support_tickets"] = finish(t, ["ticket_id", "booking_id", "client_id", "ticket_date"], "fact_support_tickets")

# ---------- summary ----------
print("=== CLEANING v2 COMPLETE ===\n")
for tbl, step, n in report:
    print(f"  {tbl:<22} {step:<55} {n:>5}")
print()
for label, df in tables.items():
    print(f"  {label:<24} raw={len(raw[label]):>5}  clean={len(df):>5}  shape={df.shape}  missing={int(df.isna().sum().sum())}")
```

**Result:** every table has zero missing values, and far fewer rows were lost than before.

| Table | Raw | Clean |
|---|---|---|
| `dim_clients` | 64 | 58 |
| `dim_travelers` | 1,225 | 1,150 |
| `dim_suppliers` | 53 | 50 |
| `fact_bookings` | 6,060 | 5,528 (5,762 after cleaning, then 234 orphans removed in section 6) |
| `fact_support_tickets` | 2,530 | 2,263 |

---

## 5. Feature Engineering

New columns turn raw fields into things a model can learn from.

| Table | Features created |
|---|---|
| `fact_bookings` | `booking_lead_days`, `trip_duration_days`, `is_same_city`, `amount_band`, `is_policy_compliant`, `booking_month`, `travel_month` |
| `fact_support_tickets` | `is_escalated`, `csat_score` (numeric), `fast_response` (first reply within 4 hours), `quick_resolution` (closed within 24 hours), `ticket_month` |
| `dim_travelers` | `tenure_bucket`, `trip_volume_bucket`, `is_loyalty_member`, plus booking and ticket aggregates: `total_bookings`, `total_spend_usd`, `avg_spend_usd`, `avg_lead_days`, `avg_trip_duration`, `policy_compliant_rate`, `same_city_rate`, `total_tickets`, `escalation_rate`, `avg_resolution_hours`, `avg_csat`, `fast_response_rate` |
| `dim_clients` | `budget_tier`, `contract_months`, `is_high_tier`, plus `client_total_bookings`, `client_total_spend_usd`, `client_avg_spend_usd`, `client_policy_compliant_rate` |

```python
# Feature engineering on cleaned tables
import pandas as pd
import numpy as np

print("=== FEATURE ENGINEERING ===\n")

clients = tables["dim_clients"].copy()
travelers = tables["dim_travelers"].copy()
suppliers = tables["dim_suppliers"].copy()
bookings = tables["fact_bookings"].copy()
tickets = tables["fact_support_tickets"].copy()

# ---------------------------------------------------------
# 1. dim_travelers features
# ---------------------------------------------------------
travelers["tenure_bucket"] = pd.cut(
    travelers["years_with_employer"],
    bins=[0, 2, 5, 10, 20, 60],
    labels=["0-2yr", "2-5yr", "5-10yr", "10-20yr", "20+yr"],
    include_lowest=True,
)
travelers["trip_volume_bucket"] = pd.cut(
    travelers["trips_last_year"],
    bins=[-0.1, 0, 3, 7, 15, 100],
    labels=["0", "1-3", "4-7", "8-15", "16+"],
    include_lowest=True,
)
travelers["is_loyalty_member"] = travelers["loyalty_member"].eq("Yes").astype(int)
print("dim_travelers: added tenure_bucket, trip_volume_bucket, is_loyalty_member")

# ---------------------------------------------------------
# 2. dim_clients features
# ---------------------------------------------------------
clients["budget_tier"] = pd.qcut(
    clients["annual_travel_budget_usd"],
    q=4,
    labels=["Low", "Medium", "High", "Premium"],
    duplicates="drop",
)
clients["contract_months"] = (
    pd.Timestamp("today").normalize() - clients["contract_start"]
).dt.days / 30.44
clients["is_high_tier"] = clients["client_tier"].isin(["Platinum", "Gold"]).astype(int)
print("dim_clients: added budget_tier, contract_months, is_high_tier")

# ---------------------------------------------------------
# 3. fact_bookings features
# ---------------------------------------------------------
bookings["booking_lead_days"] = (
    bookings["travel_date"] - bookings["booking_date"]
).dt.days
bookings["trip_duration_days"] = (
    bookings["return_date"] - bookings["travel_date"]
).dt.days
bookings["is_same_city"] = (
    bookings["origin_city"].eq(bookings["destination_city"]).astype(int)
)
bookings["amount_band"] = pd.qcut(
    bookings["amount_usd"],
    q=4,
    labels=["Low", "Medium", "High", "Premium"],
    duplicates="drop",
)
bookings["is_policy_compliant"] = bookings["policy_compliant"].astype(int)
bookings["booking_month"] = bookings["booking_date"].dt.to_period("M")
bookings["travel_month"] = bookings["travel_date"].dt.to_period("M")
print(
    "fact_bookings: added booking_lead_days, trip_duration_days, is_same_city, amount_band, is_policy_compliant, booking_month, travel_month"
)

# ---------------------------------------------------------
# 4. fact_support_tickets features
# ---------------------------------------------------------
tickets["is_escalated"] = tickets["escalated"].astype(int)
tickets["csat_score"] = pd.to_numeric(tickets["csat_score"], errors="coerce")
tickets["fast_response"] = (tickets["first_response_hours"] <= 4).astype(int)
tickets["quick_resolution"] = (tickets["resolution_hours"] <= 24).astype(int)
tickets["ticket_month"] = tickets["ticket_date"].dt.to_period("M")
print(
    "fact_support_tickets: added is_escalated, csat_score, fast_response, quick_resolution, ticket_month"
)

# ---------------------------------------------------------
# 5. Traveler-level aggregated features
# ---------------------------------------------------------
# Bookings per traveler
b_agg = (
    bookings.groupby("traveler_id")
    .agg(
        total_bookings=("booking_id", "count"),
        total_spend_usd=("amount_usd", "sum"),
        avg_spend_usd=("amount_usd", "mean"),
        avg_lead_days=("booking_lead_days", "mean"),
        avg_trip_duration=("trip_duration_days", "mean"),
        policy_compliant_rate=("is_policy_compliant", "mean"),
        same_city_rate=("is_same_city", "mean"),
    )
    .reset_index()
)

# Tickets per traveler (via booking_id join)
tickets_with_traveler = tickets.merge(
    bookings[["booking_id", "traveler_id"]], on="booking_id", how="left"
)
t_agg = (
    tickets_with_traveler.groupby("traveler_id")
    .agg(
        total_tickets=("ticket_id", "count"),
        escalation_rate=("is_escalated", "mean"),
        avg_resolution_hours=("resolution_hours", "mean"),
        avg_csat=("csat_score", "mean"),
        fast_response_rate=("fast_response", "mean"),
    )
    .reset_index()
)

# Merge traveler features
travelers = travelers.merge(b_agg, on="traveler_id", how="left")
travelers = travelers.merge(t_agg, on="traveler_id", how="left")

# Fill missing for travelers with no bookings/tickets
fill_cols = {
    "total_bookings": 0,
    "total_spend_usd": 0.0,
    "avg_spend_usd": 0.0,
    "avg_lead_days": 0.0,
    "avg_trip_duration": 0.0,
    "policy_compliant_rate": 0.0,
    "same_city_rate": 0.0,
    "total_tickets": 0,
    "escalation_rate": 0.0,
    "avg_resolution_hours": 0.0,
    "avg_csat": 0.0,
    "fast_response_rate": 0.0,
}
for c, v in fill_cols.items():
    if c in travelers.columns:
        travelers[c] = travelers[c].fillna(v)

print("Traveler-level: aggregated bookings + support tickets features merged")

# ---------------------------------------------------------
# 6. Client-level aggregated features
# ---------------------------------------------------------
c_bookings = bookings.merge(clients[["client_id"]], on="client_id", how="left")
cb_agg = (
    c_bookings.groupby("client_id")
    .agg(
        client_total_bookings=("booking_id", "count"),
        client_total_spend_usd=("amount_usd", "sum"),
        client_avg_spend_usd=("amount_usd", "mean"),
        client_policy_compliant_rate=("is_policy_compliant", "mean"),
    )
    .reset_index()
)

clients = clients.merge(cb_agg, on="client_id", how="left")
print("Client-level: aggregated booking features merged")

# ---------------------------------------------------------
# 7. Store enriched tables back
# ---------------------------------------------------------
tables["dim_clients"] = clients
tables["dim_travelers"] = travelers
tables["dim_suppliers"] = suppliers
tables["fact_bookings"] = bookings
tables["fact_support_tickets"] = tickets

print()
print("=== FEATURE ENGINEERING COMPLETE ===")
print("\nEnriched table shapes:")
for label, df in tables.items():
    print("  {:<24} shape={}  columns={}".format(label, df.shape, len(df.columns)))

print("\ndim_travelers feature columns:")
for c in travelers.columns:
    print("  - {}".format(c))
```

---

## 6. Data Modeling & Relational Schema

The clean tables form a star schema: three dimensions describe *who* and *what*, and two fact tables record *what happened*.

```
dim_clients ──────┐
dim_travelers ────┼──▶ fact_bookings ──▶ fact_support_tickets
dim_suppliers ────┘
```

| Relationship | Key |
|---|---|
| `fact_bookings` → `dim_clients` | `client_id` |
| `fact_bookings` → `dim_travelers` | `traveler_id` |
| `fact_bookings` → `dim_suppliers` | `supplier_id` |
| `fact_support_tickets` → `fact_bookings` | `booking_id` |

To enforce the relationships, the final step drops bookings whose client, traveler or supplier doesn't exist, drops tickets whose booking is gone, and keeps only dimension rows that appear in the facts. Primary keys are then checked for uniqueness. The two flattened views, `bookings_full` (5,528 × 64) and `tickets_full` (2,263 × 52), put everything on one row per booking or ticket so models can read them directly.

```python
# Star schema correction: drop orphans to enforce referential integrity
import pandas as pd
import numpy as np

print("=== CORRECTING STAR SCHEMA REFERENTIAL INTEGRITY ===\n")

# Reload cleaned/enriched state from previous cells or just work with tables
clients = tables["dim_clients"].copy()
travelers = tables["dim_travelers"].copy()
suppliers = tables["dim_suppliers"].copy()
bookings = tables["fact_bookings"].copy()
tickets = tables["fact_support_tickets"].copy()

# 1. Drop orphaned bookings
bookings_old = len(bookings)
bookings = bookings[
    bookings["client_id"].isin(clients["client_id"])
    & bookings["traveler_id"].isin(travelers["traveler_id"])
    & bookings["supplier_id"].isin(suppliers["supplier_id"])
]
print(
    "fact_bookings: dropped {} orphaned rows -> {} remaining (by client/traveler/supplier FKs)".format(
        bookings_old - len(bookings), len(bookings)
    )
)

# 2. Drop orphaned tickets
tickets_old = len(tickets)
bookings_valid_set = set(bookings["booking_id"])
tickets = tickets[tickets["booking_id"].isin(bookings_valid_set)]
print(
    "fact_support_tickets: dropped {} orphaned rows -> {} remaining (by booking_id FK)".format(
        tickets_old - len(tickets), len(tickets)
    )
)

# 3. Optional: drop dimension rows that have no fact references (prevents bloat)
clients_old = len(clients)
clients = clients[
    clients["client_id"].isin(bookings["client_id"])
    | clients["client_id"].isin(tickets["client_id"])
]
print(
    "dim_clients: kept {} of {} rows that appear in facts".format(
        len(clients), clients_old
    )
)

travelers_old = len(travelers)
travelers = travelers[travelers["traveler_id"].isin(bookings["traveler_id"])]
print(
    "dim_travelers: kept {} of {} rows that appear in bookings".format(
        len(travelers), travelers_old
    )
)

suppliers_old = len(suppliers)
suppliers = suppliers[suppliers["supplier_id"].isin(bookings["supplier_id"])]
print(
    "dim_suppliers: kept {} of {} rows that appear in bookings".format(
        len(suppliers), suppliers_old
    )
)
print()

# 4. Verify after correction
print("=== VERIFICATION AFTER CORRECTION ===")
orph_bookings_client = bookings[~bookings["client_id"].isin(clients["client_id"])]
orph_bookings_trav = bookings[~bookings["traveler_id"].isin(travelers["traveler_id"])]
orph_bookings_sup = bookings[~bookings["supplier_id"].isin(suppliers["supplier_id"])]
orph_tickets_book = tickets[~tickets["booking_id"].isin(bookings["booking_id"])]
print(
    "fact_bookings.client_id in dim_clients: {} orphaned [{}]".format(
        len(orph_bookings_client), "PASS" if len(orph_bookings_client) == 0 else "FAIL"
    )
)
print(
    "fact_bookings.traveler_id in dim_travelers: {} orphaned [{}]".format(
        len(orph_bookings_trav), "PASS" if len(orph_bookings_trav) == 0 else "FAIL"
    )
)
print(
    "fact_bookings.supplier_id in dim_suppliers: {} orphaned [{}]".format(
        len(orph_bookings_sup), "PASS" if len(orph_bookings_sup) == 0 else "FAIL"
    )
)
print(
    "fact_support_tickets.booking_id in fact_bookings: {} orphaned [{}]".format(
        len(orph_tickets_book), "PASS" if len(orph_tickets_book) == 0 else "FAIL"
    )
)
print()

# 5. PK checks
print("=== PK UNIQUENESS ===")
for name, df, pk in [
    ("dim_clients", clients, "client_id"),
    ("dim_travelers", travelers, "traveler_id"),
    ("dim_suppliers", suppliers, "supplier_id"),
    ("fact_bookings", bookings, "booking_id"),
    ("fact_support_tickets", tickets, "ticket_id"),
]:
    ok = df[pk].nunique() == len(df) and not df[pk].duplicated().any()
    print(
        "  {:<24} {:<20} total={} unique={} [{}]".format(
            name, pk, len(df), df[pk].nunique(), "PASS" if ok else "FAIL"
        )
    )

# Store corrected tables
tables["dim_clients"] = clients
tables["dim_travelers"] = travelers
tables["dim_suppliers"] = suppliers
tables["fact_bookings"] = bookings
tables["fact_support_tickets"] = tickets

# Recreate denorm views
# Suffixes stop pandas from renaming client_id when both sides carry it
bookings_full = (
    bookings.merge(clients, on="client_id", how="left", suffixes=("", "_dim_client"))
    .merge(travelers, on="traveler_id", how="left", suffixes=("", "_dim_trav"))
    .merge(suppliers, on="supplier_id", how="left", suffixes=("", "_dim_sup"))
)
tickets_full = tickets.merge(
    bookings, on="booking_id", how="left", suffixes=("", "_fact_bk")
).merge(clients, on="client_id", how="left", suffixes=("", "_dim_client"))
tables["bookings_full"] = bookings_full
tables["tickets_full"] = tickets_full

print("\n=== FINAL STAR SCHEMA STATE ===")
for label, df in [
    ("dim_clients", clients),
    ("dim_travelers", travelers),
    ("dim_suppliers", suppliers),
    ("fact_bookings", bookings),
    ("fact_support_tickets", tickets),
    ("bookings_full", bookings_full),
    ("tickets_full", tickets_full),
]:
    print("  {:<24} shape={}".format(label, df.shape))

print("\nAll referential integrity checks passed. Clean star schema finalized.")
```

---

## 7. Business Analysis

**Bookings (5,528)**

| Metric | Value |
|---|---|
| Total spend | about $7.8M (flights, hotels and car rental combined) |
| Average booking | about $1,400 |
| Policy compliance | 73.8% |
| Cancellation rate | 17.1% (946 cancelled, 137 no-shows) |
| Same-city trips | 36.3% |
| Average booking lead time | 26.7 days |
| Average trip length | 4.6 days |

- **Top suppliers by revenue:** SUP-212 (about $290K), SUP-204 (about $288K), SUP-213 (about $278K)
- **Top clients:** CLT-135 (156 bookings, $209K), CLT-123 (143, $205K), CLT-159 (135, $188K)
- **Service mix:** flights about $5.05M (64%), hotels about $2.6M (33%), car rental about $0.19M (2%)

**Support (2,263 tickets)**

| Metric | Value |
|---|---|
| Escalation rate | 29.7% |
| First reply within 4 hours | 72.6% |
| Resolved within 24 hours | 63.5% |
| Average CSAT | 3.18 / 5 |
| Average resolution time | 25.0 hours |

Complaints escalate most (about 47%) and score lowest on satisfaction (2.39 / 5).

---

## 8. Exploratory Data Analysis

**Policy compliance:** 73.8% of bookings follow policy and 26.2% don't. That is a moderate imbalance and workable for classification.

**Cancellations:** about 1 in 5 bookings is cancelled or a no-show (17.1% cancelled plus 2.5% no-shows). A model that always predicts "active" would therefore score about 80% accuracy, so cancellation results must be read against recall, not accuracy alone.

**What I checked:** service type, travel class, client tier and loyalty membership. Compliance moves only a little across client tiers (Gold 72.2%, Platinum 74.5%, Standard 74.9%), which hints that tier alone won't explain it.

**Modelling readiness:** the tables have clean types, no missing values and full key coverage:

| Dataset | Shape |
|---|---|
| Travelers | 1,150 rows |
| `bookings_full` | 5,528 × 64 |
| `tickets_full` | 2,263 × 52 |

---

## 9. Machine Learning

Five models, one per business objective, each trained on a 75/25 train/test split and judged on the held-out 25%. The results are mixed, and I report them as they came out.

| # | Objective | Model | Result on held-out data | Takeaway |
|---|---|---|---|---|
| 1 | Cancellation | Logistic Regression | 617 of 766 correct (80.5%), but only 2 of 147 cancellations caught | No better than always guessing "active" (80.8%). Not usable as built |
| 2 | Trip cost | Linear Regression on log(amount) | Points follow the diagonal with wide scatter | Moderate fit. Understands the broad price level, misses the extremes |
| 3 | Policy compliance drivers | Logistic Regression + permutation importance | Largest importance about 0.006 (under one percentage point of accuracy) | Essentially no signal. Compliance is not explained by these fields |
| 4 | Support escalation | Logistic Regression | ROC AUC 0.77 | Useful ranking signal; clearly better than chance (0.50) |
| 5 | Billing anomalies | Isolation Forest (contamination 0.02) | Flags the lowest-scoring 2% of invoices | Short audit list, saved as `detected_billing_anomalies.csv` |

**Reading these results honestly**
- **Objective 1:** cancellations are about 19% of bookings, so a model that predicts "active" every time scores about 81%. This one does almost the same and finds only 2 of 147 cancellations. Next steps are class weighting, a lower decision threshold and richer features. Judge it on recall and precision, not accuracy.
- **Objective 2:** the model is trained on log(amount), so errors are in log units. It orders bookings sensibly but is not precise enough for client budgets yet. Service type and class dominate price, and a linear model struggles with that.
- **Objective 3:** every importance score is tiny and several are negative, meaning shuffling that feature did not hurt (or slightly helped) the model. That is the signature of no learnable signal, and it is a finding in its own right: out-of-policy bookings are not predictable from these fields.
- **Objective 4:** AUC 0.77 is decent, not strong. `csat_score` is an input, and it is only known after a ticket closes, so this model suits diagnosis rather than early warning.
- **Objective 5:** the model has no labels, so "anomaly" means "unusual", not "wrong". Each flagged booking needs a human review.

### Machine learning pipeline code

One script trains all five models, saves one chart per objective and then combines them into `ml_all_objectives_validation_charts.png`.

```python
# Corporate Travel Management System: Machine Learning Pipeline
# Five-objective ML pipeline: cancellations, cost forecast, compliance mapping,
# support escalations, and billing anomaly audit.

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_theme(style="whitegrid")

PROJECT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_csv_safely(base_name: str) -> pd.DataFrame:
    """Load a CSV, tolerating spaces vs. underscores in the filename."""
    alt_name = (
        base_name.replace("_", " ") if "_" in base_name else base_name.replace(" ", "_")
    )
    for candidate in (base_name, alt_name):
        path = PROJECT_DIR / candidate
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not locate {base_name} or {alt_name} in {PROJECT_DIR}")


def clean_currency(series: pd.Series) -> pd.Series:
    """Parse dirty currency strings such as '$1,250,000', '1.25M', 'USD 1,250,000'."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^[^\d\.+-]*", "", regex=True)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("M", "e6", case=False, regex=False)
    s = s.str.replace("m", "e6", case=False, regex=False)
    s = s.str.replace(r"[^0-9eE\.\+\-]", "", regex=True)
    s = s.replace("", np.nan)
    out = pd.to_numeric(s, errors="coerce")
    out = out.where(out.isna() | (out < 1e6), out / 1e6)
    return out


def clean_boolean(series: pd.Series) -> pd.Series:
    """Map Yes/Y/TRUE/1 variants to 1 and No/N/FALSE/0 variants to 0."""
    s = series.astype(str).str.strip().str.lower()
    true_vals = {"yes", "y", "true", "1", "t"}
    false_vals = {"no", "n", "false", "0", "f"}
    out = pd.Series(index=s.index, dtype="float")
    out[s.isin(true_vals)] = 1.0
    out[s.isin(false_vals)] = 0.0
    return out


def clean_tier(series: pd.Series) -> pd.Series:
    """Map tier abbreviations (G/S/P) to full names."""
    return (
        series.astype(str)
        .str.strip()
        .str.title()
        .replace({"G": "Gold", "S": "Standard", "P": "Platinum"})
    )


def clean_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=False)


def load_and_prepare() -> dict:
    """Load all star-schema tables and return cleaned DataFrames."""
    clients = load_csv_safely("dim_clients.csv")
    travelers = load_csv_safely("dim_travelers.csv")
    suppliers = load_csv_safely("dim_suppliers.csv")
    bookings = load_csv_safely("fact_bookings (1).csv")
    tickets = load_csv_safely("fact_support_tickets.csv")

    # --- Clean dimensions ---
    clients["client_tier"] = clean_tier(clients["client_tier"])
    clients["is_high_tier"] = clients["client_tier"].isin(["Gold", "Platinum"]).astype(int)
    clients["annual_travel_budget_usd"] = clean_currency(clients["annual_travel_budget_usd"])
    clients["has_travel_policy"] = clean_boolean(clients["has_travel_policy"])
    clients["contract_start"] = clean_date(clients["contract_start"])
    clients = clients.drop_duplicates(subset=["client_id"])

    travelers["grade"] = travelers["grade"].astype(str).str.strip().str.title()
    travelers["years_with_employer"] = pd.to_numeric(travelers["years_with_employer"], errors="coerce")
    travelers["trips_last_year"] = pd.to_numeric(travelers["trips_last_year"], errors="coerce")
    travelers["loyalty_member"] = clean_boolean(travelers["loyalty_member"])
    travelers = travelers.drop_duplicates(subset=["traveler_id"])

    suppliers["service_rating"] = pd.to_numeric(suppliers["service_rating"], errors="coerce")
    suppliers["preferred_partner"] = clean_boolean(suppliers["preferred_partner"])
    suppliers = suppliers.drop_duplicates(subset=["supplier_id"])

    # --- Clean facts ---
    bookings["amount_usd"] = clean_currency(bookings["amount_usd"])
    bookings["policy_compliant"] = clean_boolean(bookings["policy_compliant"])
    bookings["refund_usd"] = clean_currency(bookings["refund_usd"])
    bookings["booking_date"] = clean_date(bookings["booking_date"])
    bookings["travel_date"] = clean_date(bookings["travel_date"])
    bookings["return_date"] = clean_date(bookings["return_date"])
    bookings["is_cancelled"] = (
        bookings["booking_status"].astype(str).str.strip().str.lower().isin(
            {"cancelled", "canceled", "no-show", "no show"}
        )
    ).astype(int)
    bookings = bookings.drop_duplicates(subset=["booking_id"])

    tickets["escalated"] = clean_boolean(tickets["escalated"])
    tickets["csat_score"] = pd.to_numeric(tickets["csat_score"], errors="coerce")
    tickets["first_response_hours"] = pd.to_numeric(tickets["first_response_hours"], errors="coerce")
    tickets["resolution_hours"] = pd.to_numeric(tickets["resolution_hours"], errors="coerce")
    tickets["agent_notes_length"] = pd.to_numeric(tickets["agent_notes_length"], errors="coerce")
    tickets["ticket_date"] = clean_date(tickets["ticket_date"])
    tickets = tickets.drop_duplicates(subset=["ticket_id"])

    return {
        "clients": clients,
        "travelers": travelers,
        "suppliers": suppliers,
        "bookings": bookings,
        "tickets": tickets,
    }


def build_master_table(data: dict) -> pd.DataFrame:
    """Join facts into dimensions to build the master evaluation table."""
    bm = (
        data["bookings"]
        .merge(data["clients"], on="client_id", how="left", suffixes=("", "_client"))
        .merge(data["travelers"], on="traveler_id", how="left", suffixes=("", "_traveler"))
        .merge(data["suppliers"], on="supplier_id", how="left", suffixes=("", "_supplier"))
    )
    return bm


# ---------------------------------------------------------------------------
# Objective 1: Cancellation Classifier (Logistic Regression pipeline)
# ---------------------------------------------------------------------------
def objective_cancellation(data: dict, bm: pd.DataFrame):
    print("Objective 1: Training Cancellation Classifier...")
    df = bm.copy()
    df["lead_days"] = (df["travel_date"] - df["booking_date"]).dt.days
    df["trip_duration"] = (df["return_date"] - df["travel_date"]).dt.days

    feature_cols = [
        "amount_usd", "lead_days", "trip_duration",
        "is_high_tier", "has_travel_policy", "years_with_employer",
        "trips_last_year", "service_rating",
    ]
    cat_cols = ["service_type", "travel_class", "booking_channel"]

    model_df = df[feature_cols + cat_cols + ["is_cancelled"]].dropna()
    X = model_df[feature_cols + cat_cols]
    y = model_df["is_cancelled"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Cancellation accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Active", "Cancelled"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Objective 1: Cancellation Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "obj1_cancellation_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    return clf, acc


# ---------------------------------------------------------------------------
# Objective 2: Cost Forecaster (log-scaled regression)
# ---------------------------------------------------------------------------
def objective_cost_forecast(data: dict, bm: pd.DataFrame):
    print("Objective 2: Training Cost Forecaster...")
    df = bm.copy()
    df = df[df["amount_usd"] > 0].copy()
    df["log_amount"] = np.log1p(df["amount_usd"])
    df["lead_days"] = (df["travel_date"] - df["booking_date"]).dt.days

    feature_cols = [
        "lead_days",
        "is_high_tier",
        "has_travel_policy",
        "years_with_employer",
        "trips_last_year",
        "service_rating",
        "annual_travel_budget_usd",
    ]
    cat_cols = ["service_type", "travel_class", "booking_channel"]

    model_df = df[feature_cols + cat_cols + ["log_amount"]].dropna()
    X = model_df[feature_cols + cat_cols]
    y = model_df["log_amount"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    reg = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  Cost forecast MAE (log scale): {mae:.3f}, R^2: {r2:.3f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_test, y_pred, alpha=0.6, color="#228B22")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("Actual Log Booking Fee")
    ax.set_ylabel("Predicted Log Booking Fee")
    ax.set_title("Objective 2: Cost Actual vs. Predicted", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "obj2_cost_forecast.png", dpi=200, bbox_inches="tight")
    plt.close()
    return reg, r2


# ---------------------------------------------------------------------------
# Objective 3: Compliance Mapping (permutation importance)
# ---------------------------------------------------------------------------
def objective_compliance(data: dict, bm: pd.DataFrame):
    print("Objective 3: Computing Compliance Feature Weights...")
    df = bm.copy()
    df["policy_compliant_num"] = df["policy_compliant"].fillna(0).astype(int)
    df["lead_days"] = (df["travel_date"] - df["booking_date"]).dt.days

    feature_cols = [
        "amount_usd", "lead_days",
        "is_high_tier", "has_travel_policy",
        "years_with_employer", "trips_last_year",
        "service_rating",
    ]
    cat_cols = ["service_type", "travel_class", "booking_channel"]

    model_df = df[feature_cols + cat_cols + ["policy_compliant_num"]].dropna()
    X = model_df[feature_cols + cat_cols]
    y = model_df["policy_compliant_num"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    clf.fit(X_train, y_train)

    result = permutation_importance(
        clf, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="accuracy"
    )
    importance = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    print("  Top compliance drivers:")
    print(importance.to_string())

    fig, ax = plt.subplots(figsize=(6, 4))
    importance.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance Score")
    ax.set_title("Objective 3: Policy Compliance Drivers", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "obj3_compliance_weights.png", dpi=200, bbox_inches="tight")
    plt.close()
    return importance


# ---------------------------------------------------------------------------
# Objective 4: Support Escalations (ROC-AUC)
# ---------------------------------------------------------------------------
def objective_escalation(data: dict):
    print("Objective 4: Training Support Escalation Model...")
    df = data["tickets"].copy()
    df = df.dropna(subset=["escalated"])

    feature_cols = [
        "first_response_hours", "resolution_hours", "csat_score",
        "agent_notes_length",
    ]
    cat_cols = ["category", "priority", "contact_channel"]

    model_df = df[feature_cols + cat_cols + ["escalated"]].dropna()
    X = model_df[feature_cols + cat_cols]
    y = model_df["escalated"].astype(int)

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print(f"  Escalation ROC-AUC: {roc_auc:.3f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#FF8C00", lw=2.5, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Objective 4: Support Escalation ROC", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "obj4_escalation_roc.png", dpi=200, bbox_inches="tight")
    plt.close()
    return clf, roc_auc


# ---------------------------------------------------------------------------
# Objective 5: Billing Anomaly Audit (Isolation Forest)
# ---------------------------------------------------------------------------
def objective_anomalies(data: dict, bm: pd.DataFrame):
    print("Objective 5: Running Billing Anomaly Audit...")
    df = bm.copy()
    df = df[df["amount_usd"] > 0].copy()
    df["lead_days"] = (df["travel_date"] - df["booking_date"]).dt.days

    feature_cols = ["amount_usd", "lead_days", "service_rating", "years_with_employer"]
    model_df = df[feature_cols].dropna()
    X = model_df[feature_cols]

    iso = IsolationForest(contamination=0.02, random_state=RANDOM_STATE, n_estimators=200)
    iso.fit(X)
    scores = iso.decision_function(X)
    preds = iso.predict(X)  # -1 = anomaly, 1 = normal
    anomaly_rate = (preds == -1).mean()
    print(f"  Billing anomalies flagged: {anomaly_rate:.2%} of invoices")

    # Attach results back for export
    model_df = model_df.copy()
    model_df["anomaly_score"] = scores
    model_df["is_anomaly"] = (preds == -1).astype(int)
    anomalies = model_df[model_df["is_anomaly"] == 1]
    anomalies.to_csv(PROJECT_DIR / "detected_billing_anomalies.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(scores, bins=45, ax=ax, color="#DC143C", kde=True)
    ax.axvline(x=np.percentile(scores, 2), color="black", linestyle="--", label="Audit Threshold (2%)")
    ax.set_xlabel("Isolation Forest Anomaly Score")
    ax.set_title("Objective 5: Billing Anomaly Scores", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "obj5_billing_anomalies.png", dpi=200, bbox_inches="tight")
    plt.close()
    return iso, anomaly_rate


# ---------------------------------------------------------------------------
# Master validation figure: stitches the five real objective charts into one dashboard
# ---------------------------------------------------------------------------
def export_master_figure(data=None):
    import matplotlib.image as mpimg
    print("Compiling 5-objective validation dashboard...")
    files = ["obj1_cancellation_matrix.png", "obj2_cost_forecast.png",
             "obj3_compliance_weights.png", "obj4_escalation_roc.png",
             "obj5_billing_anomalies.png"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, f in zip(axes.flatten(), files):
        ax.imshow(mpimg.imread(PROJECT_DIR / f))
        ax.axis("off")
    axes.flatten()[5].axis("off")
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "ml_all_objectives_validation_charts.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("-> Exported 'ml_all_objectives_validation_charts.png'")


def main():
    print("Initializing TMC ML pipeline...")
    data = load_and_prepare()
    print(f"-> Loaded {len(data['bookings'])} bookings, {len(data['tickets'])} tickets, {len(data['clients'])} clients")
    bm = build_master_table(data)

    objective_cancellation(data, bm)
    objective_cost_forecast(data, bm)
    objective_compliance(data, bm)
    objective_escalation(data)
    objective_anomalies(data, bm)
    export_master_figure(data)

    print("[DONE] TMC ML pipeline complete. Model artifacts and charts exported.")


if __name__ == "__main__":
    main()
```

---

## 10. Data Visualization & Interpretation

Two sets of charts: one for the five business objectives, one for the machine learning work.

### Business objective charts

![Five business objective charts](business_objective_charts.png)

| # | Chart | What it shows | What it means |
|---|---|---|---|
| 1 | Top 10 suppliers by revenue | SUP-212 leads at about $290K, SUP-204 follows at about $288K, and the tenth supplier (SUP-215) still earns about $255K | Revenue is spread very evenly: the gap between first and tenth is only about 12%, so no single supplier dominates |
| 2 | Policy compliance by client tier | Every tier sits between about 72% and 75% (185 bookings, about 3%, belong to clients with no recognisable tier and are left out). Standard about 75%, Platinum about 74.5%, Gold about 72% | Tier doesn't explain compliance; the problem is spread across tiers |
| 3 | Escalation rate by ticket category | Complaint about 47%, Emergency Rebooking about 31%, Itinerary Change, Visa or Document Query and Refund Request about 26%, Billing Dispute about 23% | Complaints escalate at twice the rate of billing disputes, and emergency rebookings are the clear second risk |
| 4 | Revenue by service type | Flights about $5.05M (64%), hotels about $2.6M (33%), car rental about $0.19M (2%) | Flights and hotels carry almost the whole business |
| 5 | Average spend by travel class | First about $5,300, Business about $2,750, Suite about $1,950, Premium Economy about $1,450, down to Luxury car about $450 and SUV about $350 | Premium cabins cost many times more than economy. Classes belong to different services, so compare within a service |

**Business chart code** (run after the cleaning, feature engineering and star schema cells):

```python
# Business objective charts - run AFTER the cleaning, feature engineering and star schema cells
import matplotlib.pyplot as plt
import pandas as pd

bk = bookings_full.copy()
tk = tickets_full.copy()

# Safety net: map every label variant (case, spacing, abbreviations) to one canonical label.
# Anything that does not match a known label becomes missing and is left out of the chart.
def canon_labels(series, valid, extra=None):
    lookup = {v.lower(): v for v in valid}
    lookup.update({k.lower(): v for k, v in (extra or {}).items()})
    return series.astype(str).str.strip().str.lower().map(lookup)

SERVICES = ["Flight", "Hotel", "Car Rental"]
TIERS = ["Gold", "Platinum", "Standard"]
CATEGORIES = ["Complaint", "Emergency Rebooking", "Visa or Document Query",
              "Itinerary Change", "Refund Request", "Billing Dispute"]
CLASSES = ["First", "Business", "Premium Economy", "Economy",
           "Suite", "Superior", "Standard", "Luxury", "SUV"]

bk["service_type"] = canon_labels(bk["service_type"], SERVICES)
bk["client_tier"] = canon_labels(bk["client_tier"], TIERS, {"G": "Gold", "S": "Standard", "P": "Platinum"})
bk["travel_class"] = canon_labels(bk["travel_class"], CLASSES)
tk["category"] = canon_labels(tk["category"], CATEGORIES)

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(20, 11))

# 1. Top 10 suppliers by revenue
bk.groupby("supplier_id")["amount_usd"].sum().sort_values(ascending=False).head(10).plot(
    kind="bar", ax=axes[0, 0], color="steelblue")
axes[0, 0].set(title="1. Top 10 Suppliers by Total Revenue", xlabel="Supplier ID", ylabel="Total Revenue (USD)")
axes[0, 0].tick_params(axis="x", rotation=45)

# 2. Policy compliance by client tier
bk.groupby("client_tier")["is_policy_compliant"].mean().sort_values(ascending=False).plot(
    kind="bar", ax=axes[0, 1], color="forestgreen")
axes[0, 1].set(title="2. Policy Compliance Rate by Client Tier", xlabel="Client Tier",
               ylabel="Compliance Rate", ylim=(0, 1))
axes[0, 1].tick_params(axis="x", rotation=0)

# 3. Escalation rate by ticket category
tk.groupby("category")["is_escalated"].mean().sort_values().plot(
    kind="barh", ax=axes[0, 2], color="crimson")
axes[0, 2].set(title="3. Escalation Rate by Ticket Category", xlabel="Escalation Rate", ylabel="Ticket Category")

# 4. Revenue by service type
bk.groupby("service_type")["amount_usd"].sum().sort_values(ascending=False).plot(
    kind="bar", ax=axes[1, 0], color="darkorange")
axes[1, 0].set(title="4. Revenue by Service Type", xlabel="Service Type", ylabel="Total Revenue (USD)")
axes[1, 0].tick_params(axis="x", rotation=0)

# 5. Average spend by travel class
bk.groupby("travel_class")["amount_usd"].mean().sort_values(ascending=False).plot(
    kind="bar", ax=axes[1, 1], color="slateblue")
axes[1, 1].set(title="5. Average Booking Spend by Travel Class", xlabel="Travel Class", ylabel="Average Spend (USD)")
axes[1, 1].tick_params(axis="x", rotation=45)

axes[1, 2].axis("off")
plt.tight_layout()
plt.savefig("business_objective_charts.png", dpi=200, bbox_inches="tight")
plt.show()

# Quick check: each list below should show only the canonical labels
for name, col in [("Service types", bk["service_type"]), ("Client tiers", bk["client_tier"]),
                  ("Travel classes", bk["travel_class"]), ("Ticket categories", tk["category"])]:
    print(f"{name}: {sorted(col.dropna().unique())}  (unmatched rows: {int(col.isna().sum())})")
```

### Machine learning charts

![Five machine learning objective charts](ml_all_objectives_validation_charts.png)

The five panels, read in order:

**Objective 1: Cancellation confusion matrix.** Of 619 active bookings, 615 were predicted active and 4 wrongly flagged. Of 147 real cancellations, 145 were predicted active and only 2 were caught. Accuracy is 80.5%, almost identical to the 80.8% a "never cancelled" guess would give. **Meaning:** the model has learned to say "active" almost every time. It needs class balancing before it is of any use.

**Objective 2: Cost, actual vs predicted (log scale).** Points follow the red diagonal from about 5 to 8.5 but with wide scatter, and they sit in horizontal bands because service type and class set the price level. Low-fee bookings (actual 4 to 5) are over-predicted at about 5.2, and the most expensive ones (actual 8 to 10) are under-predicted at about 7 to 8. **Meaning:** the model captures the broad price level but not the extremes. A tree-based model or per-service models would likely do better.

**Objective 3: Policy compliance drivers.** `booking_channel` (about 0.006) and `travel_class` (about 0.006) rank first, followed by `service_type`, `has_travel_policy` and `years_with_employer`. `lead_days`, `amount_usd` and `is_high_tier` are negative. **Meaning:** the top score moves accuracy by well under one percentage point, so none of these is a real driver. Compliance is not predictable from the available fields, and booking channel and travel class are the first places to look for better data.

**Objective 4: Support escalation ROC.** The curve bows well above the diagonal, with AUC 0.77. At roughly a 20% false-positive rate the model catches about 60% of escalating tickets. **Meaning:** a random escalating ticket is ranked above a random non-escalating one about 77% of the time, which is good enough to prioritise a queue but not to automate decisions.

**Objective 5: Billing anomaly scores.** Scores are left-skewed, with most bookings between 0.10 and 0.22 and a thin tail running below zero. The dashed line marks the 2% audit threshold near zero. **Meaning:** a small, clearly separated group of bookings sits far from the rest. They are the audit candidates, and each still needs a human check.

---

## 11. Validation and Quality Assurance

| Check | Outcome |
|---|---|
| Primary keys unique in every table | Passed |
| Foreign keys all resolve to a parent record | Passed, orphans removed |
| Duplicates | Removed in every table |
| Missing values after cleaning | 0 in every table |
| All six date formats parsed, travel dates in order | Passed |
| Numeric text converted | Passed |
| Booleans standardised | Passed |
| Country, city and client tier variants merged | Passed |
| Impossible amounts, scores and durations | Replaced with group medians |
| Denormalised views keep `client_id` | Passed (5,528 × 64 and 2,263 × 52) |
| Models evaluated on held-out data, with suitable metrics | Done |

**Known limitations**
- About 3% of bookings (185) belong to clients whose tier could not be matched to Gold, Platinum or Standard, so they are excluded from the tier chart
- Missing amounts are filled with a median by service type, which is a reasonable default but not the true value
- Orphan bookings (client, traveler or supplier not in the dimension tables) are removed rather than repaired
- CSAT is only available after a ticket closes, which limits the escalation model for live use
- Anomaly detection flags unusual bookings, not confirmed errors
- The data is synthetic, so real-world performance will differ

---

## 12. Key Findings & Insights

- **Flights drive revenue** (about 64%) and hotels add about 33%; premium classes give the highest value per booking
- **Escalation depends on ticket type.** Complaints escalate about 47% of the time and carry the lowest satisfaction (2.39 / 5)
- **Escalation is moderately predictable** (AUC 0.77), enough to rank tickets for triage
- **Cancellations are not yet predictable.** About 1 in 5 bookings is cancelled or a no-show, and the first model catches almost none of them
- **Policy compliance is not explained by the available fields**, and it is stable across client tiers (72% to 75%)
- **Trip cost is partly predictable**: the model gets the price level right but not the extremes
- **About 2% of bookings look unusual** and form a short audit list

---

## 13. Recommendations

- **Cancellations:** retrain with class weighting and a lower decision threshold, and add features such as supplier, route and past cancellations before relying on it
- **Cost forecasting:** try tree-based or per-service models, and report error in dollars as well as log units
- **Policy compliance:** since no field explains it, collect better data (booking channel details, approval flow, traveler history) rather than tune the model
- **Support operations:** put complaints and emergency rebookings at the front of the queue, and use the escalation model to rank the rest. Build a version that excludes CSAT for use at ticket creation
- **Suppliers:** revenue is spread across the top ten, so review them as a group and compare service ratings against the revenue they bring in
- **Revenue:** offer upgrade paths to premium classes for eligible travelers and clients
- **Billing audit:** review the flagged bookings as the starting list and record outcomes to build labels for a supervised model later

---

## 14. Conclusion

This project took messy corporate travel data, cleaned it into a validated star schema while keeping about 25% more rows than a drop-everything approach, and used it to test five practical questions. Escalation ranking (AUC 0.77) and the billing anomaly list gave usable results, trip cost was partly predictable, and the cancellation and policy compliance models showed that the available fields carry little signal. Reporting that plainly is part of the result: it shows where the next round of data collection and modelling effort should go.

**Outputs:** `clean_bookings_master.csv`, `detected_billing_anomalies.csv`, `business_objective_charts.png`, `ml_all_objectives_validation_charts.png`, `explore_travel_mgmt_raw.ipynb`

