# Horizon Global Travel Management: Predictive Decision Support System

### TRAVEL Management Operations Prediction System

<p align="left">
<img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
<img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/Models-5-2E7D32" alt="Five models">
</p>

A corporate travel team deals with the same five problems every week: bookings that cancel, trip costs that are hard to budget, bookings that break policy, support tickets that end badly and invoices that look wrong. I built one model for each, tested every one on data it had never seen, and put all five behind a Streamlit app that a non-technical user can run.

**Live app:** [Open the Streamlit app](https://horizonglobaltravelmanagementpredictivedecisionsupportsystem.streamlit.app/)  |  **Code:** [GitHub repository](https://github.com/waleoshin75-sketch/Horizon-Global-Travel-Management-Predicting-Cancellations-Cost-Policy-Compliance-and-Service-Quality)

| Page in the app | Business question | Model | Result on unseen data |
| --- | --- | --- | --- |
| Booking Cancellations | Which bookings may cancel? | Logistic Regression | Catches 59% of cancellations (ROC AUC 0.63) |
| Trip Cost Forecast | What will a trip cost? | Hist Gradient Boosting | Average error $232 per booking (18%) |
| Policy Compliance | Which bookings may break policy? | Logistic Regression | Catches 62% of out-of-policy bookings (ROC AUC 0.68) |
| Customer Satisfaction | Which tickets may end badly? | Logistic Regression | Catches 85% of dissatisfied customers (ROC AUC 0.95) |
| Booking Amount Anomalies | Which amounts look unusual? | Isolation Forest | Finds 13 of 18 anomalies with 10 false alarms (ROC AUC 0.95) |

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Dictionary](#3-data-dictionary)
4. [Data Quality Audit](#4-data-quality-audit)
5. [Data Cleaning and Error Checking](#5-data-cleaning-and-error-checking)
6. [Feature Engineering](#6-feature-engineering)
7. [Create Relationship Table](#7-create-relationship-table)
8. [Exploratory Data Analysis](#8-exploratory-data-analysis)
9. [Analytical Base Table](#9-analytical-base-table)
10. [Encode Categorical Columns](#10-encode-categorical-columns)
11. [Separate X and y](#11-separate-x-and-y)
12. [Train Test Split](#12-train-test-split)
13. [Feature Scaling of Numerical Columns](#13-feature-scaling-of-numerical-columns)
14. [Train Model](#14-train-model)
15. [Predict X_test](#15-predict-x_test)
16. [Compare X and y](#16-compare-x-and-y)
17. [Evaluate Model](#17-evaluate-model)
18. [Business Analysis](#18-business-analysis)
19. [Save Model, Scaler and Features](#19-save-model-scaler-and-features)
20. [Download Model, Scaler and Features](#20-download-model-scaler-and-features)
21. [Key Findings and Insights](#21-key-findings-and-insights)
22. [Recommendations](#22-recommendations)
23. [Conclusion](#23-conclusion)

---

## 1. Project Overview

**Horizon Global Travel Management** books flights, hotels and car rentals for 60 corporate clients. Its data lives in five tables, arrives messy, and has to answer five operational questions.

| # | Business objective | ML task | Target | Model chosen |
| --- | --- | --- | --- | --- |
| 1 | Reduce booking cancellations | Classification | `cancelled`: booking status is Cancelled (17.1% of bookings) | Logistic Regression |
| 2 | Forecast trip cost to improve client budgeting | Regression | `amount_usd`, modelled as its logarithm and converted back to dollars | Hist Gradient Boosting Regressor |
| 3 | Improve travel policy compliance | Classification | `out_of_policy`: policy_compliant is No (26.2% of bookings) | Logistic Regression |
| 4 | Raise customer satisfaction | Classification | `dissatisfied`: CSAT score of 1 or 2 (27.6% of labelled tickets) | Logistic Regression |
| 5 | Detect booking amount anomalies and billing errors | Anomaly detection (unsupervised) | none while training; the answer key is used only to score the result | Isolation Forest |

What the project covers, start to finish:

- Auditing and cleaning five dirty tables, then joining them through a relationship table.
- Engineering features and building one analytical base table.
- Training at least three models per objective (one for anomalies), comparing them on a held-out test set and choosing by the metric that fits the problem.
- Turning each result into a business decision, saving the models and serving them in a Streamlit app.

**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, joblib, Streamlit.

**Repository structure**

```text
horizon-travel-prediction-system/
├── README.md
├── app.py                              # Streamlit app (section 20)
├── requirements.txt
├── images/                             # charts used in this README
├── travel_mgmt_raw/                    # the five raw tables, data dictionary and answer key
└── models/business_objectives/
    ├── 01_booking_cancellations/       # model.joblib, pipeline.joblib, scaler.joblib, features.json
    ├── 02_trip_cost_forecasting/
    ├── 03_policy_compliance/
    ├── 04_customer_satisfaction/
    └── 05_booking_amount_anomalies/
```

---

## 2. Dataset Overview

The data is synthetic, generated for this portfolio, and deliberately dirty: duplicate rows, mixed spellings, text in number columns and unreadable dates. That is the point. It lets the cleaning work be checked against known problems.

| Table | Rows | Columns | What it holds |
| --- | ---: | ---: | --- |
| `fact_bookings` | 6,060 | 16 | One row per booking: service, class, dates, channel, amount, status, policy flag |
| `fact_support_tickets` | 2,530 | 12 | One row per support ticket: category, priority, channel, response times, escalation, CSAT |
| `dim_clients` | 64 | 8 | Corporate clients: tier, country, industry, travel policy |
| `dim_suppliers` | 53 | 7 | Airlines, hotels and car rental firms: rating, cancellation flexibility, preferred partner |
| `dim_travelers` | 1,225 | 7 | Travellers: grade and home office |
| `anomaly_answer_key` | 6,000 | 2 | Which bookings were planted as anomalies. Used only to score Objective 5 |

Loading the five tables:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "travel_mgmt_raw"

dim_clients = pd.read_csv(f"{DATA_DIR}/dim_clients.csv")
dim_clients.head(5)

dim_suppliers = pd.read_csv(f"{DATA_DIR}/dim_suppliers.csv")
dim_suppliers.head(5)

dim_travelers = pd.read_csv(f"{DATA_DIR}/dim_travelers.csv")
dim_travelers.head(5)

fact_bookings = pd.read_csv(f"{DATA_DIR}/fact_bookings.csv")
fact_bookings.head(5)

fact_support_tickets = pd.read_csv(f"{DATA_DIR}/fact_support_tickets.csv")
fact_support_tickets.head(5)
```

---

## 3. Data Dictionary

Every column, what it means and the problems known to be hiding in it.

**`dim_clients`**

| Column                   | Type       | Description                                                         | Known data issues                             |
|:-------------------------|:-----------|:--------------------------------------------------------------------|:----------------------------------------------|
| client_id                | Text (key) | Unique client code, CLT-###                                         | Duplicate rows, lower case in other tables    |
| client_name              | Text       | Corporate client name                                               | Extra spaces, mixed case                      |
| country                  | Text       | Client head office country                                          | Spelling variants (UK, KSA, RSA...)           |
| industry                 | Text       | Banking, Oil and Gas, Technology, Pharma, Consulting, Manufacturing | Mixed case, spaces                            |
| client_tier              | Text       | Standard, Gold or Platinum service tier                             | Upper/lower case, single letters (S/G/P)      |
| contract_start           | Date       | Date the client signed                                              | Six date styles, junk dates                   |
| annual_travel_budget_usd | Currency   | Yearly travel budget in USD                                         | Text such as 1.25M, $1,250,000, USD 1,250,000 |
| has_travel_policy        | Boolean    | Client has a written travel policy                                  | Yes/Y/TRUE/1 and No/N/FALSE/0                 |

**`dim_travelers`**

| Column              | Type       | Description                           | Known data issues      |
|:--------------------|:-----------|:--------------------------------------|:-----------------------|
| traveler_id         | Text (key) | Unique traveler code, TRV-####        | Duplicate rows         |
| client_id           | Text (FK)  | Links to dim_clients                  | Orphan codes           |
| grade               | Text       | Staff, Manager, Director or Executive | Mixed case, spaces     |
| home_country        | Text       | Traveler base country                 | Spelling variants      |
| years_with_employer | Integer    | Years at the client company           | Negatives, 99-200, N/A |
| loyalty_member      | Boolean    | Member of a travel loyalty programme  | Yes/Y/TRUE/1 variants  |
| trips_last_year     | Integer    | Trips taken last year                 | Missing values         |

**`dim_suppliers`**

| Column                   | Type        | Description                               | Known data issues              |
|:-------------------------|:------------|:------------------------------------------|:-------------------------------|
| supplier_id              | Text (key)  | Unique supplier code, SUP-###             | Duplicate rows                 |
| supplier_name            | Text        | Airline, hotel or car rental company      | Extra spaces, mixed case       |
| supplier_type            | Text        | Airline, Hotel or Car Rental              | Mixed case, spaces             |
| country                  | Text        | Supplier base country                     | Spelling variants              |
| preferred_partner        | Boolean     | Negotiated preferred supplier             | Yes/Y/TRUE/1 variants          |
| service_rating           | Decimal 1-5 | Company rating of service quality         | Text like '4/5', 'four', 'N/A' |
| cancellation_flexibility | Text        | Flexible, Semi-Flexible or Non-Refundable | Case, hyphen variants          |

**`fact_bookings`**

| Column           | Type       | Description                                                                         | Known data issues                                             |
|:-----------------|:-----------|:------------------------------------------------------------------------------------|:--------------------------------------------------------------|
| booking_id       | Text (key) | Unique booking code, BKG-######                                                     | Duplicate rows                                                |
| client_id        | Text (FK)  | Links to dim_clients                                                                | Some in lower case                                            |
| traveler_id      | Text (FK)  | Links to dim_travelers                                                              | Orphan codes                                                  |
| supplier_id      | Text (FK)  | Links to dim_suppliers                                                              | Missing values                                                |
| service_type     | Text       | Flight, Hotel or Car Rental                                                         | Mixed case, spaces                                            |
| travel_class     | Text       | Cabin or room/vehicle class (Economy...First, Standard...Suite, Economy/SUV/Luxury) | Mixed case, spaces                                            |
| booking_channel  | Text       | Online Tool, Mobile App, Travel Agent or Phone                                      | Case, spaces removed                                          |
| origin_city      | Text       | Trip origin                                                                         | Variants: NYC, LON, DXB, Bombay, São Paulo...                 |
| destination_city | Text       | Trip destination                                                                    | City-name and airport-code variants                           |
| booking_date     | Date       | Date the booking was made                                                           | Six date styles, junk dates                                   |
| travel_date      | Date       | Start of travel or stay                                                             | Six date styles; swapped with return_date in some rows        |
| return_date      | Date       | End of travel or stay                                                               | Mixed formats                                                 |
| amount_usd       | Currency   | Booking value in USD                                                                | Text with symbols, zero/negative values, planted price spikes |
| policy_compliant | Boolean    | Booking follows the client travel policy                                            | Yes/Y/TRUE/1 variants                                         |
| booking_status   | Text       | Completed, Cancelled or No-show                                                     | Case, 'Canceled', 'No show'                                   |
| refund_usd       | Currency   | Refund paid on cancellation (LEAKAGE for Objective 1)                               | Text with symbols                                             |

**`fact_support_tickets`**

| Column               | Type        | Description                                                                                               | Known data issues                       |
|:---------------------|:------------|:----------------------------------------------------------------------------------------------------------|:----------------------------------------|
| ticket_id            | Text (key)  | Unique ticket code, TKT-######                                                                            | Duplicate rows                          |
| booking_id           | Text (FK)   | Links to fact_bookings                                                                                    | Orphan codes                            |
| client_id            | Text (FK)   | Links to dim_clients                                                                                      | Missing values                          |
| ticket_date          | Date        | Date the ticket was opened                                                                                | Six date styles                         |
| category             | Text        | Refund Request, Itinerary Change, Visa or Document Query, Billing Dispute, Complaint, Emergency Rebooking | Mixed case, spaces                      |
| priority             | Text        | Low, Medium, High or Critical                                                                             | Mixed case, spaces                      |
| contact_channel      | Text        | Email, Phone, Chat or Portal                                                                              | Mixed case, spaces                      |
| first_response_hours | Decimal     | Hours until first reply                                                                                   | Text like '2 hrs', '1h', negatives, N/A |
| resolution_hours     | Decimal     | Hours until ticket closed                                                                                 | Negatives, 0, 9999                      |
| escalated            | Boolean     | Ticket escalated to a manager                                                                             | Yes/Y/TRUE/1 variants                   |
| csat_score           | Integer 1-5 | Customer satisfaction after closure (target for Objective 4)                                              | Blank, 'no response', 0, 6, 9, 10       |
| agent_notes_length   | Integer     | Characters in agent notes                                                                                 | -                                       |

---

## 4. Data Quality Audit

Before touching anything I counted the problems. The audit checks duplicates, missing keys, missing values, impossible values (a CSAT score of 9, a resolution time of 9999 hours), unreadable dates and broken links between tables.

**Table overview**

| table | rows | columns | duplicate_rows | missing_key | duplicate_key_rows | missing_cells |
| --- | --- | --- | --- | --- | --- | --- |
| tickets | 2530 | 12 | 30 | 0 | 60 | 28 |
| bookings | 6060 | 16 | 60 | 0 | 120 | 42 |
| clients | 64 | 8 | 4 | 0 | 8 | 0 |

**Missing values**

| table | column | missing |
| --- | --- | --- |
| tickets | client_id | 20 |
| tickets | first_response_hours | 8 |
| bookings | supplier_id | 42 |

**Rule checks on support tickets**

| check | rows |
| --- | --- |
| CSAT nonnumeric | 60 |
| CSAT outside 1-5 | 10 |
| First response missing | 8 |
| First response nonnumeric / negative | 7 |
| Resolution nonnumeric / negative | 4 |
| Resolution zero or >=9999 | 8 |
| Unparsed ticket dates | 24 |

**Broken links between tables**

| relationship | nonmissing_keys | orphan_keys |
| --- | --- | --- |
| tickets.booking_id -> bookings.booking_id | 2530 | 25 |
| tickets.client_id -> clients.client_id | 2510 | 0 |
| bookings.client_id -> clients.client_id | 6060 | 0 |

**What stands out**

- Duplicates are real: 30 tickets, 60 bookings and 4 clients appear twice.
- 60 CSAT scores are the text "no response" and 10 more fall outside 1 to 5. Those tickets cannot be used as labels.
- 25 tickets point to a booking that does not exist, and 42 bookings have no supplier.
- 24 ticket dates cannot be read.

### Full code

```python
audit_tables = {
    "tickets": (fact_support_tickets, "ticket_id"),
    "bookings": (fact_bookings, "booking_id"),
    "clients": (dim_clients, "client_id"),
}

overview = pd.DataFrame([
    {
        "table": name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_key": int(frame[key].isna().sum()),
        "duplicate_key_rows": int(frame[key].duplicated(keep=False).sum()),
        "missing_cells": int(frame.isna().sum().sum()),
    }
    for name, (frame, key) in audit_tables.items()
]).set_index("table")

missing = pd.DataFrame([
    {"table": name, "column": column, "missing": int(count)}
    for name, (frame, _) in audit_tables.items()
    for column, count in frame.isna().sum().items()
    if count
])

tickets = fact_support_tickets
csat = pd.to_numeric(tickets["csat_score"], errors="coerce")
response = pd.to_numeric(tickets["first_response_hours"], errors="coerce")
resolution = pd.to_numeric(tickets["resolution_hours"], errors="coerce")

csat_present = (
    tickets["csat_score"].notna()
    & tickets["csat_score"].astype(str).str.strip().ne("")
)
response_present = (
    tickets["first_response_hours"].notna()
    & tickets["first_response_hours"].astype(str).str.strip().ne("")
)
resolution_present = (
    tickets["resolution_hours"].notna()
    & tickets["resolution_hours"].astype(str).str.strip().ne("")
)

rules = pd.DataFrame([
    ("CSAT nonnumeric", int((csat_present & csat.isna()).sum())),
    ("CSAT outside 1-5", int((csat.notna() & ~csat.between(1, 5)).sum())),
    ("First response missing", int(tickets["first_response_hours"].isna().sum())),
    (
        "First response nonnumeric / negative",
        int((response_present & response.isna()).sum() + response.lt(0).sum()),
    ),
    (
        "Resolution nonnumeric / negative",
        int((resolution_present & resolution.isna()).sum() + resolution.lt(0).sum()),
    ),
    (
        "Resolution zero or >=9999",
        int((resolution.eq(0) | resolution.ge(9999)).sum()),
    ),
    (
        "Unparsed ticket dates",
        int(tickets["ticket_date"].map(
            lambda value: pd.isna(
                pd.to_datetime(value, format="mixed", errors="coerce")
            )
        ).sum()),
    ),
], columns=["check", "rows"])

normalize_id = lambda values: (
    values.astype("string").str.strip().str.upper()
)

client_ids = set(normalize_id(dim_clients["client_id"].dropna()))
booking_ids = set(normalize_id(fact_bookings["booking_id"].dropna()))

relationships = pd.DataFrame([
    (
        "tickets.booking_id -> bookings.booking_id",
        int(tickets["booking_id"].notna().sum()),
        int((
            ~normalize_id(tickets["booking_id"].dropna()).isin(booking_ids)
        ).sum()),
    ),
    (
        "tickets.client_id -> clients.client_id",
        int(tickets["client_id"].notna().sum()),
        int((
            ~normalize_id(tickets["client_id"].dropna()).isin(client_ids)
        ).sum()),
    ),
    (
        "bookings.client_id -> clients.client_id",
        int(fact_bookings["client_id"].notna().sum()),
        int((
            ~normalize_id(fact_bookings["client_id"].dropna()).isin(client_ids)
        ).sum()),
    ),
], columns=["relationship", "nonmissing_keys", "orphan_keys"])

display(overview)
display(missing)
display(rules)
display(relationships)
display(
    tickets["csat_score"]
    .value_counts(dropna=False)
    .rename_axis("csat_score")
    .to_frame("rows")
)
```

---

## 5. Data Cleaning and Error Checking

Cleaning keeps the raw files untouched and writes clean copies to `travel_mgmt_clean/`. Every fix is followed by a check that counts what is still wrong.

- **IDs** trimmed and upper-cased so `bkg-300001 ` and `BKG-300001` match.
- **Duplicates** removed on the key, keeping the first record.
- **Text columns** trimmed, collapsed and lower-cased. Tier shortcuts (`G`, `S`, `P`) expanded to gold, standard and platinum.
- **CSAT** kept only when it is a whole number from 1 to 5.
- **Hours** parsed from strings such as `2 hrs`; negative values dropped; resolution times of 0 or 9999 and above treated as missing.
- **Dates** parsed in mixed formats, month first when ambiguous.
- **Yes/No fields** converted to true booleans.

**Rows before and after**

| table | raw_rows | clean_rows | duplicate_rows_removed |
| --- | --- | --- | --- |
| tickets | 2530 | 2500 | 30 |
| bookings | 6060 | 6000 | 60 |
| clients | 64 | 60 | 4 |

**What is still missing after cleaning**

| check | rows |
| --- | --- |
| Ticket CSAT missing or invalid after 1-5 validation | 70 |
| Ticket dates missing or unparseable | 24 |
| First response hours missing or invalid | 11 |
| Resolution hours missing or invalid | 12 |
| Unrecognized ticket escalation values | 0 |
| Unrecognized client travel-policy values | 0 |

The leftovers are real gaps, not formatting problems. They are handled later by imputation inside the model pipelines, so no row is guessed at here.

### Full code

```python
from pathlib import Path

# Keep the raw DataFrames unchanged.
tickets_clean = fact_support_tickets.copy()
bookings_clean = fact_bookings.copy()
clients_clean = dim_clients.copy()

def normalize_text(series):
    """Trim whitespace, collapse repeated spaces, and normalize case."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )

def normalize_ids(frame, columns):
    """Standardize ID values without changing missing values."""
    for column in columns:
        frame[column] = (
            frame[column]
            .astype("string")
            .str.strip()
            .str.upper()
        )

def parse_hour_values(series):
    """Parse numbers and simple hour strings such as '2 hrs' or '1h'."""
    numeric_text = series.astype("string").str.extract(
        r"(?i)^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
        r"(?:h|hr|hrs|hours?)?\s*$",
        expand=False,
    )
    return pd.to_numeric(numeric_text, errors="coerce")

def parse_boolean(series):
    """Convert common yes/no representations to nullable booleans."""
    mapping = {
        "yes": True,
        "y": True,
        "true": True,
        "1": True,
        "no": False,
        "n": False,
        "false": False,
        "0": False,
    }
    normalized = normalize_text(series)
    parsed = normalized.map(mapping).astype("boolean")
    invalid = normalized.notna() & parsed.isna()
    return parsed, invalid

# Standardize identifiers before removing duplicate records.
normalize_ids(tickets_clean, ["ticket_id", "booking_id", "client_id"])
normalize_ids(
    bookings_clean,
    ["booking_id", "client_id", "traveler_id", "supplier_id"],
)
normalize_ids(clients_clean, ["client_id"])

# These repeated keys were exact duplicate records in the audit.
tickets_raw_rows = len(tickets_clean)
bookings_raw_rows = len(bookings_clean)
clients_raw_rows = len(clients_clean)

tickets_clean = tickets_clean.drop_duplicates(
    subset="ticket_id", keep="first"
).copy()
bookings_clean = bookings_clean.drop_duplicates(
    subset="booking_id", keep="first"
).copy()
clients_clean = clients_clean.drop_duplicates(
    subset="client_id", keep="first"
).copy()

# Normalize categories used for ticket analysis and the related booking/client data.
for frame, columns in [
    (
        tickets_clean,
        ["category", "priority", "contact_channel"],
    ),
    (bookings_clean, ["service_type"]),
    (clients_clean, ["client_tier"]),
]:
    for column in columns:
        frame[column] = normalize_text(frame[column])

# Standardize client tier abbreviations.
clients_clean["client_tier"] = clients_clean["client_tier"].replace(
    {"s": "standard", "g": "gold", "p": "platinum"}
)

# Parse ticket dates. Ambiguous numeric dates are interpreted month-first.
tickets_clean["ticket_date"] = pd.to_datetime(
    tickets_clean["ticket_date"],
    format="mixed",
    dayfirst=False,
    errors="coerce",
)

# Keep CSAT only when it is an integer rating from 1 through 5.
csat_numeric = pd.to_numeric(tickets_clean["csat_score"], errors="coerce")
valid_csat = csat_numeric.between(1, 5) & csat_numeric.mod(1).eq(0)
tickets_clean["csat_score"] = csat_numeric.where(valid_csat).astype("Int64")

# Parse response times; retain zero as a valid immediate response.
tickets_clean["first_response_hours"] = parse_hour_values(
    tickets_clean["first_response_hours"]
)
tickets_clean.loc[
    tickets_clean["first_response_hours"] < 0,
    "first_response_hours",
] = pd.NA

# Resolution time is retained for data completeness, but is post-resolution
# information and must not be used as a pre-resolution model predictor.
tickets_clean["resolution_hours"] = pd.to_numeric(
    tickets_clean["resolution_hours"],
    errors="coerce",
)
invalid_resolution = (
    (tickets_clean["resolution_hours"] <= 0)
    | (tickets_clean["resolution_hours"] >= 9999)
)
tickets_clean.loc[invalid_resolution, "resolution_hours"] = pd.NA

# Normalize boolean fields and track any values that could not be interpreted.
tickets_clean["escalated"], invalid_escalated = parse_boolean(
    tickets_clean["escalated"]
)
clients_clean["has_travel_policy"], invalid_client_policy = parse_boolean(
    clients_clean["has_travel_policy"]
)

# Summarize the cleaning results.
cleaning_summary = pd.DataFrame([
    {
        "table": "tickets",
        "raw_rows": tickets_raw_rows,
        "clean_rows": len(tickets_clean),
        "duplicate_rows_removed": tickets_raw_rows - len(tickets_clean),
    },
    {
        "table": "bookings",
        "raw_rows": bookings_raw_rows,
        "clean_rows": len(bookings_clean),
        "duplicate_rows_removed": bookings_raw_rows - len(bookings_clean),
    },
    {
        "table": "clients",
        "raw_rows": clients_raw_rows,
        "clean_rows": len(clients_clean),
        "duplicate_rows_removed": clients_raw_rows - len(clients_clean),
    },
])

cleaning_issues = pd.DataFrame([
    ("Ticket CSAT missing or invalid after 1-5 validation",
     int(tickets_clean["csat_score"].isna().sum())),
    ("Ticket dates missing or unparseable",
     int(tickets_clean["ticket_date"].isna().sum())),
    ("First response hours missing or invalid",
     int(tickets_clean["first_response_hours"].isna().sum())),
    ("Resolution hours missing or invalid",
     int(tickets_clean["resolution_hours"].isna().sum())),
    ("Unrecognized ticket escalation values",
     int(invalid_escalated.sum())),
    ("Unrecognized client travel-policy values",
     int(invalid_client_policy.sum())),
], columns=["check", "rows"])

display(cleaning_summary)
display(cleaning_issues)

# Save cleaned copies separately; the original raw CSVs are not overwritten.
CLEAN_DIR = Path("travel_mgmt_clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

tickets_clean.to_csv(CLEAN_DIR / "fact_support_tickets.csv", index=False)
bookings_clean.to_csv(CLEAN_DIR / "fact_bookings.csv", index=False)
clients_clean.to_csv(CLEAN_DIR / "dim_clients.csv", index=False)

print("Cleaned datasets saved to:", CLEAN_DIR.resolve())
```

---

## 6. Feature Engineering

The first step-by-step workflow (sections 6 to 13) runs on the support tickets, with customer dissatisfaction as the target.

| Feature | What it captures |
| --- | --- |
| `dissatisfied` (target) | 1 when CSAT is 1 or 2, 0 when 3 to 5, empty when there is no valid score |
| `ticket_month`, `ticket_day_of_week`, `ticket_is_weekend` | When the ticket was opened |
| `ticket_date_missing` | 1 when the date could not be read |
| `first_response_hours`, `first_response_missing` | How fast the team first replied, and whether that is known |
| `category`, `priority`, `contact_channel` | What the ticket is about and how it arrived |

**A deliberate choice:** `resolution_hours`, `escalated`, `agent_notes_length` and `csat_score` are left out of this workflow, because they are only known after a ticket closes. The Objective 4 model in section 14 does use resolution time and escalation, because the business objective asks for them. That makes it a tool for reviewing and closing tickets, not for predicting them at the moment they open. The app says so on that page.

**Target counts**

| Outcome | Tickets |
| --- | ---: |
| Satisfied (0) | 1,759 |
| Dissatisfied (1) | 671 |
| No valid score | 70 |

Dissatisfaction rate among labelled tickets: **27.6%**.

### Full code

```python
from pathlib import Path
import pandas as pd

CLEAN_DIR = Path("travel_mgmt_clean")

tickets_features = pd.read_csv(
    CLEAN_DIR / "fact_support_tickets.csv",
    parse_dates=["ticket_date"],
)

required_columns = {
    "ticket_date",
    "category",
    "priority",
    "contact_channel",
    "first_response_hours",
    "resolution_hours",
    "escalated",
    "csat_score",
    "agent_notes_length",
}
missing_columns = required_columns.difference(tickets_features.columns)
if missing_columns:
    raise ValueError(
        f"Cleaned ticket data is missing columns: {sorted(missing_columns)}"
    )

# Create the target from valid integer CSAT scores:
# 1–2 = dissatisfied; 3–5 = satisfied.
csat_numeric = pd.to_numeric(
    tickets_features["csat_score"],
    errors="coerce",
)
valid_csat = (
    csat_numeric.between(1, 5)
    & csat_numeric.mod(1).eq(0)
)
csat_numeric = csat_numeric.where(valid_csat)

tickets_features["dissatisfied"] = pd.Series(
    pd.NA,
    index=tickets_features.index,
    dtype="Int8",
)
labelled = csat_numeric.notna()
tickets_features.loc[labelled, "dissatisfied"] = (
    csat_numeric.loc[labelled].le(2).astype("int8")
)

# Derive calendar features from the ticket creation date.
ticket_dates = pd.to_datetime(
    tickets_features["ticket_date"],
    errors="coerce",
)
tickets_features["ticket_month"] = ticket_dates.dt.month.astype("Int8")
tickets_features["ticket_day_of_week"] = (
    ticket_dates.dt.dayofweek.astype("Int8")
)
tickets_features["ticket_is_weekend"] = (
    ticket_dates.dt.dayofweek
    .isin([5, 6])
    .where(ticket_dates.notna())
    .astype("Int8")
)
tickets_features["ticket_date_missing"] = (
    ticket_dates.isna().astype("int8")
)

# First-response time is included assuming scores are made after the first
# response but before ticket resolution.
response_hours = pd.to_numeric(
    tickets_features["first_response_hours"],
    errors="coerce",
)
tickets_features["first_response_hours"] = response_hours.where(
    response_hours.ge(0)
)
tickets_features["first_response_missing"] = (
    tickets_features["first_response_hours"].isna().astype("int8")
)

# These are the raw predictors for later modeling steps.
FEATURE_COLUMNS = [
    "category",
    "priority",
    "contact_channel",
    "first_response_hours",
    "first_response_missing",
    "ticket_month",
    "ticket_day_of_week",
    "ticket_is_weekend",
    "ticket_date_missing",
]
TARGET_COLUMN = "dissatisfied"

print("Feature-engineered ticket rows:", len(tickets_features))
print("Predictor columns:", FEATURE_COLUMNS)
print("Target counts (0=satisfied, 1=dissatisfied, NA=unlabeled):")
display(
    tickets_features[TARGET_COLUMN]
    .value_counts(dropna=False)
    .sort_index()
)
print(
    "Dissatisfaction rate among labeled tickets:",
    round(tickets_features[TARGET_COLUMN].dropna().mean(), 4),
)
print(
    "Excluded post-resolution or outcome fields:",
    ["resolution_hours", "escalated", "agent_notes_length", "csat_score"],
)
display(tickets_features[FEATURE_COLUMNS + [TARGET_COLUMN]].head())
```

---

## 7. Create Relationship Table

A relationship table records how each ticket links to a booking and a client, and whether the links hold. It checks two things before any join: every key is unique, so a join cannot multiply rows, and the client on the ticket matches the client on the booking.

| check | rows |
| --- | --- |
| Ticket rows | 2500 |
| Tickets with unmatched booking IDs | 25 |
| Ticket client IDs missing from client dimension | 20 |
| Booking client IDs missing from client dimension | 25 |
| Ticket/booking client ID mismatches | 0 |

No ticket and booking disagree about the client. The 25 unmatched bookings and 20 missing client IDs stay in the data and are flagged, not deleted.

### Full code

```python
from pathlib import Path
import pandas as pd

CLEAN_DIR = Path("travel_mgmt_clean")

# Load the cleaned related tables.
bookings_clean = pd.read_csv(CLEAN_DIR / "fact_bookings.csv")
clients_clean = pd.read_csv(CLEAN_DIR / "dim_clients.csv")

def normalize_key(series):
    return series.astype("string").str.strip().str.upper()

# Start with ticket keys only; this keeps the relationship table focused.
relationship_source = tickets_features[
    ["ticket_id", "booking_id", "client_id"]
].copy()

booking_lookup = (
    bookings_clean[["booking_id", "client_id"]]
    .rename(columns={"client_id": "booking_client_id"})
    .copy()
)
client_lookup = clients_clean[["client_id"]].copy()

# Normalize the keys consistently before matching.
for frame, columns in [
    (relationship_source, ["ticket_id", "booking_id", "client_id"]),
    (booking_lookup, ["booking_id", "booking_client_id"]),
    (client_lookup, ["client_id"]),
]:
    for column in columns:
        frame[column] = normalize_key(frame[column])

# Require unique identifiers so merges cannot silently multiply ticket rows.
for frame, key, label in [
    (relationship_source, "ticket_id", "ticket"),
    (booking_lookup, "booking_id", "booking"),
    (client_lookup, "client_id", "client"),
]:
    if frame[key].duplicated().any():
        raise ValueError(
            f"The {label} table must have unique {key} values "
            "before relationship matching."
        )

# Match each ticket to a booking. A left join preserves tickets whose
# booking ID is not found.
relationship_table = relationship_source.merge(
    booking_lookup,
    on="booking_id",
    how="left",
    validate="many_to_one",
    indicator="booking_merge",
)

relationship_table["booking_found"] = (
    relationship_table["booking_merge"].eq("both")
)

valid_client_ids = set(client_lookup["client_id"].dropna())

relationship_table["ticket_client_found"] = (
    relationship_table["client_id"]
    .isin(valid_client_ids)
    .astype("boolean")
)
relationship_table["booking_client_found"] = (
    relationship_table["booking_client_id"]
    .isin(valid_client_ids)
    .astype("boolean")
)

# Compare ticket and booking client IDs only when both are available.
comparable_clients = (
    relationship_table["client_id"].notna()
    & relationship_table["booking_client_id"].notna()
    & relationship_table["booking_found"]
)
relationship_table["ticket_booking_client_match"] = pd.Series(
    pd.NA,
    index=relationship_table.index,
    dtype="boolean",
)
relationship_table.loc[
    comparable_clients,
    "ticket_booking_client_match",
] = relationship_table.loc[
    comparable_clients, "client_id"
].eq(
    relationship_table.loc[comparable_clients, "booking_client_id"]
)

relationship_table = (
    relationship_table
    .rename(columns={"client_id": "ticket_client_id"})
    .drop(columns="booking_merge")
)

# Confirm that the relationship table still has exactly one row per ticket.
if (
    len(relationship_table) != len(tickets_features)
    or relationship_table["ticket_id"].duplicated().any()
):
    raise AssertionError(
        "The relationship table must preserve exactly one row per ticket."
    )

relationship_summary = pd.DataFrame([
    ("Ticket rows", len(relationship_table)),
    ("Tickets with unmatched booking IDs",
     int((~relationship_table["booking_found"]).sum())),
    ("Ticket client IDs missing from client dimension",
     int((~relationship_table["ticket_client_found"]).sum())),
    ("Booking client IDs missing from client dimension",
     int((~relationship_table["booking_client_found"]).sum())),
    ("Ticket/booking client ID mismatches",
     int((relationship_table["ticket_booking_client_match"] == False).sum())),
], columns=["check", "rows"])

display(relationship_summary)
display(relationship_table.head())

# Save this relationship table as its own output.
RELATIONSHIP_DIR = Path("travel_mgmt_relationships")
RELATIONSHIP_DIR.mkdir(parents=True, exist_ok=True)

relationship_table.to_csv(
    RELATIONSHIP_DIR / "ticket_booking_client_relationships.csv",
    index=False,
)
print("Relationship table saved to:", RELATIONSHIP_DIR.resolve())
```

---

## 8. Exploratory Data Analysis

Only the 2,430 tickets with a valid CSAT score are compared. **27.6% of them ended dissatisfied.**

<img width="1802" height="1287" alt="travel_eda_01_ticket_outcomes" src="https://github.com/user-attachments/assets/bcdd173a-797e-46f1-863f-8f7f2d8b7ed1" />
<p align="center"><sub><em>Dissatisfaction by ticket category, priority and contact channel</em></sub></p>

- **Complaints are the problem.** 56.2% of complaint tickets end dissatisfied, against 21% to 30% for every other category.
- **Chat works.** Only 9.4% of chat tickets end badly, against 32% to 34% for phone, email and portal.
- **Speed matters.** Satisfied customers waited a median of 1.3 hours for a first reply. Dissatisfied customers waited 6.0 hours.
- **Priority is not protective.** Low and medium priority tickets are dissatisfied about 32% to 33% of the time, high and critical about 17%.
- **No seasonal pattern.** Months with 30 or more tickets range from 19% to 36% with no trend.

<img width="1547" height="507" alt="travel_eda_02_monthly_dissatisfaction" src="https://github.com/user-attachments/assets/ea03efa2-f407-4b1e-adb0-2d0129039542" />
<p align="center"><sub><em>Monthly dissatisfaction rate, labelled tickets only. The two extreme points at the far right are months with a single ticket each</em></sub></p>

These are associations, not causes. A complaint is not unhappy because it is a complaint; it arrives unhappy.

### Full code

```python
import pandas as pd
import matplotlib.pyplot as plt

required_columns = {
    "ticket_date",
    "category",
    "priority",
    "contact_channel",
    "first_response_hours",
    "dissatisfied",
}
missing_columns = required_columns.difference(tickets_features.columns)
if missing_columns:
    raise ValueError(f"Missing required feature columns: {sorted(missing_columns)}")

# Keep the engineered data unchanged. Only labeled tickets can contribute
# to satisfaction comparisons.
eda_all = tickets_features.copy()
eda_tickets = eda_all.loc[eda_all["dissatisfied"].notna()].copy()
eda_tickets["dissatisfied"] = eda_tickets["dissatisfied"].astype("int8")

print(f"All tickets: {len(eda_all):,}")
print(f"Tickets with valid CSAT labels: {len(eda_tickets):,}")
print(
    "Dissatisfaction rate among labeled tickets:",
    f"{eda_tickets['dissatisfied'].mean():.1%}",
)

# Compare ticket volume and dissatisfaction rates across categories.
for column in ["category", "priority", "contact_channel"]:
    grouped_data = eda_tickets.copy()
    grouped_data[column] = (
        grouped_data[column].fillna("missing").astype("string")
    )

    summary = (
        grouped_data.groupby(column, dropna=False)
        .agg(
            tickets=("dissatisfied", "size"),
            dissatisfied_tickets=("dissatisfied", "sum"),
            dissatisfaction_rate=("dissatisfied", "mean"),
        )
        .sort_values("dissatisfaction_rate", ascending=False)
    )

    print(f"\nDissatisfaction by {column}:")
    display(summary)

# Compare first-response time for satisfied and dissatisfied tickets.
eda_tickets["first_response_hours"] = pd.to_numeric(
    eda_tickets["first_response_hours"],
    errors="coerce",
)

response_summary = (
    eda_tickets.groupby("dissatisfied")["first_response_hours"]
    .agg(valid_responses="count", median_hours="median", mean_hours="mean")
    .rename(index={0: "satisfied", 1: "dissatisfied"})
)

print("\nFirst-response time by outcome:")
display(response_summary)

# Explore monthly dissatisfaction rates for labeled tickets with valid dates.
eda_tickets["ticket_month"] = pd.to_datetime(
    eda_tickets["ticket_date"],
    errors="coerce",
).dt.to_period("M")

monthly_summary = (
    eda_tickets.dropna(subset=["ticket_month"])
    .groupby("ticket_month")
    .agg(
        tickets=("dissatisfied", "size"),
        dissatisfied_tickets=("dissatisfied", "sum"),
        dissatisfaction_rate=("dissatisfied", "mean"),
    )
)

print("\nMonthly dissatisfaction:")
display(monthly_summary)

# Plot outcome counts and rates by category, priority, and contact channel.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

outcome_counts = (
    eda_tickets["dissatisfied"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)
outcome_counts.index = ["Satisfied (CSAT 3-5)", "Dissatisfied (CSAT 1-2)"]
outcome_counts.plot(
    kind="bar",
    ax=axes[0, 0],
    color=["#4C9F70", "#D9534F"],
)
axes[0, 0].set_title("Labeled ticket outcomes")
axes[0, 0].set_ylabel("Ticket count")
axes[0, 0].tick_params(axis="x", rotation=15)

plot_details = [
    (axes[0, 1], "category", "Dissatisfaction by category"),
    (axes[1, 0], "priority", "Dissatisfaction by priority"),
    (axes[1, 1], "contact_channel", "Dissatisfaction by contact channel"),
]

for axis, column, title in plot_details:
    plot_data = eda_tickets.copy()
    plot_data[column] = plot_data[column].fillna("missing").astype("string")
    rates = (
        plot_data.groupby(column, dropna=False)["dissatisfied"]
        .mean()
        .sort_values(ascending=False)
    )
    rates.plot(kind="bar", ax=axis, color="#4472C4")
    axis.set_title(title)
    axis.set_ylabel("Dissatisfaction rate")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=35)

fig.tight_layout()
plt.show()

if not monthly_summary.empty:
    ax = monthly_summary["dissatisfaction_rate"].plot(
        figsize=(12, 4),
        marker="o",
        title="Monthly dissatisfaction rate (labeled tickets only)",
        ylim=(0, 1),
    )
    ax.set_ylabel("Dissatisfaction rate")
    ax.set_xlabel("Ticket month")
    plt.tight_layout()
    plt.show()
```

---

## 9. Analytical Base Table

The analytical base table has one row per ticket: the engineered features, the target, plus the booking's service type and the client's tier from the joins. Left joins keep every ticket, and the code stops if a join would duplicate a row.

| check | rows |
| --- | --- |
| ABT rows | 2500 |
| Labeled tickets | 2430 |
| Unlabeled tickets | 70 |
| Unmatched booking IDs | 25 |
| Ticket/booking client ID mismatches | 0 |

2,500 rows and 17 columns: IDs, 11 predictors, the target and two relationship flags.

### Full code

```python
from pathlib import Path
import pandas as pd

CLEAN_DIR = Path("travel_mgmt_clean")
RELATIONSHIP_DIR = Path("travel_mgmt_relationships")

# Load the cleaned lookup tables and the saved relationship table.
bookings_clean = pd.read_csv(CLEAN_DIR / "fact_bookings.csv")
clients_clean = pd.read_csv(CLEAN_DIR / "dim_clients.csv")
relationship_table = pd.read_csv(
    RELATIONSHIP_DIR / "ticket_booking_client_relationships.csv"
)

# Confirm that the inputs from earlier steps are present.
required_ticket_columns = {
    "ticket_id",
    "booking_id",
    "client_id",
    "dissatisfied",
    *FEATURE_COLUMNS,
}
missing_columns = required_ticket_columns.difference(tickets_features.columns)
if missing_columns:
    raise ValueError(
        f"Feature-engineered ticket data is missing columns: "
        f"{sorted(missing_columns)}"
    )

required_relationship_columns = {
    "ticket_id",
    "booking_found",
    "ticket_booking_client_match",
}
missing_columns = required_relationship_columns.difference(
    relationship_table.columns
)
if missing_columns:
    raise ValueError(
        f"Relationship table is missing columns: {sorted(missing_columns)}"
    )

# Require unique keys so merges cannot multiply ticket rows.
for frame, key, label in [
    (tickets_features, "ticket_id", "ticket"),
    (relationship_table, "ticket_id", "relationship"),
    (bookings_clean, "booking_id", "booking"),
    (clients_clean, "client_id", "client"),
]:
    if frame[key].duplicated().any():
        raise ValueError(
            f"{label} table contains duplicate values in {key}."
        )

booking_lookup = (
    bookings_clean[["booking_id", "service_type"]]
    .rename(columns={"service_type": "booking_service_type"})
)

client_lookup = clients_clean[["client_id", "client_tier"]]

# Keep only ticket-level predictors, target, and IDs needed to join related data.
abt = tickets_features[
    ["ticket_id", "booking_id", "client_id", *FEATURE_COLUMNS, "dissatisfied"]
].copy()

# Left joins preserve every ticket, including tickets with missing relationships.
abt = abt.merge(
    relationship_table[
        ["ticket_id", "booking_found", "ticket_booking_client_match"]
    ],
    on="ticket_id",
    how="left",
    validate="one_to_one",
)

abt = abt.merge(
    booking_lookup,
    on="booking_id",
    how="left",
    validate="many_to_one",
)

abt = abt.merge(
    client_lookup,
    on="client_id",
    how="left",
    validate="many_to_one",
)

# Define predictors separately from IDs, audit fields, and the target.
ABT_FEATURE_COLUMNS = [
    *FEATURE_COLUMNS,
    "booking_service_type",
    "client_tier",
]
ABT_TARGET_COLUMN = "dissatisfied"

# Ensure the joins kept exactly one row for every ticket.
if len(abt) != len(tickets_features):
    raise AssertionError(
        "The ABT row count must match the ticket row count."
    )
if abt["ticket_id"].duplicated().any():
    raise AssertionError(
        "The ABT must contain exactly one row per ticket."
    )

abt_summary = pd.DataFrame([
    ("ABT rows", len(abt)),
    ("Labeled tickets", int(abt[ABT_TARGET_COLUMN].notna().sum())),
    ("Unlabeled tickets", int(abt[ABT_TARGET_COLUMN].isna().sum())),
    ("Unmatched booking IDs", int((~abt["booking_found"]).sum())),
    (
        "Ticket/booking client ID mismatches",
        int((abt["ticket_booking_client_match"] == False).sum()),
    ),
], columns=["check", "rows"])

print("ABT shape:", abt.shape)
print("Predictor columns:", ABT_FEATURE_COLUMNS)
display(abt_summary)
display(abt.head())

# Keep the ABT as a separate output; raw and cleaned source files are unchanged.
ABT_DIR = Path("abt")
ABT_DIR.mkdir(parents=True, exist_ok=True)

abt.to_csv(ABT_DIR / "support_satisfaction_abt.csv", index=False)
print("Analytical base table saved to:", (ABT_DIR / "support_satisfaction_abt.csv").resolve())
```

---

## 10. Encode Categorical Columns

Models need numbers, so each category becomes a 0/1 column. The category lists are fixed in advance, with a `__missing__` slot for blanks. If a value appears that the lists do not know, the code stops instead of quietly encoding it as all zeros.

- **31 columns in total:** 6 numeric and 25 one-hot columns from 5 categorical fields (category, priority, contact channel, booking service type, client tier).
- Numeric gaps are left as they are for the scaling step.

### Full code

```python
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

ABT_PATH = Path("abt") / "support_satisfaction_abt.csv"
abt = pd.read_csv(ABT_PATH)

CATEGORICAL_FEATURES = [
    "category",
    "priority",
    "contact_channel",
    "booking_service_type",
    "client_tier",
]

NUMERIC_FEATURES = [
    "first_response_hours",
    "first_response_missing",
    "ticket_month",
    "ticket_day_of_week",
    "ticket_is_weekend",
    "ticket_date_missing",
]

TARGET_COLUMN = "dissatisfied"

required_columns = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
    + [TARGET_COLUMN, "ticket_id"]
)
missing_columns = set(required_columns) - set(abt.columns)

if missing_columns:
    raise ValueError(
        f"ABT is missing required columns: {sorted(missing_columns)}"
    )

if abt["ticket_id"].duplicated().any():
    raise ValueError("ABT must have one row per ticket before encoding.")

# Use the known categories and reserve a category for missing values.
CATEGORY_LEVELS = [
    [
        "billing dispute",
        "complaint",
        "emergency rebooking",
        "itinerary change",
        "refund request",
        "visa or document query",
        "__missing__",
    ],
    ["critical", "high", "low", "medium", "__missing__"],
    ["chat", "email", "phone", "portal", "__missing__"],
    ["car rental", "flight", "hotel", "__missing__"],
    ["gold", "platinum", "standard", "__missing__"],
]

# Surface unexpected values rather than silently encoding them as all-zero.
unknown_levels = {
    column: sorted(
        set(abt[column].dropna().astype(str))
        - (set(levels) - {"__missing__"})
    )
    for column, levels in zip(CATEGORICAL_FEATURES, CATEGORY_LEVELS)
}
unknown_levels = {
    column: levels
    for column, levels in unknown_levels.items()
    if levels
}

if unknown_levels:
    raise ValueError(
        f"Unexpected categorical values need review: {unknown_levels}"
    )

categorical_input = (
    abt[CATEGORICAL_FEATURES]
    .astype("string")
    .fillna("__missing__")
)

categorical_encoder = OneHotEncoder(
    categories=CATEGORY_LEVELS,
    handle_unknown="ignore",
    sparse_output=False,
    dtype="int8",
)

categorical_values = categorical_encoder.fit_transform(categorical_input)
categorical_feature_names = (
    categorical_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
)

categorical_encoded = pd.DataFrame(
    categorical_values,
    columns=categorical_feature_names,
    index=abt.index,
)

numeric_features = (
    abt[NUMERIC_FEATURES]
    .apply(pd.to_numeric, errors="coerce")
    .astype("float64")
)

# Model predictors only; ticket IDs and relationship-status fields are excluded.
X_encoded = pd.concat(
    [numeric_features, categorical_encoded],
    axis=1,
)
y = pd.to_numeric(
    abt[TARGET_COLUMN],
    errors="coerce",
).astype("Int8")

if len(X_encoded) != len(abt) or abt["ticket_id"].nunique() != len(abt):
    raise AssertionError("Encoding must preserve one row per ticket.")

# Save IDs and target alongside the encoded columns for traceability.
encoded_abt = pd.concat(
    [abt[["ticket_id"]], X_encoded, y.rename(TARGET_COLUMN)],
    axis=1,
)

output_path = Path("abt") / "support_satisfaction_encoded.csv"
encoded_abt.to_csv(output_path, index=False)

print("Encoded feature matrix shape:", X_encoded.shape)
print("Encoded categorical columns:", len(categorical_feature_names))
print("Target counts:", y.value_counts(dropna=False).to_dict())
print(
    "Numeric missing values retained for later preprocessing:",
    X_encoded[NUMERIC_FEATURES]
    .isna()
    .sum()
    .loc[lambda counts: counts.gt(0)]
    .to_dict(),
)
print("Saved encoded data to:", output_path.resolve())
```

---

## 11. Separate X and y

X holds the 31 predictors, y holds the target. The ticket ID is dropped so the model cannot memorise it, and the 70 tickets without a CSAT score are excluded because there is nothing to learn from them.

| | Shape |
| --- | --- |
| X | 2,430 rows, 31 columns |
| y | 2,430 values (1,759 satisfied, 671 dissatisfied) |

### Full code

```python
from pathlib import Path
import pandas as pd

encoded_abt = pd.read_csv(
    Path("abt") / "support_satisfaction_encoded.csv"
)

TARGET_COLUMN = "dissatisfied"
ID_COLUMNS = ["ticket_id"]

# Exclude rows without a known target for supervised learning.
labeled_data = encoded_abt.loc[
    encoded_abt[TARGET_COLUMN].notna()
].copy()

# Separate predictors from the target; never train on the ticket ID.
X = labeled_data.drop(columns=[*ID_COLUMNS, TARGET_COLUMN])
y = labeled_data[TARGET_COLUMN].astype("int8")

if not X.index.equals(y.index):
    raise ValueError("X and y rows are not aligned.")

if "ticket_id" in X.columns or TARGET_COLUMN in X.columns:
    raise ValueError("X must exclude both ticket_id and the target.")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Target counts (0=satisfied, 1=dissatisfied):")
display(y.value_counts().sort_index())
print(
    "Unlabeled rows excluded:",
    int(encoded_abt[TARGET_COLUMN].isna().sum()),
)

#Drop unnecessary columns 
X = labeled_data.drop(columns=["ticket_id", "dissatisfied"])
y = labeled_data["dissatisfied"].astype("int8")
```

---

## 12. Train Test Split

An 80/20 split with a fixed random seed, stratified so train and test have the same dissatisfaction rate.

| | Rows | Satisfied | Dissatisfied |
| --- | ---: | ---: | ---: |
| Train | 1,944 | 1,407 | 537 |
| Test | 486 | 352 | 134 |

### Full code

```python
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

encoded_abt = pd.read_csv(
    Path("abt") / "support_satisfaction_encoded.csv"
)

TARGET_COLUMN = "dissatisfied"

if TARGET_COLUMN not in encoded_abt.columns or "ticket_id" not in encoded_abt.columns:
    raise ValueError("Encoded ABT must contain ticket_id and dissatisfied.")

if encoded_abt["ticket_id"].duplicated().any():
    raise ValueError("ticket_id must be unique before splitting.")

# Exclude unlabeled tickets for this supervised-learning split.
labeled_data = encoded_abt.loc[
    encoded_abt[TARGET_COLUMN].notna()
].copy()

# The ID and target are not predictor features.
X = labeled_data.drop(columns=["ticket_id", TARGET_COLUMN])
y = labeled_data[TARGET_COLUMN].astype("int8")

if y.nunique() != 2 or y.value_counts().min() < 2:
    raise ValueError(
        "A stratified split requires at least two rows in each target class."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

if not X_train.index.equals(y_train.index):
    raise AssertionError("Training features and target are not aligned.")
if not X_test.index.equals(y_test.index):
    raise AssertionError("Test features and target are not aligned.")

print("X_train:", X_train.shape, "| y_train:", y_train.shape)
print("X_test:", X_test.shape, "| y_test:", y_test.shape)
print("Training class counts:", y_train.value_counts().sort_index().to_dict())
print("Test class counts:", y_test.value_counts().sort_index().to_dict())
print("Unlabeled rows excluded:", int(encoded_abt[TARGET_COLUMN].isna().sum()))
```

---

## 13. Feature Scaling of Numerical Columns

Missing values in the three continuous columns (first response hours, month, day of week) are filled with the **median**, then standardised. Both steps learn their numbers from the **training rows only**, so nothing from the test set leaks in. The 0/1 columns are left as they are.

Check: after scaling, each continuous column has mean 0 and standard deviation 1 on the training data.

### Full code

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

CONTINUOUS_NUMERIC_FEATURES = [
    "first_response_hours",
    "ticket_month",
    "ticket_day_of_week",
]

BINARY_NUMERIC_FEATURES = [
    "first_response_missing",
    "ticket_is_weekend",
    "ticket_date_missing",
]

numeric_columns = CONTINUOUS_NUMERIC_FEATURES + BINARY_NUMERIC_FEATURES
missing_columns = set(numeric_columns) - set(X_train.columns)

if missing_columns:
    raise ValueError(
        f"Expected numerical columns are missing: {sorted(missing_columns)}"
    )

# Fit only on training data to prevent leakage from the test set.
numeric_imputer = SimpleImputer(strategy="median")
numeric_scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

train_values = numeric_imputer.fit_transform(
    X_train[CONTINUOUS_NUMERIC_FEATURES]
)
test_values = numeric_imputer.transform(
    X_test[CONTINUOUS_NUMERIC_FEATURES]
)

X_train_scaled.loc[:, CONTINUOUS_NUMERIC_FEATURES] = (
    numeric_scaler.fit_transform(train_values)
)
X_test_scaled.loc[:, CONTINUOUS_NUMERIC_FEATURES] = (
    numeric_scaler.transform(test_values)
)

# Retain binary indicator features as 0/1 values.
X_train_scaled.loc[:, BINARY_NUMERIC_FEATURES] = (
    X_train_scaled[BINARY_NUMERIC_FEATURES].fillna(0)
)
X_test_scaled.loc[:, BINARY_NUMERIC_FEATURES] = (
    X_test_scaled[BINARY_NUMERIC_FEATURES].fillna(0)
)

if X_train_scaled.isna().any().any() or X_test_scaled.isna().any().any():
    raise ValueError("Missing values remain after numeric preprocessing.")

print("Scaled training shape:", X_train_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)
print(
    "Training means for continuous features:",
    X_train_scaled[CONTINUOUS_NUMERIC_FEATURES].mean().round(4).to_dict(),
)
print(
    "Training standard deviations for continuous features:",
    X_train_scaled[CONTINUOUS_NUMERIC_FEATURES].std(ddof=0).round(4).to_dict(),
)
print(
    "Train/test target alignment:",
    X_train_scaled.index.equals(y_train.index),
    X_test_scaled.index.equals(y_test.index),
)
```

---

## 14. Train Model

Sections 10 to 13 show the recipe on one table. For the five objectives the same recipe runs **inside a scikit-learn Pipeline**: fill gaps (median for numbers, most frequent for categories), scale the numbers, one-hot encode the categories, then fit the model. Because the whole chain is fitted on training rows only, the test set never influences preprocessing, and the saved pipeline repeats the same steps on new data.

| # | Objective | Features | Models compared |
| --- | --- | --- | --- |
| 1 | Reduce booking cancellations | advance days, supplier rating, cancellation flexibility, service type, client tier, booking channel, policy compliance | Logistic Regression, Random Forest, Hist Gradient Boosting |
| 2 | Forecast trip cost to improve client budgeting | service type, travel class, destination region, international trip, advance days, trip length, peak season | Linear Regression, Random Forest Regressor, Hist Gradient Boosting Regressor |
| 3 | Improve travel policy compliance | travel class, traveller grade, preferred partner, advance days, booking channel, client has a travel policy | Logistic Regression, Random Forest, Hist Gradient Boosting |
| 4 | Raise customer satisfaction | resolution hours, first response hours, escalated, ticket category, contact channel, client tier | Logistic Regression, Random Forest, Hist Gradient Boosting |
| 5 | Detect booking amount anomalies and billing errors | how far an amount sits from similar bookings (same service, class and destination region) | Isolation Forest |

All classifiers use balanced class weights because cancellations, policy breaches and dissatisfied tickets are the minority. All splits are 80/20 with seed 42, stratified for classification.

### Objective 1: Reduce booking cancellations

**Target:** `cancelled`: booking status is Cancelled (17.1% of bookings). **Models:** Logistic Regression, Random Forest, Hist Gradient Boosting.

```python
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pathlib import Path

raw_dir = Path("travel_mgmt_raw")
clean_dir = Path("travel_mgmt_clean")
cancellation_bookings = pd.read_csv(clean_dir / "fact_bookings.csv")
cancellation_suppliers = pd.read_csv(
    raw_dir / "dim_suppliers.csv"
).drop_duplicates()
cancellation_clients = pd.read_csv(clean_dir / "dim_clients.csv")

for frame, key in (
    (cancellation_bookings, "booking_id"),
    (cancellation_suppliers, "supplier_id"),
    (cancellation_clients, "client_id"),
):
    frame[key] = frame[key].astype("string").str.strip().str.upper()
    if frame[key].duplicated().any():
        raise ValueError(f"Duplicate values found in {key}.")

booking_status = (
    cancellation_bookings["booking_status"]
    .astype("string")
    .str.strip()
    .str.casefold()
)
known_statuses = {"cancelled", "canceled", "completed", "no-show", "no show"}
if not booking_status.isin(known_statuses).all():
    raise ValueError("Unrecognized booking_status values found.")

cancellation_model_data = cancellation_bookings.copy()
cancellation_model_data["cancelled"] = booking_status.isin(
    {"cancelled", "canceled"}
).astype("int8")

cancellation_model_data = cancellation_model_data.merge(
    cancellation_suppliers[
        ["supplier_id", "service_rating", "cancellation_flexibility"]
    ],
    on="supplier_id",
    how="left",
    validate="many_to_one",
).merge(
    cancellation_clients[["client_id", "client_tier"]],
    on="client_id",
    how="left",
    validate="many_to_one",
)

booking_dates = pd.to_datetime(
    cancellation_model_data["booking_date"], format="mixed", errors="coerce"
)
travel_dates = pd.to_datetime(
    cancellation_model_data["travel_date"], format="mixed", errors="coerce"
)
cancellation_model_data["advance_days"] = (
    (travel_dates - booking_dates).dt.days
    .where((travel_dates - booking_dates).dt.days.ge(0))
    .astype("float64")
)
cancellation_model_data["advance_days_missing"] = (
    cancellation_model_data["advance_days"].isna().astype("int8")
)

numeric_features = [
    "advance_days",
    "advance_days_missing",
    "service_rating",
]
categorical_features = [
    "cancellation_flexibility",
    "service_type",
    "client_tier",
    "booking_channel",
    "policy_compliant",
]
cancellation_features = numeric_features + categorical_features

# Exclude booking_id from model inputs, but keep it for traceability.
X = cancellation_model_data[cancellation_features].copy()
y = pd.to_numeric(
    cancellation_model_data["cancelled"], errors="raise"
).astype("int8")
booking_ids = cancellation_model_data["booking_id"].copy()

# Convert pandas nullable numeric types and pd.NA to standard numeric NaN.
for column in numeric_features:
    X[column] = pd.to_numeric(X[column], errors="coerce").astype("float64")

# Convert nullable strings and pd.NA to ordinary objects and np.nan.
X[categorical_features] = (
    X[categorical_features]
    .astype(object)
    .where(X[categorical_features].notna(), np.nan)
)

X_train, X_test, y_train, y_test, booking_id_train, booking_id_test = (
    train_test_split(
        X,
        y,
        booking_ids,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )),
            ]),
            categorical_features,
        ),
    ]
)

candidate_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=42,
    ),
}

trained_models = {}

for model_name, classifier in candidate_models.items():
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    model_pipeline.fit(X_train, y_train)
    trained_models[model_name] = model_pipeline
    print(f"{model_name} trained.")

print("\nTraining rows:", len(X_train))
print("Test rows reserved for later:", len(X_test))
print("Training cancellation rate:", round(y_train.mean(), 3))
print("Test cancellation rate:", round(y_test.mean(), 3))
```

### Objective 2: Forecast trip cost to improve client budgeting

**Target:** `amount_usd`, modelled as its logarithm and converted back to dollars. **Models:** Linear Regression, Random Forest Regressor, Hist Gradient Boosting Regressor.

```python
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the cleaned bookings table.
clean_dir = Path("travel_mgmt_clean")
bookings = pd.read_csv(clean_dir / "fact_bookings.csv")


def normalize_city(value):
    if pd.isna(value):
        return pd.NA

    # Normalize the corrupted São Paulo spelling present in the source data.
    text = str(value).strip().casefold().replace("\ufffd", "a")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


def normalize_category(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )


# Map the city names and airport-code variants observed in these bookings.
city_locations = {
    "bom": ("India", "South Asia"),
    "bombay": ("India", "South Asia"),
    "mumbai": ("India", "South Asia"),
    "dxb": ("United Arab Emirates", "Middle East"),
    "dubai": ("United Arab Emirates", "Middle East"),
    "fra": ("Germany", "Europe"),
    "frankfurt": ("Germany", "Europe"),
    "gru": ("Brazil", "South America"),
    "sao paulo": ("Brazil", "South America"),
    "jnb": ("South Africa", "Africa"),
    "joburg": ("South Africa", "Africa"),
    "johannesburg": ("South Africa", "Africa"),
    "lagos": ("Nigeria", "Africa"),
    "los": ("Nigeria", "Africa"),
    "lon": ("United Kingdom", "Europe"),
    "london": ("United Kingdom", "Europe"),
    "nyc": ("United States", "North America"),
    "new york": ("United States", "North America"),
    "sin": ("Singapore", "Southeast Asia"),
    "singapore": ("Singapore", "Southeast Asia"),
    "syd": ("Australia", "Oceania"),
    "sydney": ("Australia", "Oceania"),
}

regression_data = bookings.copy()

# Derive destination region and international-trip indicator.
regression_data["origin_city_key"] = regression_data["origin_city"].map(
    normalize_city
)
regression_data["destination_city_key"] = regression_data[
    "destination_city"
].map(normalize_city)

unknown_origin = sorted(
    set(regression_data["origin_city_key"].dropna()) - set(city_locations)
)
unknown_destination = sorted(
    set(regression_data["destination_city_key"].dropna()) - set(city_locations)
)

if unknown_origin or unknown_destination:
    raise ValueError(
        "Add mappings for unknown cities before training. "
        f"Origin cities: {unknown_origin}; "
        f"destination cities: {unknown_destination}"
    )

regression_data["origin_country"] = regression_data["origin_city_key"].map(
    lambda city: city_locations[city][0]
)
regression_data["destination_country"] = regression_data[
    "destination_city_key"
].map(lambda city: city_locations[city][0])
regression_data["destination_region"] = regression_data[
    "destination_city_key"
].map(lambda city: city_locations[city][1])
regression_data["international"] = (
    regression_data["origin_country"]
    != regression_data["destination_country"]
).astype("int8")

# Parse currency values such as "$1,165.07" and "USD 1,787.82".
amount_text = regression_data["amount_usd"].astype("string").str.strip()
parenthesized_negative = amount_text.str.fullmatch(r"\(.*\)", na=False)
amount_text = (
    amount_text
    .str.replace(r"(?i)\bUSD\b", "", regex=True)
    .str.replace(r"[$,()]", "", regex=True)
    .str.replace(r"[^0-9.\-]", "", regex=True)
)
regression_data["amount_usd"] = pd.to_numeric(
    amount_text,
    errors="coerce",
)
regression_data.loc[parenthesized_negative, "amount_usd"] *= -1

# A log target requires positive, finite booking amounts.
valid_amount = (
    regression_data["amount_usd"].notna()
    & np.isfinite(regression_data["amount_usd"])
    & regression_data["amount_usd"].gt(0)
)
excluded_amount_rows = int((~valid_amount).sum())
regression_data = regression_data.loc[valid_amount].copy()
regression_data["log_amount_usd"] = np.log(regression_data["amount_usd"])

# Parse dates and correct travel/return dates when those two fields were swapped.
booking_date = pd.to_datetime(
    regression_data["booking_date"],
    format="mixed",
    errors="coerce",
)
travel_date = pd.to_datetime(
    regression_data["travel_date"],
    format="mixed",
    errors="coerce",
)
return_date = pd.to_datetime(
    regression_data["return_date"],
    format="mixed",
    errors="coerce",
)

swapped_dates = (
    travel_date.notna()
    & return_date.notna()
    & return_date.lt(travel_date)
)
corrected_travel_date = travel_date.where(~swapped_dates, return_date)
corrected_return_date = return_date.where(~swapped_dates, travel_date)

advance_days = (corrected_travel_date - booking_date).dt.days
trip_length_days = (
    corrected_return_date - corrected_travel_date
).dt.days

regression_data["advance_days"] = (
    advance_days.where(advance_days >= 0).astype("float64")
)
regression_data["trip_length_days"] = (
    trip_length_days.where(trip_length_days >= 0).astype("float64")
)

# Provisional peak-season definition: June, July, August, and December.
peak_season_months = {6, 7, 8, 12}
travel_month = corrected_travel_date.dt.month
regression_data["peak_season"] = travel_month.isin(
    peak_season_months
).astype("int8")
regression_data.loc[travel_month.isna(), "peak_season"] = pd.NA

regression_data["service_type"] = normalize_category(
    regression_data["service_type"]
)
regression_data["travel_class"] = normalize_category(
    regression_data["travel_class"]
)

# Define Objective 2 features and target.
regression_features = [
    "service_type",
    "travel_class",
    "destination_region",
    "international",
    "advance_days",
    "trip_length_days",
    "peak_season",
]
numeric_features = [
    "international",
    "advance_days",
    "trip_length_days",
    "peak_season",
]
categorical_features = [
    "service_type",
    "travel_class",
    "destination_region",
]

X_regression = regression_data[regression_features].copy()
y_regression = regression_data["log_amount_usd"].astype("float64")
regression_booking_ids = regression_data["booking_id"].copy()

# Ensure nullable pandas values are handled as numeric NaN/object NaN.
for column in numeric_features:
    X_regression[column] = pd.to_numeric(
        X_regression[column],
        errors="coerce",
    ).astype("float64")

for column in categorical_features:
    X_regression[column] = (
        X_regression[column]
        .astype(object)
        .where(X_regression[column].notna(), np.nan)
    )

# Split features, target, and IDs together so booking IDs stay aligned.
(
    X_regression_train,
    X_regression_test,
    y_regression_train,
    y_regression_test,
    regression_id_train,
    regression_id_test,
) = train_test_split(
    X_regression,
    y_regression,
    regression_booking_ids,
    test_size=0.20,
    random_state=42,
)

# Impute, scale, and encode within each model pipeline.
regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )),
            ]),
            categorical_features,
        ),
    ]
)

regression_candidates = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Hist Gradient Boosting Regressor": HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        random_state=42,
    ),
}

regression_models = {}

for model_name, estimator in regression_candidates.items():
    model_pipeline = Pipeline(steps=[
        ("preprocessor", regression_preprocessor),
        ("regressor", estimator),
    ])
    model_pipeline.fit(X_regression_train, y_regression_train)
    regression_models[model_name] = model_pipeline
    print(f"{model_name} trained.")

print("\nRegression modeling rows:", len(regression_data))
print("Rows excluded for missing/non-positive amount:", excluded_amount_rows)
print("Training rows:", len(X_regression_train))
print("Test rows reserved for later:", len(X_regression_test))
print("Peak-season months used:", sorted(peak_season_months))
```

### Objective 3: Improve travel policy compliance

**Target:** `out_of_policy`: policy_compliant is No (26.2% of bookings). **Models:** Logistic Regression, Random Forest, Hist Gradient Boosting.

```python
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

raw_dir = Path("travel_mgmt_raw")
clean_dir = Path("travel_mgmt_clean")

bookings = pd.read_csv(clean_dir / "fact_bookings.csv")
travelers = pd.read_csv(raw_dir / "dim_travelers.csv")
suppliers = pd.read_csv(raw_dir / "dim_suppliers.csv")
clients = pd.read_csv(clean_dir / "dim_clients.csv")


def normalize_text(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )


def normalize_key(series):
    return series.astype("string").str.strip().str.upper()


def require_unique_key(frame, key, table_name):
    frame.drop_duplicates(inplace=True)
    if frame[key].duplicated().any():
        raise ValueError(
            f"{table_name} has conflicting duplicate values for {key}; "
            "resolve them before training."
        )


# Normalize keys before joining tables.
for frame, column in (
    (bookings, "booking_id"),
    (bookings, "client_id"),
    (bookings, "traveler_id"),
    (bookings, "supplier_id"),
    (travelers, "traveler_id"),
    (suppliers, "supplier_id"),
    (clients, "client_id"),
):
    frame[column] = normalize_key(frame[column])

for frame, key, name in (
    (bookings, "booking_id", "Bookings"),
    (travelers, "traveler_id", "Travelers"),
    (suppliers, "supplier_id", "Suppliers"),
    (clients, "client_id", "Clients"),
):
    require_unique_key(frame, key, name)

# Create the target from booking-level policy compliance.
policy_status = normalize_text(bookings["policy_compliant"])
policy_status = policy_status.replace({
    "y": "yes",
    "true": "yes",
    "1": "yes",
    "n": "no",
    "false": "no",
    "0": "no",
})
known_policy = policy_status.isin(["yes", "no"])

policy_data = bookings.loc[known_policy].copy()
policy_data["out_of_policy"] = (
    policy_status.loc[known_policy].eq("no").astype("int8")
)

# Derive lead time from mixed date formats; invalid/negative values are missing.
booking_date = pd.to_datetime(
    policy_data["booking_date"], format="mixed", errors="coerce"
)
travel_date = pd.to_datetime(
    policy_data["travel_date"], format="mixed", errors="coerce"
)
policy_data["advance_days"] = (travel_date - booking_date).dt.days
policy_data.loc[policy_data["advance_days"] < 0, "advance_days"] = np.nan
policy_data["advance_days"] = pd.to_numeric(
    policy_data["advance_days"], errors="coerce"
).astype("float64")
policy_data["advance_days_missing"] = (
    policy_data["advance_days"].isna().astype("int8")
)

# Normalize the booking and dimension attributes used by this objective.
policy_data["travel_class"] = normalize_text(policy_data["travel_class"])
policy_data["booking_channel"] = normalize_text(policy_data["booking_channel"])

traveler_features = travelers[["traveler_id", "grade"]].copy()
traveler_features["traveler_grade"] = normalize_text(
    traveler_features.pop("grade")
)

supplier_features = suppliers[["supplier_id", "preferred_partner"]].copy()
supplier_features["preferred_partner"] = normalize_text(
    supplier_features["preferred_partner"]
).replace({
    "y": "yes", "true": "yes", "1": "yes",
    "n": "no", "false": "no", "0": "no",
})

client_features = clients[["client_id", "has_travel_policy"]].copy()
client_features["has_travel_policy"] = normalize_text(
    client_features["has_travel_policy"]
).replace({
    "y": "yes", "true": "yes", "1": "yes",
    "n": "no", "false": "no", "0": "no",
})

policy_data = policy_data.merge(
    traveler_features, on="traveler_id", how="left", validate="many_to_one"
)
policy_data = policy_data.merge(
    supplier_features, on="supplier_id", how="left", validate="many_to_one"
)
policy_data = policy_data.merge(
    client_features, on="client_id", how="left", validate="many_to_one"
)

policy_features = [
    "travel_class",
    "traveler_grade",
    "preferred_partner",
    "advance_days",
    "advance_days_missing",
    "booking_channel",
    "has_travel_policy",
]

X_policy = policy_data[policy_features].copy()
y_policy = policy_data["out_of_policy"].astype("int8")
policy_booking_ids = policy_data["booking_id"].copy()

numeric_policy_features = ["advance_days", "advance_days_missing"]
categorical_policy_features = [
    "travel_class",
    "traveler_grade",
    "preferred_partner",
    "booking_channel",
    "has_travel_policy",
]

for column in categorical_policy_features:
    X_policy[column] = (
        X_policy[column].astype(object)
        .where(X_policy[column].notna(), np.nan)
    )

X_policy_train, X_policy_test, y_policy_train, y_policy_test, policy_id_train, policy_id_test = (
    train_test_split(
        X_policy,
        y_policy,
        policy_booking_ids,
        test_size=0.20,
        random_state=42,
        stratify=y_policy,
    )
)

policy_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_policy_features,
        ),
        (
            "categorical",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )),
            ]),
            categorical_policy_features,
        ),
    ]
)

policy_candidates = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=42,
    ),
}

policy_models = {}

for model_name, classifier in policy_candidates.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", policy_preprocessor),
        ("classifier", classifier),
    ])
    pipeline.fit(X_policy_train, y_policy_train)
    policy_models[model_name] = pipeline
    print(f"Objective 3 — {model_name} trained.")

print("\nTraining rows:", len(X_policy_train))
print("Test rows reserved for later:", len(X_policy_test))
print("Training out-of-policy rate:", round(y_policy_train.mean(), 3))
print("Test out-of-policy rate:", round(y_policy_test.mean(), 3))
```

### Objective 4: Raise customer satisfaction

**Target:** `dissatisfied`: CSAT score of 1 or 2 (27.6% of labelled tickets). **Models:** Logistic Regression, Random Forest, Hist Gradient Boosting.

```python
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

clean_dir = Path("travel_mgmt_clean")

tickets = pd.read_csv(clean_dir / "fact_support_tickets.csv")
bookings = pd.read_csv(clean_dir / "fact_bookings.csv")
clients = pd.read_csv(clean_dir / "dim_clients.csv")


def normalize_text(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )


def normalize_key(series):
    return series.astype("string").str.strip().str.upper()


# Normalize keys and ensure lookup tables do not duplicate ticket rows.
tickets["ticket_id"] = normalize_key(tickets["ticket_id"])
tickets["booking_id"] = normalize_key(tickets["booking_id"])
tickets["client_id"] = normalize_key(tickets["client_id"])
bookings["booking_id"] = normalize_key(bookings["booking_id"])
bookings["client_id"] = normalize_key(bookings["client_id"])
clients["client_id"] = normalize_key(clients["client_id"])

if tickets["ticket_id"].duplicated().any():
    raise ValueError("Tickets contain duplicate ticket IDs.")
if bookings["booking_id"].duplicated().any():
    raise ValueError("Bookings contain duplicate booking IDs.")
if clients["client_id"].duplicated().any():
    raise ValueError("Clients contain duplicate client IDs.")

# Use the ticket client ID when available; otherwise fall back to its booking.
ticket_client_lookup = bookings[["booking_id", "client_id"]].rename(
    columns={"client_id": "booking_client_id"}
)
satisfaction_data = tickets.merge(
    ticket_client_lookup,
    on="booking_id",
    how="left",
    validate="many_to_one",
)
satisfaction_data["client_id"] = satisfaction_data["client_id"].fillna(
    satisfaction_data["booking_client_id"]
)

client_tiers = clients[["client_id", "client_tier"]].copy()
client_tiers["client_tier"] = normalize_text(client_tiers["client_tier"])
satisfaction_data = satisfaction_data.merge(
    client_tiers,
    on="client_id",
    how="left",
    validate="many_to_one",
)

# Keep only tickets with valid labeled CSAT values (1 through 5).
csat = pd.to_numeric(satisfaction_data["csat_score"], errors="coerce")
labeled = csat.between(1, 5)
satisfaction_data = satisfaction_data.loc[labeled].copy()
csat = csat.loc[labeled]
satisfaction_data["dissatisfied"] = csat.le(2).astype("int8")

# Prepare the objective's specified features.
satisfaction_features = [
    "resolution_hours",
    "first_response_hours",
    "escalated",
    "category",
    "contact_channel",
    "client_tier",
]
numeric_satisfaction_features = [
    "resolution_hours",
    "first_response_hours",
]
categorical_satisfaction_features = [
    "escalated",
    "category",
    "contact_channel",
    "client_tier",
]

for column in numeric_satisfaction_features:
    satisfaction_data[column] = pd.to_numeric(
        satisfaction_data[column], errors="coerce"
    ).astype("float64")

for column in categorical_satisfaction_features:
    satisfaction_data[column] = normalize_text(satisfaction_data[column])
    satisfaction_data[column] = (
        satisfaction_data[column].astype(object)
        .where(satisfaction_data[column].notna(), np.nan)
    )

X_satisfaction = satisfaction_data[satisfaction_features].copy()
y_satisfaction = satisfaction_data["dissatisfied"].astype("int8")
satisfaction_ticket_ids = satisfaction_data["ticket_id"].copy()

X_satisfaction_train, X_satisfaction_test, y_satisfaction_train, y_satisfaction_test, satisfaction_id_train, satisfaction_id_test = (
    train_test_split(
        X_satisfaction,
        y_satisfaction,
        satisfaction_ticket_ids,
        test_size=0.20,
        random_state=42,
        stratify=y_satisfaction,
    )
)

satisfaction_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_satisfaction_features,
        ),
        (
            "categorical",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )),
            ]),
            categorical_satisfaction_features,
        ),
    ]
)

satisfaction_candidates = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=42,
    ),
}

satisfaction_models = {}

for model_name, classifier in satisfaction_candidates.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", satisfaction_preprocessor),
        ("classifier", classifier),
    ])
    pipeline.fit(X_satisfaction_train, y_satisfaction_train)
    satisfaction_models[model_name] = pipeline
    print(f"Objective 4 — {model_name} trained.")

print("\nTraining rows:", len(X_satisfaction_train))
print("Test rows reserved for later:", len(X_satisfaction_test))
print("Training dissatisfied rate:", round(y_satisfaction_train.mean(), 3))
print("Test dissatisfied rate:", round(y_satisfaction_test.mean(), 3))
```

### Objective 5: Detect booking amount anomalies and billing errors

**Target:** none while training; the answer key is used only to score the result. **Models:** Isolation Forest.

```python
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the cleaned bookings. The answer key is intentionally not loaded here.
bookings = pd.read_csv(
    Path("travel_mgmt_clean") / "fact_bookings.csv"
)


def normalize_city(value):
    if pd.isna(value):
        return "missing"

    text = str(value).strip().casefold().replace("\ufffd", "a")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


def normalize_category(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
        .fillna("missing")
    )


# Standardize city aliases observed in this dataset.
city_regions = {
    "bom": "South Asia",
    "bombay": "South Asia",
    "mumbai": "South Asia",
    "dxb": "Middle East",
    "dubai": "Middle East",
    "fra": "Europe",
    "frankfurt": "Europe",
    "gru": "South America",
    "sao paulo": "South America",
    "jnb": "Africa",
    "joburg": "Africa",
    "johannesburg": "Africa",
    "lagos": "Africa",
    "los": "Africa",
    "lon": "Europe",
    "london": "Europe",
    "nyc": "North America",
    "new york": "North America",
    "sin": "Southeast Asia",
    "singapore": "Southeast Asia",
    "syd": "Oceania",
    "sydney": "Oceania",
}

anomaly_data = bookings[
    [
        "booking_id",
        "service_type",
        "travel_class",
        "destination_city",
        "amount_usd",
    ]
].copy()

anomaly_data["booking_id"] = (
    anomaly_data["booking_id"].astype("string").str.strip().str.upper()
)
anomaly_data["service_type"] = normalize_category(
    anomaly_data["service_type"]
)
anomaly_data["travel_class"] = normalize_category(
    anomaly_data["travel_class"]
)
anomaly_data["destination_region"] = (
    anomaly_data["destination_city"].map(normalize_city).map(city_regions)
)

if anomaly_data["destination_region"].isna().any():
    unknown = (
        anomaly_data.loc[
            anomaly_data["destination_region"].isna(),
            "destination_city",
        ]
        .dropna()
        .unique()
        .tolist()
    )
    raise ValueError(f"Add destination-region mappings for: {unknown}")

# Parse currency values while retaining zero and negative amounts as potential anomalies.
amount_text = anomaly_data["amount_usd"].astype("string").str.strip()
parenthesized_negative = amount_text.str.fullmatch(r"\(.*\)", na=False)
amount_text = (
    amount_text
    .str.replace(r"(?i)\bUSD\b", "", regex=True)
    .str.replace(r"[$,()]", "", regex=True)
    .str.replace(r"[^0-9.\-]", "", regex=True)
)
anomaly_data["amount_usd"] = pd.to_numeric(amount_text, errors="coerce")
anomaly_data.loc[parenthesized_negative, "amount_usd"] *= -1

group_columns = [
    "service_type",
    "travel_class",
    "destination_region",
]

# Hold out test bookings before calculating peer-group reference statistics.
anomaly_train_data, anomaly_test_data = train_test_split(
    anomaly_data,
    test_size=0.20,
    random_state=42,
)

# Learn comparable-booking amount references from training data only.
positive_train = anomaly_train_data.loc[
    anomaly_train_data["amount_usd"] > 0,
    group_columns + ["amount_usd"],
].copy()

if positive_train.empty:
    raise ValueError("No positive training booking amounts are available.")

positive_train["log_amount"] = np.log(positive_train["amount_usd"])
global_log_median = positive_train["log_amount"].median()
global_log_mad = (
    positive_train["log_amount"] - global_log_median
).abs().median()

# Avoid a zero or undefined scale in the robust deviation feature.
if pd.isna(global_log_mad) or global_log_mad <= 0:
    global_log_mad = 0.1

group_stats = positive_train.groupby(group_columns)["log_amount"].agg(
    group_median="median",
    group_count="size",
)
group_stats["group_mad"] = positive_train.groupby(group_columns)[
    "log_amount"
].apply(lambda values: (values - values.median()).abs().median())

group_stats_lookup = group_stats.to_dict("index")


def build_anomaly_features(frame):
    amount = pd.to_numeric(frame["amount_usd"], errors="coerce").astype(float)
    log_amount = np.log(amount.where(amount > 0))

    peer_medians = []
    peer_mads = []

    for key in frame[group_columns].itertuples(index=False, name=None):
        stats = group_stats_lookup.get(tuple(key))

        # Fall back to the global training reference for small/unseen groups.
        if stats is None or stats["group_count"] < 20:
            peer_medians.append(global_log_median)
            peer_mads.append(global_log_mad)
        else:
            peer_medians.append(stats["group_median"])
            group_mad = stats["group_mad"]
            peer_mads.append(
                global_log_mad
                if pd.isna(group_mad) or group_mad <= 0
                else group_mad
            )

    peer_median = np.asarray(peer_medians, dtype="float64")
    peer_mad = np.asarray(peer_mads, dtype="float64")
    log_deviation = log_amount.to_numpy() - peer_median
    robust_scale = np.maximum(1.4826 * peer_mad, 0.05)

    return pd.DataFrame(
        {
            "log_amount_abs": np.log1p(amount.abs()).to_numpy(),
            "log_amount_deviation": log_deviation,
            "robust_peer_deviation": np.abs(log_deviation) / robust_scale,
            "amount_nonpositive": (
                amount.notna() & amount.le(0)
            ).astype("int8").to_numpy(),
            "amount_missing": amount.isna().astype("int8").to_numpy(),
        },
        index=frame.index,
    )


X_anomaly_train = build_anomaly_features(anomaly_train_data)
X_anomaly_test = build_anomaly_features(anomaly_test_data)
anomaly_booking_id_test = anomaly_test_data["booking_id"].copy()

# Fit the model using only training features. The answer key is not used.
anomaly_model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("isolation_forest", IsolationForest(
        n_estimators=500,
        contamination=0.02,
        random_state=42,
    )),
])

anomaly_model.fit(X_anomaly_train)

print("Objective 5 Isolation Forest trained.")
print("Training bookings:", len(X_anomaly_train))
print("Test bookings reserved for prediction:", len(X_anomaly_test))
print("Training feature count:", X_anomaly_train.shape[1])
print("Answer key used:", False)
```

---

## 15. Predict X_test

Each model predicts the rows it never saw. Classifiers give a label and a probability. The cost model predicts a log amount that is converted back to dollars. The anomaly model gives a flag and a score, where higher means more unusual. The predictions are saved to CSV. Preview for the winning model of each objective:

### Objective 1: Reduce booking cancellations

| booking_id | actual_cancelled | predicted_cancelled | probability_cancelled | model_name |
| --- | --- | --- | --- | --- |
| BKG-301019 | 0 | 0 | 0.331 | Logistic Regression |
| BKG-300843 | 0 | 1 | 0.577 | Logistic Regression |
| BKG-305703 | 0 | 1 | 0.604 | Logistic Regression |
| BKG-304073 | 0 | 0 | 0.383 | Logistic Regression |
| BKG-300969 | 0 | 0 | 0.296 | Logistic Regression |

```python
import numpy as np
import pandas as pd

prediction_rows = []

for model_name, model in trained_models.items():
    predicted_labels = model.predict(X_test)
    predicted_probabilities = model.predict_proba(X_test)

    # Get the probability column corresponding to cancelled = 1.
    cancelled_column = np.where(model.classes_ == 1)[0][0]

    model_predictions = pd.DataFrame({
        "booking_id": booking_id_test.to_numpy(),
        "actual_cancelled": y_test.to_numpy(),
        "predicted_cancelled": predicted_labels,
        "probability_cancelled": predicted_probabilities[:, cancelled_column],
        "model_name": model_name,
    })

    prediction_rows.append(model_predictions)

all_cancellation_predictions = pd.concat(
    prediction_rows,
    ignore_index=True,
)

all_cancellation_predictions.to_csv(
    "booking_cancellation_test_predictions.csv",
    index=False,
)

print("Saved: booking_cancellation_test_predictions.csv")
print("Prediction rows:", len(all_cancellation_predictions))
display(all_cancellation_predictions.head(10))
```

### Objective 2: Forecast trip cost to improve client budgeting

| booking_id | actual_amount_usd | predicted_amount_usd | model_name |
| --- | --- | --- | --- |
| BKG-300312 | 731.6 | 903.71 | Hist Gradient Boosting Regressor |
| BKG-300304 | 2645.72 | 2580.48 | Hist Gradient Boosting Regressor |
| BKG-304410 | 3046.08 | 2797.79 | Hist Gradient Boosting Regressor |
| BKG-301593 | 3054.61 | 3168.67 | Hist Gradient Boosting Regressor |
| BKG-301114 | 5650.1 | 5055.82 | Hist Gradient Boosting Regressor |

```python
import numpy as np
import pandas as pd

regression_prediction_rows = []

# Convert the actual test targets from log dollars back to dollars.
actual_amount_usd = np.exp(y_regression_test.to_numpy())

for model_name, model in regression_models.items():
    # Predict log(amount_usd), then convert predictions back to dollars.
    predicted_log_amount = model.predict(X_regression_test)
    predicted_amount_usd = np.exp(predicted_log_amount)

    regression_prediction_rows.append(pd.DataFrame({
        "booking_id": regression_id_test.to_numpy(),
        "actual_amount_usd": actual_amount_usd,
        "predicted_amount_usd": predicted_amount_usd,
        "model_name": model_name,
    }))

all_regression_predictions = pd.concat(
    regression_prediction_rows,
    ignore_index=True,
)

all_regression_predictions.to_csv(
    "trip_cost_test_predictions.csv",
    index=False,
)

print("Saved: trip_cost_test_predictions.csv")
print("Prediction rows:", len(all_regression_predictions))
display(
    all_regression_predictions
    .groupby("model_name", group_keys=False)
    .head(5)
    .round(2)
)
```

### Objective 3: Improve travel policy compliance

| booking_id | actual_out_of_policy | predicted_out_of_policy | probability_out_of_policy | model_name |
| --- | --- | --- | --- | --- |
| BKG-304813 | 0 | 1 | 0.565 | Logistic Regression |
| BKG-301026 | 0 | 1 | 0.58 | Logistic Regression |
| BKG-303319 | 0 | 1 | 0.679 | Logistic Regression |
| BKG-301786 | 1 | 0 | 0.333 | Logistic Regression |
| BKG-302476 | 0 | 0 | 0.261 | Logistic Regression |

```python
import numpy as np
import pandas as pd

policy_prediction_rows = []

for model_name, model in policy_models.items():
    predicted_labels = model.predict(X_policy_test)
    predicted_probabilities = model.predict_proba(X_policy_test)

    out_of_policy_column = np.where(model.classes_ == 1)[0][0]

    policy_prediction_rows.append(pd.DataFrame({
        "booking_id": policy_id_test.to_numpy(),
        "actual_out_of_policy": y_policy_test.to_numpy(),
        "predicted_out_of_policy": predicted_labels,
        "probability_out_of_policy": predicted_probabilities[
            :, out_of_policy_column
        ],
        "model_name": model_name,
    }))

all_policy_predictions = pd.concat(
    policy_prediction_rows,
    ignore_index=True,
)

all_policy_predictions.to_csv(
    "policy_compliance_test_predictions.csv",
    index=False,
)

print("Saved: policy_compliance_test_predictions.csv")
print("Prediction rows:", len(all_policy_predictions))
display(all_policy_predictions.head(5))
```

### Objective 4: Raise customer satisfaction

| ticket_id | actual_dissatisfied | predicted_dissatisfied | probability_dissatisfied | model_name |
| --- | --- | --- | --- | --- |
| TKT-500700 | 0 | 0 | 0.016 | Logistic Regression |
| TKT-500809 | 0 | 0 | 0.273 | Logistic Regression |
| TKT-502182 | 1 | 1 | 0.957 | Logistic Regression |
| TKT-500016 | 1 | 1 | 0.999 | Logistic Regression |
| TKT-501477 | 0 | 0 | 0.063 | Logistic Regression |

```python
satisfaction_prediction_rows = []

for model_name, model in satisfaction_models.items():
    predicted_labels = model.predict(X_satisfaction_test)
    predicted_probabilities = model.predict_proba(X_satisfaction_test)

    dissatisfied_column = np.where(model.classes_ == 1)[0][0]

    satisfaction_prediction_rows.append(pd.DataFrame({
        "ticket_id": satisfaction_id_test.to_numpy(),
        "actual_dissatisfied": y_satisfaction_test.to_numpy(),
        "predicted_dissatisfied": predicted_labels,
        "probability_dissatisfied": predicted_probabilities[
            :, dissatisfied_column
        ],
        "model_name": model_name,
    }))

all_satisfaction_predictions = pd.concat(
    satisfaction_prediction_rows,
    ignore_index=True,
)

all_satisfaction_predictions.to_csv(
    "customer_satisfaction_test_predictions.csv",
    index=False,
)

print("Saved: customer_satisfaction_test_predictions.csv")
print("Prediction rows:", len(all_satisfaction_predictions))
display(all_satisfaction_predictions.head(5))
```

### Objective 5: Detect booking amount anomalies and billing errors

| booking_id | predicted_anomaly | anomaly_score |
| --- | --- | --- |
| BKG-303085 | 1 | 0.124 |
| BKG-300445 | 1 | 0.092 |
| BKG-305379 | 1 | 0.09 |
| BKG-300082 | 1 | 0.089 |
| BKG-301497 | 1 | 0.089 |

```python
import pandas as pd

# Isolation Forest labels: -1 = anomaly, 1 = normal.
isolation_labels = anomaly_model.predict(X_anomaly_test)

# Higher scores indicate observations more likely to be anomalous.
anomaly_scores = -anomaly_model.decision_function(X_anomaly_test)

anomaly_test_predictions = pd.DataFrame({
    "booking_id": anomaly_booking_id_test.to_numpy(),
    "predicted_anomaly": (isolation_labels == -1).astype("int8"),
    "anomaly_score": anomaly_scores,
})

anomaly_test_predictions.to_csv(
    "booking_anomaly_test_predictions.csv",
    index=False,
)

print("Saved: booking_anomaly_test_predictions.csv")
print("Test bookings predicted:", len(anomaly_test_predictions))
print(
    "Bookings flagged as anomalous:",
    int(anomaly_test_predictions["predicted_anomaly"].sum()),
)

display(
    anomaly_test_predictions
    .sort_values("anomaly_score", ascending=False)
    .head(5)
)
```

---

## 16. Compare X and y

Here the predictions are lined up against the truth, row by row. It is the quickest way to see where a model is right and wrong before trusting a score.

One lesson from this step: **raw accuracy is a poor judge when the outcome is rare.** In Objective 1 the Random Forest has the most correct predictions (823 of 1,200) simply because it flags fewer bookings, while Logistic Regression has the fewest (744) but catches the most cancellations. That is why the final choice uses average precision and recall, not accuracy.

### Objective 1: Reduce booking cancellations

| model_name | booking_id | actual_cancelled | predicted_cancelled | probability_cancelled | correct_prediction |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | BKG-301019 | 0 | 0 | 0.331 | True |
| Logistic Regression | BKG-300843 | 0 | 1 | 0.577 | False |
| Logistic Regression | BKG-305703 | 0 | 1 | 0.604 | False |
| Logistic Regression | BKG-304073 | 0 | 0 | 0.383 | True |
| Logistic Regression | BKG-300969 | 0 | 0 | 0.296 | True |

**Correct and incorrect predictions by model**

| model_name | test_bookings | correct_predictions | incorrect_predictions |
| --- | --- | --- | --- |
| Hist Gradient Boosting | 1200 | 774 | 426 |
| Logistic Regression | 1200 | 744 | 456 |
| Random Forest | 1200 | 823 | 377 |

```python
# Compare actual and predicted cancellation labels
cancellation_comparison = all_cancellation_predictions[
    [
        "model_name",
        "booking_id",
        "actual_cancelled",
        "predicted_cancelled",
        "probability_cancelled",
    ]
].copy()

cancellation_comparison["correct_prediction"] = (
    cancellation_comparison["actual_cancelled"]
    == cancellation_comparison["predicted_cancelled"]
)

print("First 10 comparisons for each model:")
display(
    cancellation_comparison
    .groupby("model_name", group_keys=False)
    .head(5)
)

print("\nCorrect and incorrect predictions by model:")
display(
    cancellation_comparison
    .groupby("model_name")
    .agg(
        test_bookings=("correct_prediction", "size"),
        correct_predictions=("correct_prediction", "sum"),
        incorrect_predictions=(
            "correct_prediction",
            lambda values: (~values).sum(),
        ),
    )
)
```

### Objective 2: Forecast trip cost to improve client budgeting

| booking_id | actual_amount_usd | predicted_amount_usd | model_name |
| --- | --- | --- | --- |
| BKG-300312 | 731.6 | 903.71 | Hist Gradient Boosting Regressor |
| BKG-300304 | 2645.72 | 2580.48 | Hist Gradient Boosting Regressor |
| BKG-304410 | 3046.08 | 2797.79 | Hist Gradient Boosting Regressor |
| BKG-301593 | 3054.61 | 3168.67 | Hist Gradient Boosting Regressor |
| BKG-301114 | 5650.1 | 5055.82 | Hist Gradient Boosting Regressor |

```python
import numpy as np
import pandas as pd

regression_prediction_rows = []

# Convert the log-transformed actual test target back to dollars.
actual_amount_usd = np.exp(y_regression_test.to_numpy())

for model_name, model in regression_models.items():
    predicted_log_amount = model.predict(X_regression_test)
    predicted_amount_usd = np.exp(predicted_log_amount)

    regression_prediction_rows.append(pd.DataFrame({
        "booking_id": regression_id_test.to_numpy(),
        "actual_amount_usd": actual_amount_usd,
        "predicted_amount_usd": predicted_amount_usd,
        "actual_log_amount": y_regression_test.to_numpy(),
        "predicted_log_amount": predicted_log_amount,
        "model_name": model_name,
    }))

all_regression_predictions = pd.concat(
    regression_prediction_rows,
    ignore_index=True,
)

print("Actual versus predicted trip costs (first 10 rows per model):")
display(
    all_regression_predictions
    .groupby("model_name", group_keys=False)
    .head(5)
    .round(2)
)
```

### Objective 3: Improve travel policy compliance

| model_name | booking_id | actual_out_of_policy | predicted_out_of_policy | probability_out_of_policy | correct_prediction |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | BKG-304813 | 0 | 1 | 0.565 | False |
| Logistic Regression | BKG-301026 | 0 | 1 | 0.58 | False |
| Logistic Regression | BKG-303319 | 0 | 1 | 0.679 | False |
| Logistic Regression | BKG-301786 | 1 | 0 | 0.333 | False |
| Logistic Regression | BKG-302476 | 0 | 0 | 0.261 | True |

**Correct and incorrect predictions by model**

| model_name | test_bookings | correct_predictions | incorrect_predictions |
| --- | --- | --- | --- |
| Hist Gradient Boosting | 1200 | 804 | 396 |
| Logistic Regression | 1200 | 788 | 412 |
| Random Forest | 1200 | 779 | 421 |

```python
policy_comparison = all_policy_predictions[
    [
        "model_name",
        "booking_id",
        "actual_out_of_policy",
        "predicted_out_of_policy",
        "probability_out_of_policy",
    ]
].copy()

policy_comparison["correct_prediction"] = (
    policy_comparison["actual_out_of_policy"]
    == policy_comparison["predicted_out_of_policy"]
)

print("Actual vs. predicted (first 10 rows per model):")
display(
    policy_comparison
    .groupby("model_name", group_keys=False)
    .head(5)
)

print("Correct and incorrect predictions by model:")
display(
    policy_comparison
    .groupby("model_name")
    .agg(
        test_bookings=("correct_prediction", "size"),
        correct_predictions=("correct_prediction", "sum"),
        incorrect_predictions=(
            "correct_prediction",
            lambda values: (~values).sum(),
        ),
    )
)
```

### Objective 4: Raise customer satisfaction

| model_name | ticket_id | actual_dissatisfied | predicted_dissatisfied | probability_dissatisfied | correct_prediction |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | TKT-500700 | 0 | 0 | 0.016 | True |
| Logistic Regression | TKT-500809 | 0 | 0 | 0.273 | True |
| Logistic Regression | TKT-502182 | 1 | 1 | 0.957 | True |
| Logistic Regression | TKT-500016 | 1 | 1 | 0.999 | True |
| Logistic Regression | TKT-501477 | 0 | 0 | 0.063 | True |

**Correct and incorrect predictions by model**

| model_name | test_tickets | correct_predictions | incorrect_predictions |
| --- | --- | --- | --- |
| Hist Gradient Boosting | 486 | 412 | 74 |
| Logistic Regression | 486 | 424 | 62 |
| Random Forest | 486 | 419 | 67 |

```python
satisfaction_comparison = all_satisfaction_predictions[
    [
        "model_name",
        "ticket_id",
        "actual_dissatisfied",
        "predicted_dissatisfied",
        "probability_dissatisfied",
    ]
].copy()

satisfaction_comparison["correct_prediction"] = (
    satisfaction_comparison["actual_dissatisfied"]
    == satisfaction_comparison["predicted_dissatisfied"]
)

print("Actual vs. predicted (first 10 rows per model):")
display(
    satisfaction_comparison
    .groupby("model_name", group_keys=False)
    .head(5)
)

print("Correct and incorrect predictions by model:")
display(
    satisfaction_comparison
    .groupby("model_name")
    .agg(
        test_tickets=("correct_prediction", "size"),
        correct_predictions=("correct_prediction", "sum"),
        incorrect_predictions=(
            "correct_prediction",
            lambda values: (~values).sum(),
        ),
    )
)
```

### Objective 5: Detect booking amount anomalies and billing errors

**Bookings flagged as anomalous (top 8 by score)**

| booking_id | service_type | travel_class | destination_region | amount_usd | anomaly_score |
| --- | --- | --- | --- | --- | --- |
| BKG-303085 | flight | business | Africa | 21169.4 | 0.124 |
| BKG-300445 | flight | business | Africa | 13364.3 | 0.092 |
| BKG-305379 | car rental | economy | North America | 3668.85 | 0.09 |
| BKG-300082 | flight | economy | Southeast Asia | 5635.63 | 0.089 |
| BKG-301497 | flight | economy | Europe | 6435.7 | 0.089 |
| BKG-300711 | hotel | suite | South America | 20644.6 | 0.081 |
| BKG-301609 | hotel | suite | South America | 20449.9 | 0.079 |
| BKG-302203 | car rental | suv | Southeast Asia | 73.47 | 0.073 |

```python
# Show each test booking's input features alongside its anomaly prediction.
anomaly_comparison = anomaly_test_data[
    [
        "booking_id",
        "service_type",
        "travel_class",
        "destination_region",
        "amount_usd",
    ]
].copy()

# Add the numeric features supplied to Isolation Forest.
anomaly_comparison = anomaly_comparison.join(X_anomaly_test)

# Add the model's flag and anomaly score for each booking.
anomaly_comparison = anomaly_comparison.merge(
    anomaly_test_predictions[
        ["booking_id", "predicted_anomaly", "anomaly_score"]
    ],
    on="booking_id",
    how="left",
    validate="one_to_one",
)

print("Test features alongside predictions:")
display(
    anomaly_comparison
    .sort_values("anomaly_score", ascending=False)
    .head(5)
)

print("Bookings flagged as anomalous:")
display(
    anomaly_comparison
    .loc[anomaly_comparison["predicted_anomaly"] == 1]
    .sort_values("anomaly_score", ascending=False)
)
```

---

## 17. Evaluate Model

Metrics are chosen to fit each problem. For the three classifiers I ranked models by **average precision**, which suits rare outcomes, and checked balanced accuracy, precision, recall, F1 and ROC AUC. For cost I used **mean absolute error in dollars**. For anomalies I scored the flags against the answer key at the very end.

### Objective 1: Reduce booking cancellations

| model | balanced_accuracy | precision_cancelled | recall_cancelled | f1_cancelled | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.608 | 0.245 | 0.59 | 0.347 | 0.633 | 0.253 |
| Random Forest | 0.553 | 0.228 | 0.351 | 0.276 | 0.584 | 0.232 |
| Hist Gradient Boosting | 0.554 | 0.217 | 0.415 | 0.285 | 0.583 | 0.221 |

Logistic Regression wins on average precision (0.253) and catches 121 of 205 cancellations (59%). Precision is low at 24.5%, so most flagged bookings will not cancel. This is a weak signal and the app treats it that way.

```python
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

evaluation_rows = []

for model_name, model_data in all_cancellation_predictions.groupby("model_name"):
    y_true = model_data["actual_cancelled"]
    y_pred = model_data["predicted_cancelled"]
    y_prob = model_data["probability_cancelled"]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    evaluation_rows.append({
        "model": model_name,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_cancelled": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall_cancelled": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "f1_cancelled": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    })

cancellation_evaluation = (
    pd.DataFrame(evaluation_rows)
    .sort_values("average_precision", ascending=False)
    .reset_index(drop=True)
)

display(cancellation_evaluation.round(3))
```

### Objective 2: Forecast trip cost to improve client budgeting

| model | MAE_USD | RMSE_USD | R2 | WAPE |
| --- | --- | --- | --- | --- |
| Hist Gradient Boosting Regressor | 231.85 | 1012.34 | 0.506 | 0.184 |
| Random Forest Regressor | 251.397 | 1029.19 | 0.49 | 0.2 |
| Linear Regression | 373.648 | 1075.67 | 0.443 | 0.297 |

Hist Gradient Boosting is the most accurate: typical error $232, or 18.4% of spend, and R-squared 0.51. Half of all predictions are within about $78. The RMSE of $1,012 shows a tail of large bookings that are hard to predict.

```python
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

regression_evaluation_rows = []

for model_name, model_data in all_regression_predictions.groupby("model_name"):
    y_true_usd = model_data["actual_amount_usd"]
    y_pred_usd = model_data["predicted_amount_usd"]

    regression_evaluation_rows.append({
        "model": model_name,
        "MAE_USD": mean_absolute_error(y_true_usd, y_pred_usd),
        "RMSE_USD": np.sqrt(mean_squared_error(y_true_usd, y_pred_usd)),
        "R2": r2_score(y_true_usd, y_pred_usd),
        "WAPE": (
            np.abs(y_true_usd - y_pred_usd).sum()
            / y_true_usd.sum()
        ),
    })

regression_evaluation = (
    pd.DataFrame(regression_evaluation_rows)
    .sort_values("MAE_USD")
    .reset_index(drop=True)
)

display(regression_evaluation.round(3))
```

### Objective 3: Improve travel policy compliance

| model | balanced_accuracy | precision_out_of_policy | recall_out_of_policy | f1_out_of_policy | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.646 | 0.401 | 0.622 | 0.488 | 0.68 | 0.452 |
| Hist Gradient Boosting | 0.628 | 0.404 | 0.54 | 0.462 | 0.677 | 0.441 |
| Random Forest | 0.622 | 0.385 | 0.565 | 0.458 | 0.673 | 0.425 |

Logistic Regression wins with average precision 0.452. It catches 196 of 315 out-of-policy bookings (62%) with 40% precision. Useful for choosing what to review first, not for deciding compliance.

```python
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

policy_evaluation_rows = []

for model_name, model_data in all_policy_predictions.groupby("model_name"):
    y_true = model_data["actual_out_of_policy"]
    y_pred = model_data["predicted_out_of_policy"]
    y_prob = model_data["probability_out_of_policy"]

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    policy_evaluation_rows.append({
        "model": model_name,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_out_of_policy": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall_out_of_policy": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "f1_out_of_policy": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "roc_auc": (
            roc_auc_score(y_true, y_prob)
            if y_true.nunique() == 2 else np.nan
        ),
        "average_precision": (
            average_precision_score(y_true, y_prob)
            if y_true.nunique() == 2 else np.nan
        ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    })

policy_evaluation = (
    pd.DataFrame(policy_evaluation_rows)
    .sort_values("average_precision", ascending=False)
    .reset_index(drop=True)
)

display(policy_evaluation.round(3))
```

### Objective 4: Raise customer satisfaction

| model | balanced_accuracy | precision_dissatisfied | recall_dissatisfied | f1_dissatisfied | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.866 | 0.731 | 0.851 | 0.786 | 0.945 | 0.893 |
| Random Forest | 0.849 | 0.719 | 0.821 | 0.767 | 0.929 | 0.873 |
| Hist Gradient Boosting | 0.828 | 0.7 | 0.784 | 0.739 | 0.926 | 0.861 |

The strongest result: 114 of 134 dissatisfied tickets caught (85%), 73% precision, ROC AUC 0.945 and 424 of 486 tickets classified correctly. Remember that resolution time and escalation help here, so it works best on closing tickets.

```python
satisfaction_evaluation_rows = []

for model_name, model_data in all_satisfaction_predictions.groupby("model_name"):
    y_true = model_data["actual_dissatisfied"]
    y_pred = model_data["predicted_dissatisfied"]
    y_prob = model_data["probability_dissatisfied"]

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    satisfaction_evaluation_rows.append({
        "model": model_name,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_dissatisfied": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall_dissatisfied": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "f1_dissatisfied": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "roc_auc": (
            roc_auc_score(y_true, y_prob)
            if y_true.nunique() == 2 else np.nan
        ),
        "average_precision": (
            average_precision_score(y_true, y_prob)
            if y_true.nunique() == 2 else np.nan
        ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    })

satisfaction_evaluation = (
    pd.DataFrame(satisfaction_evaluation_rows)
    .sort_values("average_precision", ascending=False)
    .reset_index(drop=True)
)

display(satisfaction_evaluation.round(3))
```

### Objective 5: Detect booking amount anomalies and billing errors

| Metric | Value |
| --- | ---: |
| Test bookings | 1,200 |
| Actual anomalies | 18 |
| Flagged | 23 |
| Precision | 0.565 |
| Recall | 0.722 |
| F1 | 0.634 |
| Balanced accuracy | 0.857 |
| ROC AUC | 0.947 |
| Average precision | 0.678 |

13 of 18 planted anomalies found, with 10 false alarms among 1,182 normal bookings. Only 18 anomalies were in the test set, so treat the numbers as a guide, not a guarantee.

```python
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

answer_key = pd.read_csv(
    Path("travel_mgmt_raw") / "anomaly_answer_key.csv"
)

# Normalize IDs and labels before matching the held-out predictions.
predictions = anomaly_test_predictions.copy()
predictions["booking_id"] = (
    predictions["booking_id"].astype("string").str.strip().str.upper()
)

answer_key["booking_id"] = (
    answer_key["booking_id"].astype("string").str.strip().str.upper()
)
answer_key["is_anomaly"] = pd.to_numeric(
    answer_key["is_anomaly"], errors="raise"
).astype("int8")

if predictions["booking_id"].duplicated().any():
    raise ValueError("Test predictions contain duplicate booking IDs.")
if answer_key["booking_id"].duplicated().any():
    raise ValueError("The anomaly answer key contains duplicate booking IDs.")
if not answer_key["is_anomaly"].isin([0, 1]).all():
    raise ValueError("Answer-key labels must be 0 (normal) or 1 (anomaly).")

# Match answer-key labels only to the held-out test bookings.
evaluation_data = predictions.merge(
    answer_key[["booking_id", "is_anomaly"]],
    on="booking_id",
    how="left",
    validate="one_to_one",
)

if len(evaluation_data) != len(predictions):
    raise ValueError("Matching with the answer key changed the test row count.")
if evaluation_data["is_anomaly"].isna().any():
    raise ValueError("Some test bookings are missing answer-key labels.")

y_true = evaluation_data["is_anomaly"]
y_pred = evaluation_data["predicted_anomaly"]
y_score = evaluation_data["anomaly_score"]

if y_true.nunique() != 2:
    raise ValueError(
        "ROC AUC and average precision require both normal and anomalous "
        "bookings in the test set."
    )

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
).ravel()

print("Objective 5 — held-out anomaly detection evaluation")
print("Test bookings:", len(evaluation_data))
print("Actual anomalies:", int(y_true.sum()))
print("Bookings flagged:", int(y_pred.sum()))
print(f"Anomaly prevalence: {y_true.mean():.3%}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.3f}")
print(f"F1 score: {f1_score(y_true, y_pred, zero_division=0):.3f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")
print(f"ROC AUC: {roc_auc_score(y_true, y_score):.3f}")
print(f"Average precision: {average_precision_score(y_true, y_score):.3f}")

print("\nConfusion matrix (rows = actual, columns = predicted; labels [0, 1]):")
print([[tn, fp], [fn, tp]])
```

**Models chosen**

| # | Objective | Chosen model | Why |
| --- | --- | --- | --- |
| 1 | Cancellations | Logistic Regression | Highest average precision (0.253) |
| 2 | Trip cost | Hist Gradient Boosting | Lowest error (MAE $231.85) |
| 3 | Policy compliance | Logistic Regression | Highest average precision (0.452) |
| 4 | Satisfaction | Logistic Regression | Highest average precision (0.893) |
| 5 | Anomalies | Isolation Forest | Only detector tested; scored against the answer key |

These are the best of the candidates tested, not a claim that nothing better exists. Each result comes from one held-out sample.

---

## 18. Business Analysis

Before reading model scores, here is where each problem sits in the business. These charts describe the data; they are associations, not proof of cause.

### Objective 1: Reduce booking cancellations

<img width="1677" height="643" alt="travel_business_01_cancellations" src="https://github.com/user-attachments/assets/465ec2f2-94d7-45c9-b449-03693325f8f1" />

**17.6%** of completed or cancelled bookings cancel. Travel-agent bookings cancel most (21.3%) and mobile-app bookings least (14.0%). Car rentals (20.2%) and flights (19.4%) cancel more than hotels (14.3%).

### Objective 2: Forecast trip cost

<img width="1677" height="643" alt="travel_business_02_trip_cost" src="https://github.com/user-attachments/assets/55fc3cd0-1989-416e-91d1-e458c0d3ebb5" />

Flights carry **$5.05M** of the spend and have a median booking of $1,107. Hotels follow with $2.60M and car rentals are small at $0.20M. Cost accuracy matters most on flights.

### Objective 3: Improve policy compliance

<img width="1677" height="643" alt="travel_business_03_policy_compliance" src="https://github.com/user-attachments/assets/21c1d761-bfb1-44f6-a283-435f5948dd2e" />

**26.2%** of bookings are out of policy. Phone (34.3%) and travel-agent (31.4%) bookings break policy far more than the online tool (20.6%). Premium cabins and rooms stand out: first class 43.7%, suites 41.0%, business 39.0%, against 18.8% for standard rooms.

### Objective 4: Raise customer satisfaction

<img width="1807" height="643" alt="travel_business_04_customer_satisfaction" src="https://github.com/user-attachments/assets/482211ce-4946-495c-a97d-b517dfd5fe85" />

Complaints (56.2%) and billing disputes (29.6%) sit above the 27.6% average. Chat is the only channel well below it (9.4%).

### Objective 5: Detect booking amount anomalies

<img width="1676" height="643" alt="travel_business_05_anomalies" src="https://github.com/user-attachments/assets/13a9e686-058b-4cdb-953a-f3f07b81d018" />

138 of 6,000 bookings (2.3%) are planted anomalies, spread evenly across services. Their median amount is **$4,688** against $929 for normal bookings, so an amount far from its peers is the signal to look for.

### Full code

```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_DIR = Path("travel_mgmt_raw")
CLEAN_DIR = Path("travel_mgmt_clean")


def normalize_category(series):
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )


def plot_rate_bars(axis, table, title, ylabel, color):
    bars = axis.bar(table.index.astype(str), table["rate"] * 100, color=color)
    axis.set_title(title, fontsize=12, weight="bold")
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, max(10, float(table["rate"].max() * 120)))
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    for bar, rate, count in zip(bars, table["rate"], table["rows"]):
        axis.annotate(
            f"{rate:.1%}\n(n={count:,})",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def rate_by_group(frame, group_column, target_column, minimum_rows=100):
    return (
        frame.groupby(group_column, dropna=False)[target_column]
        .agg(rows="size", rate="mean")
        .query("rows >= @minimum_rows")
        .sort_values("rate", ascending=False)
    )


bookings = pd.read_csv(CLEAN_DIR / "fact_bookings.csv")
bookings["service"] = normalize_category(bookings["service_type"])
bookings["channel"] = normalize_category(bookings["booking_channel"]).replace(
    {"onlinetool": "online tool"}
)
bookings["travel_class_group"] = normalize_category(bookings["travel_class"])
bookings["status"] = normalize_category(bookings["booking_status"]).replace(
    {"canceled": "cancelled", "no show": "no-show"}
)
bookings["amount_usd"] = pd.to_numeric(
    bookings["amount_usd"]
    .astype("string")
    .str.replace(r"[^0-9.\-]", "", regex=True),
    errors="coerce",
)

# 1. Cancellation: compare completed/cancelled bookings; exclude no-shows.
cancellation_data = bookings.loc[
    bookings["status"].isin(["completed", "cancelled"])
].copy()
cancellation_data["cancelled"] = cancellation_data["status"].eq("cancelled").astype(int)
cancellation_by_channel = rate_by_group(cancellation_data, "channel", "cancelled")
cancellation_by_service = rate_by_group(cancellation_data, "service", "cancelled")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_rate_bars(
    axes[0], cancellation_by_channel, "Cancellation rate by booking channel",
    "Cancelled bookings (%)", "#4472C4",
)
plot_rate_bars(
    axes[1], cancellation_by_service, "Cancellation rate by service",
    "Cancelled bookings (%)", "#5B9BD5",
)
fig.suptitle(
    f"Objective 1 — Reduce cancellations ({cancellation_data['cancelled'].mean():.1%} overall)",
    fontsize=15, weight="bold",
)
fig.tight_layout()
plt.show()

# 2. Cost forecasting: show where positive booking spend is concentrated.
positive_spend = bookings.loc[bookings["amount_usd"].gt(0)]
spend_by_service = (
    positive_spend.groupby("service")["amount_usd"]
    .agg(total_spend="sum", median_booking="median", bookings="size")
    .sort_values("total_spend", ascending=False)
)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
spend_millions = spend_by_service["total_spend"] / 1_000_000
bars = axes[0].bar(spend_by_service.index, spend_millions, color="#70AD47")
axes[0].set_title("Positive booking spend by service", weight="bold")
axes[0].set_ylabel("Total spend (USD millions)")
axes[0].tick_params(axis="x", rotation=20)
axes[0].grid(axis="y", alpha=0.25)
for bar, value in zip(bars, spend_millions):
    axes[0].annotate(
        f"${value:.2f}M", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9,
    )
bars = axes[1].bar(
    spend_by_service.index, spend_by_service["median_booking"], color="#A5A5A5"
)
axes[1].set_title("Median booking amount by service", weight="bold")
axes[1].set_ylabel("Median amount (USD)")
axes[1].tick_params(axis="x", rotation=20)
axes[1].grid(axis="y", alpha=0.25)
for bar, value in zip(bars, spend_by_service["median_booking"]):
    axes[1].annotate(
        f"${value:,.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9,
    )
fig.suptitle(
    f"Objective 2 — Forecast trip cost ({len(positive_spend):,} positive amounts)",
    fontsize=15, weight="bold",
)
fig.tight_layout()
plt.show()

# 3. Policy compliance: compare noncompliance by channel and travel class.
bookings["policy"] = normalize_category(bookings["policy_compliant"]).replace(
    {"y": "yes", "true": "yes", "1": "yes", "n": "no", "false": "no", "0": "no"}
)
policy_data = bookings.loc[bookings["policy"].isin(["yes", "no"])].copy()
policy_data["out_of_policy"] = policy_data["policy"].eq("no").astype(int)
policy_by_channel = rate_by_group(policy_data, "channel", "out_of_policy")
policy_by_class = rate_by_group(
    policy_data, "travel_class_group", "out_of_policy"
)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_rate_bars(
    axes[0], policy_by_channel, "Out-of-policy rate by booking channel",
    "Out-of-policy bookings (%)", "#ED7D31",
)
plot_rate_bars(
    axes[1], policy_by_class, "Out-of-policy rate by travel class",
    "Out-of-policy bookings (%)", "#FFC000",
)
fig.suptitle(
    f"Objective 3 — Improve policy compliance ({policy_data['out_of_policy'].mean():.1%} overall)",
    fontsize=15, weight="bold",
)
fig.tight_layout()
plt.show()

# 4. Satisfaction: compare dissatisfaction by ticket category and channel.
tickets = pd.read_csv(CLEAN_DIR / "fact_support_tickets.csv")
tickets["csat_score"] = pd.to_numeric(tickets["csat_score"], errors="coerce")
tickets = tickets.loc[tickets["csat_score"].between(1, 5)].copy()
tickets["dissatisfied"] = tickets["csat_score"].le(2).astype(int)
tickets["category_group"] = normalize_category(tickets["category"])
tickets["contact_channel_group"] = normalize_category(tickets["contact_channel"])
satisfaction_by_category = rate_by_group(
    tickets, "category_group", "dissatisfied", minimum_rows=20
)
satisfaction_by_channel = rate_by_group(
    tickets, "contact_channel_group", "dissatisfied", minimum_rows=20
)
overall_dissatisfaction = tickets["dissatisfied"].mean()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_rate_bars(
    axes[0], satisfaction_by_category, "Dissatisfaction by ticket category",
    "Dissatisfied tickets (%)", "#C00000",
)
plot_rate_bars(
    axes[1], satisfaction_by_channel, "Dissatisfaction by contact channel",
    "Dissatisfied tickets (%)", "#ED7D31",
)
for axis in axes:
    axis.axhline(
        overall_dissatisfaction * 100, color="#404040", linestyle="--",
        linewidth=1.5, label=f"Overall {overall_dissatisfaction:.1%}",
    )
    axis.legend()
fig.suptitle(
    f"Objective 4 — Raise customer satisfaction ({len(tickets):,} labeled tickets)",
    fontsize=15, weight="bold",
)
fig.tight_layout()
plt.show()

# 5. Amount anomalies: use answer-key labels only for retrospective analysis.
answer_key = pd.read_csv(RAW_DIR / "anomaly_answer_key.csv")
answer_key["booking_id"] = (
    answer_key["booking_id"].astype("string").str.strip().str.upper()
)
bookings["booking_id"] = (
    bookings["booking_id"].astype("string").str.strip().str.upper()
)
answer_key["is_anomaly"] = pd.to_numeric(
    answer_key["is_anomaly"], errors="raise"
).astype(int)
if answer_key["booking_id"].duplicated().any():
    raise ValueError("Anomaly answer key contains duplicate booking IDs.")
anomaly_data = bookings.merge(
    answer_key[["booking_id", "is_anomaly"]],
    on="booking_id", how="left", validate="one_to_one",
)
if anomaly_data["is_anomaly"].isna().any():
    raise ValueError("Some bookings are missing anomaly answer-key labels.")
anomaly_by_service = (
    anomaly_data.groupby("service")["is_anomaly"]
    .agg(rows="size", rate="mean")
    .sort_values("rate", ascending=False)
)
amount_by_label = anomaly_data.groupby("is_anomaly")["amount_usd"].median()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plot_rate_bars(
    axes[0], anomaly_by_service, "Answer-key anomaly rate by service",
    "Anomalous bookings (%)", "#7030A0",
)
labels = ["Normal", "Anomaly"]
values = [amount_by_label.get(0, np.nan), amount_by_label.get(1, np.nan)]
bars = axes[1].bar(labels, values, color=["#A5A5A5", "#7030A0"])
axes[1].set_title("Median amount by answer-key label", weight="bold")
axes[1].set_ylabel("Median booking amount (USD)")
axes[1].grid(axis="y", alpha=0.25)
for bar, value in zip(bars, values):
    axes[1].annotate(
        f"${value:,.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9,
    )
fig.suptitle(
    f"Objective 5 — Detect booking amount anomalies ({int(anomaly_data['is_anomaly'].sum()):,} labeled anomalies)",
    fontsize=15, weight="bold",
)
fig.tight_layout()
plt.show()

print(
    "Interpretation: segment differences are descriptive associations, not proof "
    "of causation. Anomaly flags should be reviewed, not treated as confirmed errors."
)
```

---

## 19. Save Model, Scaler and Features

Each objective is saved in its own folder so the app can load it without retraining:

| File | What it holds |
| --- | --- |
| `model.joblib` | A dictionary with the full pipeline, the scaler and the feature details. The app loads this one |
| `pipeline.joblib` | The complete fitted pipeline on its own |
| `scaler.joblib` | The fitted numeric scaler |
| `features.json` | The feature list, derived-feature rules and the selection metric, in readable form |

The anomaly model also stores its peer reference tables (median and spread of log amounts for every service, class and region group), because its inputs depend on them.

### Full code

```python
from pathlib import Path

import json

import joblib

ARTIFACT_ROOT = Path("models") / "business_objectives"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


def fitted_numeric_scaler(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    return numeric_pipeline.named_steps["scaler"]


def save_objective_artifacts(
    objective_id,
    pipeline,
    scaler,
    feature_spec,
    extra_bundle=None,
):
    output_dir = ARTIFACT_ROOT / objective_id
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "pipeline": pipeline,
        "scaler": scaler,
        "feature_spec": feature_spec,
    }
    if extra_bundle:
        bundle.update(extra_bundle)

    joblib.dump(bundle, output_dir / "model.joblib")
    joblib.dump(pipeline, output_dir / "pipeline.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")

    feature_metadata = {"feature_spec": feature_spec}
    if extra_bundle:
        feature_metadata.update(extra_bundle)
    if "feature_reference" in feature_metadata:
        feature_metadata["feature_reference"] = dict(feature_metadata["feature_reference"])
    feature_reference = feature_metadata.get("feature_reference")
    if feature_reference and "group_stats_lookup" in feature_reference:
        feature_reference["group_stats_lookup"] = [
            {"group_values": list(group_key), **stats}
            for group_key, stats in feature_reference["group_stats_lookup"].items()
        ]

    def json_safe(value):
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return json_safe(value.item())
        if value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
            return None
        return value

    with (output_dir / "features.json").open("w", encoding="utf-8") as feature_file:
        json.dump(json_safe(feature_metadata), feature_file, indent=2, allow_nan=False)

    print(f"Saved {objective_id} artifacts to {output_dir.resolve()}")


objective_artifacts = [
    (
        "01_booking_cancellations",
        trained_models["Logistic Regression"],
        fitted_numeric_scaler(trained_models["Logistic Regression"]),
        {
            "objective": "Reduce booking cancellations",
            "selected_model": "Logistic Regression",
            "target": "cancelled (1 if booking_status is Cancelled/Canceled, else 0)",
            "model_features": [
                "advance_days", "advance_days_missing", "service_rating",
                "cancellation_flexibility", "service_type", "client_tier",
                "booking_channel", "policy_compliant",
            ],
            "numeric_features": [
                "advance_days", "advance_days_missing", "service_rating",
            ],
            "categorical_features": [
                "cancellation_flexibility", "service_type", "client_tier",
                "booking_channel", "policy_compliant",
            ],
            "required_source_columns": [
                "booking_date", "travel_date", "booking_status", "supplier_id",
                "service_type", "client_id", "booking_channel", "policy_compliant",
            ],
            "required_lookups": {
                "supplier_id": ["service_rating", "cancellation_flexibility"],
                "client_id": ["client_tier"],
            },
            "derived_features": {
                "advance_days": "Parsed travel_date minus booking_date; negative or invalid values become missing.",
                "advance_days_missing": "1 when advance_days is missing, else 0.",
            },
            "selection_metric": "Highest held-out average precision among evaluated candidate models (0.253).",
        },
    ),
    (
        "02_trip_cost_forecasting",
        regression_models["Hist Gradient Boosting Regressor"],
        fitted_numeric_scaler(regression_models["Hist Gradient Boosting Regressor"]),
        {
            "objective": "Forecast trip cost",
            "selected_model": "Hist Gradient Boosting Regressor",
            "target": "log_amount_usd (invert predictions with exp to obtain USD)",
            "model_features": list(regression_features),
            "numeric_features": [
                "international", "advance_days", "trip_length_days", "peak_season",
            ],
            "categorical_features": [
                "service_type", "travel_class", "destination_region",
            ],
            "required_source_columns": [
                "origin_city", "destination_city", "booking_date", "travel_date",
                "return_date", "amount_usd", "service_type", "travel_class",
            ],
            "derived_features": {
                "destination_region": "Mapped from normalized destination city using the notebook city-location map.",
                "international": "1 when normalized origin and destination countries differ, else 0.",
                "advance_days": "Corrected travel date minus booking date, in days; negative values become missing.",
                "trip_length_days": "Corrected return date minus corrected travel date, in days; negative values become missing.",
                "peak_season": "1 for travel months June, July, August, or December; otherwise 0.",
            },
            "selection_metric": "Lowest held-out MAE ($231.850); RMSE $1,012.344, R2 0.506, WAPE 0.184.",
        },
    ),
    (
        "03_policy_compliance",
        policy_models["Logistic Regression"],
        fitted_numeric_scaler(policy_models["Logistic Regression"]),
        {
            "objective": "Improve travel policy compliance",
            "selected_model": "Logistic Regression",
            "target": "out_of_policy (1 if policy_compliant is No, else 0)",
            "model_features": list(policy_features),
            "numeric_features": list(numeric_policy_features),
            "categorical_features": list(categorical_policy_features),
            "required_source_columns": [
                "travel_class", "booking_date", "travel_date", "booking_channel",
                "policy_compliant", "traveler_id", "supplier_id", "client_id",
            ],
            "required_lookups": {
                "traveler_id": ["grade"],
                "supplier_id": ["preferred_partner"],
                "client_id": ["has_travel_policy"],
            },
            "derived_features": {
                "advance_days": "Parsed travel_date minus booking_date; negative or invalid values become missing.",
                "advance_days_missing": "1 when advance_days is missing, else 0.",
                "traveler_grade": "Joined from the traveler dimension using traveler_id.",
                "preferred_partner": "Joined from the supplier dimension using supplier_id.",
                "has_travel_policy": "Joined from the client dimension using client_id.",
            },
            "selection_metric": "Highest held-out average precision (0.452); balanced accuracy 0.646, F1 0.488.",
        },
    ),
    (
        "04_customer_satisfaction",
        satisfaction_models["Logistic Regression"],
        fitted_numeric_scaler(satisfaction_models["Logistic Regression"]),
        {
            "objective": "Raise customer satisfaction",
            "selected_model": "Logistic Regression",
            "target": "dissatisfied (1 if valid csat_score <= 2, else 0)",
            "model_features": list(satisfaction_features),
            "numeric_features": list(numeric_satisfaction_features),
            "categorical_features": list(categorical_satisfaction_features),
            "required_source_columns": [
                "resolution_hours", "first_response_hours", "escalated",
                "category", "contact_channel", "ticket_id", "booking_id",
                "client_id", "csat_score",
            ],
            "required_lookups": {
                "booking_id": ["client_id"],
                "client_id": ["client_tier"],
            },
            "derived_features": {
                "client_id": "Use ticket client_id when present, otherwise fill from its booking.",
                "client_tier": "Joined from the client dimension using resolved client_id.",
                "dissatisfied": "1 for valid CSAT scores 1 or 2; 0 for scores 3 through 5.",
            },
            "selection_metric": "Highest held-out average precision (0.893); balanced accuracy 0.866, F1 0.786.",
        },
    ),
    (
        "05_booking_amount_anomalies",
        anomaly_model,
        anomaly_model.named_steps["scaler"],
        {
            "objective": "Detect booking amount anomalies",
            "selected_model": "Isolation Forest (only anomaly model evaluated)",
            "target": "Unsupervised anomaly score; answer key is for evaluation only.",
            "model_features": list(X_anomaly_train.columns),
            "numeric_features": list(X_anomaly_train.columns),
            "categorical_features": [],
            "required_source_columns": [
                "service_type", "travel_class", "destination_city", "amount_usd",
            ],
            "derived_features": {
                "destination_region": "Normalize destination city and map through the notebook city-region map.",
                "peer_reference": "Training-only medians and MADs by service_type, travel_class, and destination_region; unseen or groups with fewer than 20 rows fall back to global training references.",
                "model_features": list(X_anomaly_train.columns),
            },
            "selection_metric": "Only anomaly model evaluated; held-out average precision 0.678, ROC AUC 0.947, F1 0.634.",
        },
        {
            "feature_reference": {
                "city_regions": city_regions,
                "group_columns": group_columns,
                "group_stats_lookup": group_stats_lookup,
                "global_log_median": float(global_log_median),
                "global_log_mad": float(global_log_mad),
                "minimum_group_count": 20,
            },
        },
    ),
]

for objective_id, pipeline, scaler, feature_spec, *extra in objective_artifacts:
    save_objective_artifacts(
        objective_id,
        pipeline,
        scaler,
        feature_spec,
        extra_bundle=extra[0] if extra else None,
    )

print("Saved model bundles, pipelines, fitted scalers, and feature metadata for all objectives.")
```

---

## 20. Download Model, Scaler and Features

This cell zips each objective's files and shows a download link for every one of them. It checks that all four files exist first, and stops with a clear message if one is missing.

### Full code

```python
from IPython.display import FileLink, HTML, display
import zipfile

artifact_names = ("model.joblib", "pipeline.joblib", "scaler.joblib", "features.json")

for objective_id, *_ in objective_artifacts:
    objective_dir = ARTIFACT_ROOT / objective_id
    missing_artifacts = [
        name for name in artifact_names if not (objective_dir / name).is_file()
    ]
    if missing_artifacts:
        raise FileNotFoundError(
            f"Missing artifacts for {objective_id}: {', '.join(missing_artifacts)}. "
            "Run the artifact-saving cell first."
        )

    archive_path = ARTIFACT_ROOT / f"{objective_id}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in artifact_names:
            archive.write(objective_dir / name, arcname=f"{objective_id}/{name}")

    display(HTML(f"<h4>{objective_id}</h4>"))
    for name in artifact_names:
        display(FileLink(str(objective_dir / name), result_html_prefix=f"{name}: "))
    display(FileLink(str(archive_path), result_html_prefix="Download all: "))
```

### The Streamlit app

The saved files power [`app.py`](app.py). It has a Home page and one page for each model: **Booking Cancellations, Trip Cost Forecast, Policy Compliance, Customer Satisfaction and Booking Amount Anomalies.**

- **One booking or ticket:** fill in a short form and press Predict. The result shows the risk percentage, a High, Medium or Low tier, a suggested action and the main factors behind the score.
- **Many rows:** download a CSV template, upload your filled file and download the scored results.
- **Derived inputs** such as advance days, trip length, peak season and destination region are worked out by the app from the dates and cities.
- Model features were checked against the notebook on the test rows, and the results matched.


Live app: [Open the Streamlit app](https://horizonglobaltravelmanagementpredictivedecisionsupportsystem.streamlit.app/)

<!-- Add a screenshot: <p align="center"><img src="images/app_screenshot.png" width="100%"></p> -->

### Full code: app.py

```python
# TRAVEL BUSINESS OPERATIONS PREDICTION SYSTEM  (Horizon Global Travel Management)
# A Streamlit app that uses the five saved models from the notebook.
#
#   Page                      Business problem                        Model
#   Booking Cancellations     Which bookings are likely to cancel?    Logistic Regression
#   Trip Cost Forecast        What will this trip cost?               Hist Gradient Boosting
#   Policy Compliance         Is this booking likely out of policy?   Logistic Regression
#   Customer Satisfaction     Will this ticket end dissatisfied?      Logistic Regression
#   Booking Amount Anomalies  Is this booking amount unusual?         Isolation Forest

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
```

---

## 21. Key Findings and Insights

| # | Objective | Result on unseen data | What it means |
| --- | --- | --- | --- |
| 1 | Cancellations | ROC AUC 0.63, recall 59%, precision 24.5% | Weak but usable for ranking, not for certain calls |
| 2 | Trip cost | Error $232 (18.4%), R-squared 0.51 | Good for planning ranges |
| 3 | Policy compliance | ROC AUC 0.68, recall 62%, precision 40% | A review-order signal |
| 4 | Satisfaction | ROC AUC 0.95, recall 85%, precision 73% | Strongest model |
| 5 | Anomalies | 13 of 18 found, 10 false alarms | Good first filter |

- **Data quality.** After cleaning there are 6,000 bookings and 2,500 tickets, and 2,430 tickets carry a usable satisfaction label. 70 were dropped because they had no valid score.
- **Complaints drive dissatisfaction.** 56.2% of complaint tickets end dissatisfied, compared with 27.6% overall. Chat tickets are the best handled at 9.4%.
- **Response speed shows up.** Satisfied customers waited a median of 1.3 hours for a first reply, dissatisfied ones 6.0 hours.
- **Premium cabins and phone bookings break policy most.** First class 43.7%, suites 41.0%, phone 34.3%, against 20.6% for the online tool.
- **Scores rank risk well even when accuracy looks modest.** On test data, tickets scored under 40% were dissatisfied 4% of the time and tickets scored over 70% were dissatisfied 80% of the time. For policy, under 40% means 16% out of policy and over 70% means 58%. For cancellations, under 40% means 10% cancel and over 70% means 32%.
- **A simple model won three of four supervised objectives.** Logistic Regression beat Random Forest and Hist Gradient Boosting on cancellations, policy and satisfaction. Extra complexity did not help.
- **Anomalies are about distance from peers.** The model compares each amount with similar bookings, which is why a $4,688 median for anomalies stands out against $929.

**Limits to keep in mind**

- Cancellation and policy models have modest discrimination. Many flagged bookings will not turn out to be problems.
- The satisfaction model uses resolution time and escalation, which are known late in a ticket's life.
- Each result comes from one 80/20 split, and the anomaly test set has only 18 anomalies.
- The data is synthetic, so the results show the method, not the performance of a real company.

---

## 22. Recommendations

1. **Start with satisfaction triage.** It is the strongest model. Score tickets as they close, give High-tier tickets a same-day follow-up call and treat the score as a prompt for human help, never a reason to reduce support. Look at complaints first, since more than half end badly.
2. **Move more customers to chat.** Chat tickets end dissatisfied only 9.4% of the time. Test whether routing suitable requests to chat holds that result before assuming it will.
3. **Reply faster.** Dissatisfied customers waited about 4.7 hours longer for a first reply. Set a first-response target and track it.
4. **Pilot cancellation outreach cheaply.** With 24.5% precision, only low-cost, reversible steps make sense, such as a reminder or a flexible-change offer. Compare against a control group to prove it works before scaling.
5. **Review bookings for policy by tier.** Check High-tier bookings before ticketing, especially phone and travel-agent bookings and premium classes. Verify against the client's actual policy, since the score is only a signal.
6. **Use cost estimates as ranges.** Quote clients the planning range (about plus or minus 18%), and review large bookings by hand because the model is least reliable on them.
7. **Send flagged amounts to billing review.** Check each against the supplier invoice before payment. A flag means unusual, not wrong. Record which flags are confirmed so the threshold can be tuned.
8. **Roll out in shadow mode, then monitor.** Run alongside current processes, track false alarms, missed cases and drift every month, and retrain on fresh data. Keep a person in the loop for every flagged case.

---

## 23. Impact

These results show where the models can help, but they do **not** prove realised savings or better service. Every figure below comes from held-out test data. Real impact has to be confirmed in a monitored pilot.

### What the models can do for the business

| Area | Model | What the test results say | How to use it |
|---|---|---|---|
| Customer experience | Logistic Regression (satisfaction) | Balanced accuracy **0.866**, ROC AUC **0.945**, F1 **0.786**. Precision **0.731** and recall **0.851**: about 73 of every 100 tickets it flags are truly dissatisfied, and it finds about 85 of every 100 dissatisfied tickets | Rank tickets so staff follow up on the riskiest first. This is the strongest candidate for a first pilot |
| Cancellations | Logistic Regression | Recall **0.590**, precision **0.245**, ROC AUC **0.633** against a **17.1%** cancellation rate. It finds about 6 in 10 cancellations, but roughly 3 in 4 flagged bookings will not cancel | Low-cost, reversible outreach only. Set the threshold to fit team capacity |
| Policy compliance | Logistic Regression | Recall **0.622**, precision **0.401**, balanced accuracy **0.646** | Order the review queue. Never use a score to penalise a traveller |
| Billing anomalies | Isolation Forest | On 1,200 held-out bookings (18 true anomalies) it flagged **23**. About **13** were real (precision **0.565**) and it caught **72.2%** of the anomalies, missing about 5 (ROC AUC **0.947**) | Verify every flag against source records. A flag marks an unusual booking, not confirmed error or fraud |
| Trip-cost planning | Hist Gradient Boosting | MAE about **$231.85**, WAPE **18.4%**, R² **0.506**, RMSE about **$1,012** | Use as a planning range, not a quote. Review large misses before committing to a price |

### Customer experience

**27.6%** of labelled tickets ended in dissatisfaction, rising to **56.2%** for complaints and falling to **9.4%** for chat. The satisfaction model is accurate enough to put the riskiest tickets at the front of the queue. These are patterns in the data. They do not prove that earlier follow-up will reduce dissatisfaction, so the pilot must compare results with the current process.

### Operational efficiency and risk control

Cancellation, policy and anomaly scores can order the work in human-review queues. Their precision (**0.245**, **0.401** and **0.565**) means some flagged cases will be false alarms, so scores should never trigger automatic penalties or customer-facing actions. Any efficiency gain depends on saving staff time without missing more cases or treating groups of customers unfairly.

### Planning and cost management

The cost model is typically off by about **$232** per booking (**18.4%** of booking value), but a few bookings are missed by much more, which is why RMSE (about **$1,012**) is far above MAE. Treat the output as guidance and not as a guaranteed price or a saving.

### How to measure the impact

Start with a limited support-triage pilot and compare it against the current process or a control group.

| What to track | Why |
|---|---|
| Time to first response and to resolution | Shows whether prioritised tickets are handled faster |
| Dissatisfaction after follow-up | The main customer-experience outcome |
| Staff review time per flag | Shows whether the queues save time or add work |
| Confirmed vs unconfirmed flags, and missed cases | Tests precision and recall in real use |
| Cost-forecast error | Confirms the MAE and WAPE hold on new bookings |
| Results across customer groups | Catches unfair outcomes early |
| Intervention cost, and any recovered or avoided cost | Needed before claiming any financial benefit |

Expand only if the measured improvements justify the extra workload and risk.

---

## 23. Conclusion

This project took five travel-management problems from raw, messy tables to a working prediction app. Raw data was audited, cleaned and joined; features were built; at least three models were compared on each supervised objective; the best of each was confirmed on unseen data and saved with its scaler and features. Logistic Regression won cancellations, policy compliance and satisfaction, Hist Gradient Boosting won trip cost, and Isolation Forest handled anomalies.

The models are decision support. Satisfaction is ready for a first pilot. Cancellation and policy scores should only rank where to look first. Cost estimates are planning ranges, and anomaly flags are prompts for a human check. The next steps are to replace the synthetic data with live bookings, pilot each tier next to current practice, measure the difference and retrain as the business changes.

---

**Author:** Adewale Oshinukan | Data Scientist | [GitHub](https://github.com/waleoshin75-sketch/Horizon-Global-Travel-Management-Predicting-Cancellations-Cost-Policy-Compliance-and-Service-Quality) | [Live App](https://horizonglobaltravelmanagementpredictivedecisionsupportsystem.streamlit.app/)
