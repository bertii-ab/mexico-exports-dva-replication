"""
Stage 1: Data Acquisition & Preprocessing
Extracts trade and macroeconomic data from OECD SDMX, BIS, and Banxico APIs.
"""

import io
import pandas as pd
import requests

# 1. TiVA URL
base_url = "https://sdmx.oecd.org/sti-public/rest/data/OECD.STI.PIE,DSD_TIVA_MAINLV@DF_MAINLV,1.1/EXGR_DVA.MEX._T.W..A?startPeriod=1995"

# 2. Tell the OECD API we want the data formatted as a CSV with text labels
csv_url = base_url + "&format=csvfilewithlabels"

print("Fetching data from OECD...")

# 3. Make the request with a standard user-agent header to avoid getting blocked
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
response = requests.get(csv_url, headers=headers)

if response.status_code == 200:
    # 4. Load the CSV content directly into a pandas DataFrame
    df = pd.read_csv(io.StringIO(response.text))

    print("\n Success! Data loaded into DataFrame.")
    print(f"Shape of DataFrame: {df.shape}\n")

    # 5. Display the first few rows and useful columns
    # Common OECD SDMX columns: 'TIME_PERIOD' (Year) and 'OBS_VALUE' (The actual value)
    available_cols = [
        col
        for col in ["TIME_PERIOD", "OBS_VALUE", "Structured Code", "Description"]
        if col in df.columns
    ]

    if available_cols:
        print(df[available_cols].head(10))
    else:
        # Fallback to showing whatever columns returned
        print(df.head(10))

else:
    print(
        f" Failed to fetch data. Status code: {response.status_code}"
    )
    print(response.text)

df_dva = df.copy()
print("DataFrame 'df' renamed to 'df_dva'.")

import pandas as pd

# 1. Target the official BIS v2 API with an explicit SDMX key
# Key Layout: FREQ (M = Monthly) . EER_TYPE (N+R = Nominal & Real) . EER_BASKET (B = Broad Basket) . REF_AREA (MX = Mexico)
url = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/M.N+R.B.MX?format=csv"

print("Connecting to the official BIS Data Gateway...")

try:
    # 2. Read the CSV streaming payload directly from the BIS server
    df_raw = pd.read_csv(url)

    # 3. Identify columns dynamically (makes your code immune to future API header label updates)
    time_col = [
        c
        for c in df_raw.columns
        if "TIME_PERIOD" in c.upper() or "PERIOD" in c.upper()
    ][0]
    val_col = [
        c for c in df_raw.columns if "OBS_VALUE" in c.upper() or "VALUE" in c.upper()
    ][0]
    type_col = [
        c for c in df_raw.columns if "EER_TYPE" in c.upper() or "TYPE" in c.upper()
    ][0]

    # 4. Extract critical dimensions
    df_clean = df_raw[[time_col, type_col, val_col]].copy()
    df_clean.columns = ["Month", "Type", "Value"]

    # 5. Pivot the table so REER and NEER align side-by-side chronologically
    df_pivot = df_clean.pivot(index="Month", columns="Type", values="Value")

    # 6. Dynamically map BIS structural codes to academic headers
    rename_dict = {}
    for col in df_pivot.columns:
        if str(col).upper().startswith("R"):
            rename_dict[col] = "REER (Real - Broad)"
        elif str(col).upper().startswith("N"):
            rename_dict[col] = "NEER (Nominal - Broad)"

    df_pivot.rename(columns=rename_dict, inplace=True)
    df_pivot.sort_index(inplace=True)  # Ensure chronological order

    print("\n Success! Official BIS Monthly Data loaded into DataFrame.")
    print(f"Total historical months retrieved: {len(df_pivot)}\n")

    # Display data snapshot
    print("--- Earliest Observations ---")
    print(df_pivot.head(10))
    print("\n...")
    print("\n--- Most Recent Observations ---")
    print(df_pivot.tail(10))

    # 7. Automatically export a verified copy for your thesis replication folder
    df_pivot.to_csv("mexico_monthly_eer_bis_official.csv")

except Exception as e:
    print(f"\n❌ Error pulling or parsing BIS data: {e}")
    print(
        "Verify your network connection or check if the BIS server is undergoing scheduled maintenance."
    )



import pandas as pd
import requests

import os

# 1. Configuration & Mapping
# To fetch fresh data from Banxico SIE API, obtain a free token at https://www.banxico.org.mx/SieAPIRest/
# and export BANXICO_TOKEN="your_token_here"
BANXICO_TOKEN = os.getenv("BANXICO_TOKEN", "YOUR_BANXICO_TOKEN_HERE")

# Dictionary to map Banxico IDs to your custom names
RENAME_DICT = {
    "SR16575": "ValueAdded",
    "SE36593": "TotalExports",
    "SE35401": "Inter_goods_NO",
}


# 2. Core helper function to request and parse Banxico's JSON payload
def get_banxico_dataframe(series_ids, frequency_flag):
    url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series_ids}/datos"
    headers = {"Bmx-Token": BANXICO_TOKEN, "Accept": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        raw_json = response.json()

        series_list = raw_json["bmx"]["series"]
        all_series_dfs = []

        for series in series_list:
            s_id = series["idSerie"]

            if "datos" not in series:
                print(f"⚠️ Series {s_id} returned no observation data.")
                continue

            # Load individual series observations
            df_temp = pd.DataFrame(series["datos"])

            # Cast date strings
            df_temp["fecha"] = pd.to_datetime(df_temp["fecha"], format="%d/%m/%Y")

            # Strip out thousands-separator commas before converting to float
            df_temp["dato"] = (
                df_temp["dato"].astype(str).str.replace(",", "", regex=False)
            )
            df_temp["dato"] = pd.to_numeric(df_temp["dato"], errors="coerce")

            # Map to readable columns and establish datetime index
            df_temp.rename(columns={"fecha": "Date", "dato": s_id}, inplace=True)
            df_temp.set_index("Date", inplace=True)

            all_series_dfs.append(df_temp)

        if all_series_dfs:
            # Outer join combinations to match dates perfectly
            df_combined = pd.concat(all_series_dfs, axis=1)
            df_combined.sort_index(inplace=True)

            # Enforce academic period syntax formatting based on frequency
            if frequency_flag == "Q":
                df_combined.index = df_combined.index.to_period("Q")
                df_combined.index.name = "Quarter"
            elif frequency_flag == "M":
                df_combined.index = df_combined.index.to_period("M")
                df_combined.index.name = "Month"

            # Apply the column renaming mapping
            df_combined.rename(columns=RENAME_DICT, inplace=True)

            return df_combined
        else:
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        print(f"❌ Network or Token validation error for [{series_ids}]: {e}")
        return pd.DataFrame()


# ==========================================
# 3. Execution
# ==========================================

print("Querying Banco de México API Gateway and renaming series...\n")

# --- Fetch and process Quarterly Data ---
df_quarterly = get_banxico_dataframe("SR16575", frequency_flag="Q")

# --- Fetch and process Monthly Data ---
df_monthly = get_banxico_dataframe("SE35401,SE36593", frequency_flag="M")

# ==========================================
# 4. Display Renamed Results
# ==========================================

if not df_quarterly.empty:
    print("--- 📊 Quarterly Dataframe (df_quarterly) ---")
    print(f"Shape: {df_quarterly.shape}")
    print(df_quarterly.tail(10))
    print("-" * 50)

if not df_monthly.empty:
    print("\n--- 📊 Monthly Dataframe (df_monthly) ---")
    print(f"Shape: {df_monthly.shape}")
    print(df_monthly.tail(10))
    print("-" * 50)

import pandas as pd

# Direct open URL to stream the INDPRO dataset
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=INDPRO"

print("Connecting to FRED Open Data Server...")

try:
    # 1. Read the raw data from FRED
    df_fred_monthly = pd.read_csv(url)

    # 2. DYNAMIC FIX: Identify columns by position instead of exact text names
    raw_date_col = df_fred_monthly.columns[0]  # Always the 1st column
    raw_value_col = df_fred_monthly.columns[1]  # Always the 2nd column

    # 3. Coerce the date column to actual datetime objects
    df_fred_monthly[raw_date_col] = pd.to_datetime(df_fred_monthly[raw_date_col])

    # 4. Standardize period syntax to 'YYYY-MM' to match your Banxico monthly DataFrame
    df_fred_monthly["Month"] = df_fred_monthly[raw_date_col].dt.to_period("M")
    df_fred_monthly.set_index("Month", inplace=True)

    # Drop the original raw date tracker column safely
    df_fred_monthly.drop(columns=[raw_date_col], inplace=True)

    # 5. Handle any potential FRED missing data placeholders safely
    df_fred_monthly[raw_value_col] = pd.to_numeric(
        df_fred_monthly[raw_value_col], errors="coerce"
    )

    # 6. Rename the series to your clean academic designation
    df_fred_monthly.rename(
        columns={raw_value_col: "US_IndustrialProduction"}, inplace=True
    )

    print("\n Success! FRED Monthly Data loaded into DataFrame.")
    print(f"Total historical months retrieved: {len(df_fred_monthly)}\n")

    # Display data snapshot
    print("--- 📊 Monthly Dataframe (df_fred_monthly) ---")
    print(df_fred_monthly.tail(12))
    print("-" * 50)

except Exception as e:
    print(f"\n❌ Error pulling or parsing data from FRED: {e}")

import pandas as pd

# Official download URL for the New York Fed's GSCPI spreadsheet
url = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"

print("Connecting to the Federal Reserve Bank of New York Server...")

try:
    # CRITICAL FIX: Add usecols=[0, 1] to force pandas to only read Columns A and B,
    # completely dropping any ghost columns created by the banner formatting.
    df_raw = pd.read_excel(
        url, sheet_name=1, skiprows=5, header=None, usecols=[0, 1]
    )

    # Now it is guaranteed to have exactly 2 columns
    df_raw.columns = ["Date", "GSCPI"]

    # Convert the date strings (e.g., '31-Jan-1998') into datetime objects
    df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")

    # Drop any trailing empty rows or unparseable artifacts safely
    df_raw.dropna(subset=["Date"], inplace=True)

    # Standardize the timeline index to 'YYYY-MM' to match your workspace
    df_raw["Month"] = df_raw["Date"].dt.to_period("M")
    df_raw.set_index("Month", inplace=True)

    # Isolate the final clean economic index column
    df_gscpi = df_raw[["GSCPI"]].copy()
    df_gscpi["GSCPI"] = pd.to_numeric(df_gscpi["GSCPI"], errors="coerce")
    df_gscpi.sort_index(inplace=True)

    print(
        f"\n Success! GSCPI layout handled. Total months retrieved: {len(df_gscpi)}\n"
    )
    print("--- 📊 Cleaned GSCPI DataFrame (df_gscpi) ---")
    print(df_gscpi.head(5))
    print("\n...")
    print(df_gscpi.tail(5))
    print("-" * 50)

    # Save a clean version for your replication folder
    df_gscpi.to_csv("global_supply_chain_pressure_index.csv")

except Exception as e:
    print(f"\n❌ Error parsing the Excel file layout: {e}")
    df_gscpi = pd.DataFrame(columns=["GSCPI"])
    df_gscpi.index.name = "Month"



import pandas as pd

# Ensure df_gscpi exists even if the NY Fed download failed
if 'df_gscpi' not in dir():
    df_gscpi = pd.DataFrame(columns=["GSCPI"])
    df_gscpi.index.name = "Month"

print("Beginning multi-frequency dataset consolidation...\n")

# ==========================================
# 1. CONSOLIDATE MONTHLY DATAFREQ
# ==========================================
master_monthly = pd.concat(
    [df_monthly, df_fred_monthly, df_gscpi], axis=1, join="outer"
)
start_period_monthly = pd.Period('1993-01', freq='M')
master_monthly.sort_index(inplace=True)
master_monthly.index.name = "Month"
master_monthly = master_monthly.loc[start_period_monthly:]


# ==========================================
# 2. PROCESS AND CONSOLIDATE ANNUAL DATA (TiVA OECD)
# ==========================================
def clean_annual_tiva(dataframe, target_column_name):
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    # Isolate core observation values
    df_clean = dataframe[["TIME_PERIOD", "OBS_VALUE"]].copy()

    # FIXED: Using pd.PeriodIndex to correctly build the annual tracker
    df_clean["Year"] = pd.PeriodIndex(
        df_clean["TIME_PERIOD"].astype(str), freq="Y"
    )
    df_clean.set_index("Year", inplace=True)

    # Assign distinct academic variable names to avoid OBS_VALUE collisions
    df_clean.rename(columns={"OBS_VALUE": target_column_name}, inplace=True)
    df_clean.drop(columns=["TIME_PERIOD"], inplace=True)

    return df_clean


# Process both TiVA layers
df_tiva_base = clean_annual_tiva(df, target_column_name="TiVA_GrossExports")
df_tiva_dva = clean_annual_tiva(df_dva, target_column_name="TiVA_DVA")

# Combine annual metrics side-by-side
if not df_tiva_base.empty and not df_tiva_dva.empty:
    master_annual = pd.concat([df_tiva_base, df_tiva_dva], axis=1, join="outer")
    master_annual.sort_index(inplace=True)
    master_annual.index.name = "Year"
else:
    master_annual = df_tiva_base if not df_tiva_base.empty else df_tiva_dva


# ==========================================
# 3. CONSOLIDATE QUARTERLY DATA
# ==========================================
master_quarterly = df_quarterly.copy()
master_quarterly.sort_index(inplace=True)
master_quarterly.index.name = "Quarter"


# ==========================================
# 4. VALIDATE AND DISPLAY MATRIX METRICS
# ==========================================
print("--- 📊 Consolidated Monthly Master (master_monthly) ---")
print(f"Time Horizon: {master_monthly.index.min()} to {master_monthly.index.max()}")
print(f"Available Columns: {list(master_monthly.columns)}")
print(master_monthly.tail(6))
print("-" * 60)

print("\n--- 📊 Consolidated Quarterly Master (master_quarterly) ---")
print(
    f"Time Horizon: {master_quarterly.index.min()} to {master_quarterly.index.max()}"
)
print(master_quarterly.tail(6))
print("-" * 60)

if not master_annual.empty:
    print("\n--- 📊 Consolidated Annual TiVA Master (master_annual) ---")
    print(f"Time Horizon: {master_annual.index.min()} to {master_annual.index.max()}")
    print(master_annual.tail(6))
    print("-" * 60)

print("Saving master_annual.csv...")
master_annual.to_csv('master_annual.csv')
print("master_annual.csv saved.")

print("Saving master_monthly.csv...")
master_monthly.to_csv('master_monthly.csv')
print("master_monthly.csv saved.")

print("Saving master_quarterly.csv...")
master_quarterly.to_csv('master_quarterly.csv')
print("master_quarterly.csv saved.")