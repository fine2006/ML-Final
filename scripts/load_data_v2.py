"""
DECISION LOG:
- Data loading from 4 regions with flexible file format handling
- Per-region missing value strategies
- Preserve maximum data
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


REGION_CONFIG = {
    " Bhatagaon DCR": {
        "path": "Pollution Data Raipur/Bhatagaon DCR/DCR bhatagaon/DCR {year}/{month}/",
        "pm25_col": "PM2_5__BHATAGAON",
        "date_col": "Date & Time",
        "file_pattern": "Hourly",
        "interpolation": "linear",
        "fallback": "ffill_bfill",
    },
    "DCR AIIMS": {
        "path": "Pollution Data Raipur/DCR AIIMS/{year}/{month}/",
        "pm25_col": "PM2_5_AIIM",
        "date_col": "Date & Time",
        "file_pattern": "Hourly",
        "interpolation": "linear",
        "fallback": "spline",
    },
    "IGKV DCR": {
        "path": "Pollution Data Raipur/IGKV DCR/Year {year}/{month}/",
        "pm25_col": "PM 2.5",
        "date_col": "Date & Time",
        "file_pattern": "Hourly",
        "month_transform": lambda m: (
            m.upper()
            if m.lower()
            in [
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            ]
            else m.upper()[:3].upper() + m[3:]
            if len(m) > 3
            else m.upper()
        ),
        "interpolation": "linear",
        "fallback": "ffill",
    },
    "SILTARA DCR": {
        "path": "Pollution Data Raipur/SILTARA DCR/{year}/{month}/",
        "pm25_col": "PM2_5_SILTARA",
        "date_col": "Date & Time",
        "file_pattern": "Hourly",
        "interpolation": "linear",
        "fallback": "ffill_bfill",
    },
}


def find_files(region: str, year: int) -> List[str]:
    """Find all hourly data files for a region and year."""
    config = REGION_CONFIG[region]
    base_path = config["path"].format(year=year, month="*")
    month_transform = config.get("month_transform", lambda m: m)

    files = []
    for month_dir in Path(".").glob(base_path):
        if month_dir.is_dir():
            for f in month_dir.iterdir():
                if (
                    f.is_file()
                    and config["file_pattern"] in f.name
                    and not f.name.startswith("~")
                    and not f.name.startswith("~$")
                ):
                    files.append(str(f))
    return sorted(files)


def load_single_file(filepath: str, region: str) -> pd.DataFrame:
    """Load a single Excel file and extract PM2.5 data with max preservation."""
    config = REGION_CONFIG[region]
    pm25_col = config["pm25_col"]
    date_col = config["date_col"]

    try:
        xl = pd.ExcelFile(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

    all_data = []

    for sheet in xl.sheet_names:
        if not sheet.isdigit():
            continue

        try:
            df = pd.read_excel(xl, sheet_name=sheet, header=None)

            date_col_idx = None
            pm25_col_idx = None

            for idx, row in df.iterrows():
                if idx < 6:
                    continue
                if idx == 6:
                    for col_idx, val in enumerate(row):
                        if val == date_col:
                            date_col_idx = col_idx
                        if val == pm25_col:
                            pm25_col_idx = col_idx
                    if date_col_idx is None or pm25_col_idx is None:
                        break
                    continue

                if idx >= 8:
                    date_val = row.iloc[date_col_idx]
                    pm25_val = row.iloc[pm25_col_idx]

                    if pd.isna(date_val):
                        continue

                    if isinstance(date_val, str):
                        if date_val.strip() in [
                            "Min",
                            "Max",
                            "Avg",
                            "Stdev",
                            "Num",
                            "DCR",
                            "Status",
                            "Done by",
                            "Comment",
                        ]:
                            break
                        date_val = date_val.strip()

                    if isinstance(pm25_val, str):
                        if pm25_val.strip() in [
                            "Data F",
                            "Maint.",
                            "Invalid",
                            "Pw.Off",
                            "Link F",
                            "Calib.",
                            "",
                        ]:
                            pm25_val = np.nan
                        else:
                            try:
                                pm25_val = float(pm25_val)
                            except ValueError:
                                pm25_val = np.nan
                    else:
                        if pd.isna(pm25_val):
                            pm25_val = np.nan

                    if not pd.isna(pm25_val):
                        all_data.append({"datetime": date_val, "pm25_raw": pm25_val})

        except Exception as e:
            continue

    if all_data:
        df_result = pd.DataFrame(all_data)

        def parse_datetime(dt_val):
            dt_str = str(dt_val).strip()
            if " 24:00" in dt_str:
                dt_str = dt_str.replace(" 24:00", " 00:00")
                try:
                    return pd.to_datetime(dt_str, dayfirst=True) + pd.Timedelta(days=1)
                except:
                    pass
            try:
                return pd.to_datetime(dt_str, dayfirst=True)
            except:
                return pd.NaT

        df_result["datetime"] = df_result["datetime"].apply(parse_datetime)
        df_result = df_result.dropna(subset=["datetime"])

        if not df_result.empty:
            df_result = df_result.sort_values("datetime").reset_index(drop=True)

        return df_result

    return pd.DataFrame()


def load_region_data(region: str, years: List[int]) -> pd.DataFrame:
    """Load all data for a region across multiple years."""
    all_data = []

    for year in years:
        files = find_files(region, year)
        for f in files:
            df = load_single_file(f, region)
            if not df.empty:
                df["source_file"] = os.path.basename(f)
                df["source_year"] = year
                all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"])
    combined = combined.sort_values("datetime").reset_index(drop=True)

    return combined


def load_all_regions(years: List[int] = [2022, 2023, 2024, 2025]) -> pd.DataFrame:
    """Load data from all regions for given years."""
    all_data = []

    for region in REGION_CONFIG.keys():
        print(f"Loading {region.strip()}...")
        df = load_region_data(region, years)
        if not df.empty:
            df["region"] = region.strip()
            all_data.append(df)
            print(f"  -> {len(df)} records")
        else:
            print(f"  -> No data found")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def load_quarter_hourly_data(
    years: List[int] = [2022, 2023, 2024, 2025],
) -> pd.DataFrame:
    """Load quarter-hourly data files."""
    all_data = []

    QUAT_CONFIG = {
        " Bhatagaon DCR": {
            "path": "Pollution Data Raipur/Bhatagaon DCR/DCR bhatagaon/DCR {year}/{month}/",
            "pm25_col": "PM2_5__BHATAGAON",
            "file_pattern": "QUAT",
        },
        "DCR AIIMS": {
            "path": "Pollution Data Raipur/DCR AIIMS/{year}/{month}/",
            "pm25_col": "PM2_5_AIIM",
            "file_pattern": "QUAT",
        },
        "IGKV DCR": {
            "path": "Pollution Data Raipur/IGKV DCR/Year {year}/{month}/",
            "pm25_col": "PM 2.5",
            "file_pattern": "Quat",
            "month_transform": lambda m: (
                m.upper()
                if m.lower()
                in [
                    "january",
                    "february",
                    "march",
                    "april",
                    "may",
                    "june",
                    "july",
                    "august",
                    "september",
                    "october",
                    "november",
                    "december",
                ]
                else m.upper()[:3].upper() + m[3:]
                if len(m) > 3
                else m.upper()
            ),
        },
        "SILTARA DCR": {
            "path": "Pollution Data Raipur/SILTARA DCR/{year}/{month}/",
            "pm25_col": "PM2_5_SILTARA",
            "file_pattern": "QUAT",
        },
    }

    for region, config in QUAT_CONFIG.items():
        print(f"Loading QUAT {region.strip()}...")

        for year in years:
            base_path = config["path"].format(year=year, month="*")

            for month_dir in Path(".").glob(base_path):
                if month_dir.is_dir():
                    for f in month_dir.iterdir():
                        if (
                            f.is_file()
                            and config["file_pattern"] in f.name
                            and not f.name.startswith("~")
                        ):
                            try:
                                xl = pd.ExcelFile(str(f))

                                for sheet in xl.sheet_names:
                                    if not sheet.isdigit():
                                        continue

                                    df = pd.read_excel(
                                        xl, sheet_name=sheet, header=None
                                    )

                                    date_col_idx = None
                                    pm25_col_idx = None

                                    for idx, row in df.iterrows():
                                        if idx < 6:
                                            continue
                                        if idx == 6:
                                            for col_idx, val in enumerate(row):
                                                if val == "Date & Time":
                                                    date_col_idx = col_idx
                                                if val == config["pm25_col"]:
                                                    pm25_col_idx = col_idx
                                            if (
                                                date_col_idx is None
                                                or pm25_col_idx is None
                                            ):
                                                break
                                            continue

                                        if idx >= 8:
                                            date_val = (
                                                row.iloc[date_col_idx]
                                                if date_col_idx is not None
                                                else None
                                            )
                                            pm25_val = (
                                                row.iloc[pm25_col_idx]
                                                if pm25_col_idx is not None
                                                else None
                                            )

                                            if pd.isna(date_val):
                                                continue

                                            if isinstance(date_val, str):
                                                if date_val.strip() in [
                                                    "Min",
                                                    "Max",
                                                    "Avg",
                                                    "Stdev",
                                                    "Num",
                                                    "DCR",
                                                    "Status",
                                                ]:
                                                    continue

                                            if isinstance(pm25_val, str):
                                                if pm25_val.strip() in [
                                                    "Data F",
                                                    "Maint.",
                                                    "Invalid",
                                                    "Pw.Off",
                                                ]:
                                                    pm25_val = np.nan
                                                else:
                                                    try:
                                                        pm25_val = float(pm25_val)
                                                    except:
                                                        pm25_val = np.nan

                                            if not pd.isna(pm25_val) and not pd.isna(
                                                date_val
                                            ):
                                                all_data.append(
                                                    {
                                                        "datetime": date_val,
                                                        "pm25_raw": pm25_val,
                                                        "region": region.strip(),
                                                    }
                                                )
                                            break
                            except Exception as e:
                                continue

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    def parse_datetime(dt_val):
        dt_str = str(dt_val).strip()
        if " 24:00" in dt_str:
            dt_str = dt_str.replace(" 24:00", " 00:00")
            try:
                return pd.to_datetime(dt_str, dayfirst=True) + pd.Timedelta(days=1)
            except:
                pass
        try:
            return pd.to_datetime(dt_str, dayfirst=True)
        except:
            return pd.NaT

    df["datetime"] = df["datetime"].apply(parse_datetime)
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"Total QUAT records: {len(df)}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Loading Hourly Data")
    print("=" * 60)
    df_hourly = load_all_regions([2022, 2023, 2024, 2025])
    print(f"\nHourly total: {len(df_hourly)}")
    print(f"Regions: {df_hourly['region'].unique().tolist()}")
    print(f"Date range: {df_hourly['datetime'].min()} to {df_hourly['datetime'].max()}")

    print("\n" + "=" * 60)
    print("Loading Quarter-Hourly Data")
    print("=" * 60)
    df_quat = load_quarter_hourly_data([2022, 2023, 2024, 2025])
    print(f"\nQuarter-hourly total: {len(df_quat)}")
    print(
        f"Regions: {df_quat['region'].unique().tolist() if not df_quat.empty else 'N/A'}"
    )
