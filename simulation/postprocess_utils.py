#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
postprocess_utils.py

Utility functions for post-generation analysis of simulated life-course data.

Main Function:
    compute_age_completed_education(df, educ_prefix="educ_")

Description:
    Computes the age when an individual's education trajectory stops changing,
    i.e., when the highest education level stabilizes.

Usage Example:
    from postprocess_utils import compute_age_completed_education
    df["age_completed_education"] = compute_age_completed_education(df)
"""

import pandas as pd
import numpy as np


def compute_age_finished_education(df, educ_prefix="education_"):
    """
    Compute the age when education stops changing for each individual.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing yearly education states with columns like
        'educ_14', 'educ_15', ..., 'educ_60'.
    educ_prefix : str, optional
        Prefix of education trajectory columns (default: 'educ_').

    Returns
    -------
    pandas.Series
        A pandas Series of length equal to df, containing the estimated
        age when education stabilized. NaN if trajectory data missing.

    Notes
    -----
    - Columns must follow the pattern 'educ_<age>'.
    - The function assumes monotonic or stable education changes:
      once a new level is reached, it remains stable or increases.
    - If education never changes, returns the last available non-null age.
    - If all values are missing, returns NaN.
    """

    # --- Identify columns that match the pattern 'educ_<age>' ---
    educ_cols = [c for c in df.columns if c.startswith(educ_prefix)]
    if not educ_cols:
        print(f"[postprocess_utils] No columns found with prefix '{educ_prefix}'.")
        return pd.Series([np.nan] * len(df), index=df.index)

    # Sort columns numerically by age
    educ_cols = sorted(educ_cols, key=lambda c: int(c.split("_")[1]))
    ages = [int(c.split("_")[1]) for c in educ_cols]
    def _find_last_change(row):
        """Helper to find the last age when education changed.
        If there is no change, return the first valid age.
        """
        levels = [row[c] for c in educ_cols]
        last_level, last_change_age = None, None

        for age, level in zip(ages, levels):
            if pd.isna(level):
                continue

            if last_level is None:
                last_level, last_change_age = level, age
            elif level != last_level:
                last_level, last_change_age = level, age

        # If no change but valid data exists → return *first* valid age
        if last_change_age is None:
            valid_ages = [a for a, l in zip(ages, levels) if pd.notna(l)]
            return valid_ages[0] if valid_ages else np.nan

        return last_change_age

    return df.apply(_find_last_change, axis=1)

def compute_age_at_first_marriage(df, marital_prefix="marital_status_"):
    """
    Compute the age at first marriage (first age at which status becomes 'married'),
    regardless of previous status.
    """

    marital_cols = [c for c in df.columns if c.startswith(marital_prefix)]
    if not marital_cols:
        print(f"[postprocess_utils] No columns found with prefix '{marital_prefix}'.")
        return pd.Series([np.nan] * len(df), index=df.index)

    # Sort by age
    marital_cols = sorted(marital_cols, key=lambda c: int(c.split("_")[2]))
    ages = [int(c.split("_")[2]) for c in marital_cols]

    def _find_first_marriage(row):
        for age, status in zip(ages, [row[c] for c in marital_cols]):
            if pd.isna(status):
                continue

            status = str(status).lower()
            if status == "married":      # 第一次出现 married
                return age

        return np.nan  # never married

    return df.apply(_find_first_marriage, axis=1)

def compute_ever_divorced(df, marital_prefix="marital_status_"):
    """
    Compute whether an individual has ever been divorced or widowed.
    Returns:
        "ever divorced" if ANY year has marital_status in 
            ['divorced', 'widowed', 'divorced or widowed']
        else "never divorced"
    """

    allowed = {"divorced", "widowed", "divorced or widowed"}

    # Identify marital status columns
    marital_cols = [c for c in df.columns if c.startswith(marital_prefix)]
    if not marital_cols:
        print(f"[postprocess_utils] No columns found with prefix '{marital_prefix}'.")
        return pd.Series([np.nan] * len(df), index=df.index)

    # Sort columns numerically
    marital_cols = sorted(marital_cols, key=lambda c: int(c.split("_")[2]))

    def _ever_divorced(row):
        for raw_status in [row[c] for c in marital_cols]:
            if pd.isna(raw_status):
                continue
            status = str(raw_status).strip().lower()
            if status in allowed:
                return "ever divorced"
        return "never divorced"

    return df.apply(_ever_divorced, axis=1)

def compute_age_started_work(df, work_prefix="employment_"):
    """
    Compute the age at which an individual first becomes employed
    (first year where employment_status == 'employed', case-insensitive).
    """

    # Identify employment status columns
    work_cols = [c for c in df.columns if c.startswith(work_prefix)]
    if not work_cols:
        print(f"[postprocess_utils] No columns found with prefix '{work_prefix}'.")
        return pd.Series([np.nan] * len(df), index=df.index)

    # Sort columns numerically by age
    work_cols = sorted(work_cols, key=lambda c: int(c.split("_")[1]))
    ages = [int(c.split("_")[1]) for c in work_cols]

    def _find_started_work(row):
        for age, status in zip(ages, [row[c] for c in work_cols]):
            if pd.isna(status):
                continue

            status = str(status).lower()
            if status == "employed":    # 第一次出现 employed
                return age

        return np.nan  # never employed

    return df.apply(_find_started_work, axis=1)

def compute_age_at_first_child(df, child_prefix="children_number_"):
    """
    Compute the age at first childbirth (first age where child count >= 1).
    """

    child_cols = [c for c in df.columns if c.startswith(child_prefix)]
    if not child_cols:
        print(f"[postprocess_utils] No columns found with prefix '{child_prefix}'.")
        return pd.Series([np.nan] * len(df), index=df.index)

    child_cols = sorted(child_cols, key=lambda c: int(c.split("_")[2]))
    ages = [int(c.split("_")[2]) for c in child_cols]

    def _find_first_child(row):
        for age, raw in zip(ages, [row[c] for c in child_cols]):

            # --- Convert to numeric safely ---
            try:
                num = float(raw)
            except:
                continue  # skip invalid strings

            if pd.isna(num):
                continue

            if num >= 1:
                return age

        return np.nan  # never had a child

    return df.apply(_find_first_child, axis=1)

def compute_child_number(df, child_prefix="children_number_"):
    child_cols = sorted(
        [c for c in df.columns if c.startswith(child_prefix)],
        key=lambda c: int(c.split("_")[2])
    )
    def _last_valid(row):
        for c in reversed(child_cols):
            if pd.notna(row[c]):
                return row[c]
        return np.nan
    return df.apply(_last_valid, axis=1)


def compute_occupation_30_40(df, occ_prefix="occupation_"):
    """
    Compute the most frequent occupation between ages 30 and 40 (inclusive).

    Parameters
    ----------
    df : pandas.DataFrame
        Contains yearly occupation columns such as:
        'occupation_14', 'occupation_15', ..., 'occupation_60'.
    occ_prefix : str, optional
        Prefix of occupation trajectory columns (default: 'occupation_').

    Returns
    -------
    pandas.Series
        Most frequent occupation during ages 30-40. NaN if no valid data.
    """

    # Identify occupation trajectory columns
    occ_cols = [c for c in df.columns if c.startswith(occ_prefix)]
    if not occ_cols:
        print(f"[postprocess_utils] No columns found with prefix '{occ_prefix}'.")
        return pd.Series([np.nan] * len(df), index=df.index)

    # Sort numerically by age
    occ_cols = sorted(occ_cols, key=lambda c: int(c.split("_")[1]))
    ages = [int(c.split("_")[1]) for c in occ_cols]

    # Select only ages 30–40
    selected = [(age, col) for age, col in zip(ages, occ_cols) if 30 <= age <= 40]

    if not selected:
        return pd.Series([np.nan] * len(df), index=df.index)

    selected_cols = [col for age, col in selected]

    def _mode_occupation(row):
        vals = [row[c] for c in selected_cols if pd.notna(row[c])]
        if not vals:
            return np.nan
        # mode: most frequent
        return pd.Series(vals).mode().iloc[0]

    return df.apply(_mode_occupation, axis=1)

def compute_age_at_first_sex(df, prefix="ever_sex_sequential_"):
    """
    Compute age at first sex.
    From columns like ever_sex_sequential_14, ever_sex_sequential_15, ...
    Find the smallest age where value == "ever sex".
    """

    # 找 sequential 性行为列
    sex_cols = [c for c in df.columns if c.startswith(prefix)]
    if not sex_cols:
        print(f"[postprocess_utils] No columns found with prefix '{prefix}'.")
        return pd.Series([np.nan] * len(df), index=df.index)

    # 从列名提取 age
    sex_cols = sorted(sex_cols, key=lambda c: int(c.split("_")[-1]))
    ages = [int(c.split("_")[-1]) for c in sex_cols]

    def _first_sex(row):
        for col, age in zip(sex_cols, ages):
            val = row[col]
            if pd.isna(val):
                continue
            val = str(val).lower().strip()
            if val == "ever sex":      # 若你还有 "yes" 等，也可加进来
                return age
        return np.nan

    return df.apply(_first_sex, axis=1)
def compute_mean_income_30_40(df, income_prefix="income_"):
    """
    Compute the average income between ages 30 and 40 (inclusive),
    ignoring NaN values.

    Parameters
    ----------
    df : pandas.DataFrame
        Contains yearly income columns such as:
        'income_14', 'income_15', ..., 'income_60'.
    income_prefix : str, optional
        Prefix of income trajectory columns (default: 'income_').

    Returns
    -------
    pandas.Series
        Mean income during ages 30-40. NaN if no valid values.
    """

    # Identify income columns
    income_cols = [c for c in df.columns if c.startswith(income_prefix)]
    if not income_cols:
        print(f"[postprocess_utils] No columns found with prefix '{income_prefix}'.")
        return pd.Series([np.nan] * len(df), index=df.index)

    # Sort numerically by age
    income_cols = sorted(income_cols, key=lambda c: int(c.split("_")[1]))
    ages = [int(c.split("_")[1]) for c in income_cols]

    # Select only ages 30–40
    selected = [(age, col) for age, col in zip(ages, income_cols) if 30 <= age <= 40]
    if not selected:
        return pd.Series([np.nan] * len(df), index=df.index)

    selected_cols = [col for age, col in selected]

    def _mean_income(row):
        vals = [row[c] for c in selected_cols if pd.notna(row[c])]
        if not vals:
            return np.nan  # all missing
        return float(np.mean(vals))

    return df.apply(_mean_income, axis=1)

def compute_highest_education(df, educ_prefix="education_"):
    """
    Compute the highest education level, defined as the last non-missing
    education value in the trajectory.

    Parameters
    ----------
    df : pandas.DataFrame
        Contains yearly education states such as:
        'educ_14', 'educ_15', ..., 'educ_60'.
    educ_prefix : str, optional
        Prefix of education trajectory columns (default: 'educ_').

    Returns
    -------
    pandas.Series
        Last observed (non-NaN) education level. NaN if all missing.
    """

    # Identify education columns
    educ_cols = [c for c in df.columns if c.startswith(educ_prefix)]
    if not educ_cols:
        print(f"[postprocess_utils] No columns found with prefix '{educ_prefix}'.")
        return pd.Series([np.nan] * len(df), index=df.index)

    # Sort by age
    educ_cols = sorted(educ_cols, key=lambda c: int(c.split("_")[1]))

    def _last_education(row):
        # reverse order to find last valid
        for col in reversed(educ_cols):
            val = row[col]
            if pd.notna(val):
                return val
        return np.nan  # all missing

    return df.apply(_last_education, axis=1)