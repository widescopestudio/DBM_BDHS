# -*- coding: utf-8 -*-

!pip install pyreadstat --quiet

import os, io, zipfile, warnings, glob
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pyreadstat
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW
from scipy import stats
from scipy.stats import chi2_contingency, linregress

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

RED, BLUE, GREEN  = "#C0392B", "#2471A3", "#27AE60"
ORANGE, PURPLE    = "#E67E22", "#7D3C98"
GREY, BLACK       = "#7F8C8D", "#1a1a1a"
WAVE_COLORS       = {2011:"#2E86AB", 2014:"#A23B72", 2018:"#F18F01", 2022:"#C73E1D"}

DIVISION_MAP = {1:"Barishal", 2:"Chattogram", 3:"Dhaka",    4:"Khulna",
                5:"Mymensingh", 6:"Rajshahi", 7:"Rangpur",  8:"Sylhet"}
WEALTH_MAP   = {1:"Poorest", 2:"Poorer", 3:"Middle", 4:"Richer", 5:"Richest"}
CVI_COMPONENTS = ["dep_water","dep_sanit","dep_floor",
                  "dep_wall","dep_roof","dep_elec","dep_crowd"]

SEP  = "─" * 75
SEP2 = "═" * 75

print("✓ Imports complete.")


def find_uploaded_zip():
    search_paths = [
        "/content/*.zip",
        "/content/drive/MyDrive/*.zip",
        "/content/drive/My Drive/*.zip",
        "./*.zip",
    ]
    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "Could not find the zip file.\n"
        "Please upload 'all_recodes.zip' using the Files panel (folder icon "
        "on the left sidebar) and make sure it is in /content/."
    )


def extract_all_waves(outer_zip_path: str, extract_dir: str = "/content/bdhs_data") -> dict:
    os.makedirs(extract_dir, exist_ok=True)
    print(f"  Outer zip  : {outer_zip_path}")

    WAVE_ID = {
        "2011": 2011,
        "2014": 2014,
        "2017": 2018,
        "2022": 2022,
    }
    RECODE_PREFIX = {
        2011: ("BDIR61FL", "BDHR61FL"),
        2014: ("BDIR72FL", "BDHR72FL"),
        2018: ("BDIR7RFL", "BDHR7RFL"),
        2022: ("BDIR81FL", "BDHR81FL"),
    }

    wave_paths = {}

    with zipfile.ZipFile(outer_zip_path, "r") as outer:
        inner_zips = [n for n in outer.namelist() if n.lower().endswith(".zip")]
        print(f"  Wave zips found inside: {inner_zips}")

        for inner_name in inner_zips:
            year = None
            for tag, yr in WAVE_ID.items():
                if tag in inner_name:
                    year = yr
                    break
            if year is None:
                print(f"  [SKIP] Could not identify year for: {inner_name}")
                continue

            ir_prefix, hr_prefix = RECODE_PREFIX[year]
            wave_dir = os.path.join(extract_dir, str(year))
            os.makedirs(wave_dir, exist_ok=True)

            inner_bytes = outer.read(inner_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                extracted = {"ir": None, "hr": None}
                for member in inner.namelist():
                    fname = os.path.basename(member).upper()
                    if fname == f"{ir_prefix}.SAV":
                        dest = os.path.join(wave_dir, fname)
                        if not os.path.exists(dest):
                            inner.extract(member, wave_dir)
                            extracted_at = os.path.join(wave_dir, member)
                            if extracted_at != dest:
                                os.rename(extracted_at, dest)
                        extracted["ir"] = os.path.join(wave_dir, fname)
                    elif fname == f"{hr_prefix}.SAV":
                        dest = os.path.join(wave_dir, fname)
                        if not os.path.exists(dest):
                            inner.extract(member, wave_dir)
                            extracted_at = os.path.join(wave_dir, member)
                            if extracted_at != dest:
                                os.rename(extracted_at, dest)
                        extracted["hr"] = os.path.join(wave_dir, fname)

                for root, _, files in os.walk(wave_dir):
                    for f in files:
                        fu = f.upper()
                        src = os.path.join(root, f)
                        dst = os.path.join(wave_dir, fu)
                        if fu in (f"{ir_prefix}.SAV", f"{hr_prefix}.SAV") and src != dst:
                            os.rename(src, dst)
                        if fu == f"{ir_prefix}.SAV":
                            extracted["ir"] = dst
                        if fu == f"{hr_prefix}.SAV":
                            extracted["hr"] = dst

            if extracted["ir"] and extracted["hr"]:
                wave_paths[year] = extracted
                print(f"  ✓ {year}: IR={os.path.basename(extracted['ir'])}  "
                      f"HR={os.path.basename(extracted['hr'])}")
            else:
                print(f"  [WARN] {year}: missing IR or HR — IR={extracted['ir']}  HR={extracted['hr']}")

    return wave_paths


print(SEP2)
print("STEP 1 — Locating and extracting BDHS wave files")
print(SEP2)

ZIP_PATH   = find_uploaded_zip()
WAVE_PATHS = extract_all_waves(ZIP_PATH)
print(f"\n  Waves ready: {sorted(WAVE_PATHS.keys())}")


def wpct(mask, wts):
    return int(mask.sum()), wts[mask].sum() / wts.sum() * 100

def wmean_sd(series, wts):
    valid = series.notna()
    if valid.sum() == 0:
        return np.nan, np.nan
    d = DescrStatsW(series[valid], weights=wts[valid], ddof=1)
    return d.mean, d.std

def weighted_chi2_stat(col_cat, col_outcome, wts):
    wct = pd.crosstab(col_cat, col_outcome, values=wts, aggfunc="sum").fillna(0)
    chi2, p, dof, _ = chi2_contingency(wct)
    return chi2, p, dof

def sig_star(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def erreygers_ci(y, rank, weights=None, n_boot=1000, seed=42):
    y    = np.asarray(y,    dtype=float)
    rank = np.asarray(rank, dtype=float)
    if weights is None:
        weights = np.ones_like(y)
    weights = np.asarray(weights, dtype=float)
    valid   = ~(np.isnan(y) | np.isnan(rank) | np.isnan(weights))
    y, rank, weights = y[valid], rank[valid], weights[valid]
    w        = weights / weights.sum()
    idx_sort = np.argsort(rank, kind="stable")
    y_s, w_s = y[idx_sort], w[idx_sort]
    rank_s   = rank[idx_sort]
    cum_w    = np.cumsum(w_s)
    frac_r   = cum_w - w_s / 2
    mu       = np.sum(w_s * y_s)
    cov_num  = np.sum(w_s * y_s * frac_r) - mu * np.sum(w_s * frac_r)
    mean_r   = np.sum(w_s * frac_r)
    var_r    = np.sum(w_s * (frac_r - mean_r) ** 2)
    CI       = cov_num / var_r / mu * 2
    ECI      = 4 * mu * CI
    rng      = np.random.default_rng(seed)
    n        = len(y_s)
    eci_b    = np.empty(n_boot)
    for b in range(n_boot):
        i      = rng.integers(0, n, size=n)
        yb, wb = y_s[i], w_s[i]
        wb     = wb / wb.sum()
        cwb    = np.cumsum(wb)
        frb    = cwb - wb / 2
        mu_b   = np.sum(wb * yb)
        if mu_b <= 0: eci_b[b] = np.nan; continue
        cb     = np.sum(wb * yb * frb) - mu_b * np.sum(wb * frb)
        vb     = np.sum(wb * (frb - np.sum(wb * frb)) ** 2)
        if vb  <= 0: eci_b[b] = np.nan; continue
        eci_b[b] = 4 * mu_b * (cb / vb / mu_b * 2)
    se   = np.nanstd(eci_b)
    p_v  = 2 * (1 - stats.norm.cdf(abs(ECI / se))) if se > 0 else np.nan
    return {"mu": mu, "CI": CI, "ECI": ECI, "se_ECI": se, "p_ECI": p_v,
            "ECI_lo": ECI - 1.96 * se, "ECI_hi": ECI + 1.96 * se}

def run_logit(outcome_col, df_in, reg_vars, label, weights_col="wt"):
    sub = df_in[[outcome_col, weights_col] + reg_vars].dropna().copy()
    if len(sub) < 50:
        print(f"  [SKIP] {label}: too few rows ({len(sub)})")
        return {}
    X = sm.add_constant(sub[reg_vars].astype(float))
    y = sub[outcome_col].astype(float)
    w = sub[weights_col].astype(float)
    w_sc = w / w.mean()
    try:
        model = sm.Logit(y, X, freq_weights=w_sc).fit(disp=0, maxiter=300)
        params, ci, pv = model.params, model.conf_int(), model.pvalues
        results = {var: {"OR":  np.exp(params[var]),
                         "CI_lo": np.exp(ci.loc[var, 0]),
                         "CI_hi": np.exp(ci.loc[var, 1]),
                         "p":   pv[var]}
                   for var in reg_vars if var in params}
        results["_n"]         = len(sub)
        results["_pseudo_r2"] = model.prsquared
        results["_aic"]       = model.aic
        return results
    except Exception as e:
        print(f"  [WARN] {label}: {e}")
        return {}

print("✓ Helper functions defined.")


WAVE_CONFIG = {
    2011: {"label": "BDHS 2011",    "has_mobile": False},
    2014: {"label": "BDHS 2014",    "has_mobile": False},
    2018: {"label": "BDHS 2017-18", "has_mobile": True},
    2022: {"label": "BDHS 2022",    "has_mobile": True},
}

IR_BASE = ["V001","V002","V005","V012","V024","V025","V106","V190",
           "V445","V213","V743A","V743B","V743D","V743F","V169A"]
HR_BASE = ["HV001","HV002","HV201","HV205","HV206",
           "HV213","HV214","HV215","HV009"]

AUT_VARS = ["V743A","V743B","V743D","V743F"]

print(SEP2)
print("STEP 2 — Loading and harmonising all waves")
print(SEP2)

wave_dfs = {}

for year, cfg in sorted(WAVE_CONFIG.items()):
    if year not in WAVE_PATHS:
        print(f"  [SKIP] {year}: files not found")
        continue

    print(f"\n  Loading {cfg['label']} ...")
    ir_path = WAVE_PATHS[year]["ir"]
    hr_path = WAVE_PATHS[year]["hr"]

    ir_load = [v for v in IR_BASE if v != "V169A" or cfg["has_mobile"]]

    ir, _ = pyreadstat.read_sav(ir_path, usecols=ir_load,  apply_value_formats=False)
    hr, _ = pyreadstat.read_sav(hr_path, usecols=HR_BASE,  apply_value_formats=False)
    ir.columns = [c.upper() for c in ir.columns]
    hr.columns = [c.upper() for c in hr.columns]

    df = ir[
        ir["V445"].notna() &
        ir["V445"].between(1000, 6000) &
        (ir["V213"] != 1)
    ].copy()

    df["HV001"] = df["V001"]
    df["HV002"] = df["V002"]
    df = df.merge(hr, on=["HV001","HV002"], how="left")

    df["wt"]  = df["V005"] / 1_000_000

    df["BMI"]              = df["V445"] / 100
    df["underweight"]      = (df["BMI"] < 18.5).astype(int)
    df["normal"]           = (df["BMI"].between(18.5, 24.999)).astype(int)
    df["overweight"]       = (df["BMI"].between(25.0,  29.999)).astype(int)
    df["obese"]            = (df["BMI"] >= 30.0).astype(int)
    df["overweight_obese"] = (df["BMI"] >= 25.0).astype(int)

    df["dep_water"] = (df["HV201"].notna() & (df["HV201"] >= 30)).astype(int)
    df["dep_sanit"] = df["HV205"].isin({31, 42, 43, 96}).astype(int)
    df["dep_floor"] = df["HV213"].isin({11, 12}).astype(int)
    df["dep_wall"]  = df["HV214"].isin({11,12,13,21,22,23,24,25,26}).astype(int)
    df["dep_roof"]  = df["HV215"].isin({11, 12, 13}).astype(int)
    df["dep_elec"]  = (df["HV206"] == 0).astype(int)
    df["dep_crowd"] = (df["HV009"] >= 6).astype(int)
    df["CVI"]       = df[CVI_COMPONENTS].sum(axis=1)
    df["CVI_cat"]   = pd.cut(df["CVI"], bins=[-1, 0, 1, 7],
                              labels=["Low (CVI=0)","Medium (CVI=1)","High (CVI≥2)"])

    df["age"]      = df["V012"]
    df["wealth"]   = df["V190"]
    df["edu"]      = df["V106"]
    df["urban"]    = (df["V025"] == 1).astype(int)
    df["division"] = df["V024"].map(DIVISION_MAP)
    df["autonomy"] = df[AUT_VARS].apply(
        lambda r: sum(1 for v in AUT_VARS if pd.notna(r[v]) and r[v] in {1.0, 2.0}),
        axis=1
    )
    df["mobile"] = (df["V169A"] == 1).astype(int) if cfg["has_mobile"] else np.nan

    df["wave"]       = year
    df["wave_label"] = cfg["label"]

    must_have = CVI_COMPONENTS + ["BMI","wt","age","wealth","edu","urban","autonomy"]
    df = df.dropna(subset=must_have)

    wave_dfs[year] = df
    print(f"    ✓  Analytic n = {len(df):,}  |  Weight sum = {df['wt'].sum():,.1f}")

pool = pd.concat(wave_dfs.values(), ignore_index=True)
pool["wave_c"] = pool["wave"] - 2011

for yr in [2014, 2018, 2022]:
    pool[f"wave_{yr}"] = (pool["wave"] == yr).astype(int)

YEARS   = sorted(wave_dfs.keys())
YR_LBLS = [WAVE_CONFIG[y]["label"].replace("BDHS ","") for y in YEARS]

print(f"\n  ✓ POOLED sample: n = {len(pool):,}  |  Waves: {YEARS}")


print(f"\n{SEP2}")
print("TABLE T1 — DBM PREVALENCE TREND ACROSS WAVES")
print(SEP2)

trend_rows = []
hdr = f"  {'Wave':<14} {'n':>7}  {'UW%':>7}  {'OW/OB%':>8}  {'MeanBMI':>9}  {'MeanCVI':>9}  {'CVI=0%':>8}  {'CVI≥2%':>8}"
print(hdr); print("  " + SEP)

for yr in YEARS:
    df  = wave_dfs[yr]
    wts = df["wt"].values
    uw  = (df["underweight"].values * wts).sum() / wts.sum() * 100
    ow  = (df["overweight_obese"].values * wts).sum() / wts.sum() * 100
    bm  = DescrStatsW(df["BMI"], weights=wts, ddof=1).mean
    cv  = DescrStatsW(df["CVI"], weights=wts, ddof=1).mean
    p0  = wts[df["CVI"].values == 0].sum() / wts.sum() * 100
    p2  = wts[df["CVI"].values >= 2].sum() / wts.sum() * 100
    trend_rows.append({"wave":yr,"n":len(df),"UW":uw,"OW":ow,
                        "BMI":bm,"CVI":cv,"CVI0":p0,"CVI2":p2})
    print(f"  {WAVE_CONFIG[yr]['label']:<14} {len(df):>7,}  {uw:>7.1f}  {ow:>8.1f}  "
          f"{bm:>9.2f}  {cv:>9.2f}  {p0:>8.1f}  {p2:>8.1f}")

trend_df = pd.DataFrame(trend_rows)
waves_x  = trend_df["wave"].values
lr_uw = linregress(waves_x, trend_df["UW"].values)
lr_ow = linregress(waves_x, trend_df["OW"].values)
lr_cv = linregress(waves_x, trend_df["CVI"].values)
print(f"\n  Linear trend — UW  : slope={lr_uw.slope:+.3f} pp/yr  R²={lr_uw.rvalue**2:.3f}  p={lr_uw.pvalue:.4f}")
print(f"  Linear trend — OW/OB: slope={lr_ow.slope:+.3f} pp/yr  R²={lr_ow.rvalue**2:.3f}  p={lr_ow.pvalue:.4f}")
print(f"  Linear trend — CVI  : slope={lr_cv.slope:+.4f}/yr     R²={lr_cv.rvalue**2:.3f}  p={lr_cv.pvalue:.4f}")


print(f"\n{SEP2}")
print("TABLE T2 — CVI COMPONENT PREVALENCE BY WAVE (weighted %)")
print(SEP2)

COMP_LABELS = {
    "dep_water": "Unimproved water",
    "dep_sanit": "Unimproved sanitation",
    "dep_floor": "Earthen/sand floor",
    "dep_wall":  "Vulnerable wall",
    "dep_roof":  "Vulnerable roof",
    "dep_elec":  "No electricity",
    "dep_crowd": "Overcrowding (≥6 members)",
}

yr_hdrs = "  ".join([f"{y:>8}" for y in YEARS])
print(f"\n  {'Component':<30} {yr_hdrs}  Trend p")
print("  " + SEP)

cvi_comp_data = {}
for comp, lbl in COMP_LABELS.items():
    vals = []
    for yr in YEARS:
        df  = wave_dfs[yr]; wts = df["wt"].values
        vals.append((df[comp].values * wts).sum() / wts.sum() * 100)
    cvi_comp_data[comp] = vals
    lr    = linregress(YEARS, vals)
    pstr  = "p<0.001" if lr.pvalue < 0.001 else f"p={lr.pvalue:.3f}"
    vstr  = "  ".join([f"{v:>7.1f}%" for v in vals])
    print(f"  {lbl:<30} {vstr}  {pstr}")


print(f"\n{SEP2}")
print("TABLE T3 — CVI GRADIENT BY WAVE")
print(SEP2)

CVI_CATS = ["Low (CVI=0)", "Medium (CVI=1)", "High (CVI≥2)"]
wave_cvi_rows = []

for yr in YEARS:
    df  = wave_dfs[yr]; wts = df["wt"].values
    for cat in CVI_CATS:
        mask   = (df["CVI_cat"] == cat).values
        sub_wt = wts * mask
        if sub_wt.sum() == 0: continue
        uw = (df["underweight"].values * sub_wt).sum() / sub_wt.sum() * 100
        ow = (df["overweight_obese"].values * sub_wt).sum() / sub_wt.sum() * 100
        wave_cvi_rows.append({"wave":yr,"CVI_cat":cat,"n":mask.sum(),"UW":uw,"OW":ow})

wave_cvi_df = pd.DataFrame(wave_cvi_rows)
print(f"\n  {'Wave':<14} {'CVI Category':<22} {'n':>6}  {'UW%':>7}  {'OW/OB%':>9}")
print("  " + SEP)
for yr in YEARS:
    sub = wave_cvi_df[wave_cvi_df["wave"]==yr]
    for _, r in sub.iterrows():
        print(f"  {WAVE_CONFIG[yr]['label']:<14} {r['CVI_cat']:<22} "
              f"{r['n']:>6,}  {r['UW']:>7.1f}  {r['OW']:>9.1f}")
    df = wave_dfs[yr]
    chi_uw, p_uw, _ = weighted_chi2_stat(df["CVI_cat"], df["underweight"],       df["wt"].values)
    chi_ow, p_ow, _ = weighted_chi2_stat(df["CVI_cat"], df["overweight_obese"],  df["wt"].values)
    print(f"    χ²(UW)={chi_uw:.2f} {'p<0.001' if p_uw<0.001 else f'p={p_uw:.4f}'}  |  "
          f"χ²(OW/OB)={chi_ow:.2f} {'p<0.001' if p_ow<0.001 else f'p={p_ow:.4f}'}")
    print()


print(f"\n{SEP2}")
print("TABLE T4 — WAVE-SPECIFIC WEIGHTED LOGISTIC REGRESSION (CVI effect)")
print(SEP2)

REG_BASE   = ["CVI","age","wealth","edu","urban","autonomy"]
REG_MOBILE = ["CVI","age","wealth","edu","urban","mobile","autonomy"]

wave_reg_results = {}
print(f"\n  {'Wave':<14} {'Outcome':<10} {'CVI aOR':>8}  {'95% CI':>18}  {'p':>8}  Sig")
print("  " + SEP)

for yr in YEARS:
    df    = wave_dfs[yr].copy()
    rvars = REG_MOBILE if WAVE_CONFIG[yr]["has_mobile"] else REG_BASE
    r_ow  = run_logit("overweight_obese", df, rvars, f"{yr} OW")
    r_uw  = run_logit("underweight",      df, rvars, f"{yr} UW")
    wave_reg_results[yr] = {"ow": r_ow, "uw": r_uw}
    for tag, res in [("OW/OB", r_ow), ("UW", r_uw)]:
        if "CVI" in res:
            c = res["CVI"]
            print(f"  {WAVE_CONFIG[yr]['label']:<14} {tag:<10} {c['OR']:>8.3f}  "
                  f"({c['CI_lo']:.3f}–{c['CI_hi']:.3f})  {c['p']:>8.4f}  {sig_star(c['p'])}")


print(f"\n{SEP2}")
print("TABLE T5 — POOLED WEIGHTED LOGISTIC REGRESSION (wave fixed effects, n=61,006)")
print(SEP2)

POOL_VARS = ["CVI","age","wealth","edu","urban","autonomy",
             "wave_2014","wave_2018","wave_2022"]
POOL_LBLS = {
    "CVI":       "CVI (continuous)",
    "age":       "Age (per year)",
    "wealth":    "Wealth quintile",
    "edu":       "Education level",
    "urban":     "Urban residence",
    "autonomy":  "Autonomy score",
    "wave_2014": "Wave 2014 (ref=2011)",
    "wave_2018": "Wave 2017-18",
    "wave_2022": "Wave 2022",
}

pool_c      = pool[POOL_VARS + ["overweight_obese","underweight","wt"]].dropna()
res_pool_ow = run_logit("overweight_obese", pool_c, POOL_VARS, "Pooled OW")
res_pool_uw = run_logit("underweight",      pool_c, POOL_VARS, "Pooled UW")

for tag, res in [("Overweight/Obese", res_pool_ow), ("Underweight", res_pool_uw)]:
    print(f"\n  Outcome: {tag}  (n={res.get('_n',0):,}  Pseudo-R²={res.get('_pseudo_r2',np.nan):.4f})")
    print(f"  {'Variable':<30} {'aOR':>7}  {'95% CI':>18}  {'p':>8}  Sig")
    print("  " + SEP)
    for var in POOL_VARS:
        if var in res:
            c = res[var]
            print(f"  {POOL_LBLS.get(var,var):<30} {c['OR']:>7.3f}  "
                  f"({c['CI_lo']:.3f}–{c['CI_hi']:.3f})  {c['p']:>8.4f}  {sig_star(c['p'])}")


print(f"\n{SEP2}")
print("TABLE T6 — CVI × WAVE INTERACTION (gradient steepening over time)")
print(SEP2)

pool["CVI_x_wave"] = pool["CVI"] * pool["wave_c"]
INT_VARS = ["CVI","wave_c","CVI_x_wave","age","wealth","edu","urban","autonomy"]
INT_LBLS = {
    "CVI":        "CVI (main effect)",
    "wave_c":     "Wave (centred, per 3-yr step)",
    "CVI_x_wave": "CVI × Wave interaction",
    "age":        "Age", "wealth":"Wealth quintile",
    "edu":        "Education", "urban":"Urban", "autonomy":"Autonomy",
}

int_c       = pool[INT_VARS + ["overweight_obese","underweight","wt"]].dropna()
res_int_ow  = run_logit("overweight_obese", int_c, INT_VARS, "Int OW")
res_int_uw  = run_logit("underweight",      int_c, INT_VARS, "Int UW")

for tag, res in [("Overweight/Obese", res_int_ow), ("Underweight", res_int_uw)]:
    print(f"\n  Outcome: {tag}  (n={res.get('_n',0):,})")
    print(f"  {'Variable':<34} {'aOR':>7}  {'95% CI':>18}  {'p':>8}  Sig")
    print("  " + SEP)
    for var in INT_VARS:
        if var in res:
            c = res[var]
            print(f"  {INT_LBLS.get(var,var):<34} {c['OR']:>7.3f}  "
                  f"({c['CI_lo']:.3f}–{c['CI_hi']:.3f})  {c['p']:>8.4f}  {sig_star(c['p'])}")


print(f"\n{SEP2}")
print("TABLE T7 — ERREYGERS CORRECTED CONCENTRATION INDEX TREND (2011–2022)")
print(SEP2)

eci_trend = []
print(f"\n  {'Wave':<14} {'Outcome':<10} {'Wtd mean':>9}  {'ECI':>8}  {'95% CI':>22}  Sig")
print("  " + SEP)

for yr in YEARS:
    df = wave_dfs[yr]
    for col, lbl in [("overweight_obese","OW/OB"), ("underweight","UW")]:
        r = erreygers_ci(df[col], df["wealth"], df["wt"])
        eci_trend.append({"wave":yr,"outcome":lbl,
                           "mu":r["mu"],"ECI":r["ECI"],
                           "ECI_lo":r["ECI_lo"],"ECI_hi":r["ECI_hi"],"p":r["p_ECI"]})
        print(f"  {WAVE_CONFIG[yr]['label']:<14} {lbl:<10} {r['mu']:>9.3f}  "
              f"{r['ECI']:>+8.4f}  ({r['ECI_lo']:+.4f} to {r['ECI_hi']:+.4f})  "
              f"{sig_star(r['p_ECI'])}")

eci_df = pd.DataFrame(eci_trend)
for lbl in ["OW/OB", "UW"]:
    sub = eci_df[eci_df["outcome"] == lbl]
    lr  = linregress(sub["wave"].values, sub["ECI"].values)
    print(f"\n  ECI trend ({lbl}): slope={lr.slope:+.4f}/yr  p={lr.pvalue:.4f}")


print(f"\n{SEP2}")
print("TABLE T8 — DIVISION-LEVEL PANEL")
print(SEP2)

div_panel = []
for yr in YEARS:
    df = wave_dfs[yr]
    for div in sorted(DIVISION_MAP.values()):
        sub = df[df["division"] == div]
        if len(sub) < 20: continue
        ws  = sub["wt"].values
        div_panel.append({
            "wave":yr,"division":div,"n":len(sub),
            "UW": (sub["underweight"].values * ws).sum() / ws.sum() * 100,
            "OW": (sub["overweight_obese"].values * ws).sum() / ws.sum() * 100,
            "mean_CVI": DescrStatsW(sub["CVI"], weights=ws, ddof=1).mean,
        })

div_df = pd.DataFrame(div_panel)
hdrs   = "  ".join([f"{y:>6}" for y in YEARS])
print(f"\n  {'Division':<14} {hdrs}  (UW%)   ||  {hdrs}  (OW%)")
print("  " + "─" * 90)
for div in sorted(DIVISION_MAP.values()):
    sub    = div_df[div_df["division"] == div]
    uw_str = "  ".join([f"{sub[sub['wave']==y]['UW'].values[0]:>5.1f}" if len(sub[sub['wave']==y])>0 else "    —" for y in YEARS])
    ow_str = "  ".join([f"{sub[sub['wave']==y]['OW'].values[0]:>5.1f}" if len(sub[sub['wave']==y])>0 else "    —" for y in YEARS])
    print(f"  {div:<14} {uw_str}          {ow_str}")

print(f"\n{SEP2}")
print("TABLE T9 — WEALTH-STRATIFIED PANEL")
print(SEP2)
print(f"\n  {'Wealth':<12} {hdrs}  (UW%)   ||  {hdrs}  (OW%)")
print("  " + "─" * 90)
for k in range(1, 6):
    uw_vals, ow_vals = [], []
    for yr in YEARS:
        df = wave_dfs[yr]; sub = df[df["wealth"] == k]; ws = sub["wt"].values
        uw_vals.append((sub["underweight"].values * ws).sum() / ws.sum() * 100 if len(sub) > 0 else np.nan)
        ow_vals.append((sub["overweight_obese"].values * ws).sum() / ws.sum() * 100 if len(sub) > 0 else np.nan)
    uw_s = "  ".join([f"{v:>5.1f}" if not np.isnan(v) else "    —" for v in uw_vals])
    ow_s = "  ".join([f"{v:>5.1f}" if not np.isnan(v) else "    —" for v in ow_vals])
    print(f"  {WEALTH_MAP[k]:<12} {uw_s}          {ow_s}")


print(f"\n{SEP2}")
print("TABLE T10 — SENSITIVITY ANALYSES")
print(SEP2)

sens_configs = [
    ("Pooled CVI≥1",  pool[pool["CVI"] >= 1].copy()),
    ("Urban only",    pool[pool["urban"] == 1].copy()),
    ("2022 wave only",wave_dfs[2022].copy()),
]

for label, df_s in sens_configs:
    print(f"\n  ({label}  n={len(df_s):,})")
    for tag, out in [("OW/OB","overweight_obese"), ("UW","underweight")]:
        res = run_logit(out, df_s, POOL_VARS if "Pooled" in label or "Urban" in label
                        else REG_MOBILE, f"{label} {tag}")
        if "CVI" in res:
            c = res["CVI"]
            print(f"    CVI aOR ({tag}): {c['OR']:.3f} ({c['CI_lo']:.3f}–{c['CI_hi']:.3f}) "
                  f"p={c['p']:.4f} {sig_star(c['p'])}")


print(f"\n{SEP2}")
print("TABLE T11 — AUTONOMY × CVI INTERACTION (pooled)")
print(SEP2)

pool["CVI_x_aut"] = pool["CVI"] * pool["autonomy"]
AUT_VARS2 = ["CVI","autonomy","CVI_x_aut","age","wealth","edu","urban",
             "wave_2014","wave_2018","wave_2022"]
aut_c    = pool[AUT_VARS2 + ["overweight_obese","underweight","wt"]].dropna()
r_aut_ow = run_logit("overweight_obese", aut_c, AUT_VARS2, "Aut OW")
r_aut_uw = run_logit("underweight",      aut_c, AUT_VARS2, "Aut UW")

for tag, res in [("OW/OB", r_aut_ow), ("UW", r_aut_uw)]:
    for var in ["CVI","autonomy","CVI_x_aut"]:
        if var in res:
            c = res[var]
            print(f"  {var:<20} ({tag}): aOR={c['OR']:.3f} ({c['CI_lo']:.3f}–{c['CI_hi']:.3f}) "
                  f"p={c['p']:.4f} {sig_star(c['p'])}")


print(f"\n{SEP2}")
print("GENERATING 12-PANEL COMPOSITE FIGURE")
print(SEP2)

fig = plt.figure(figsize=(24, 20), facecolor="white")
gs  = GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38)

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("white")
    ax.tick_params(colors=BLACK, labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#cccccc")
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8, color=BLACK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=BLACK)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=BLACK)
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.7)
    ax.set_axisbelow(True)

ax = fig.add_subplot(gs[0, 0])
uw_prev = trend_df["UW"].values
ow_prev = trend_df["OW"].values
ax.plot(YEARS, ow_prev, "o-",  color=RED,  lw=2.5, ms=7, label="Overweight/Obese")
ax.plot(YEARS, uw_prev, "s--", color=BLUE, lw=2.5, ms=7, label="Underweight")
for x, y in zip(YEARS, ow_prev): ax.annotate(f"{y:.1f}%", (x,y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=RED)
for x, y in zip(YEARS, uw_prev): ax.annotate(f"{y:.1f}%", (x,y), xytext=(0,-14), textcoords="offset points", ha="center", fontsize=8, color=BLUE)
ax.set_xticks(YEARS); ax.set_xticklabels(YR_LBLS, fontsize=8)
ax.set_ylim(0, 50); ax.legend(fontsize=8, loc="center left")
style_ax(ax, "(A) DBM Prevalence Trend 2011–2022", "Survey year", "Weighted prevalence (%)")

ax = fig.add_subplot(gs[0, 1])
cvi0 = trend_df["CVI0"].values
cvi2 = trend_df["CVI2"].values
cvi1 = 100 - cvi0 - cvi2
x4   = np.arange(4)
ax.bar(x4 - 0.25, cvi0, width=0.22, color=GREEN,  alpha=0.85, label="CVI=0 (Low)")
ax.bar(x4,        cvi1, width=0.22, color=ORANGE,  alpha=0.85, label="CVI=1 (Medium)")
ax.bar(x4 + 0.25, cvi2, width=0.22, color=RED,    alpha=0.85, label="CVI≥2 (High)")
ax.set_xticks(x4); ax.set_xticklabels(YR_LBLS, fontsize=8)
ax.set_ylim(0, 75); ax.legend(fontsize=8)
style_ax(ax, "(B) CVI Distribution by Wave", "Survey year", "Weighted % of women")

ax = fig.add_subplot(gs[0, 2])
comp_clrs = [BLUE, "#117A65", RED, ORANGE, PURPLE, GREY, "#922B21"]
for (comp, lbl), clr in zip(COMP_LABELS.items(), comp_clrs):
    ax.plot(YEARS, cvi_comp_data[comp], "o-", color=clr, lw=2, ms=5,
            label=lbl.split()[0], alpha=0.85)
ax.set_xticks(YEARS); ax.set_xticklabels(YR_LBLS, fontsize=8)
ax.set_ylim(0, 80); ax.legend(fontsize=7, ncol=2)
style_ax(ax, "(C) CVI Component Trends", "Survey year", "Weighted % deprived")

ax = fig.add_subplot(gs[1, 0])
x3 = np.arange(3)
for i, yr in enumerate(YEARS):
    sub   = wave_cvi_df[wave_cvi_df["wave"] == yr]
    ow_v  = [sub[sub["CVI_cat"]==c]["OW"].values[0] if len(sub[sub["CVI_cat"]==c])>0 else np.nan for c in CVI_CATS]
    ax.plot(x3, ow_v, "o-", color=list(WAVE_COLORS.values())[i], lw=2.2, ms=6, label=YR_LBLS[i])
ax.set_xticks(x3); ax.set_xticklabels(["Low\n(CVI=0)","Medium\n(CVI=1)","High\n(CVI≥2)"], fontsize=8)
ax.set_ylim(0, 65); ax.legend(fontsize=8)
style_ax(ax, "(D) OW/OB Gradient × CVI × Wave", "CVI Category", "Weighted OW/OB prevalence (%)")

ax = fig.add_subplot(gs[1, 1])
for i, yr in enumerate(YEARS):
    sub   = wave_cvi_df[wave_cvi_df["wave"] == yr]
    uw_v  = [sub[sub["CVI_cat"]==c]["UW"].values[0] if len(sub[sub["CVI_cat"]==c])>0 else np.nan for c in CVI_CATS]
    ax.plot(x3, uw_v, "s--", color=list(WAVE_COLORS.values())[i], lw=2.2, ms=6, label=YR_LBLS[i])
ax.set_xticks(x3); ax.set_xticklabels(["Low\n(CVI=0)","Medium\n(CVI=1)","High\n(CVI≥2)"], fontsize=8)
ax.set_ylim(0, 45); ax.legend(fontsize=8)
style_ax(ax, "(E) Underweight Gradient × CVI × Wave", "CVI Category", "Weighted UW prevalence (%)")

ax = fig.add_subplot(gs[1, 2])
ax.axvline(1.0, color="#aaaaaa", lw=1.2, ls="--")
for i, yr in enumerate(YEARS):
    r_ow = wave_reg_results[yr]["ow"]
    r_uw = wave_reg_results[yr]["uw"]
    if "CVI" in r_ow:
        c = r_ow["CVI"]
        ax.plot([c["CI_lo"],c["CI_hi"]], [i+0.15, i+0.15], color=RED,  lw=2.8)
        ax.plot(c["OR"], i+0.15, "D", color=RED,  ms=7)
    if "CVI" in r_uw:
        c = r_uw["CVI"]
        ax.plot([c["CI_lo"],c["CI_hi"]], [i-0.15, i-0.15], color=BLUE, lw=2.8)
        ax.plot(c["OR"], i-0.15, "s", color=BLUE, ms=7)
ax.set_yticks(np.arange(4)); ax.set_yticklabels(YR_LBLS, fontsize=9)
ax.set_xlabel("Adjusted Odds Ratio (CVI per unit)", fontsize=9, color=BLACK)
ax.legend(handles=[Line2D([0],[0],color=RED, marker="D",ms=6,label="OW/OB"),
                   Line2D([0],[0],color=BLUE,marker="s",ms=6,label="UW",linestyle="--")],
          fontsize=8)
ax.yaxis.grid(False); ax.xaxis.grid(True, color="#eeeeee", lw=0.7)
ax.set_facecolor("white")
for sp in ax.spines.values(): sp.set_edgecolor("#cccccc")
ax.set_title("(F) CVI aOR Forest Plot by Wave", fontsize=10.5, fontweight="bold", color=BLACK)

ax = fig.add_subplot(gs[2, 0])
eci_ow = eci_df[eci_df["outcome"]=="OW/OB"].sort_values("wave")
eci_uw = eci_df[eci_df["outcome"]=="UW"].sort_values("wave")
ax.plot(eci_ow["wave"], eci_ow["ECI"], "o-",  color=RED,  lw=2.5, ms=7, label="OW/OB")
ax.fill_between(eci_ow["wave"], eci_ow["ECI_lo"], eci_ow["ECI_hi"], color=RED,  alpha=0.15)
ax.plot(eci_uw["wave"], eci_uw["ECI"], "s--", color=BLUE, lw=2.5, ms=7, label="Underweight")
ax.fill_between(eci_uw["wave"], eci_uw["ECI_lo"], eci_uw["ECI_hi"], color=BLUE, alpha=0.15)
ax.axhline(0, color="grey", lw=1, ls=":")
ax.set_xticks(YEARS); ax.set_xticklabels(YR_LBLS, fontsize=8); ax.legend(fontsize=8)
style_ax(ax, "(G) ECI Trend 2011–2022", "Survey year", "Erreygers Corrected CI")

ax = fig.add_subplot(gs[2, 1])
divs  = sorted(DIVISION_MAP.values())
x_d   = np.arange(len(divs)); w_bar = 0.18
for i, yr in enumerate(YEARS):
    ow_d = [div_df[(div_df["wave"]==yr)&(div_df["division"]==d)]["OW"].values[0]
             if len(div_df[(div_df["wave"]==yr)&(div_df["division"]==d)])>0 else 0 for d in divs]
    ax.bar(x_d + (i-1.5)*w_bar, ow_d, width=w_bar,
           color=list(WAVE_COLORS.values())[i], alpha=0.85, label=YR_LBLS[i])
ax.set_xticks(x_d); ax.set_xticklabels([d[:3] for d in divs], fontsize=8, rotation=30)
ax.legend(fontsize=7, ncol=2)
style_ax(ax, "(H) OW/OB by Division & Wave", "Division", "Weighted OW/OB prevalence (%)")

ax = fig.add_subplot(gs[2, 2])
for k, wlbl in WEALTH_MAP.items():
    clr  = [BLUE,"#117A65",ORANGE,PURPLE,RED][k-1]
    ow_w = []
    for yr in YEARS:
        df  = wave_dfs[yr]; sub = df[df["wealth"]==k]; ws = sub["wt"].values
        ow_w.append((sub["overweight_obese"].values*ws).sum()/ws.sum()*100 if len(sub)>0 else np.nan)
    ax.plot(YEARS, ow_w, "o-", color=clr, lw=2, ms=5, label=wlbl)
ax.set_xticks(YEARS); ax.set_xticklabels(YR_LBLS, fontsize=8); ax.legend(fontsize=8)
style_ax(ax, "(I) OW/OB by Wealth Quintile & Wave", "Survey year", "Weighted OW/OB (%)")

ax = fig.add_subplot(gs[3, 0])
df22 = wave_dfs[2022].copy().sort_values("wealth")
df22["cum_wt"] = df22["wt"].cumsum() / df22["wt"].sum()
for col, clr, lbl, eci_val in [
    ("overweight_obese", RED,  f"OW/OB (ECI={eci_df[(eci_df['wave']==2022)&(eci_df['outcome']=='OW/OB')]['ECI'].values[0]:+.2f})", None),
    ("underweight",      BLUE, f"UW    (ECI={eci_df[(eci_df['wave']==2022)&(eci_df['outcome']=='UW')]['ECI'].values[0]:+.2f})", None),
]:
    df22["wt_out"]  = df22["wt"] * df22[col]
    df22["cum_out"] = df22["wt_out"].cumsum() / df22["wt_out"].sum()
    ax.plot(df22["cum_wt"], df22["cum_out"], color=clr, lw=2, label=lbl)
ax.plot([0,1],[0,1], "k--", lw=1.2, label="Line of equality")
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(fontsize=8)
ax.set_xlabel("Cumulative wealth rank", fontsize=9, color=BLACK)
ax.set_ylabel("Cumulative nutritional outcome", fontsize=9, color=BLACK)
ax.set_title("(J) Concentration Curves — 2022", fontsize=10.5, fontweight="bold", color=BLACK)
ax.set_facecolor("white")
for sp in ax.spines.values(): sp.set_edgecolor("#cccccc")

ax = fig.add_subplot(gs[3, 1])
fp_vars = ["CVI","wealth","edu","urban","autonomy","wave_2014","wave_2018","wave_2022"]
fp_lbls = ["CVI","Wealth","Education","Urban","Autonomy","2014 vs 2011","2017-18 vs 2011","2022 vs 2011"]
ax.axvline(1.0, color="#aaaaaa", lw=1.2, ls="--")
for i, var in enumerate(fp_vars):
    for res, clr, mkr, off in [(res_pool_ow,RED,"D",0.15),(res_pool_uw,BLUE,"s",-0.15)]:
        if var in res:
            c = res[var]
            ax.plot([c["CI_lo"],c["CI_hi"]], [i+off,i+off], color=clr, lw=2.5)
            ax.plot(c["OR"], i+off, mkr, color=clr, ms=6)
ax.set_yticks(np.arange(len(fp_vars))); ax.set_yticklabels(fp_lbls, fontsize=8)
ax.set_xlabel("Adjusted Odds Ratio", fontsize=9, color=BLACK)
ax.legend(handles=[Line2D([0],[0],color=RED,marker="D",ms=6,label="OW/OB"),
                   Line2D([0],[0],color=BLUE,marker="s",ms=6,label="UW",linestyle="--")],
          fontsize=8)
ax.yaxis.grid(False); ax.xaxis.grid(True, color="#eeeeee", lw=0.7)
ax.set_facecolor("white")
for sp in ax.spines.values(): sp.set_edgecolor("#cccccc")
ax.set_title("(K) Pooled Regression Forest Plot", fontsize=10.5, fontweight="bold", color=BLACK)

ax = fig.add_subplot(gs[3, 2])
ns   = trend_df["n"].values
bars = ax.bar(np.arange(4), ns, color=list(WAVE_COLORS.values()), alpha=0.85, edgecolor="none")
for b, n in zip(bars, ns):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 100,
            f"{n:,}", ha="center", fontsize=9, fontweight="bold", color=BLACK)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(YR_LBLS, fontsize=9)
ax.set_ylim(0, max(ns) * 1.15)
style_ax(ax, "(L) Analytic Sample Size by Wave", "Survey year", "n (non-pregnant women)")

fig.suptitle(
    "Household Deprivation, Climate Vulnerability & Double Burden of Malnutrition\n"
    "Among Women of Reproductive Age — BDHS 2011–2022 [Multi-Wave Analysis]",
    fontsize=13, fontweight="bold", color=BLACK, y=0.998,
)

FIG_OUT = "/content/BDHS_Multiwave_Figure.png"
fig.savefig(FIG_OUT, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\n  ✓ Figure saved to {FIG_OUT}")

try:
    from google.colab import files
    files.download(FIG_OUT)
    print("  ✓ Download triggered.")
except Exception:
    print("  (Not in Colab — figure saved locally.)")

print(f"\n{SEP2}")
print("ALL ANALYSES COMPLETE")
print(SEP2)
