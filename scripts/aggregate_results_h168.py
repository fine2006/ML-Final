#!/usr/bin/env python3
"""
Results Aggregation for h168 (168-hour) Forecasting

Calculates:
1. Skill Score vs Persistence: Skill_Pers = 1 - RMSE_Model / RMSE_Persistence
2. Skill Score vs Climatology: Skill_Clim = 1 - RMSE_Model / RMSE_Climatology
3. Regional Leaderboard: Rank regions by Skill Score
4. Gate Pass Report: Tiered operational gates

Tier Framework:
- Tier 1 (Operational - PM2.5/PM10): RMSE/Mean < 0.5, Coverage ∈ [0.85, 0.95]
- Tier 2 (Rhythm - NO2): Focus on diurnal correlation, peak timing
- Tier 3 (Discovery - O3): Negative R² = feature gap (UV missing)

"Volatility Warning" for Siltara: Reliability map, not just leaderboard

Run: python scripts/aggregate_results_h168.py
Output: models/experiments/aggregated_h168_results.json
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

import argparse
parser = argparse.ArgumentParser(description="Aggregate h168 results")
parser.add_argument("--output", type=str, default=None, help="Output path")
args_cli = parser.parse_args()

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "models" / "experiments"

# Gate thresholds
GATE_RMSE_MEAN_PM = 0.50      # PM2.5/PM10: RMSE/Mean < 0.5
GATE_RMSE_MEAN_GAS = 0.70     # NO2/O3: RMSE/Mean < 0.7
GATE_COVERAGE_MIN = 0.85
GATE_COVERAGE_MAX = 0.95

# ============================================================================
# LOADING RESULTS
# ============================================================================


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_evaluation_summary() -> Dict:
    """Load final evaluation summary."""
    return load_json(OUTPUT_DIR / "final_h168_operational_snapshot_evaluation_summary.json")


def load_wfcv_results() -> Dict:
    """Load WFCV results if available."""
    wfcv_path = OUTPUT_DIR / "wfcv_h168_results.json"
    if wfcv_path.exists():
        return load_json(wfcv_path)
    return {}


def load_optuna_results() -> Dict:
    """Load Optuna results if available."""
    optuna_path = OUTPUT_DIR / "optuna_h168_best_configs.json"
    if optuna_path.exists():
        return load_json(optuna_path)
    return {}


# ============================================================================
# SKILL SCORE CALCULATION
# ============================================================================


def calculate_skill_scores(evaluation_summary: Dict) -> List[Dict]:
    """
    Calculate Skill Score for each pollutant.
    
    Skill_Persistence = 1 - RMSE_Model / RMSE_Persistence
    Skill_Climatology = 1 - RMSE_Model / RMSE_Climatology
    
    Negative skill = model worse than baseline
    """
    fair_comparison = evaluation_summary.get("fair_comparison", [])
    
    skill_results = []
    
    for comparison in fair_comparison:
        pollutant = comparison["pollutant"]
        rmse_model = comparison["lstm_rmse"]
        rmse_persistence = comparison["persistence_rmse_raw_yt"]
        rmse_climatology = comparison["climatology_rmse_month_hour"]
        
        # Skill scores
        skill_pers = 1.0 - (rmse_model / rmse_persistence) if rmse_persistence > 0 else 0.0
        skill_clim = 1.0 - (rmse_model / rmse_climatology) if rmse_climatology > 0 else 0.0
        
        skill_results.append({
            "pollutant": pollutant,
            "rmse_model": rmse_model,
            "rmse_persistence": rmse_persistence,
            "rmse_climatology": rmse_climatology,
            "skill_persistence": skill_pers,
            "skill_climatology": skill_clim,
            "beats_persistence": comparison.get("beats_persistence", False),
            "beats_climatology": comparison.get("beats_climatology", False),
        })
    
    return skill_results


# ============================================================================
# REGIONAL LEADERBOARD
# ============================================================================


def calculate_regional_skill_scores(evaluation_summary: Dict) -> Dict:
    """
    Calculate Skill Score per region.
    
    Returns regional leaderboard ranked by Skill_Climatology.
    """
    lstm_pollutants = evaluation_summary.get("lstm", {}).get("pollutants", {})
    
    regional_results = {}
    
    for pollutant, data in lstm_pollutants.items():
        by_horizon = data.get("by_horizon", {})
        h168_data = by_horizon.get("h168", {})
        by_region = h168_data.get("by_region", {})
        
        # Get persistence RMSE for this pollutant
        fair_comparison = evaluation_summary.get("fair_comparison", [])
        persistence_rmse = None
        climatology_rmse = None
        
        for fc in fair_comparison:
            if fc["pollutant"] == pollutant:
                persistence_rmse = fc["persistence_rmse_raw_yt"]
                climatology_rmse = fc["climatology_rmse_month_hour"]
                break
        
        regional_skill = []
        
        for region, metrics in by_region.items():
            rmse_region = metrics["rmse"]
            
            skill_pers = 1.0 - (rmse_region / persistence_rmse) if persistence_rmse else 0.0
            skill_clim = 1.0 - (rmse_region / climatology_rmse) if climatology_rmse else 0.0
            
            regional_skill.append({
                "region": region,
                "rmse": rmse_region,
                "skill_persistence": skill_pers,
                "skill_climatology": skill_clim,
            })
        
        # Sort by skill_climatology (descending)
        regional_skill.sort(key=lambda x: x["skill_climatology"], reverse=True)
        
        regional_results[pollutant] = regional_skill
    
    return regional_results


# ============================================================================
# GATE PASS REPORT
# ============================================================================


def generate_gate_pass_report(skill_results: List[Dict], 
                              regional_skill: Dict) -> List[Dict]:
    """
    Generate gate pass report with tiered operational gates.
    
    Tier 1 (Operational - PM): Skill_Clim > 0, RMSE/Mean < 0.5, Coverage ∈ [0.85, 0.95]
    Tier 2 (Rhythm - NO2): Skill_Clim > -0.2, Peak timing alignment
    Tier 3 (Discovery - O3): Explain negative R² as feature gap
    """
    # Load evaluation summary for coverage data
    eval_summary = load_evaluation_summary()
    
    gate_report = []
    
    for sr in skill_results:
        pollutant = sr["pollutant"]
        rmse = sr["rmse_model"]
        skill_clim = sr["skill_climatology"]
        
        # Get coverage from evaluation summary
        lstm_pollutants = eval_summary.get("lstm", {}).get("pollutants", {})
        coverage = None
        r2 = None
        
        if pollutant in lstm_pollutants:
            h168_data = lstm_pollutants[pollutant].get("by_horizon", {}).get("h168", {})
            coverage = h168_data.get("coverage_p05_p95")
            r2 = h168_data.get("r2")
        
        # Get mean concentration for RMSE/Mean ratio
        fair_comparison = eval_summary.get("fair_comparison", [])
        mean_conc = None
        
        for fc in fair_comparison:
            if fc["pollutant"] == pollutant:
                mean_conc = fc.get("mean_concentration", rmse)
                break
        
        rmse_mean_ratio = rmse / mean_conc if mean_conc else float("nan")
        
        # Determine tier
        if pollutant in ["pm25", "pm10"]:
            tier = "Operational"
            tier_description = "High-confidence forecasts for air quality alerts"
            
            # Gate checks
            gate_pass = (
                skill_clim > 0 and
                rmse_mean_ratio < GATE_RMSE_MEAN_PM and
                coverage is not None and
                GATE_COVERAGE_MIN <= coverage <= GATE_COVERAGE_MAX
            )
            
            gate_details = {
                "skill_clim_positive": skill_clim > 0,
                "rmse_mean_threshold": rmse_mean_ratio < GATE_RMSE_MEAN_PM,
                "coverage_in_range": coverage is not None and GATE_COVERAGE_MIN <= coverage <= GATE_COVERAGE_MAX,
            }
            
        elif pollutant == "no2":
            tier = "Rhythm"
            tier_description = "Human activity patterns - utility for traffic management"
            
            # Gate checks (more lenient for NO2)
            gate_pass = (
                skill_clim > -0.2 and
                rmse_mean_ratio < GATE_RMSE_MEAN_GAS
            )
            
            gate_details = {
                "skill_clim_above_threshold": skill_clim > -0.2,
                "rmse_mean_threshold": rmse_mean_ratio < GATE_RMSE_MEAN_GAS,
            }
            
        else:  # o3
            tier = "Discovery"
            tier_description = "Photochemical limitation - negative R² indicates feature gap (UV missing)"
            
            # Gate checks (lenient - we expect challenges)
            gate_pass = skill_clim > -0.5  # Allow some negative skill
            
            gate_details = {
                "skill_clim_above_threshold": skill_clim > -0.5,
                "negative_r2_explained": "Missing UV/solar radiation data prevents photochemical modeling",
            }
        
        # Volatility warning for Siltara
        volatility_warning = None
        if pollutant in regional_skill and "SILTARA" in regional_skill[pollutant]:
            siltara_skill = regional_skill[pollutant][0]  # First is best
            if siltara_skill["skill_climatology"] < -0.3:
                volatility_warning = (
                    f"SILTARA shows low reliability (Skill_Clim={siltara_skill['skill_climatology']:.3f}). "
                    "Consider region-specific calibration or sensor quality review."
                )
        
        gate_report.append({
            "pollutant": pollutant,
            "tier": tier,
            "tier_description": tier_description,
            "rmse": rmse,
            "rmse_mean_ratio": rmse_mean_ratio,
            "coverage": coverage,
            "r2": r2,
            "skill_persistence": sr["skill_persistence"],
            "skill_climatology": skill_clim,
            "gate_pass": gate_pass,
            "gate_details": gate_details,
            "volatility_warning": volatility_warning,
        })
    
    return gate_report


# ============================================================================
# NARRATIVE FRAMEWORK
# ============================================================================


def generate_narrative_summary(gate_report: List[Dict]) -> str:
    """
    Generate narrative summary with Honest Failure framework.
    
    - PM2.5/PM10 = Operational Tier (high-confidence forecasts)
    - NO2 = Rhythm Tier (human activity match)
    - O3 = Discovery Tier (photochemical limitation explained)
    """
    narrative = []
    
    for gr in gate_report:
        pollutant = gr["pollutant"].upper()
        tier = gr["tier"]
        
        narrative.append(f"\n{'=' * 80}")
        narrative.append(f"{pollutant} - {tier} Tier")
        narrative.append(f"{'=' * 80}")
        narrative.append(gr["tier_description"])
        narrative.append("")
        
        narrative.append("Performance:")
        narrative.append(f"  RMSE: {gr['rmse']:.4f}")
        narrative.append(f"  RMSE/Mean: {gr['rmse_mean_ratio']:.4f}")
        if gr["coverage"]:
            narrative.append(f"  Coverage (p05-p95): {gr['coverage']:.4f}")
        if gr["r2"]:
            narrative.append(f"  R²: {gr['r2']:.4f}")
        narrative.append(f"  Skill vs Climatology: {gr['skill_climatology']:.4f}")
        narrative.append(f"  Gate Pass: {'✓ YES' if gr['gate_pass'] else '✗ NO'}")
        narrative.append("")
        
        if gr["volatility_warning"]:
            narrative.append("⚠️  Volatility Warning:")
            narrative.append(f"  {gr['volatility_warning']}")
    
    return "\n".join(narrative)


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run aggregation and generate reports."""
    
    print("=" * 80)
    print("RESULTS AGGREGATION FOR h168")
    print("=" * 80)
    
    # Load results
    print("\nLoading evaluation summary...")
    eval_summary = load_evaluation_summary()
    
    print("Loading WFCV results...")
    wfcv_results = load_wfcv_results()
    
    print("Loading Optuna results...")
    optuna_results = load_optuna_results()
    
    # Calculate skill scores
    print("\nCalculating skill scores...")
    skill_results = calculate_skill_scores(eval_summary)
    
    # Regional leaderboard
    print("Calculating regional skill scores...")
    regional_skill = calculate_regional_skill_scores(eval_summary)
    
    # Gate pass report
    print("Generating gate pass report...")
    gate_report = generate_gate_pass_report(skill_results, regional_skill)
    
    # Narrative summary
    narrative = generate_narrative_summary(gate_report)
    
    # Aggregate all results
    aggregated_results = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "skill_scores": skill_results,
        "regional_leaderboard": regional_skill,
        "gate_pass_report": gate_report,
        "narrative_summary": narrative,
    }
    
    # Add WFCV results if available
    if wfcv_results:
        aggregated_results["wfcv_results"] = wfcv_results
    
    # Add Optuna results if available
    if optuna_results:
        aggregated_results["optuna_champions"] = optuna_results
    
    # Save results
    output_path = OUTPUT_DIR / "aggregated_h168_results.json"
    with open(output_path, "w") as f:
        json.dump(aggregated_results, f, indent=2, default=str)
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS SAVED TO: {output_path}")
    print(f"{'=' * 80}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("GATE PASS SUMMARY")
    print("=" * 80)
    
    for gr in gate_report:
        status = "✓ PASS" if gr["gate_pass"] else "✗ FAIL"
        print(f"\n{gr['pollutant'].upper()} ({gr['tier']} Tier): {status}")
        print(f"  Skill_Clim: {gr['skill_climatology']:.4f}")
        print(f"  RMSE/Mean: {gr['rmse_mean_ratio']:.4f}")
        if gr["coverage"]:
            print(f"  Coverage: {gr['coverage']:.4f}")
    
    print("\n" + narrative)
    
    return aggregated_results


if __name__ == "__main__":
    main()
