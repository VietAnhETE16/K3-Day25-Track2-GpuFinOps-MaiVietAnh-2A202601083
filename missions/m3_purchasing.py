"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # Extension 5: Carbon-Aware Scheduling Analysis
    from finops import sustainability
    carbon_analysis = sustainability.carbon_aware_workload_savings(jobs, cat, base_region="us-east-1", target_region="europe-north1", days=DAYS)
    region_matrix = sustainability.region_comparison_matrix(carbon_analysis["total_energy_kwh"] * 1000)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n--- Extension 5: Carbon-Aware Scheduling (Interruptible Jobs) ---")
        print(f"Total interruptible energy: {carbon_analysis['total_energy_kwh']:,.1f} kWh / month")
        print(f"{'Region':18}{'Carbon(g/kWh)':>15}{'Elec($/kWh)':>14}{'Carbon(kgCO2e)':>16}{'Elec Cost($)':>14}")
        for reg in region_matrix:
            print(f"{reg['region']:18}{reg['gco2_per_kwh']:>15}{reg['price_per_kwh_usd']:>14.3f}{reg['carbon_kg']:>16.1f}${reg['energy_cost_usd']:>13.2f}")
        print(f"\nMigrating interruptible jobs from {carbon_analysis['base_region']} -> {carbon_analysis['target_region']}:")
        print(f"  Carbon saved: {carbon_analysis['carbon_saved_kg']:,.1f} kgCO2e/month ({carbon_analysis['carbon_saved_pct']}% reduction!)")
        print(f"  Electricity saved: ${carbon_analysis['electricity_saved_usd']:,.2f}/month")

    return {
        "recommendations": recs,
        "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "carbon_aware_scheduling": carbon_analysis,
        "region_matrix": region_matrix,
    }


if __name__ == "__main__":
    run()

