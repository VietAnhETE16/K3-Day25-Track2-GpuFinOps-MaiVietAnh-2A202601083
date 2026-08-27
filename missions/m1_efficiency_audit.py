"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # Extension 2: VRAM efficiency and MBU-aware right-sizing
    vram_eff = metrics.vram_cost_efficiency(cat)
    mbu_recs = metrics.mbu_rightsizing_recommendation(summary, cat)
    total_mbu_save = sum(r["monthly_savings_usd"] for r in mbu_recs)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n--- Extension 2: MBU & VRAM Right-Sizing Analysis ---")
        print(f"{'GPU Type':10}{'VRAM(GB)':>10}{'$/hr':>8}{'$/GB-hr':>12}{'BW(TB/s)':>12}{'$/(TB/s)-hr':>14}")
        for v in vram_eff:
            print(f"{v['gpu_type']:10}{v['hbm_gb']:>10.0f}${v['on_demand_hr']:>7.2f}${v['cost_per_gb_vram_hr']:>11.4f}{v['peak_bw_tbs']:>12.2f}${v['cost_per_tbs_bw_hr']:>13.4f}")

        print("\nMBU-aware Right-sizing Recommendations:")
        for r in mbu_recs:
            print(f"  {r['gpu_id']} ({r['current_type']}) ach_bw={r['achieved_bw_tbs']} TB/s -> recommend {r['recommended_type']} (bw={r['recommended_bw_tbs']} TB/s) | Save: ${r['monthly_savings_usd']:,.2f}/month")
        print(f"Total potential MBU right-sizing savings: ${total_mbu_save:,.2f}/month")

    return {
        "summary": summary,
        "lies": lies,
        "idle_waste_daily": round(idle_waste, 2),
        "vram_efficiency": vram_eff,
        "mbu_rightsizing": mbu_recs,
        "mbu_savings_monthly": round(total_mbu_save, 2),
    }


if __name__ == "__main__":
    run()

