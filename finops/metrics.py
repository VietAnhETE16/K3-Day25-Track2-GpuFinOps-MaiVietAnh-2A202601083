"""Efficiency metrics — the numbers that actually drive GPU cost.

Key teaching point (deck §5): nvidia-smi "GPU-Util %" is a *time-active* clock,
not an efficiency metric. A GPU can read 100% util while its MFU is ~20% — you
are paying the full GPU-hour for a fraction of the FLOPs you rented.
"""
from __future__ import annotations


def compute_mfu(achieved_tflops: float, peak_tflops: float) -> float:
    """Model FLOPs Utilization = achieved / peak (clamped to 0..1).

    Good training MFU is ~0.35-0.45; >0.50 is excellent. Returns 0 if peak<=0.
    """
    if peak_tflops <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_tflops / peak_tflops))


def compute_mbu(achieved_bw_tbs: float, peak_bw_tbs: float) -> float:
    """Model Bandwidth Utilization = achieved HBM BW / peak BW (clamped 0..1).

    The right metric for memory-bound decode; target ~0.60 on H100-80GB batch-1.
    """
    if peak_bw_tbs <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_bw_tbs / peak_bw_tbs))


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOP / byte for a workload (the x-axis of the roofline model)."""
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved


def roofline_regime(intensity: float, ridge_point: float) -> str:
    """Below the ridge point a workload is memory-bound; at/above it is compute-bound.

    H100 ridge ~295 FLOP/byte (BF16). LLM decode (~1-2) is memory-bound; prefill
    (~455) is compute-bound — which is *why* prefill/decode disaggregation pays off.
    """
    return "compute-bound" if intensity >= ridge_point else "memory-bound"


def flag_util_lies(rows, util_threshold: float = 0.90, mfu_threshold: float = 0.30):
    """Return the rows where GPU-Util is high but MFU is low — money leaking.

    `rows` is an iterable of dicts each having 'gpu_util_pct' (0-100) and 'mfu' (0-1).
    These are GPUs you are billed full-rate for while they do little real compute.
    """
    out = []
    for r in rows:
        util = float(r.get("gpu_util_pct", 0)) / 100.0
        mfu = float(r.get("mfu", 0))
        if util >= util_threshold and mfu < mfu_threshold:
            out.append(r)
    return out


def idle_waste_usd(idle_hours: float, on_demand_hr: float) -> float:
    """Dollars burned by a GPU left running idle (training done, instance up)."""
    return max(0.0, idle_hours) * max(0.0, on_demand_hr)


def vram_cost_efficiency(catalog: dict) -> list[dict]:
    """Calculate $/GB-VRAM-hr and $/TB/s-BW-hr across catalog GPUs (Extension 2)."""
    res = []
    for gtype, spec in catalog.items():
        od = float(spec.get("on_demand_hr", 0.0))
        hbm = float(spec.get("hbm_gb", 0.0))
        bw = float(spec.get("peak_bw_tbs", 0.0))
        cost_per_gb = od / hbm if hbm > 0 else 0.0
        cost_per_tbs = od / bw if bw > 0 else 0.0
        res.append({
            "gpu_type": gtype,
            "on_demand_hr": od,
            "hbm_gb": hbm,
            "peak_bw_tbs": bw,
            "cost_per_gb_vram_hr": round(cost_per_gb, 4),
            "cost_per_tbs_bw_hr": round(cost_per_tbs, 4),
        })
    return sorted(res, key=lambda x: x["cost_per_gb_vram_hr"])


def mbu_rightsizing_recommendation(summary: list[dict], catalog: dict) -> list[dict]:
    """Recommend optimal replacement GPU for memory-bound or util-lie GPUs based on achieved BW (Extension 2)."""
    recs = []
    for s in summary:
        cur_type = s["gpu_type"]
        cur_peak_bw = float(catalog[cur_type]["peak_bw_tbs"])
        achieved_bw = s["mbu"] * cur_peak_bw
        cur_cost = float(catalog[cur_type]["on_demand_hr"])

        # Find cheapest GPU that satisfies achieved BW requirement with safety margin (1.2x)
        target_bw = achieved_bw * 1.2
        viable_gpus = []
        for gtype, spec in catalog.items():
            peak_bw = float(spec.get("peak_bw_tbs", 0.0))
            od = float(spec.get("on_demand_hr", 0.0))
            if peak_bw >= target_bw and od < cur_cost:
                viable_gpus.append((od, gtype, peak_bw))

        if viable_gpus:
            viable_gpus.sort()
            opt_cost, opt_type, opt_bw = viable_gpus[0]
            monthly_save = (cur_cost - opt_cost) * 24 * 30
            recs.append({
                "gpu_id": s["gpu_id"],
                "current_type": cur_type,
                "current_cost_hr": cur_cost,
                "achieved_bw_tbs": round(achieved_bw, 2),
                "recommended_type": opt_type,
                "recommended_cost_hr": opt_cost,
                "recommended_bw_tbs": opt_bw,
                "monthly_savings_usd": round(monthly_save, 2),
            })
    return recs

