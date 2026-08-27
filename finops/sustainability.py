"""Sustainability economics — energy and carbon as governed cost levers (deck §11).

Region selection cuts $ and carbon together; reasoning queries are an energy bomb.
"""
from __future__ import annotations

# Grid carbon intensity (gCO2 / kWh) — illustrative 2026 snapshot.
REGION_CARBON = {
    "us-east-1": 380,
    "us-west-2": 120,   # Oregon hydro
    "europe-north1": 30,  # Norway
    "europe-central2": 660,  # Poland (dirtiest)
    "us-east-wa": 90,
}
# Electricity price (USD / kWh) — illustrative.
REGION_PRICE_KWH = {
    "us-east-1": 0.12,
    "us-west-2": 0.07,
    "europe-north1": 0.09,
    "europe-central2": 0.18,
    "us-east-wa": 0.055,
}

REASONING_ENERGY_MULTIPLIER = 80.0  # deck: reasoning ~74-86x a small-model query


def wh_per_query(total_tokens: int, wh_per_1k_tokens: float = 0.30, is_reasoning: bool = False) -> float:
    """Energy for one query. Median Gemini prompt ~0.24 Wh; reasoning ~74-86x."""
    base = (total_tokens / 1000.0) * wh_per_1k_tokens
    return base * (REASONING_ENERGY_MULTIPLIER if is_reasoning else 1.0)


def carbon_g(wh: float, region: str = "us-east-1") -> float:
    """Grams CO2 for an energy amount in a region."""
    gco2_kwh = REGION_CARBON.get(region, 400)
    return (wh / 1000.0) * gco2_kwh


def energy_cost_usd(wh: float, region: str = "us-east-1") -> float:
    """Electricity cost of an energy amount in a region."""
    return (wh / 1000.0) * REGION_PRICE_KWH.get(region, 0.12)


def tokens_per_watt(total_tokens: int, wh: float, seconds: float = 1.0) -> float:
    """Energy efficiency of serving: tokens per watt (higher is better)."""
    watts = (wh * 3600.0) / seconds if seconds > 0 else 0.0
    return total_tokens / watts if watts > 0 else 0.0


def region_comparison_matrix(wh: float) -> list[dict]:
    """Compare energy cost and carbon emissions across all supported regions (Extension 5)."""
    res = []
    for reg, gco2 in REGION_CARBON.items():
        price = REGION_PRICE_KWH.get(reg, 0.12)
        carb_g = (wh / 1000.0) * gco2
        cost_usd = (wh / 1000.0) * price
        res.append({
            "region": reg,
            "gco2_per_kwh": gco2,
            "price_per_kwh_usd": price,
            "carbon_g": round(carb_g, 3),
            "carbon_kg": round(carb_g / 1000.0, 4),
            "energy_cost_usd": round(cost_usd, 4),
        })
    return sorted(res, key=lambda x: x["carbon_g"])


def carbon_aware_workload_savings(
    workloads: list[dict],
    catalog: dict,
    base_region: str = "us-east-1",
    target_region: str = "europe-north1",
    days: int = 30,
) -> dict:
    """Calculate CO2 and electricity cost reduction by migrating interruptible jobs to green regions (Extension 5)."""
    total_wh = 0.0
    job_details = []
    for j in workloads:
        interruptible = bool(int(j.get("interruptible", 0)))
        if not interruptible:
            continue
        gtype = j["gpu_type"]
        ngpu = int(j.get("num_gpus", 1))
        hpd = float(j.get("hours_per_day", 0))
        watts = float(catalog[gtype].get("watts", 700))

        # Total energy in Wh = power (W) * hours * GPUs * days
        job_wh = watts * hpd * ngpu * days
        total_wh += job_wh

        base_carb_kg = (job_wh / 1000.0) * REGION_CARBON.get(base_region, 380) / 1000.0
        target_carb_kg = (job_wh / 1000.0) * REGION_CARBON.get(target_region, 30) / 1000.0

        job_details.append({
            "job_id": j.get("job_id"),
            "gpu_type": gtype,
            "energy_kwh": round(job_wh / 1000.0, 1),
            "base_carbon_kg": round(base_carb_kg, 2),
            "target_carbon_kg": round(target_carb_kg, 2),
            "carbon_saved_kg": round(base_carb_kg - target_carb_kg, 2),
        })

    base_carb_total_kg = (total_wh / 1000.0) * REGION_CARBON.get(base_region, 380) / 1000.0
    target_carb_total_kg = (total_wh / 1000.0) * REGION_CARBON.get(target_region, 30) / 1000.0
    carb_saved_kg = base_carb_total_kg - target_carb_total_kg
    pct_saved = (carb_saved_kg / base_carb_total_kg * 100.0) if base_carb_total_kg > 0 else 0.0

    base_elec_cost = (total_wh / 1000.0) * REGION_PRICE_KWH.get(base_region, 0.12)
    target_elec_cost = (total_wh / 1000.0) * REGION_PRICE_KWH.get(target_region, 0.09)
    cost_saved_usd = base_elec_cost - target_elec_cost

    return {
        "total_energy_kwh": round(total_wh / 1000.0, 2),
        "base_region": base_region,
        "target_region": target_region,
        "base_carbon_kg": round(base_carb_total_kg, 2),
        "target_carbon_kg": round(target_carb_total_kg, 2),
        "carbon_saved_kg": round(carb_saved_kg, 2),
        "carbon_saved_pct": round(pct_saved, 1),
        "base_electricity_usd": round(base_elec_cost, 2),
        "target_electricity_usd": round(target_elec_cost, 2),
        "electricity_saved_usd": round(cost_saved_usd, 2),
        "jobs": job_details,
    }

