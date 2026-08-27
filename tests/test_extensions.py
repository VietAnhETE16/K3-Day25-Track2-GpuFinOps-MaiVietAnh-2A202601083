import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finops import pricing, metrics, sustainability
from missions._common import load_csv, catalog_by_type


def test_extension1_recommend_tier_enhanced():
    # Test backward compatibility
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"

    # Test duration awareness
    assert pricing.recommend_tier(16, False, job_days=30) == "on_demand"
    assert pricing.recommend_tier(24, False, job_days=365) == "reserved"


def test_extension2_vram_and_mbu_rightsizing():
    cat = catalog_by_type()
    eff = metrics.vram_cost_efficiency(cat)
    assert len(eff) > 0
    assert all("cost_per_gb_vram_hr" in e and "cost_per_tbs_bw_hr" in e for e in eff)

    # Test MBU rightsizing
    sample_summary = [
        {"gpu_id": "gpu-test", "gpu_type": "H100", "mbu": 0.20, "mfu": 0.20, "idle_hours": 0}
    ]
    recs = metrics.mbu_rightsizing_recommendation(sample_summary, cat)
    assert len(recs) == 1
    assert recs[0]["monthly_savings_usd"] > 0
    assert recs[0]["recommended_type"] in ("A10G", "A100", "L4")


def test_extension3_cache_economics():
    # Large model break-even: write_cost=3.75, read_price=3.0, discount=0.10 -> 3.75 / (3.0 * 0.9) = 1.3888
    be_large = pricing.cache_break_even_reads(write_cost_per_m=3.75, read_price_per_m=3.0, read_discount=0.10)
    assert abs(be_large - 1.38888) < 1e-3

    # cache_is_worth_it returns True if avg_reads >= break-even
    assert pricing.cache_is_worth_it(avg_cache_reads=2.0, write_cost_per_m=3.75, read_price_per_m=3.0) is True
    assert pricing.cache_is_worth_it(avg_cache_reads=1.0, write_cost_per_m=3.75, read_price_per_m=3.0) is False


def test_extension4_reasoning_energy_and_budget():
    tokens = 1000
    std_wh = sustainability.wh_per_query(tokens, is_reasoning=False)
    reasoning_wh = sustainability.wh_per_query(tokens, is_reasoning=True)
    assert abs(reasoning_wh - std_wh * 80.0) < 1e-6


def test_extension5_carbon_aware_scheduling():
    cat = catalog_by_type()
    jobs = load_csv("workloads.csv")
    carbon_res = sustainability.carbon_aware_workload_savings(jobs, cat, base_region="us-east-1", target_region="europe-north1")
    assert carbon_res["carbon_saved_kg"] > 0
    assert carbon_res["carbon_saved_pct"] > 85.0  # (380 - 30) / 380 = 92.1% reduction

    matrix = sustainability.region_comparison_matrix(10000)
    assert len(matrix) == 5
    assert matrix[0]["region"] == "europe-north1"  # cleanest
