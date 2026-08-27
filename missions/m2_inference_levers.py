"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Extension 3: Cache Economics Analysis
    # Gemini/Anthropic style cache write cost ~1.25x input price
    large_in_price = MODEL_PRICES["large"][0]
    small_in_price = MODEL_PRICES["small"][0]
    large_be_reads = pricing.cache_break_even_reads(write_cost_per_m=large_in_price * 1.25, read_price_per_m=large_in_price)
    small_be_reads = pricing.cache_break_even_reads(write_cost_per_m=small_in_price * 1.25, read_price_per_m=small_in_price)
    cached_requests = sum(1 for r in rows if int(num(r["cached_input_tokens"])) > 0)
    cached_tokens_total = sum(int(num(r["cached_input_tokens"])) for r in rows)

    # Extension 4: Reasoning Traffic & Budget Analysis
    from finops import sustainability
    reasoning_reqs = [r for r in rows if bool(int(num(r.get("is_reasoning", 0))))]
    std_reqs = [r for r in rows if not bool(int(num(r.get("is_reasoning", 0))))]
    
    reasoning_tokens = sum(int(num(r["input_tokens"])) + int(num(r["output_tokens"])) for r in reasoning_reqs)
    std_tokens = sum(int(num(r["input_tokens"])) + int(num(r["output_tokens"])) for r in std_reqs)

    reasoning_wh = sum(sustainability.wh_per_query(int(num(r["input_tokens"])) + int(num(r["output_tokens"])), is_reasoning=True) for r in reasoning_reqs)
    std_wh = sum(sustainability.wh_per_query(int(num(r["input_tokens"])) + int(num(r["output_tokens"])), is_reasoning=False) for r in std_reqs)

    reasoning_cost = sum(pricing.request_cost(int(num(r["input_tokens"])), int(num(r["output_tokens"])),
                                              MODEL_PRICES[r["route_tier"]][0], MODEL_PRICES[r["route_tier"]][1],
                                              cached_in=int(num(r["cached_input_tokens"])),
                                              batch=bool(int(num(r["is_batch"])))) for r in reasoning_reqs)

    # Capping reasoning to max 10% traffic policy (if current > 10%)
    current_reasoning_pct = len(reasoning_reqs) / len(rows) * 100
    target_cap_pct = 10.0
    if current_reasoning_pct > target_cap_pct:
        capped_excess_ratio = (current_reasoning_pct - target_cap_pct) / current_reasoning_pct
        capped_saved_wh = reasoning_wh * capped_excess_ratio * (1.0 - 1.0 / sustainability.REASONING_ENERGY_MULTIPLIER)
        capped_saved_cost = reasoning_cost * capped_excess_ratio * 0.5  # estimate savings moving to standard
    else:
        capped_saved_wh = 0.0
        capped_saved_cost = 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n--- Extension 3: Prompt Cache Economics ---")
        print(f"Cache hit requests: {cached_requests:,} / {len(rows):,} ({cached_requests/len(rows):.1%})")
        print(f"Total cached tokens: {cached_tokens_total:,}")
        print(f"Break-even read multiplier: Large Model = {large_be_reads:.2f}x reads | Small Model = {small_be_reads:.2f}x reads")
        print(f"Cache is viable: {pricing.cache_is_worth_it(avg_cache_reads=3.5, write_cost_per_m=large_in_price*1.25, read_price_per_m=large_in_price)} (assumes typical prefix reuse >= 1.39x)")

        print("\n--- Extension 4: Reasoning Traffic & Budget Analysis ---")
        print(f"Reasoning queries: {len(reasoning_reqs):,} ({current_reasoning_pct:.1f}% of requests), {reasoning_tokens:,} tokens ({reasoning_tokens/total_tokens:.1%})")
        print(f"Reasoning spend: ${reasoning_cost:.2f}/day ({reasoning_cost/opt_cost:.1%} of optimized daily spend)")
        print(f"Energy consumed: Reasoning = {reasoning_wh/1000:.2f} kWh vs Standard = {std_wh/1000:.2f} kWh ({reasoning_wh/(reasoning_wh+std_wh):.1%} of total inference energy!)")
        print(f"Policy recommendation: Cap reasoning to 10% traffic -> Save ~${capped_saved_cost*30:.2f}/mo & {capped_saved_wh*30/1000:.2f} kWh/mo")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_economics": {
            "cached_requests": cached_requests,
            "cached_tokens": cached_tokens_total,
            "large_be_reads": round(large_be_reads, 2),
            "small_be_reads": round(small_be_reads, 2),
        },
        "reasoning_analysis": {
            "reasoning_reqs": len(reasoning_reqs),
            "reasoning_pct": round(current_reasoning_pct, 1),
            "reasoning_cost_daily": round(reasoning_cost, 2),
            "reasoning_wh": round(reasoning_wh, 2),
            "std_wh": round(std_wh, 2),
            "energy_share_pct": round(reasoning_wh / (reasoning_wh + std_wh) * 100, 1),
        },
    }


if __name__ == "__main__":
    run()

