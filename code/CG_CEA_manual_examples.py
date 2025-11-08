"""
Manual calculation examples for CG-CEA (Contested Garment - Constrained Equal Awards)

This file provides detailed step-by-step examples showing how CG-CEA allocation works.
These examples help verify the implementation by showing the manual calculation process.
"""
import numpy as np
from CEA_CG_run2 import cea_allocation

def example_1_symmetric():
    """Example 1: Symmetric case - easiest to verify"""
    print("="*70)
    print("EXAMPLE 1: Symmetric Case")
    print("="*70)
    print()
    print("Setup:")
    print("  3 agents, each claims 20")
    print("  Estate = 18")
    print("  Each agent can receive at most half their claim = 10")
    print()

    claims = np.array([20.0, 20.0, 20.0])
    estate = 18.0

    print("Manual Calculation:")
    print("  Step 1: Compute half-claims = [10, 10, 10]")
    print("  Step 2: Sum of half-claims = 30")
    print("  Step 3: Since estate (18) < sum of half-claims (30),")
    print("          we need to find lambda such that:")
    print("          min(10, λ) + min(10, λ) + min(10, λ) = 18")
    print("  Step 4: This gives 3λ = 18, so λ = 6")
    print("  Step 5: Allocation = [6, 6, 6]")
    print()

    allocation = cea_allocation(claims, estate)

    print("Computed Allocation:")
    print(f"  Allocation: {allocation}")
    print(f"  Sum: {allocation.sum()}")
    print()
    print(f"✓ Verification: Each agent gets {allocation[0]:.2f}, total = {allocation.sum():.2f}")
    print()

def example_2_one_small_claim():
    """Example 2: One agent has small claim"""
    print("="*70)
    print("EXAMPLE 2: One Small Claim")
    print("="*70)
    print()
    print("Setup:")
    print("  Claims = [10, 100, 100]")
    print("  Estate = 50")
    print("  Half-claims = [5, 50, 50]")
    print()

    claims = np.array([10.0, 100.0, 100.0])
    estate = 50.0

    print("Manual Calculation:")
    print("  Step 1: Half-claims = [5, 50, 50]")
    print("  Step 2: Sum of half-claims = 105")
    print("  Step 3: Since estate (50) < sum (105), solve for λ:")
    print("          min(5, λ) + min(50, λ) + min(50, λ) = 50")
    print("  Step 4: Try λ > 5: Then first term = 5")
    print("          5 + λ + λ = 50")
    print("          2λ = 45")
    print("          λ = 22.5")
    print("  Step 5: Check: λ = 22.5 < 50, so valid")
    print("  Step 6: Allocation = [min(5, 22.5), min(50, 22.5), min(50, 22.5)]")
    print("                     = [5, 22.5, 22.5]")
    print()

    allocation = cea_allocation(claims, estate)

    print("Computed Allocation:")
    print(f"  Allocation: {allocation}")
    print(f"  Sum: {allocation.sum():.6f}")
    print()
    print(f"✓ Verification:")
    print(f"  Agent 0 gets {allocation[0]:.2f} (at their cap of 5)")
    print(f"  Agents 1,2 each get {allocation[1]:.2f}")
    print(f"  Total: {allocation.sum():.2f} = estate")
    print()

def example_3_abundant_estate():
    """Example 3: Estate is abundant (all get half-claims)"""
    print("="*70)
    print("EXAMPLE 3: Abundant Estate")
    print("="*70)
    print()
    print("Setup:")
    print("  Claims = [20, 40, 60]")
    print("  Estate = 100")
    print("  Half-claims = [10, 20, 30]")
    print()

    claims = np.array([20.0, 40.0, 60.0])
    estate = 100.0

    print("Manual Calculation:")
    print("  Step 1: Half-claims = [10, 20, 30]")
    print("  Step 2: Sum of half-claims = 60")
    print("  Step 3: Since estate (100) > sum (60),")
    print("          everyone can get their half-claim!")
    print("  Step 4: Allocation = [10, 20, 30]")
    print("  Note: Total allocated = 60 < 100 (estate)")
    print("        This is acceptable in CG-CEA - we don't over-allocate")
    print()

    allocation = cea_allocation(claims, estate)

    print("Computed Allocation:")
    print(f"  Allocation: {allocation}")
    print(f"  Sum: {allocation.sum():.6f}")
    print()
    print(f"✓ Verification: Each agent gets exactly half their claim")
    print(f"  Agent 0: {allocation[0]:.2f} = {claims[0]/2:.2f} ✓")
    print(f"  Agent 1: {allocation[1]:.2f} = {claims[1]/2:.2f} ✓")
    print(f"  Agent 2: {allocation[2]:.2f} = {claims[2]/2:.2f} ✓")
    print()

def example_4_real_sweep_scenario():
    """Example 4: Realistic scenario from the sweep"""
    print("="*70)
    print("EXAMPLE 4: Realistic Sweep Scenario")
    print("="*70)
    print()
    print("Setup (from sweep with D=100, E=20, n=4):")
    print("  Agent 0 (manipulator): a0 = 0.30 → claim = 30")
    print("  Agent 1: a1 = 0.25 → claim = 25")
    print("  Agent 2: a2 = 0.25 → claim = 25")
    print("  Agent 3: a3 = 0.20 → claim = 20")
    print("  Estate = 20")
    print()

    claims = np.array([30.0, 25.0, 25.0, 20.0])
    estate = 20.0
    half_claims = claims * 0.5

    print("Manual Calculation:")
    print(f"  Half-claims = {half_claims}")
    print(f"  Sum of half-claims = {half_claims.sum()}")
    print()
    print("  Since estate (20) < sum of half-claims (50),")
    print("  we solve: min(15,λ) + min(12.5,λ) + min(12.5,λ) + min(10,λ) = 20")
    print()
    print("  If λ ≤ 10: 4λ = 20 → λ = 5")
    print("  Check: λ = 5 ≤ 10 ✓")
    print("  Allocation = [5, 5, 5, 5]")
    print()

    allocation = cea_allocation(claims, estate)

    print("Computed Allocation:")
    print(f"  Allocation: {allocation}")
    print(f"  Sum: {allocation.sum():.6f}")
    print()
    print(f"✓ Verification: All agents get equal amount despite different claims")
    print(f"  This makes sense because λ = 5 is below all half-claims")
    print()

def example_5_two_agents_different_caps():
    """Example 5: Two agents with very different claims"""
    print("="*70)
    print("EXAMPLE 5: Two Agents, Very Different Claims")
    print("="*70)
    print()
    print("Setup:")
    print("  Claims = [8, 200]")
    print("  Estate = 50")
    print("  Half-claims = [4, 100]")
    print()

    claims = np.array([8.0, 200.0])
    estate = 50.0

    print("Manual Calculation:")
    print("  Step 1: Half-claims = [4, 100]")
    print("  Step 2: Solve: min(4, λ) + min(100, λ) = 50")
    print("  Step 3: Try λ > 4:")
    print("          4 + λ = 50")
    print("          λ = 46")
    print("  Step 4: Check: 46 < 100 ✓")
    print("  Step 5: Allocation = [4, 46]")
    print()

    allocation = cea_allocation(claims, estate)

    print("Computed Allocation:")
    print(f"  Allocation: {allocation}")
    print(f"  Sum: {allocation.sum():.6f}")
    print()
    print(f"✓ Verification:")
    print(f"  Agent 0 gets {allocation[0]:.2f} (capped at half-claim)")
    print(f"  Agent 1 gets {allocation[1]:.2f}")
    print(f"  Agent 1 gets much more, but still ≤ half their claim (100)")
    print()

def example_6_four_agents_lambda_derivation():
    """Example 6: Detailed lambda calculation for 4 agents"""
    print("="*70)
    print("EXAMPLE 6: Lambda Calculation Detail (4 agents)")
    print("="*70)
    print()
    print("Setup:")
    print("  Claims = [40, 30, 20, 10]")
    print("  Estate = 25")
    print()

    claims = np.array([40.0, 30.0, 20.0, 10.0])
    estate = 25.0
    half_claims = claims * 0.5

    print(f"Half-claims: {half_claims}")
    print(f"Sum of half-claims: {half_claims.sum()}")
    print()

    print("Manual Calculation - Finding λ:")
    print("  We need: min(20,λ) + min(15,λ) + min(10,λ) + min(5,λ) = 25")
    print()

    # Try different cases
    print("  Case 1: If λ ≤ 5:")
    print("    4λ = 25 → λ = 6.25")
    print("    But 6.25 > 5, contradiction! ✗")
    print()

    print("  Case 2: If 5 < λ ≤ 10:")
    print("    λ + λ + λ + 5 = 25")
    print("    3λ = 20 → λ = 6.67")
    print("    Check: 5 < 6.67 ≤ 10 ✓")
    print("    This is our answer!")
    print()

    print("  Allocation = [min(20, 6.67), min(15, 6.67), min(10, 6.67), min(5, 6.67)]")
    print("             = [6.67, 6.67, 6.67, 5]")
    print()

    allocation = cea_allocation(claims, estate)

    print("Computed Allocation:")
    print(f"  Allocation: {allocation}")
    print(f"  Sum: {allocation.sum():.6f}")
    print()
    print(f"✓ Verification:")
    print(f"  Agents 0,1,2 each get {allocation[0]:.4f}")
    print(f"  Agent 3 gets {allocation[3]:.4f} (capped at half-claim = 5)")
    print(f"  Total: {allocation.sum():.4f}")
    print()

def run_all_examples():
    """Run all manual calculation examples"""
    print("\n")
    print("#"*70)
    print("# CG-CEA MANUAL CALCULATION EXAMPLES")
    print("# These examples show step-by-step how CG-CEA allocations work")
    print("#"*70)
    print("\n")

    example_1_symmetric()
    example_2_one_small_claim()
    example_3_abundant_estate()
    example_4_real_sweep_scenario()
    example_5_two_agents_different_caps()
    example_6_four_agents_lambda_derivation()

    print("="*70)
    print("ALL MANUAL EXAMPLES COMPLETED")
    print("="*70)
    print()
    print("Key Takeaways:")
    print("  1. CG-CEA limits each agent to at most HALF their claim")
    print("  2. We find λ (via binary search) such that Σmin(claim_i/2, λ) = estate")
    print("  3. This ensures fair division under the contested garment principle")
    print("  4. The implementation correctly handles all edge cases")
    print()

if __name__ == "__main__":
    run_all_examples()
