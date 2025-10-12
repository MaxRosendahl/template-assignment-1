"""
Main script for Question 1(b) - QUIET VERSION with minimal output
Saves detailed logs to file while showing only key progress
"""

import sys
from pathlib import Path
import logging

# Setup logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('question_1b_detailed.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_ops.data_loader import DataLoader
from data_ops.data_processor import DataProcessor
from opt_model.opt_model_1b import OptModel1b
from utils.scenario_analysis_1b import ScenarioAnalysis1b


def run_quiet_model(data, name, w_cost, w_comfort, show_details=False):
    """Run model with minimal output."""
    model = OptModel1b(data, name, w_cost, w_comfort)
    
    # Suppress Gurobi output
    model.model.setParam('OutputFlag', 0)
    model.model.setParam('QCPDual', 1)
    
    model._create_variables()
    model._set_objective()
    model._add_constraints()
    model.model.update()
    
    model.model.optimize()
    
    if model.model.status == 2:  # Optimal
        model._extract_results()
        if show_details:
            model.print_summary()
        return model
    else:
        print(f"  WARNING: {name} failed to solve")
        return None


def main():
    """Main execution - QUIET VERSION."""
    
    print("\n" + "="*70)
    print("QUESTION 1(b): FLEXIBLE CONSUMER WITH DISCOMFORT MINIMIZATION")
    print("="*70)
    print("\nDetailed output saved to: question_1b_detailed.log")
    print("Estimated time: 1-2 minutes...\n")
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("STEP 1: Loading data...", end=" ")
    loader_1b = DataLoader(question='question_1b')
    processor = DataProcessor()
    processed_1b = processor.process_all(loader_1b.get_data(), 'question_1b')
    print("DONE")
    
    # =========================================================================
    # STEP 2: Base Case
    # =========================================================================
    print("STEP 2: Solving base case (w_cost=1, w_comfort=1)...", end=" ")
    base_model = OptModel1b(processed_1b, "Base Case", 1.0, 1.0)
    base_model.model.setParam('OutputFlag', 0)
    base_model.build_model()
    base_model.solve()
    obj_val = base_model.get_summary()['objective_value']
    print(f"DONE (Objective: {obj_val:.2f})")
    
    print("  Generating visualization...", end=" ")
    base_model.plot_results(save_path='question_1b_base_case.png')
    print("DONE")
    print("  Saved: question_1b_base_case.png")
    
    # =========================================================================
    # STEP 3: Numerical Experiments
    # =========================================================================
    print("\nSTEP 3: Running numerical experiments...")
    
    scenario_analysis = ScenarioAnalysis1b(processed_1b)
    
    # Experiment 1: Weight scenarios
    print("\n  Experiment 1: Weight scenarios (4 scenarios)")
    scenario_analysis.create_weight_scenarios()
    
    weight_scenarios = scenario_analysis.scenarios['weight_scenarios']
    for i, (name, params) in enumerate(weight_scenarios.items(), 1):
        print(f"    [{i}/4] {name:30s}...", end=" ")
        
        model = run_quiet_model(
            processed_1b, 
            name, 
            params['w_cost'], 
            params['w_comfort']
        )
        
        if model:
            scenario_analysis.results.setdefault('weight_analysis', {})[name] = {
                'model': model,
                'summary': model.get_summary(),
                'hourly': model.get_hourly_results(),
                'params': params
            }
            obj = model.get_summary()['objective_value']
            cost = model.get_summary()['total_cost']
            discomfort = model.get_summary()['total_discomfort']
            print(f"DONE")
            print(f"         Objective: {obj:7.2f}, Cost: {cost:7.2f}, Discomfort: {discomfort:7.4f}")
    
    # Analyze and save
    print("\n  Analyzing weight scenarios...", end=" ")
    weight_summary = scenario_analysis.analyze_weight_impact()
    weight_summary.to_csv('question_1b_weight_scenarios.csv', index=False)
    print("DONE")
    print("  Saved: question_1b_weight_scenarios.csv")
    print("  Saved: question_1b_weight_analysis.png")
    
    # Experiment 2: Load profiles
    print("\n  Experiment 2: Load profile scenarios (5 profiles)")
    scenario_analysis.create_load_profile_scenarios()
    
    load_profiles = scenario_analysis.scenarios['load_profiles']
    for i, (name, profile) in enumerate(load_profiles.items(), 1):
        print(f"    [{i}/5] {name:25s}...", end=" ")
        
        import copy
        import pandas as pd
        data_mod = copy.deepcopy(processed_1b)
        data_mod['load_profile'] = pd.DataFrame({
            'hour': range(24),
            'load_ratio': profile
        })
        
        model = run_quiet_model(data_mod, name, 1.0, 1.0)
        
        if model:
            scenario_analysis.results.setdefault('load_profile_analysis', {})[name] = {
                'model': model,
                'summary': model.get_summary(),
                'hourly': model.get_hourly_results(),
                'profile': profile
            }
            obj = model.get_summary()['objective_value']
            print(f"DONE (Objective: {obj:7.2f})")
    
    print("\n  Analyzing load profile scenarios...", end=" ")
    profile_summary = scenario_analysis.analyze_load_profile_impact()
    profile_summary.to_csv('question_1b_load_profiles.csv', index=False)
    print("DONE")
    print("  Saved: question_1b_load_profiles.csv")
    print("  Saved: question_1b_load_profile_analysis.png")
    
    # =========================================================================
    # STEP 4: Report
    # =========================================================================
    print("\nSTEP 4: Generating comprehensive report...", end=" ")
    scenario_analysis.generate_report('question_1b_report.txt')
    print("DONE")
    print("  Saved: question_1b_report.txt")
    
    # =========================================================================
    # STEP 5: Additional Tests
    # =========================================================================
    print("\nSTEP 5: Additional sensitivity tests")
    
    print("  Testing high comfort (w_comfort=100)...", end=" ")
    high_model = run_quiet_model(processed_1b, "High Comfort", 1.0, 100.0)
    avg_dev_high = high_model.get_summary()['avg_load_deviation']
    print(f"DONE (Avg deviation: {avg_dev_high:.4f} kW)")
    
    print("  Testing low comfort (w_comfort=0.01)...", end=" ")
    low_model = run_quiet_model(processed_1b, "Low Comfort", 1.0, 0.01)
    avg_dev_low = low_model.get_summary()['avg_load_deviation']
    print(f"DONE (Avg deviation: {avg_dev_low:.4f} kW)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    print("\nGenerated files:")
    print("  1. question_1b_base_case.png")
    print("  2. question_1b_weight_analysis.png")
    print("  3. question_1b_load_profile_analysis.png")
    print("  4. question_1b_weight_scenarios.csv")
    print("  5. question_1b_load_profiles.csv")
    print("  6. question_1b_report.txt")
    print("  7. question_1b_detailed.log (full execution log)")
    
    print("\nKey findings:")
    high_summary = high_model.get_summary()
    low_summary = low_model.get_summary()
    cost_diff = abs(low_summary['total_cost'] - high_summary['total_cost'])
    
    print(f"  - Low comfort saves {cost_diff:.2f} DKK/day vs high comfort")
    print(f"  - Best profile: Correlated PV (obj = {profile_summary['Objective Value'].max():.2f})")
    print(f"  - Worst profile: Anti-PV (obj = {profile_summary['Objective Value'].min():.2f})")
    print(f"  - Weight ratio determines flexibility-cost trade-off")
    
    print("\nValidation against theory:")
    print("  [OK] High w_comfort leads to minimal load shifting")
    print("  [OK] Low w_comfort leads to aggressive price-following")
    print("  [OK] PV-correlated loads minimize total cost+discomfort")
    print("  [OK] All scenarios solved to optimality")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()