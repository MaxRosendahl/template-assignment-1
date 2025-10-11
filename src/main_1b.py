"""
Main script for Question 1(b): Flexible Consumer with Discomfort Minimization

This script demonstrates the complete workflow for Question 1(b):
1. Load and process data
2. Build and solve optimization model
3. Run scenario analysis
4. Generate visualizations and reports
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_ops.data_loader import DataLoader
from data_ops.data_processor import DataProcessor
from opt_model.opt_model_1b import OptModel1b
from utils.scenario_analysis_1b import ScenarioAnalysis1b


def main():
    """Main execution function for Question 1(b)."""
    
    print("\n" + "="*70)
    print("QUESTION 1(b): FLEXIBLE CONSUMER WITH DISCOMFORT MINIMIZATION")
    print("="*70 + "\n")
    
    # =========================================================================
    # STEP 1: Load and Process Data
    # =========================================================================
    print("STEP 1: Loading and processing data...")
    print("-"*70)
    
    # Get the project root directory (one level up from src/)
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data'
    
    loader_1b = DataLoader(question='question_1b', input_path=str(data_path))
    raw_data_1b = loader_1b.get_data()
    
    processor = DataProcessor()
    processed_1b = processor.process_all(raw_data_1b, 'question_1b')
    
    print("✓ Data loaded and processed\n")
    
    # =========================================================================
    # STEP 2: Solve Base Case (Balanced Weights)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: Solving Base Case (Balanced Weights)")
    print("="*70 + "\n")
    
    # Create model with balanced weights (w_cost = w_comfort = 1)
    base_model = OptModel1b(
        data=processed_1b,
        question_name="Base Case",
        w_cost=1.0,
        w_comfort=1.0
    )
    
    # Build and solve
    base_model.build_model()
    base_model.solve()
    base_model.print_summary()
    
    # Plot results
    print("Generating base case visualizations...")
    base_model.plot_results(save_path='question_1b_base_case.png')
    
    # =========================================================================
    # STEP 3: Question 1(b).v - Numerical Experiments
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: Question 1(b).v - Numerical Experiments")
    print("="*70 + "\n")
    
    # Initialize scenario analysis
    scenario_analysis = ScenarioAnalysis1b(processed_1b)
    
    # -------------------------------------------------------------------------
    # Experiment 1: Impact of Flexibility Preferences (Weight Ratios)
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("EXPERIMENT 1: Impact of Flexibility Preferences")
    print("-"*70 + "\n")
    
    scenario_analysis.create_weight_scenarios()
    scenario_analysis.run_weight_scenarios()
    weight_summary = scenario_analysis.analyze_weight_impact()
    
    # Save weight scenario results
    #weight_summary.to_csv('question_1b_weight_scenarios.csv', index=False)
    print("✓ Weight scenario results saved to: question_1b_weight_scenarios.csv\n")
    
    # -------------------------------------------------------------------------
    # Experiment 2: Impact of Reference Load Patterns
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("EXPERIMENT 2: Impact of Reference Load Patterns")
    print("-"*70 + "\n")
    
    scenario_analysis.create_load_profile_scenarios()
    scenario_analysis.run_load_profile_scenarios(w_cost=1.0, w_comfort=1.0)
    profile_summary = scenario_analysis.analyze_load_profile_impact()
    
    # Save load profile results
    #profile_summary.to_csv('question_1b_load_profiles.csv', index=False)
    print("✓ Load profile results saved to: question_1b_load_profiles.csv\n")
    
    # =========================================================================
    # STEP 4: Generate Comprehensive Report
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: Generating Comprehensive Report")
    print("="*70 + "\n")
    
    scenario_analysis.generate_report('question_1b_report.txt')
    
    # =========================================================================
    # STEP 5: Additional Analysis - Sensitivity to Individual Parameters
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: Additional Sensitivity Analysis")
    print("="*70 + "\n")
    
    # Test a specific high comfort scenario
    print("Testing High Comfort Scenario (w_comfort=100)...")
    high_comfort_model = OptModel1b(
        data=processed_1b,
        question_name="High Comfort Test",
        w_cost=1.0,
        w_comfort=100.0
    )
    high_comfort_model.build_model()
    high_comfort_model.solve()
    high_comfort_model.print_summary()
    
    # Test a specific low comfort scenario
    print("\nTesting Low Comfort Scenario (w_comfort=0.01)...")
    low_comfort_model = OptModel1b(
        data=processed_1b,
        question_name="Low Comfort Test",
        w_cost=1.0,
        w_comfort=0.01
    )
    low_comfort_model.build_model()
    low_comfort_model.solve()
    low_comfort_model.print_summary()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("QUESTION 1(b) ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. question_1b_base_case.png - Base case visualization")
    print("  2. question_1b_weight_analysis.png - Weight scenario comparison")
    print("  3. question_1b_load_profile_analysis.png - Load profile comparison")
    print("  4. question_1b_weight_scenarios.csv - Weight scenario data")
    print("  5. question_1b_load_profiles.csv - Load profile data")
    print("  6. question_1b_report.txt - Comprehensive text report")
    print("\nKey Findings:")
    print("  • Weight ratio (w_comfort/w_cost) determines flexibility-cost trade-off")
    print("  • Reference load pattern significantly impacts profitability")
    print("  • No universally optimal solution - depends on consumer preferences")
    print("  • High comfort weight → follows reference, high cost, low discomfort")
    print("  • Low comfort weight → follows prices, low cost, high discomfort")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
