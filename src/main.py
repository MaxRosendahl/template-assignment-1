"""
Placeholder for main function to execute the model runner. This function creates a single/multiple instance of the Runner class, prepares input data,
and runs a single/multiple simulation.

Suggested structure:
- Import necessary modules and functions.
- Define a main function to encapsulate the workflow (e.g. Create an instance of your the Runner class, Run a single simulation or multiple simulations, Save results and generate plots if necessary.)
- Prepare input data for a single simulation or multiple simulations.
- Execute main function when the script is run directly.
"""
from data_ops.data_loader import DataLoader
from data_ops.data_processor import DataProcessor
from data_ops.data_visualizer import DataVisualizer
from opt_model.opt_model import OptModel
from opt_model.Battery_investment_model import BatteryInvestmentModel



# Load
loader_1a = DataLoader(question='question_1a')
loader_1b = DataLoader(question='question_1b')
loader_1c = DataLoader(question='question_1c')
raw_data_1a = loader_1a.get_data()
raw_data_1b = loader_1b.get_data()
raw_data_1c = loader_1c.get_data()

# Process
processor = DataProcessor()
processed_1a = processor.process_all(raw_data_1a, 'question_1a')
processed_1b = processor.process_all(raw_data_1b, 'question_1b')
processed_1c = processor.process_all(raw_data_1c, 'question_1c')


# # Visualize
# visualizer = DataVisualizer()
#visualizer.create_all_plots(processed, 'question_1a')

#optimize
opt_model = OptModel(processed_1a, 'question_1a')
opt_model.build_model()  # Create variables and constraints
opt_model.solve()        # Solve optimization
opt_model.print_summary()  # Display results

#Alternate Scenarios

opt = OptModel(processed_1a, "Scenario_Analysis")

opt.create_scenarios()
opt.run_scenarios()
summary_table = opt.analyze_scenarios()

opt.plot_dual_prices('Base Case')
opt.plot_dual_prices('Constant Prices')
opt.plot_dual_prices('High Import Tariff')
opt.plot_dual_prices('No Tariffs')

for name, res in opt.Scenario_results.items():
    gamma = res['duals']['gamma']
    mu_range = [min(res['duals']['mu']), max(res['duals']['mu'])]
    print(f"{name}: γ = {gamma:.4f}, μ_t ∈ [{mu_range[0]:.2f}, {mu_range[1]:.2f}]")

#============================================================================================
#Question 2b. Battery Investment
#============================================================================================

battery_model = BatteryInvestmentModel(processed_1c, 'question_2b', c_battery = 1500)
battery_model.build_model()
battery_model.solve()
battery_model.print_summary()