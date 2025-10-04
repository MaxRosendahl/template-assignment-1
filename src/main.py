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



# Load
loader = DataLoader(question='question_1c')
raw_data = loader.get_data()

# Process
processor = DataProcessor()
processed = processor.process_all(raw_data, 'question_1c')

# # Visualize
# visualizer = DataVisualizer()
# visualizer.create_all_plots(processed, 'question_1c')

#printing the processed data for verification
for key, df in processed.items():
    print(f"\n--- {key} ---")
    print(df)