# Question 1(b): Flexible Consumer with Discomfort Minimization

## Overview

This implementation covers Question 1(b) of Assignment 1, which extends the base case (1a) by:
- Adding a **quadratic discomfort term** to minimize deviations from a reference load profile
- Removing the **minimum daily energy constraint**
- Introducing **weight parameters** (w_cost, w_comfort) to balance cost vs. comfort trade-offs

## Problem Formulation

### Objective Function
```
minimize: w_cost × Cost + w_comfort × Discomfort

where:
  Cost = Σ[(λ_t + τ_import)×P_import - (λ_t - τ_export)×P_export]
  Discomfort = Σ(P_load - P_ref)²
```

### Key Changes from Question 1(a)
1. **Added**: Quadratic discomfort term with reference load profile
2. **Added**: Weight parameters (w_cost, w_comfort)
3. **Removed**: Minimum daily energy constraint
4. **Changed**: Problem type from LP to QP (Quadratic Program)

## File Structure

```
src/
├── opt_model_1b.py              # Main optimization model class
├── scenario_analysis_1b.py       # Scenario analysis and experiments
├── main_1b.py                    # Main execution script
└── data_ops/                     # Data loading and processing (shared)
```

## Implementation Details

### OptModel1b Class

**Purpose**: Implements the quadratic optimization model for Question 1(b)

**Key Features**:
- Quadratic objective with cost and discomfort terms
- Configurable weight parameters
- Dual variable extraction for sensitivity analysis
- Comprehensive result visualization

**Usage Example**:
```python
from opt_model_1b import OptModel1b

# Create model with balanced weights
model = OptModel1b(
    data=processed_data,
    question_name="Balanced Case",
    w_cost=1.0,
    w_comfort=1.0
)

# Build and solve
model.build_model()
model.solve()
model.print_summary()
model.plot_results()
```

### ScenarioAnalysis1b Class

**Purpose**: Conducts numerical experiments for Question 1(b).v

**Experiments Implemented**:

1. **Weight Scenarios** - Impact of flexibility preferences:
   - High Comfort (w_comfort=100): Comfort-prioritizing consumer
   - Balanced (w_comfort=1): Moderate consumer
   - Low Comfort (w_comfort=0.01): Cost-prioritizing consumer
   - Very Low Comfort (w_comfort=0.001): Extremely price-sensitive

2. **Load Profile Scenarios** - Impact of reference patterns:
   - Flat Load: Uniform consumption
   - Evening Peak: High 6-9 PM
   - Day Peak: High 9 AM - 5 PM
   - Anti-correlated PV: High when PV low
   - Correlated PV: High when PV high (best case)

**Usage Example**:
```python
from scenario_analysis_1b import ScenarioAnalysis1b

# Initialize analysis
analysis = ScenarioAnalysis1b(processed_data)

# Run weight scenarios
analysis.create_weight_scenarios()
analysis.run_weight_scenarios()
summary = analysis.analyze_weight_impact()

# Run load profile scenarios
analysis.create_load_profile_scenarios()
analysis.run_load_profile_scenarios()
profile_summary = analysis.analyze_load_profile_impact()

# Generate report
analysis.generate_report()
```

## Running the Code

### Quick Start

```bash
# Navigate to src directory
cd src

# Run complete Question 1(b) analysis
python main_1b.py
```

This will:
1. Load and process data from `data/question_1b/`
2. Solve base case with balanced weights
3. Run all weight scenarios
4. Run all load profile scenarios
5. Generate visualizations and reports

### Expected Output Files

After running `main_1b.py`, you should have:

1. **Figures** (PNG):
   - `question_1b_base_case.png` - Base case visualization
   - `question_1b_weight_analysis.png` - Weight scenario comparison (9 subplots)
   - `question_1b_load_profile_analysis.png` - Load profile comparison

2. **Data** (CSV):
   - `question_1b_weight_scenarios.csv` - Numerical results for weight scenarios
   - `question_1b_load_profiles.csv` - Numerical results for load profiles

3. **Report** (TXT):
   - `question_1b_report.txt` - Comprehensive text report with insights

### Running Individual Components

```python
# Option 1: Just solve one scenario
from opt_model_1b import OptModel1b
from data_ops import DataLoader, DataProcessor

loader = DataLoader('question_1b')
processor = DataProcessor()
data = processor.process_all(loader.get_data(), 'question_1b')

model = OptModel1b(data, "Test", w_cost=1.0, w_comfort=1.0)
model.build_model()
model.solve()
model.print_summary()

# Option 2: Run specific experiment
from scenario_analysis_1b import ScenarioAnalysis1b

analysis = ScenarioAnalysis1b(data)
analysis.run_weight_scenarios()  # Only weight scenarios
analysis.analyze_weight_impact()
```

## Key Results and Insights

### Expected Findings

Based on the mathematical analysis:

1. **Weight Ratio Impact**:
   - High w_comfort → Minimal load shifting, follows reference closely
   - Low w_comfort → Aggressive load shifting, follows price signals
   - Trade-off: Cost savings vs. lifestyle comfort

2. **Reference Load Pattern Impact**:
   - PV-correlated: Best case (low cost + low discomfort)
   - Anti-PV: Worst case (high cost or high discomfort)
   - Flat: Neutral (moderate results)

3. **Dual Variables**:
   - π_t^balance includes both economic and comfort components
   - Higher when consuming below reference (comfort gradient positive)
   - Lower when consuming above reference

4. **Binding Constraints**:
   - Energy balance: Always binding (equality)
   - Load capacity: More likely when reference high
   - Grid limits: Never binding (too large)

## Advanced Usage

### Custom Scenarios

```python
# Create custom weight scenario
custom_model = OptModel1b(
    data=processed_data,
    question_name="Custom",
    w_cost=1.0,
    w_comfort=50.0  # Custom weight
)
custom_model.build_model()
custom_model.solve()

# Create custom reference load
import numpy as np
custom_profile = np.sin(np.linspace(0, 2*np.pi, 24)) * 0.3 + 0.5

data_custom = processed_data.copy()
data_custom['load_profile'] = pd.DataFrame({
    'hour': range(24),
    'load_ratio': custom_profile
})

model = OptModel1b(data_custom, "Custom Profile")
model.build_model()
model.solve()
```

### Sensitivity Analysis

```python
# Test range of weight ratios
ratios = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
results = []

for ratio in ratios:
    model = OptModel1b(data, f"Ratio_{ratio}", 
                      w_cost=1.0, w_comfort=ratio)
    model.build_model()
    model.solve()
    results.append({
        'ratio': ratio,
        'cost': model.get_summary()['total_cost'],
        'discomfort': model.get_summary()['total_discomfort']
    })
