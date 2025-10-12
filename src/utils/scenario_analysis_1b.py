import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from opt_model.opt_model_1b import OptModel1b


class ScenarioAnalysis1b:
    """
    Scenario analysis for Question 1(b).v: Numerical Experiments
    
    Analyzes how flexibility preferences (w_comfort/w_cost ratio) and 
    reference load patterns impact consumer flexibility and profits.
    """
    
    def __init__(self, base_data):
        """
        Initialize scenario analysis.
        
        Args:
            base_data: Base processed data dictionary
        """
        self.base_data = base_data
        self.scenarios = {}
        self.results = {}
        
        print(f"\n{'='*60}")
        print("QUESTION 1(b).v: SCENARIO ANALYSIS")
        print(f"{'='*60}\n")
    
    def create_weight_scenarios(self):
        """
        Create scenarios varying comfort/cost weight ratios.
        
        Scenarios from assignment guidelines:
        - High comfort weight (comfort-prioritizing consumer)
        - Balanced weights (moderate consumer)
        - Low comfort weight (cost-prioritizing consumer)
        """
        print("Creating weight scenarios...")
        
        self.scenarios['weight_scenarios'] = {
            'High Comfort (w_comfort=100)': {
                'w_cost': 1.0,
                'w_comfort': 100.0,
                'description': 'Comfort-prioritizing consumer (elderly, health-dependent)'
            },
            'Balanced (w_comfort=1)': {
                'w_cost': 1.0,
                'w_comfort': 1.0,
                'description': 'Moderate consumer (typical residential)'
            },
            'Low Comfort (w_comfort=0.01)': {
                'w_cost': 1.0,
                'w_comfort': 0.01,
                'description': 'Cost-prioritizing consumer (price-sensitive, flexible)'
            },
            'Very Low Comfort (w_comfort=0.001)': {
                'w_cost': 1.0,
                'w_comfort': 0.001,
                'description': 'Extremely price-sensitive consumer'
            }
        }
        
        print(f"Created {len(self.scenarios['weight_scenarios'])} weight scenarios\n")
    
    def create_load_profile_scenarios(self):
        """
        Create scenarios with different reference load patterns.
        
        Reference load patterns from assignment guidelines:
        - Flat load: uniform consumption
        - Evening peak: high during 6-9 PM
        - Day peak: high during 9 AM - 5 PM
        - Anti-correlated with PV: high when PV low
        - Correlated with PV: high when PV high (best case)
        """
        print("Creating load profile scenarios...")
        
        hours = np.arange(24)
        
        # 1. Flat load
        flat_load = np.full(24, 0.5)
        
        # 2. Evening peak (6-9 PM high)
        evening_peak = np.array([
            0.3, 0.25, 0.25, 0.25, 0.3, 0.4,   # 0-5
            0.5, 0.6, 0.7, 0.6, 0.5, 0.5,      # 6-11
            0.5, 0.5, 0.5, 0.6, 0.7, 0.8,      # 12-17
            0.95, 0.9, 0.8, 0.6, 0.4, 0.35     # 18-23
        ])
        
        # 3. Day peak (9 AM - 5 PM high)
        day_peak = np.array([
            0.2, 0.2, 0.2, 0.2, 0.25, 0.3,     # 0-5
            0.4, 0.5, 0.7, 0.85, 0.9, 0.95,    # 6-11
            0.9, 0.95, 0.9, 0.85, 0.8, 0.6,    # 12-17
            0.4, 0.35, 0.3, 0.25, 0.25, 0.2    # 18-23
        ])
        
        # 4. Anti-correlated with PV (from data)
        pv_profile = self.base_data['solar_profile']['production_ratio'].values
        # Invert PV profile (high load when PV low)
        anti_pv = 1.0 - pv_profile
        anti_pv = anti_pv / anti_pv.max() * 0.8  # Normalize to reasonable range
        
        # 5. Correlated with PV (best case)
        corr_pv = pv_profile / pv_profile.max() * 0.8
        
        self.scenarios['load_profiles'] = {
            'Flat Load': flat_load,
            'Evening Peak': evening_peak,
            'Day Peak': day_peak,
            'Anti-correlated PV': anti_pv,
            'Correlated PV': corr_pv
        }
        
        print(f"Created {len(self.scenarios['load_profiles'])} load profile scenarios\n")
    
    def run_weight_scenarios(self):
        """Run optimization for all weight scenarios."""
        if 'weight_scenarios' not in self.scenarios:
            self.create_weight_scenarios()
        
        print(f"\n{'='*60}")
        print("RUNNING WEIGHT SCENARIOS")
        print(f"{'='*60}\n")
        
        self.results['weight_analysis'] = {}
        
        for name, params in self.scenarios['weight_scenarios'].items():
            print(f"\n{'-'*60}")
            print(f"Scenario: {name}")
            print(f"  {params['description']}")
            print(f"  w_cost = {params['w_cost']}, w_comfort = {params['w_comfort']}")
            print(f"{'-'*60}")
            
            # Create model with specific weights
            model = OptModel1b(
                data=self.base_data,
                question_name=name,
                w_cost=params['w_cost'],
                w_comfort=params['w_comfort']
            )
            
            model.build_model()
            success = model.solve()
            
            if success:
                self.results['weight_analysis'][name] = {
                    'model': model,
                    'summary': model.get_summary(),
                    'hourly': model.get_hourly_results(),
                    'params': params
                }
            else:
                print(f"WARNING: {name} failed to solve optimally")
        
        print(f"\n{'='*60}")
        print(f"Completed {len(self.results['weight_analysis'])} weight scenarios")
        print(f"{'='*60}\n")
    
    def run_load_profile_scenarios(self, w_cost=1.0, w_comfort=1.0):
        """
        Run optimization for all load profile scenarios.
        
        Args:
            w_cost: Cost weight to use (default=1.0)
            w_comfort: Comfort weight to use (default=1.0)
        """
        if 'load_profiles' not in self.scenarios:
            self.create_load_profile_scenarios()
        
        print(f"\n{'='*60}")
        print("RUNNING LOAD PROFILE SCENARIOS")
        print(f"Using w_cost={w_cost}, w_comfort={w_comfort}")
        print(f"{'='*60}\n")
        
        self.results['load_profile_analysis'] = {}
        
        for name, profile in self.scenarios['load_profiles'].items():
            print(f"\n{'-'*60}")
            print(f"Load Profile: {name}")
            print(f"{'-'*60}")
            
            # Create modified data with new reference load
            data_mod = copy.deepcopy(self.base_data)
            data_mod['load_profile'] = pd.DataFrame({
                'hour': range(24),
                'load_ratio': profile
            })
            
            # Create and solve model
            model = OptModel1b(
                data=data_mod,
                question_name=name,
                w_cost=w_cost,
                w_comfort=w_comfort
            )
            
            model.build_model()
            success = model.solve()
            
            if success:
                self.results['load_profile_analysis'][name] = {
                    'model': model,
                    'summary': model.get_summary(),
                    'hourly': model.get_hourly_results(),
                    'profile': profile
                }
            else:
                print(f"WARNING: {name} failed to solve optimally")
        
        print(f"\n{'='*60}")
        print(f"Completed {len(self.results['load_profile_analysis'])} load profile scenarios")
        print(f"{'='*60}\n")
    
    def analyze_weight_impact(self):
        """
        Analyze and visualize impact of weight ratios.
        Returns summary table and creates visualizations.
        """
        if 'weight_analysis' not in self.results or len(self.results['weight_analysis']) == 0:
            print("WARNING: No weight analysis results. Run run_weight_scenarios() first.")
            return None
        
        print(f"\n{'='*60}")
        print("WEIGHT SCENARIO ANALYSIS RESULTS")
        print(f"{'='*60}\n")
        
        # Create summary table
        summaries = []
        for name, res in self.results['weight_analysis'].items():
            summary = res['summary']
            params = res['params']
            
            summaries.append({
                'Scenario': name,
                'w_comfort/w_cost': params['w_comfort'] / params['w_cost'],
                'Total Cost [DKK]': summary['total_cost'],
                'Total Discomfort': summary['total_discomfort'],
                'Objective Value': summary['objective_value'],
                'Total Load [kWh]': summary['total_load'],
                'Total Import [kWh]': summary['total_import'],
                'Total Export [kWh]': summary['total_export'],
                'Max Deviation [kW]': summary['max_load_deviation'],
                'Avg Deviation [kW]': summary['avg_load_deviation']
            })
        
        summary_df = pd.DataFrame(summaries)
        print(summary_df.to_string(index=False))
        print()
        
        # Create visualizations
        self._plot_weight_comparison()
        
        return summary_df
    
    def _plot_weight_comparison(self):
        """Create comprehensive comparison plots for weight scenarios."""
        
        if len(self.results['weight_analysis']) == 0:
            print("WARNING: No results to plot")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Question 1(b): Impact of Flexibility Preferences (Weight Ratios)',
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Extract data for plotting
        scenarios = list(self.results['weight_analysis'].keys())
        colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']
        
        # Plot 1: Cost vs Discomfort Trade-off
        ax1 = fig.add_subplot(gs[0, 0])
        for i, (name, res) in enumerate(self.results['weight_analysis'].items()):
            summary = res['summary']
            ax1.scatter(summary['total_discomfort'], summary['total_cost'],
                       s=200, alpha=0.7, color=colors[i % len(colors)], 
                       label=name.split('(')[0].strip())
        ax1.set_xlabel('Total Discomfort', fontsize=10)
        ax1.set_ylabel('Total Cost [DKK]', fontsize=10)
        ax1.set_title('Cost vs Discomfort Trade-off', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Objective value comparison
        ax2 = fig.add_subplot(gs[0, 1])
        obj_vals = [self.results['weight_analysis'][s]['summary']['objective_value'] 
                    for s in scenarios]
        bars = ax2.bar(range(len(scenarios)), obj_vals, color=colors[:len(scenarios)], alpha=0.7)
        ax2.set_xticks(range(len(scenarios)))
        ax2.set_xticklabels([s.split('(')[0].strip() for s in scenarios], 
                            rotation=15, ha='right', fontsize=9)
        ax2.set_ylabel('Objective Value', fontsize=10)
        ax2.set_title('Total Objective Value', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Load deviation metrics
        ax3 = fig.add_subplot(gs[0, 2])
        max_dev = [self.results['weight_analysis'][s]['summary']['max_load_deviation'] 
                   for s in scenarios]
        avg_dev = [self.results['weight_analysis'][s]['summary']['avg_load_deviation'] 
                   for s in scenarios]
        x = np.arange(len(scenarios))
        width = 0.35
        ax3.bar(x - width/2, max_dev, width, label='Max Deviation', alpha=0.7)
        ax3.bar(x + width/2, avg_dev, width, label='Avg Deviation', alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.split('(')[0].strip() for s in scenarios], 
                            rotation=15, ha='right', fontsize=9)
        ax3.set_ylabel('Deviation [kW]', fontsize=10)
        ax3.set_title('Load Deviation from Reference', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4-6: Load profiles for first 3 scenarios
        for idx in range(min(3, len(scenarios))):
            name = scenarios[idx]
            res = self.results['weight_analysis'][name]
            ax = fig.add_subplot(gs[1 + idx//2, idx%2])
            df = res['hourly']
            
            ax.plot(df['hour'], df['p_ref'], 's--', label='Reference', 
                   color='gray', alpha=0.5, markersize=3, linewidth=1.5)
            ax.plot(df['hour'], df['P_load'], 'o-', label='Optimal Load', 
                   color=colors[idx], linewidth=2, markersize=4)
            ax.fill_between(df['hour'], df['p_ref'], df['P_load'], 
                           alpha=0.2, color=colors[idx])
            
            ax.set_xlabel('Hour', fontsize=9)
            ax.set_ylabel('Load [kW]', fontsize=9)
            ax.set_title(f'{name.split("(")[0].strip()}', 
                        fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, 23.5)
        
        # Plot 7: Energy flows comparison
        ax7 = fig.add_subplot(gs[2, 0])
        import_vals = [self.results['weight_analysis'][s]['summary']['total_import'] 
                       for s in scenarios]
        export_vals = [self.results['weight_analysis'][s]['summary']['total_export'] 
                       for s in scenarios]
        x = np.arange(len(scenarios))
        width = 0.35
        ax7.bar(x - width/2, import_vals, width, label='Import', 
               color='red', alpha=0.7)
        ax7.bar(x + width/2, export_vals, width, label='Export', 
               color='blue', alpha=0.7)
        ax7.set_xticks(x)
        ax7.set_xticklabels([s.split('(')[0].strip() for s in scenarios], 
                           rotation=15, ha='right', fontsize=9)
        ax7.set_ylabel('Energy [kWh]', fontsize=10)
        ax7.set_title('Grid Import/Export', fontsize=11, fontweight='bold')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Plot 8: Dual prices for one scenario (Balanced)
        if 'Balanced (w_comfort=1)' in self.results['weight_analysis']:
            ax8 = fig.add_subplot(gs[2, 1])
            balanced = self.results['weight_analysis']['Balanced (w_comfort=1)']
            df = balanced['hourly']
            duals = balanced['model'].results['duals']['pi_balance']
            ax8.plot(df['hour'], duals, 'o-', color='purple', linewidth=2, 
                    label='pi_t (Shadow Price)')
            ax8.plot(df['hour'], df['price'], '--', color='black', 
                    alpha=0.5, linewidth=1.5, label='Market Price')
            ax8.set_xlabel('Hour', fontsize=9)
            ax8.set_ylabel('Price [DKK/kWh]', fontsize=9)
            ax8.set_title('Shadow Prices (Balanced Case)', fontsize=10, fontweight='bold')
            ax8.legend(fontsize=8)
            ax8.grid(True, alpha=0.3)
        
       
        
        plt.savefig('question_1b_weight_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved figure: question_1b_weight_analysis.png\n")
        plt.close()
    
    def analyze_load_profile_impact(self):
        """
        Analyze and visualize impact of different load profiles.
        Returns summary table and creates visualizations.
        """
        if 'load_profile_analysis' not in self.results or len(self.results['load_profile_analysis']) == 0:
            print("WARNING: No load profile analysis results. Run run_load_profile_scenarios() first.")
            return None
        
        print(f"\n{'='*60}")
        print("LOAD PROFILE ANALYSIS RESULTS")
        print(f"{'='*60}\n")
        
        # Create summary table
        summaries = []
        for name, res in self.results['load_profile_analysis'].items():
            summary = res['summary']
            
            summaries.append({
                'Profile': name,
                'Total Cost [DKK]': summary['total_cost'],
                'Total Discomfort': summary['total_discomfort'],
                'Objective Value': summary['objective_value'],
                'Max Deviation [kW]': summary['max_load_deviation'],
                'Avg Deviation [kW]': summary['avg_load_deviation'],
                'Total Import [kWh]': summary['total_import'],
                'Total Export [kWh]': summary['total_export']
            })
        
        summary_df = pd.DataFrame(summaries)
        print(summary_df.to_string(index=False))
        print()
        
        # Create visualizations
        self._plot_load_profile_comparison()
        
        return summary_df
    
    def _plot_load_profile_comparison(self):
        """Create comparison plots for load profile scenarios."""
        
        if len(self.results['load_profile_analysis']) == 0:
            print("WARNING: No results to plot")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle('Question 1(b): Impact of Reference Load Patterns',
                     fontsize=14, fontweight='bold')
        
        profiles = list(self.results['load_profile_analysis'].keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(profiles)))
        
        # Plot 1: All reference profiles
        ax = axes[0, 0]
        for i, (name, res) in enumerate(self.results['load_profile_analysis'].items()):
            profile = res['profile']
            ax.plot(range(24), profile, 'o-', label=name, 
                   color=colors[i], linewidth=2, alpha=0.7)
        ax.set_xlabel('Hour')
        ax.set_ylabel('Reference Load')
        ax.set_title('Reference Load Profiles')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Objective values
        ax = axes[0, 1]
        obj_vals = [self.results['load_profile_analysis'][p]['summary']['objective_value'] 
                    for p in profiles]
        bars = ax.bar(range(len(profiles)), obj_vals, color=colors, alpha=0.7)
        ax.set_xticks(range(len(profiles)))
        ax.set_xticklabels(profiles, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Objective Value')
        ax.set_title('Total Objective by Profile')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Cost breakdown
        ax = axes[1, 0]
        costs = [self.results['load_profile_analysis'][p]['summary']['total_cost'] 
                 for p in profiles]
        discomforts = [self.results['load_profile_analysis'][p]['summary']['total_discomfort'] 
                       for p in profiles]
        x = np.arange(len(profiles))
        width = 0.35
        ax.bar(x - width/2, costs, width, label='Cost', alpha=0.7)
        ax.bar(x + width/2, discomforts, width, label='Discomfort', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(profiles, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Value')
        ax.set_title('Cost vs Discomfort')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Load deviations
        ax = axes[1, 1]
        max_devs = [self.results['load_profile_analysis'][p]['summary']['max_load_deviation'] 
                    for p in profiles]
        avg_devs = [self.results['load_profile_analysis'][p]['summary']['avg_load_deviation'] 
                    for p in profiles]
        x = np.arange(len(profiles))
        width = 0.35
        ax.bar(x - width/2, max_devs, width, label='Max', alpha=0.7)
        ax.bar(x + width/2, avg_devs, width, label='Average', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(profiles, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Deviation [kW]')
        ax.set_title('Load Deviations from Reference')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 5-6: Example load profiles (best and worst cases)
        # Find best and worst by objective value
        obj_dict = {p: self.results['load_profile_analysis'][p]['summary']['objective_value'] 
                    for p in profiles}
        best_profile = max(obj_dict, key=obj_dict.get)
        worst_profile = min(obj_dict, key=obj_dict.get)
        
        for idx, profile_name in enumerate([best_profile, worst_profile]):
            ax = axes[2, idx]
            res = self.results['load_profile_analysis'][profile_name]
            df = res['hourly']
            
            ax.plot(df['hour'], df['p_ref'], 's--', label='Reference', 
                   color='gray', alpha=0.5, markersize=3)
            ax.plot(df['hour'], df['P_load'], 'o-', label='Optimal', 
                   linewidth=2)
            ax.fill_between(df['hour'], df['p_ref'], df['P_load'], 
                           alpha=0.2)
            
            ax.set_xlabel('Hour')
            ax.set_ylabel('Load [kW]')
            title_prefix = 'Best Case' if idx == 0 else 'Worst Case'
            ax.set_title(f'{title_prefix}: {profile_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('question_1b_load_profile_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved figure: question_1b_load_profile_analysis.png\n")
        plt.close()
    
    def generate_report(self, save_path='question_1b_report.txt'):
        """Generate comprehensive text report of all results."""
        
        report = []
        report.append("="*70)
        report.append("QUESTION 1(b): NUMERICAL ANALYSIS REPORT")
        report.append("Flexible Consumer with Discomfort Minimization")
        report.append("="*70)
        report.append("")
        
        # Weight scenarios summary
        if 'weight_analysis' in self.results and len(self.results['weight_analysis']) > 0:
            report.append("\n" + "-"*70)
            report.append("1. IMPACT OF FLEXIBILITY PREFERENCES (Weight Ratios)")
            report.append("-"*70)
            report.append("")
            
            for name, res in self.results['weight_analysis'].items():
                summary = res['summary']
                params = res['params']
                
                report.append(f"\n{name}:")
                report.append(f"  Description: {params['description']}")
                report.append(f"  w_cost/w_comfort ratio: {params['w_cost']}/{params['w_comfort']}")
                report.append(f"  Total Cost: {summary['total_cost']:.2f} DKK")
                report.append(f"  Total Discomfort: {summary['total_discomfort']:.4f}")
                report.append(f"  Objective Value: {summary['objective_value']:.2f}")
                report.append(f"  Max Load Deviation: {summary['max_load_deviation']:.4f} kW")
                report.append(f"  Avg Load Deviation: {summary['avg_load_deviation']:.4f} kW")
        
        # Load profile scenarios summary
        if 'load_profile_analysis' in self.results and len(self.results['load_profile_analysis']) > 0:
            report.append("\n" + "-"*70)
            report.append("2. IMPACT OF REFERENCE LOAD PATTERNS")
            report.append("-"*70)
            report.append("")
            
            for name, res in self.results['load_profile_analysis'].items():
                summary = res['summary']
                
                report.append(f"\n{name}:")
                report.append(f"  Total Cost: {summary['total_cost']:.2f} DKK")
                report.append(f"  Total Discomfort: {summary['total_discomfort']:.4f}")
                report.append(f"  Objective Value: {summary['objective_value']:.2f}")
                report.append(f"  Max Load Deviation: {summary['max_load_deviation']:.4f} kW")
        
        # Key insights
        report.append("\n" + "="*70)
        report.append("KEY INSIGHTS AND CONCLUSIONS")
        report.append("="*70)
        report.append("")
        report.append("1. Weight ratio (w_comfort/w_cost) fundamentally determines trade-off:")
        report.append("   - High w_comfort leads to minimal load shifting, high costs, low discomfort")
        report.append("   - Low w_comfort leads to aggressive load shifting, low costs, high discomfort")
        report.append("")
        report.append("2. Reference load pattern significantly impacts profitability:")
        report.append("   - PV-correlated loads lead to lowest cost and discomfort (aligned incentives)")
        report.append("   - Anti-PV loads lead to high cost or high discomfort (conflicting incentives)")
        report.append("")
        report.append("3. No universally optimal solution - depends on consumer type:")
        report.append("   - Elderly/health-dependent consumers prefer high comfort weight")
        report.append("   - Young/price-sensitive consumers prefer low comfort weight")
        report.append("")
        report.append("="*70)
        
        # Save report
        report_text = "\n".join(report)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"\nReport saved to: {save_path}\n")
        print(report_text)
        
        return report_text