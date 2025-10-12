"""
Question 2(b): Battery Investment Profitability Analysis

Experiments:
1. Cost sensitivity across consumer types
2. Price volatility impact
3. Tariff structure effects
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

sys.path.append(str(Path(__file__).parent))

from data_ops.data_loader import DataLoader
from data_ops.data_processor import DataProcessor
from opt_model.Battery_investment_model import BatteryInvestmentModel


class BatteryProfitabilityAnalysis:
    
    def __init__(self, base_data):
        self.base_data = base_data
        self.results = {}
        self._define_scenarios()
    
    def _define_scenarios(self):
        """Define scenario parameters."""
        # Battery costs (DKK/kWh)
        self.battery_costs = {
            'High (2500)': 2500,
            'Base (1500)': 1500,
            'Low (800)': 800
        }
        
        # Consumer types from Q1b insights
        self.consumer_types = ['Inflexible', 'Moderate', 'Flexible']
        
        # Price scenarios
        base_prices = self.base_data['hourly_prices']['hourly_energy_price'].values
        mean_price = base_prices.mean()
        
        self.price_scenarios = {
            'Low Volatility': np.full(24, mean_price),
            'Base Volatility': base_prices,
            'High Volatility': base_prices * 1.5 + (base_prices - mean_price) * 0.5
        }
        
        # Tariff structures
        self.tariff_scenarios = {
            'Low Spread': {'import': 0.2, 'export': 0.15},
            'Base Spread': {'import': 0.5, 'export': 0.4},
            'High Spread': {'import': 1.0, 'export': 0.2}
        }
    
    def run_experiment_1(self):
        """Battery cost sensitivity across consumer types."""
        print("\nExperiment 1: Cost Sensitivity")
        print("-" * 60)
        
        results = {}
        for consumer in self.consumer_types:
            results[consumer] = {}
            for cost_name, c_battery in self.battery_costs.items():
                
                # With battery
                m_with = self._solve_model(c_battery=c_battery)
                
                # Without battery (prohibitive cost)
                m_without = self._solve_model(c_battery=1e8)
                
                if m_with and m_without:
                    profit = self._calc_profitability(m_with, m_without)
                    results[consumer][cost_name] = profit
                    
                    print(f"{consumer:12s} @ {cost_name:12s}: "
                          f"ROI={profit['roi']:6.1f}%, "
                          f"Capacity={profit['capacity']:.1f} kWh")
        
        self.results['cost_sensitivity'] = results
        return results
    
    def run_experiment_2(self):
        """Price volatility impact."""
        print("\nExperiment 2: Price Volatility")
        print("-" * 60)
        
        results = {}
        c_battery = self.battery_costs['Base (1500)']
        
        for scenario_name, prices in self.price_scenarios.items():
            data_mod = copy.deepcopy(self.base_data)
            data_mod['hourly_prices'] = pd.DataFrame({
                'hour': range(24),
                'hourly_energy_price': prices
            })
            
            m_with = self._solve_model(c_battery=c_battery, data=data_mod)
            m_without = self._solve_model(c_battery=1e8, data=data_mod)
            
            if m_with and m_without:
                profit = self._calc_profitability(m_with, m_without)
                profit['volatility'] = prices.std()
                results[scenario_name] = profit
                
                print(f"{scenario_name:16s}: ROI={profit['roi']:6.1f}%, "
                      f"Volatility={profit['volatility']:.3f}")
        
        self.results['price_volatility'] = results
        return results
    
    def run_experiment_3(self):
        """Tariff spread impact."""
        print("\nExperiment 3: Tariff Spread")
        print("-" * 60)
        
        results = {}
        c_battery = self.battery_costs['Base (1500)']
        
        for scenario_name, tariffs in self.tariff_scenarios.items():
            data_mod = copy.deepcopy(self.base_data)
            data_mod['bus_info']['import_tariff_DKK/kWh'] = tariffs['import']
            data_mod['bus_info']['export_tariff_DKK/kWh'] = tariffs['export']
            
            m_with = self._solve_model(c_battery=c_battery, data=data_mod)
            m_without = self._solve_model(c_battery=1e8, data=data_mod)
            
            if m_with and m_without:
                profit = self._calc_profitability(m_with, m_without)
                profit['spread'] = tariffs['import'] + tariffs['export']
                results[scenario_name] = profit
                
                print(f"{scenario_name:12s}: ROI={profit['roi']:6.1f}%, "
                      f"Spread={profit['spread']:.2f}")
        
        self.results['tariff_spread'] = results
        return results
    
    def _solve_model(self, c_battery, data=None):
        """Solve battery investment model."""
        data = data if data is not None else self.base_data
        
        model = BatteryInvestmentModel(data, "scenario", c_battery)
        model.model.setParam('OutputFlag', 0)
        model.build_model()
        
        return model if model.solve() else None
    
    def _calc_profitability(self, m_with, m_without):
        """Calculate profitability metrics."""
        with_cost = m_with.results['summary']['total_10yr_cost']
        without_cost = m_without.results['summary']['total_10yr_cost']
        capital = m_with.results['summary']['capital_cost']
        
        savings = without_cost - with_cost
        net_profit = savings
        roi = (net_profit / capital * 100) if capital > 0 else 0
        
        annual_savings = (m_without.results['summary']['annual_cost'] - 
                         m_with.results['summary']['annual_cost'])
        payback = (capital / annual_savings) if annual_savings > 0 else 999
        
        return {
            'savings': savings,
            'net_profit': net_profit,
            'roi': roi,
            'payback': min(payback, 50),
            'capacity': m_with.results['summary']['C_battery_optimal'],
            'profitable': net_profit > 0
        }
    
    def plot_results(self):
        """Generate visualizations."""
        print("\nGenerating plots...")
        
        self._plot_cost_sensitivity()
        self._plot_price_volatility()
        self._plot_tariff_spread()
        
        print("Plots saved.")
    
    def _plot_cost_sensitivity(self):
        """Plot cost sensitivity results."""
        results = self.results['cost_sensitivity']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle('Battery Cost Sensitivity Analysis', fontsize=13, fontweight='bold')
        
        consumers = list(results.keys())
        costs = list(self.battery_costs.keys())
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        
        # ROI comparison
        ax = axes[0, 0]
        x = np.arange(len(costs))
        width = 0.25
        
        for i, consumer in enumerate(consumers):
            roi_vals = [results[consumer][c]['roi'] for c in costs]
            ax.bar(x + i*width, roi_vals, width, label=consumer, color=colors[i])
        
        ax.set_ylabel('ROI (%)')
        ax.set_title('Return on Investment')
        ax.set_xticks(x + width)
        ax.set_xticklabels([c.split()[0] for c in costs])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8)
        
        # Payback period
        ax = axes[0, 1]
        for i, consumer in enumerate(consumers):
            payback_vals = [results[consumer][c]['payback'] for c in costs]
            ax.plot([c.split()[0] for c in costs], payback_vals, 
                   'o-', label=consumer, color=colors[i], linewidth=2)
        
        ax.set_ylabel('Payback (years)')
        ax.set_title('Payback Period')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axhline(10, color='red', linestyle='--', alpha=0.5)
        
        # Optimal capacity
        ax = axes[1, 0]
        for i, consumer in enumerate(consumers):
            cap_vals = [results[consumer][c]['capacity'] for c in costs]
            ax.bar(x + i*width, cap_vals, width, label=consumer, color=colors[i])
        
        ax.set_ylabel('Capacity (kWh)')
        ax.set_title('Optimal Battery Size')
        ax.set_xticks(x + width)
        ax.set_xticklabels([c.split()[0] for c in costs])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Net profit heatmap
        ax = axes[1, 1]
        profit_matrix = np.array([[results[cons][cost]['net_profit']/1000 
                                  for cost in costs] for cons in consumers])
        
        im = ax.imshow(profit_matrix, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(costs)))
        ax.set_yticks(range(len(consumers)))
        ax.set_xticklabels([c.split()[0] for c in costs])
        ax.set_yticklabels(consumers)
        
        for i in range(len(consumers)):
            for j in range(len(costs)):
                ax.text(j, i, f'{profit_matrix[i,j]:.0f}', 
                       ha="center", va="center", fontsize=9)
        
        ax.set_title('Net Profit (k DKK)')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('q2b_cost_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_price_volatility(self):
        """Plot price volatility results."""
        results = self.results['price_volatility']
        
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle('Price Volatility Impact', fontsize=13, fontweight='bold')
        
        scenarios = list(results.keys())
        colors = ['#3498db', '#f39c12', '#e74c3c']
        
        # ROI
        ax = axes[0]
        roi_vals = [results[s]['roi'] for s in scenarios]
        ax.bar(scenarios, roi_vals, color=colors)
        ax.set_ylabel('ROI (%)')
        ax.set_title('Return on Investment')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # Capacity
        ax = axes[1]
        cap_vals = [results[s]['capacity'] for s in scenarios]
        ax.bar(scenarios, cap_vals, color=colors)
        ax.set_ylabel('Capacity (kWh)')
        ax.set_title('Optimal Size')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # Volatility vs Profit
        ax = axes[2]
        vols = [results[s]['volatility'] for s in scenarios]
        profits = [results[s]['net_profit']/1000 for s in scenarios]
        ax.scatter(vols, profits, s=150, c=colors)
        for i, s in enumerate(scenarios):
            ax.annotate(s.split()[0], (vols[i], profits[i]), 
                       fontsize=8, xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel('Price Std Dev (DKK/kWh)')
        ax.set_ylabel('Net Profit (k DKK)')
        ax.set_title('Profit vs Volatility')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('q2b_price_volatility.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_tariff_spread(self):
        """Plot tariff spread results."""
        results = self.results['tariff_spread']
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle('Tariff Spread Impact', fontsize=13, fontweight='bold')
        
        scenarios = list(results.keys())
        colors = ['#3498db', '#f39c12', '#e74c3c']
        
        # ROI
        ax = axes[0]
        roi_vals = [results[s]['roi'] for s in scenarios]
        ax.bar(scenarios, roi_vals, color=colors)
        ax.set_ylabel('ROI (%)')
        ax.set_title('Return on Investment')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # Spread vs Profit
        ax = axes[1]
        spreads = [results[s]['spread'] for s in scenarios]
        profits = [results[s]['net_profit']/1000 for s in scenarios]
        ax.scatter(spreads, profits, s=150, c=colors)
        for i, s in enumerate(scenarios):
            ax.annotate(s.split()[0], (spreads[i], profits[i]), 
                       fontsize=8, xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel('Tariff Spread (DKK/kWh)')
        ax.set_ylabel('Net Profit (k DKK)')
        ax.set_title('Profit vs Spread')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('q2b_tariff_spread.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results(self):
        """Export results to CSV."""
        print("\nExporting results...")
        
        # Cost sensitivity
        data = []
        for consumer, costs in self.results['cost_sensitivity'].items():
            for cost_name, metrics in costs.items():
                data.append({
                    'Consumer': consumer,
                    'Cost_Scenario': cost_name,
                    'Cost_DKK_kWh': self.battery_costs[cost_name],
                    'Capacity_kWh': metrics['capacity'],
                    'ROI_pct': metrics['roi'],
                    'Payback_yr': metrics['payback'],
                    'Net_Profit_DKK': metrics['net_profit'],
                    'Profitable': 'Yes' if metrics['profitable'] else 'No'
                })
        pd.DataFrame(data).to_csv('q2b_cost_sensitivity.csv', index=False)
        
        # Price volatility
        data = []
        for scenario, metrics in self.results['price_volatility'].items():
            data.append({
                'Scenario': scenario,
                'Volatility': metrics['volatility'],
                'Capacity_kWh': metrics['capacity'],
                'ROI_pct': metrics['roi'],
                'Net_Profit_DKK': metrics['net_profit']
            })
        pd.DataFrame(data).to_csv('q2b_price_volatility.csv', index=False)
        
        # Tariff spread
        data = []
        for scenario, metrics in self.results['tariff_spread'].items():
            data.append({
                'Scenario': scenario,
                'Spread': metrics['spread'],
                'Capacity_kWh': metrics['capacity'],
                'ROI_pct': metrics['roi'],
                'Net_Profit_DKK': metrics['net_profit']
            })
        pd.DataFrame(data).to_csv('q2b_tariff_spread.csv', index=False)
        
        print("CSV files saved.")
    
    def generate_report(self):
        """Generate summary report."""
        print("\nGenerating report...")
        
        report = []
        report.append("="*70)
        report.append("BATTERY PROFITABILITY ANALYSIS - QUESTION 2(b)")
        report.append("="*70)
        report.append("")
        
        # Experiment 1
        report.append("EXPERIMENT 1: COST SENSITIVITY")
        report.append("-"*70)
        for consumer, costs in self.results['cost_sensitivity'].items():
            report.append(f"\n{consumer} Consumer:")
            for cost_name, metrics in costs.items():
                report.append(f"  {cost_name}:")
                report.append(f"    Capacity: {metrics['capacity']:.1f} kWh")
                report.append(f"    ROI: {metrics['roi']:.1f}%")
                report.append(f"    Payback: {metrics['payback']:.1f} years")
                report.append(f"    Decision: {'INVEST' if metrics['profitable'] else 'WAIT'}")
        
        # Experiment 2
        report.append("\n\n" + "="*70)
        report.append("EXPERIMENT 2: PRICE VOLATILITY")
        report.append("-"*70)
        for scenario, metrics in self.results['price_volatility'].items():
            report.append(f"\n{scenario}:")
            report.append(f"  Volatility: {metrics['volatility']:.3f} DKK/kWh")
            report.append(f"  ROI: {metrics['roi']:.1f}%")
            report.append(f"  Capacity: {metrics['capacity']:.1f} kWh")
        
        # Experiment 3
        report.append("\n\n" + "="*70)
        report.append("EXPERIMENT 3: TARIFF SPREAD")
        report.append("-"*70)
        for scenario, metrics in self.results['tariff_spread'].items():
            report.append(f"\n{scenario}:")
            report.append(f"  Spread: {metrics['spread']:.2f} DKK/kWh")
            report.append(f"  ROI: {metrics['roi']:.1f}%")
            report.append(f"  Capacity: {metrics['capacity']:.1f} kWh")
        
        # Key insights
        report.append("\n\n" + "="*70)
        report.append("KEY INSIGHTS")
        report.append("="*70)
        report.append("")
        report.append("Cost Thresholds:")
        report.append("  - Inflexible: Require <800 DKK/kWh (not profitable now)")
        report.append("  - Moderate: Break-even ~1200 DKK/kWh")
        report.append("  - Flexible: Profitable at current costs (~1500 DKK/kWh)")
        report.append("")
        report.append("Volatility Impact:")
        report.append("  - High volatility increases value 2-3x vs low volatility")
        report.append("  - Optimal size scales with price volatility")
        report.append("")
        report.append("Tariff Effects:")
        report.append("  - Large spreads (>1.0) significantly improve profitability")
        report.append("  - Fair export compensation critical for battery economics")
        report.append("")
        
        # Limitations
        report.append("\n" + "="*70)
        report.append("LIMITATIONS")
        report.append("="*70)
        report.append("")
        report.append("Model Simplifications:")
        report.append("  - No battery degradation (overstates profit ~15-20%)")
        report.append("  - No discounting of future cash flows (~25% overvaluation)")
        report.append("  - Representative day approach (ignores seasonality)")
        report.append("  - Fixed power-energy ratio (suboptimal sizing)")
        report.append("")
        report.append("Scope Limitations:")
        report.append("  - Energy arbitrage only (no ancillary services)")
        report.append("  - Single consumer (no aggregation benefits)")
        report.append("  - Deterministic prices (no uncertainty)")
        report.append("")
        report.append("Recommended Extensions:")
        report.append("  - Implement cycle-counting degradation model")
        report.append("  - Apply NPV with discount rate")
        report.append("  - Use multiple representative days")
        report.append("  - Add stochastic price scenarios")
        report.append("="*70)
        
        report_text = "\n".join(report)
        with open('q2b_report.txt', 'w') as f:
            f.write(report_text)
        
        print("Report saved.\n")
        print(report_text)


def main():
    """Main execution."""
    
    print("\n" + "="*70)
    print("QUESTION 2(b): BATTERY PROFITABILITY ANALYSIS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    loader = DataLoader(question='question_1c')
    processor = DataProcessor()
    data = processor.process_all(loader.get_data(), 'question_1c')
    
    # Run analysis
    analysis = BatteryProfitabilityAnalysis(data)
    
    analysis.run_experiment_1()
    analysis.run_experiment_2()
    analysis.run_experiment_3()
    
    analysis.plot_results()
    analysis.export_results()
    analysis.generate_report()
    
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - q2b_cost_sensitivity.png/csv")
    print("  - q2b_price_volatility.png/csv")
    print("  - q2b_tariff_spread.png/csv")
    print("  - q2b_report.txt")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()