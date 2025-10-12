
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class OptModel1b:
    """
    Optimization model for Question 1(b): Flexible Consumer with Discomfort Minimization.
    
    Key changes from 1(a):
    - Added quadratic discomfort term in objective
    - Removed minimum daily energy constraint
    - Added reference load profile as input
    - Added comfort/cost weight parameters
    """
    
    def __init__(self, data, question_name, w_cost=1.0, w_comfort=1.0):
        """
        Initialize the optimization model for Question 1(b).
        
        Args:
            data: Processed data dictionary from DataProcessor
            question_name: Name of the scenario/question
            w_cost: Weight for cost minimization objective (default=1.0)
            w_comfort: Weight for discomfort minimization objective (default=1.0)
        """
        self.question_name = question_name
        self.data = data
        self.w_cost = w_cost
        self.w_comfort = w_comfort
        
        self._extract_parameters()
        
        self.model = gp.Model(f"Energy_Opt_1b_{question_name}")
        self.variables = {}
        self.results = {}
    
    def _extract_parameters(self):
        """Extract all necessary parameters from input data."""
        print(f"\n{'='*60}")
        print(f"Extracting parameters for Question 1(b): {self.question_name}")
        print(f"{'='*60}\n")
        
        # Time parameters
        self.T = 24
        self.hours = list(range(self.T))
        
        # Economic parameters
        bus_info = self.data['bus_info']
        self.prices = self.data['hourly_prices']['hourly_energy_price'].values
        self.tau_import = float(bus_info['import_tariff_DKK/kWh'].iloc[0])
        self.tau_export = float(bus_info['export_tariff_DKK/kWh'].iloc[0])
        
        # PV parameters
        self.P_PV_max = self.data['DER']['max_power_kW'].values[0]
        self.r_pv = self.data['solar_profile']['production_ratio'].values
        
        # Load parameters
        self.P_load_max = self.data['load']['max_load_kWh_per_hour'].values[0]
        
        # Reference load profile (NEW for 1(b))
        if 'load_profile' in self.data:
            self.p_ref = self.data['load_profile']['load_ratio'].values
            print(f"✓ Reference load profile loaded (24 hours)")
        else:
            # Default to uniform load if not provided
            self.p_ref = np.full(24, 0.5)
            print(f"⚠ Warning: No reference load profile found, using uniform load")
        
        # Grid limits
        self.P_import_max = bus_info['max_import_kW'].values[0]
        self.P_export_max = bus_info['max_export_kW'].values[0]
        
        # Weight parameters
        print(f"\nObjective weights:")
        print(f"  w_cost = {self.w_cost}")
        print(f"  w_comfort = {self.w_comfort}")
        
        print(f"\nSystem parameters:")
        print(f"  PV capacity: {self.P_PV_max} kW")
        print(f"  Load capacity: {self.P_load_max} kW")
        print(f"  Import tariff: {self.tau_import} DKK/kWh")
        print(f"  Export tariff: {self.tau_export} DKK/kWh")
        print(f"  Price range: [{self.prices.min():.2f}, {self.prices.max():.2f}] DKK/kWh")
        print(f"  Reference load range: [{self.p_ref.min():.3f}, {self.p_ref.max():.3f}]")
        print(f"{'='*60}\n")
    
    def build_model(self):
        """Build the complete optimization model."""
        print("Building optimization model for Question 1(b)...")
        
        self._create_variables()
        self._set_objective()
        self._add_constraints()
        
        self.model.setParam('OutputFlag', 1)
        self.model.setParam('QCPDual', 1)  # Enable QP dual variables
        self.model.update()
        
        print(f"\nModel statistics:")
        print(f"  Variables: {self.model.NumVars}")
        print(f"  Constraints: {self.model.NumConstrs}")
        print(f"  Problem type: Quadratic Program (QP)")
        print("Model built successfully\n")
    
    def _create_variables(self):
        """Create all decision variables."""
        print("Creating decision variables...")
        
        # PV power used at each hour [kW]
        self.variables['P_PV'] = self.model.addVars(
            self.hours,
            lb=0,
            ub=[self.P_PV_max * self.r_pv[t] for t in self.hours],
            name="P_PV"
        )
        
        # Load consumption at each hour [kW]
        self.variables['P_load'] = self.model.addVars(
            self.hours,
            lb=0,
            ub=self.P_load_max,
            name="P_load"
        )
        
        # Grid import at each hour [kW]
        self.variables['P_import'] = self.model.addVars(
            self.hours,
            lb=0,
            ub=self.P_import_max,
            name="P_import"
        )
        
        # Grid export at each hour [kW]
        self.variables['P_export'] = self.model.addVars(
            self.hours,
            lb=0,
            ub=self.P_export_max,
            name="P_export"
        )
        
        print(f"  Created {self.model.NumVars} decision variables (4 types × 24 hours)\n")
    
    def _set_objective(self):
        """
        Set the objective function: minimize weighted sum of cost and discomfort.
        
        Objective = w_cost × Cost + w_comfort × Discomfort
        
        Where:
        - Cost = Σ[(λ_t + τ_import)×P_import - (λ_t - τ_export)×P_export]
        - Discomfort = Σ(P_load - P_ref)²
        """
        print("Setting objective function...")
        
        # Cost term (same as 1(a) but weighted)
        import_cost = gp.quicksum(
            (float(self.prices[t]) + self.tau_import) * self.variables['P_import'][t]
            for t in self.hours
        )
        
        export_revenue = gp.quicksum(
            (float(self.prices[t]) - self.tau_export) * self.variables['P_export'][t]
            for t in self.hours
        )
        
        cost_term = self.w_cost * (import_cost - export_revenue)
        
        # Discomfort term (NEW for 1(b)) - quadratic penalty
        discomfort_term = self.w_comfort * gp.quicksum(
            (self.variables['P_load'][t] - self.p_ref[t])**2
            for t in self.hours
        )
        
        # Total objective
        total_objective = cost_term + discomfort_term
        
        self.model.setObjective(total_objective, GRB.MINIMIZE)
        
        print("  Objective: MIN [w_cost × Cost + w_comfort × Discomfort]")
        print(f"    Cost term: Weighted by {self.w_cost}")
        print(f"    Discomfort term: Weighted by {self.w_comfort}")
        print("  ✓ Objective set (Quadratic Programming)\n")
    
    def _add_constraints(self):
        """Add all constraints to the model."""
        print("Adding constraints...")
        
        # C1: Energy balance at each hour (EQUALITY)
        # P_PV + P_import = P_load + P_export
        self.model.addConstrs(
            (self.variables['P_PV'][t] + self.variables['P_import'][t] ==
             self.variables['P_load'][t] + self.variables['P_export'][t]
             for t in self.hours),
            name="Energy_Balance"
        )
        print("  ✓ Energy balance constraints (24 equality)")
        
        # NOTE: Minimum daily energy constraint REMOVED in Question 1(b)
        # This is the key difference from Question 1(a)
        
        print("  ✓ Variable bounds already set (implicit constraints)")
        print(f"\nTotal constraints: {self.model.NumConstrs}")
        print("  - 24 energy balance (equality)")
        print("  - 96 bounds on variables (implicit)")
        print(f"{'='*60}\n")
    
    def solve(self):
        """Solve the optimization problem."""
        print(f"\n{'='*60}")
        print("SOLVING OPTIMIZATION PROBLEM")
        print(f"{'='*60}\n")
        
        self.model.optimize()
        
        if self.model.status == GRB.OPTIMAL:
            print(f"\n{'='*60}")
            print("✓ OPTIMAL SOLUTION FOUND")
            print(f"{'='*60}\n")
            self._extract_results()
            return True
        else:
            print(f"\n⚠ Warning: Solver status = {self.model.status}")
            return False
    
    def _extract_results(self):
        """Extract and organize optimization results."""
        print("Extracting results...\n")
        
        # Extract primal variables
        results_df = pd.DataFrame({
            'hour': self.hours,
            'price': self.prices,
            'p_ref': self.p_ref,
            'P_PV': [self.variables['P_PV'][t].X for t in self.hours],
            'P_load': [self.variables['P_load'][t].X for t in self.hours],
            'P_import': [self.variables['P_import'][t].X for t in self.hours],
            'P_export': [self.variables['P_export'][t].X for t in self.hours],
        })
        
        # Calculate derived quantities
        results_df['pv_available'] = self.P_PV_max * self.r_pv
        results_df['pv_curtailment'] = results_df['pv_available'] - results_df['P_PV']
        results_df['load_deviation'] = results_df['P_load'] - results_df['p_ref']
        results_df['discomfort'] = results_df['load_deviation']**2
        results_df['import_cost'] = (results_df['price'] + self.tau_import) * results_df['P_import']
        results_df['export_revenue'] = (results_df['price'] - self.tau_export) * results_df['P_export']
        results_df['net_cost'] = results_df['import_cost'] - results_df['export_revenue']
        
        self.results['hourly'] = results_df
        
        # Calculate summary statistics
        total_cost = results_df['net_cost'].sum()
        total_discomfort = results_df['discomfort'].sum()
        
        self.results['summary'] = {
            'objective_value': self.model.objVal,
            'total_cost': total_cost,
            'total_discomfort': total_discomfort,
            'weighted_cost': self.w_cost * total_cost,
            'weighted_discomfort': self.w_comfort * total_discomfort,
            'total_import': results_df['P_import'].sum(),
            'total_export': results_df['P_export'].sum(),
            'total_load': results_df['P_load'].sum(),
            'total_pv_used': results_df['P_PV'].sum(),
            'total_pv_available': results_df['pv_available'].sum(),
            'total_pv_curtailed': results_df['pv_curtailment'].sum(),
            'max_load_deviation': abs(results_df['load_deviation']).max(),
            'avg_load_deviation': abs(results_df['load_deviation']).mean(),
        }
        
        # Extract dual variables (shadow prices)
        self._extract_dual_variables()
        
        print("✓ Results extracted successfully\n")
    
    def _extract_dual_variables(self):
        """Extract dual variables (shadow prices) from the solution."""
        print("Extracting dual variables...")
        
        self.results['duals'] = {}
        
        # π_t^balance: Shadow price of energy balance at each hour
        # Represents marginal cost of power supply (economic + comfort)
        try:
            energy_balance_constrs = [
                self.model.getConstrByName(f"Energy_Balance[{t}]")
                for t in self.hours
            ]
            self.results['duals']['pi_balance'] = [c.Pi for c in energy_balance_constrs]
            print("  ✓ Energy balance duals (π_t^balance) extracted")
        except:
            print("  ⚠ Could not extract energy balance duals")
            self.results['duals']['pi_balance'] = [0] * self.T
        
        # Note: In QP, reduced costs for variables with active bounds
        # are also available but require special handling
        
        print("✓ Dual variables extracted\n")
    
    def print_summary(self):
        """Print comprehensive results summary."""
        if not self.results:
            print("⚠ No results available. Run solve() first.")
            return
        
        summary = self.results['summary']
        
        print(f"\n{'='*60}")
        print(f"QUESTION 1(b) RESULTS SUMMARY")
        print(f"Scenario: {self.question_name}")
        print(f"{'='*60}")
        
        print(f"\nWEIGHTS:")
        print(f"  Cost weight (w_cost): {self.w_cost}")
        print(f"  Comfort weight (w_comfort): {self.w_comfort}")
        print(f"  Ratio (w_comfort/w_cost): {self.w_comfort/self.w_cost:.2f}")
        
        print(f"\nOBJECTIVE VALUE:")
        print(f"  Total: {summary['objective_value']:.2f} (weighted)")
        print(f"    = {summary['weighted_cost']:.2f} (cost component)")
        print(f"    + {summary['weighted_discomfort']:.2f} (discomfort component)")
        
        print(f"\nCOST BREAKDOWN:")
        print(f"  Total Cost: {summary['total_cost']:.2f} DKK")
        print(f"  Import Cost: {self.results['hourly']['import_cost'].sum():.2f} DKK")
        print(f"  Export Revenue: {self.results['hourly']['export_revenue'].sum():.2f} DKK")
        
        print(f"\nDISCOMFORT METRICS:")
        print(f"  Total Discomfort: {summary['total_discomfort']:.4f} (sum of squared deviations)")
        print(f"  Max Load Deviation: {summary['max_load_deviation']:.4f} kW")
        print(f"  Avg Absolute Deviation: {summary['avg_load_deviation']:.4f} kW")
        
        print(f"\nENERGY FLOWS (Daily):")
        print(f"  Total Load: {summary['total_load']:.2f} kWh")
        print(f"  Total Import: {summary['total_import']:.2f} kWh")
        print(f"  Total Export: {summary['total_export']:.2f} kWh")
        print(f"  PV Used: {summary['total_pv_used']:.2f} kWh")
        print(f"  PV Curtailed: {summary['total_pv_curtailed']:.2f} kWh")
        
        print(f"\nDUAL VARIABLES:")
        duals = self.results['duals']['pi_balance']
        print(f"  π_t^balance range: [{min(duals):.4f}, {max(duals):.4f}] DKK/kWh")
        print(f"  Average: {np.mean(duals):.4f} DKK/kWh")
        
        print(f"\n{'='*60}\n")
    
    def get_hourly_results(self):
        """Return hourly results DataFrame."""
        return self.results.get('hourly', pd.DataFrame())
    
    def get_summary(self):
        """Return summary statistics dictionary."""
        return self.results.get('summary', {})
    
    def plot_results(self, save_path=None):
        """
        Create comprehensive visualization of results.
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.results:
            print("⚠ No results to plot. Run solve() first.")
            return
        
        df = self.results['hourly']
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'Question 1(b) Results: {self.question_name}\n' +
                     f'w_cost={self.w_cost}, w_comfort={self.w_comfort}',
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Prices over time
        ax = axes[0, 0]
        ax.plot(df['hour'], df['price'], 'o-', color='black', linewidth=2)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Price [DKK/kWh]')
        ax.set_title('Electricity Prices')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Load vs Reference
        ax = axes[0, 1]
        ax.plot(df['hour'], df['p_ref'], 's--', label='Reference Load', 
                color='gray', alpha=0.7, markersize=4)
        ax.plot(df['hour'], df['P_load'], 'o-', label='Optimal Load', 
                color='green', linewidth=2)
        ax.fill_between(df['hour'], df['p_ref'], df['P_load'], 
                        alpha=0.2, color='orange', label='Deviation')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Load [kW]')
        ax.set_title('Load Consumption vs Reference')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: PV Generation
        ax = axes[1, 0]
        ax.fill_between(df['hour'], 0, df['pv_available'], 
                        alpha=0.3, color='orange', label='PV Available')
        ax.plot(df['hour'], df['P_PV'], 'o-', color='darkorange', 
                linewidth=2, label='PV Used')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Power [kW]')
        ax.set_title('PV Generation and Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Grid Interactions
        ax = axes[1, 1]
        ax.bar(df['hour'], df['P_import'], width=0.8, 
               label='Import', color='red', alpha=0.7)
        ax.bar(df['hour'], -df['P_export'], width=0.8, 
               label='Export', color='blue', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Power [kW]')
        ax.set_title('Grid Import/Export')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Dual Prices
        ax = axes[2, 0]
        duals = self.results['duals']['pi_balance']
        ax.plot(df['hour'], duals, 'o-', color='purple', linewidth=2, 
                label='π_t (Shadow Price)')
        ax.plot(df['hour'], df['price'] + self.tau_import, '--', 
                color='red', alpha=0.5, label='Price + τ_import')
        ax.plot(df['hour'], df['price'] - self.tau_export, '--', 
                color='blue', alpha=0.5, label='Price - τ_export')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Price [DKK/kWh]')
        ax.set_title('Shadow Prices (Dual Variables)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Discomfort over time
        ax = axes[2, 1]
        ax.bar(df['hour'], df['discomfort'], width=0.8, 
               color='coral', alpha=0.7)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Discomfort [(kW)²]')
        ax.set_title(f'Hourly Discomfort (Total={df["discomfort"].sum():.2f})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Figure saved to {save_path}")
        
        plt.show()
