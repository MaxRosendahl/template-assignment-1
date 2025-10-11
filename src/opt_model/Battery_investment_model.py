import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BatteryInvestmentModel:
    """
    Optimization model for battery investment decision.
    Determines optimal battery capacity and daily operations to minimize
    total cost (capital + 10 years operational).
    """
    
    def __init__(self, data, question_name, c_battery=1500):
        """
        Initialize battery investment model.
        
        Args:
            data: Processed data dictionary
            question_name: Name of scenario
            c_battery: Capital cost per kWh of battery [DKK/kWh]
        """
        self.question_name = question_name
        self.data = data
        self.c_battery = c_battery  # Capital cost
        
        self._extract_parameters()
        
        self.model = gp.Model(f"Battery_Investment_{question_name}")
        self.variables = {}
        self.results = {}
    
    def _extract_parameters(self):
        """Extract parameters from data."""
        self.T = 24
        self.hours = list(range(self.T))
        
        # Economic parameters
        bus_info = self.data['bus_info']
        self.prices = self.data['hourly_prices']['hourly_energy_price'].values
        self.tau_import = bus_info['import_tariff_DKK/kWh']
        self.tau_export = bus_info['export_tariff_DKK/kWh']
        
        # PV parameters
        self.P_PV_max = self.data['DER']['max_power_kW'].values[0]
        self.r_pv = self.data['solar_profile']['production_ratio'].values
        
        # Load parameters
        self.P_load_max = self.data['load']['max_load_kWh_per_hour'].values[0]
        self.E_min = 24  # Minimum daily energy
        
        # Grid limits
        self.P_import_max = bus_info['max_import_kW'].values[0]
        self.P_export_max = bus_info['max_export_kW'].values[0]
        
        # Battery reference parameters (from question 1c)
        self.C_ref = 6.0  # Reference capacity [kWh]
        self.charge_ratio = 0.15  # Max charge rate
        self.discharge_ratio = 0.30  # Max discharge rate
        self.eta_charge = 0.9  # Charging efficiency
        self.eta_discharge = 0.9  # Discharging efficiency
        self.SOC_initial = 0.5  # Initial SOC
        self.SOC_final = 0.5  # Final SOC
        
        # Time horizon
        self.years = 10
        self.days_per_year = 365
        self.total_days = self.years * self.days_per_year

        #Battery Capacity limits
        self.C_min = 0.0  # Minimum capacity [kWh]
        self.C_max = 20.0  # Maximum capacity [kWh]
        
        
        print(f"\n{'='*60}")
        print(f"Battery Investment Model: {self.question_name}")
        print(f"{'='*60}")
        print(f"Capital cost: {self.c_battery} DKK/kWh")
        print(f"Time horizon: {self.years} years ({self.total_days} days)")
        print(f"Price range: {self.prices.min():.2f} - {self.prices.max():.2f} DKK/kWh")
        print(f"Battery capacity range: {self.C_min} - {self.C_max} kWh")
        print(f"{'='*60}\n")
    
    def build_model(self):
        """Build the optimization model."""
        print("Building battery investment model...")
        
        self._create_variables()
        self._set_objective()
        self._add_constraints()
        
        self.model.setParam('OutputFlag', 1)
        print("Model built\n")
    
    def _create_variables(self):
        """Create decision variables."""
        print("Creating variables...")
        
        # INVESTMENT VARIABLE
        self.variables['C_battery'] = self.model.addVar(
            lb=self.C_min,
            ub=self.C_max,
            name="C_battery"
        )
        
        # OPERATIONAL VARIABLES (same as before, plus battery)
        
        # PV production
        self.variables['P_PV'] = self.model.addVars(
            self.hours, lb=0,
            ub=[self.P_PV_max * self.r_pv[t] for t in self.hours],
            name="P_PV"
        )
        
        # Load consumption
        self.variables['P_load'] = self.model.addVars(
            self.hours, lb=0, ub=self.P_load_max,
            name="P_load"
        )
        
        # Grid import
        self.variables['P_import'] = self.model.addVars(
            self.hours, lb=0, ub=self.P_import_max,
            name="P_import"
        )
        
        # Grid export
        self.variables['P_export'] = self.model.addVars(
            self.hours, lb=0, ub=self.P_export_max,
            name="P_export"
        )
        
        # BATTERY OPERATIONAL VARIABLES
        
        # Battery charging power
        self.variables['P_charge'] = self.model.addVars(
            self.hours, lb=0,
            name="P_charge"
        )
        
        # Battery discharging power
        self.variables['P_discharge'] = self.model.addVars(
            self.hours, lb=0,
            name="P_discharge"
        )
        
        # Battery state of charge (energy stored)
        self.variables['E_battery'] = self.model.addVars(
            self.hours, lb=0,
            name="E_battery"
        )
        
        print(f"Created {1 + 7*24} variables (1 investment + {7*24} operational)\n")
    
    def _set_objective(self):
        """Set objective: minimize capital + 10-year operational cost."""
        print("Setting objective...")
        
        C_battery = self.variables['C_battery']
        
        # Capital cost
        capital_cost = self.c_battery * C_battery
        
        # Daily operational cost
        daily_import_cost = gp.quicksum(
            (float(self.prices[t]) + float(self.tau_import)) * self.variables['P_import'][t]
            for t in self.hours
        )
        
        daily_export_revenue = gp.quicksum(
            (float(self.prices[t]) - float(self.tau_export)) * self.variables['P_export'][t]
            for t in self.hours
        )
        
        daily_cost = daily_import_cost - daily_export_revenue
        
        # Total cost over 10 years
        total_cost = capital_cost + self.total_days * daily_cost
        
        self.model.setObjective(total_cost, GRB.MINIMIZE)
        
        print("Objective: Minimize (Capital Cost + 3650 × Daily Cost)\n")
    
    def _add_constraints(self):
        """Add all constraints."""
        print("Adding constraints...")
        
        C_battery = self.variables['C_battery']
        
        # 1. ENERGY BALANCE
        self.model.addConstrs(
            (self.variables['P_PV'][t] + self.variables['P_import'][t] + self.variables['P_discharge'][t] ==
             self.variables['P_load'][t] + self.variables['P_export'][t] + self.variables['P_charge'][t]
             for t in self.hours),
            name="Energy_Balance"
        )
        
        # 2. MINIMUM DAILY ENERGY
        self.model.addConstr(
            gp.quicksum(self.variables['P_load'][t] for t in self.hours) >= self.E_min,
            name="Min_Daily_Energy"
        )
        
        # 3. BATTERY CHARGING LIMIT
        self.model.addConstrs(
            (self.variables['P_charge'][t] <= self.charge_ratio * C_battery
             for t in self.hours),
            name="Charge_Limit"
        )
        
        # 4. BATTERY DISCHARGING LIMIT
        self.model.addConstrs(
            (self.variables['P_discharge'][t] <= self.discharge_ratio * C_battery
             for t in self.hours),
            name="Discharge_Limit"
        )
        
        # 5. BATTERY SOC DYNAMICS
        
        # First hour (initial condition)
        self.model.addConstr(
            self.variables['E_battery'][0] == 
            self.SOC_initial * C_battery + 
            self.eta_charge * self.variables['P_charge'][0] -
            (1/self.eta_discharge) * self.variables['P_discharge'][0],
            name="SOC_Dynamics_0"
        )
        
        # Remaining hours
        for t in range(1, self.T):
            self.model.addConstr(
                self.variables['E_battery'][t] ==
                self.variables['E_battery'][t-1] +
                self.eta_charge * self.variables['P_charge'][t] -
                (1/self.eta_discharge) * self.variables['P_discharge'][t],
                name=f"SOC_Dynamics_{t}"
            )
        
        
        # 6. BATTERY SOC LIMITS (couples with investment!)
        self.model.addConstrs(
            (self.variables['E_battery'][t] <= C_battery
             for t in self.hours),
            name="SOC_Limit"
        )
        
        # 7. BATTERY END CONDITION (cyclic)
        self.model.addConstr(
            self.variables['E_battery'][self.T-1] == self.SOC_final * C_battery,
            name="SOC_Final"
        )
        
        print(f"Total constraints: {self.model.NumConstrs}\n")
    
    def solve(self):
        """Solve the optimization."""
        print(f"\n{'='*60}")
        print("Solving battery investment problem...")
        print(f"{'='*60}\n")
        
        self.model.optimize()
        
        if self.model.status == GRB.OPTIMAL:
            print(f"\n{'='*60}")
            print("OPTIMAL SOLUTION FOUND")
            print(f"{'='*60}\n")
            self._extract_results()
            return True
        else:
            print(f"\nSolver status: {self.model.status}")
            return False
    
    def _extract_results(self):
        """Extract solution."""
        print("Extracting results...")
        
        # Investment decision
        C_battery_opt = self.variables['C_battery'].X
        
        # Operational variables
        results_df = pd.DataFrame({
            'hour': self.hours,
            'price': self.prices,
            'P_PV': [self.variables['P_PV'][t].X for t in self.hours],
            'P_load': [self.variables['P_load'][t].X for t in self.hours],
            'P_import': [self.variables['P_import'][t].X for t in self.hours],
            'P_export': [self.variables['P_export'][t].X for t in self.hours],
            'P_charge': [self.variables['P_charge'][t].X for t in self.hours],
            'P_discharge': [self.variables['P_discharge'][t].X for t in self.hours],
            'E_battery': [self.variables['E_battery'][t].X for t in self.hours],
        })
        
        # Derived columns
        results_df['pv_available'] = self.P_PV_max * self.r_pv
        results_df['import_cost'] = (results_df['price'] + self.tau_import) * results_df['P_import']
        results_df['export_revenue'] = (results_df['price'] - self.tau_export) * results_df['P_export']
        results_df['net_cost'] = results_df['import_cost'] - results_df['export_revenue']
        
        self.results['hourly'] = results_df
        
        # Calculate costs
        daily_cost = results_df['net_cost'].sum()
        capital_cost = self.c_battery * C_battery_opt
        total_10yr_cost = capital_cost + self.total_days * daily_cost
        
        # Summary
        self.results['summary'] = {
            'C_battery_optimal': C_battery_opt,
            'capital_cost': capital_cost,
            'daily_cost': daily_cost,
            'annual_cost': daily_cost * 365,
            'total_10yr_operational': daily_cost * self.total_days,
            'total_10yr_cost': total_10yr_cost,
            'total_load': results_df['P_load'].sum(),
            'total_import': results_df['P_import'].sum(),
            'total_export': results_df['P_export'].sum(),
            'total_pv_used': results_df['P_PV'].sum(),
            'total_charge': results_df['P_charge'].sum(),
            'total_discharge': results_df['P_discharge'].sum(),
            'battery_cycles': results_df['P_charge'].sum() / C_battery_opt if C_battery_opt > 0 else 0,
        }
        
        # Extract dual variables
        self.results['duals'] = {}
        
        # Energy balance duals
        energy_balance_constrs = self.model.getConstrByName("Energy_Balance[0]").index
        self.results['duals']['mu'] = []
        for t in self.hours:
            constr = self.model.getConstrs()[energy_balance_constrs + t]
            self.results['duals']['mu'].append(constr.Pi)
        
        # Min energy dual
        min_energy_constr = self.model.getConstrByName("Min_Daily_Energy")
        self.results['duals']['gamma'] = min_energy_constr.Pi
        
        print("Results extracted\n")
    
    def print_summary(self):
        """Print results summary."""
        if not self.results:
            print("No results. Run solve() first.")
            return
        
        summary = self.results['summary']
        
        print(f"\n{'='*60}")
        print(f"BATTERY INVESTMENT RESULTS")
        print(f"{'='*60}")
        
        print(f"\nINVESTMENT DECISION:")
        print(f"  Optimal Battery Capacity: {summary['C_battery_optimal']:.2f} kWh")
        print(f"  Capital Cost: {summary['capital_cost']:.2f} DKK")
        
        if summary['C_battery_optimal'] > 0.01:
            print(f"  Max Charge Power: {summary['C_battery_optimal'] * self.charge_ratio:.2f} kW")
            print(f"  Max Discharge Power: {summary['C_battery_optimal'] * self.discharge_ratio:.2f} kW")
            print(f"  Daily Cycles: {summary['battery_cycles']:.2f}")
        else:
            print(f"  → NO BATTERY INVESTMENT (not profitable)")
        
        print(f"\nCOST BREAKDOWN:")
        print(f"  Daily Operational Cost: {summary['daily_cost']:.2f} DKK/day")
        print(f"  Annual Operational Cost: {summary['annual_cost']:.2f} DKK/year")
        print(f"  10-Year Operational Cost: {summary['total_10yr_operational']:.2f} DKK")
        print(f"  10-Year Total Cost: {summary['total_10yr_cost']:.2f} DKK")
        
        print(f"\nENERGY FLOWS (Daily):")
        print(f"  Total Load: {summary['total_load']:.2f} kWh")
        print(f"  Total Import: {summary['total_import']:.2f} kWh")
        print(f"  Total Export: {summary['total_export']:.2f} kWh")
        print(f"  Total PV Used: {summary['total_pv_used']:.2f} kWh")
        print(f"  Total Charge: {summary['total_charge']:.2f} kWh")
        print(f"  Total Discharge: {summary['total_discharge']:.2f} kWh")
        
        print(f"\n{'='*60}\n")