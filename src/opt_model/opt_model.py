from pathlib import Path
import numpy as np
import pandas as pd
import gurobipy as gp

class OptModel:

    def __init__(self, data, question_name):
        self.question_name = question_name
        self.data = data
        self._extract_parameters()
        self.model = gp.Model(f"Energy_Opt_{question_name}")
        self.variables = {}
        self.results = {}   

    def _extract_parameters(self):
        self.T = 24
        self.hours = list(range(self.T))

        # get economic parameters
        bus_info = self.data['bus_info']
        self.prices = self.data['hourly_prices']['hourly_energy_price'].values
        self.tau_import = bus_info['import_tariff_DKK/kWh']
        self.tau_export = bus_info['export_tariff_DKK/kWh']

        # PV stuff
        self.P_PV_max = self.data['DER']['max_power_kW'].values[0]
        self.r_pv = self.data['solar_profile']['production_ratio'].values

        # Load stuff
        self.P_load_max = self.data['load']['max_load_kWh_per_hour'].values[0]

        # minimum energy 
        self.E_min = 24

        # Grid limits
        self.P_import_max = bus_info['max_import_kW'].values[0]
        self.P_export_max = bus_info['max_export_kW'].values[0]

        print(f"\nParameters for {self.question_name}")
        print(f"PV capacity: {self.P_PV_max} kW")
        print(f"Load capacity: {self.P_load_max} kW")
        print(f"Min energy: {self.E_min} kWh\n")


    def build_model(self):
        print("Building model...")

        # make variables
        self.create_variables()

        # set objective
        self.set_objective()

        # add constraints
        self.add_constraints()

        self.model.setParam('OutputFlag', 1)
        print("Model built\n")

    def create_variables(self):
        print("Creating variables...")

        # PV production
        self.variables['P_PV'] = self.model.addVars(
            self.hours,
            lb = 0,
            ub = [self.P_PV_max * self.r_pv[t] for t in self.hours],
            name="P_PV")    
        
        # load consumption
        self.variables['P_load'] = self.model.addVars(
            self.hours,
            lb = 0,
            ub = self.P_load_max,
            name="P_load")
        
        # grid import
        self.variables['P_import'] = self.model.addVars(
            self.hours,
            lb = 0,
            ub = self.P_import_max,
            name="P_import")
        
        # grid export
        self.variables['P_export'] = self.model.addVars(
            self.hours,
            lb = 0,
            ub = self.P_export_max,
            name="P_export")
        
        print("Variables created\n")

    def set_objective(self):
        print("Setting objective...")

        # cost of importing
        import_cost = gp.quicksum(
            (float(self.prices[t]) + float(self.tau_import)) * self.variables['P_import'][t]
            for t in self.hours
            )

        # revenue from exporting
        export_revenue = gp.quicksum(
            (float(self.prices[t]) - float(self.tau_export)) * self.variables['P_export'][t]
            for t in self.hours
            )

        # minimize cost - revenue
        self.model.setObjective(import_cost - export_revenue, gp.GRB.MINIMIZE)

    def add_constraints(self):
        print("Adding constraints...")

        # energy balance each hour
        self.model.addConstrs(
            (self.variables['P_PV'][t] + self.variables['P_import'][t] == 
             self.variables['P_load'][t] + self.variables['P_export'][t] for t in self.hours),
            name="Energy_Balance"
        )

        # minimum daily energy
        self.model.addConstr(
            gp.quicksum(self.variables['P_load'][t] for t in self.hours) >= self.E_min,
            name="Min_Daily_Energy"
        )

    def solve(self):
        print("Solving...")

        self.model.optimize()
        
        if self.model.status == gp.GRB.OPTIMAL:
            print("Optimal solution found")
            self._extract_results()
        else:
            print(f"Status: {self.model.status}")

    def _extract_results(self):
        print("Extracting results...")
        
        # make results dataframe
        results_df = pd.DataFrame({
            'hour': self.hours,
            'price': self.prices,
            'P_PV': [self.variables['P_PV'][t].X for t in self.hours],
            'P_load': [self.variables['P_load'][t].X for t in self.hours],
            'P_import': [self.variables['P_import'][t].X for t in self.hours],
            'P_export': [self.variables['P_export'][t].X for t in self.hours],
        })
        
        # calculate some extra stuff
        results_df['pv_available'] = self.P_PV_max * self.r_pv
        results_df['pv_curtailment'] = results_df['pv_available'] - results_df['P_PV']
        results_df['import_cost'] = (results_df['price'] + self.tau_import) * results_df['P_import']
        results_df['export_revenue'] = (results_df['price'] - self.tau_export) * results_df['P_export']
        results_df['net_cost'] = results_df['import_cost'] - results_df['export_revenue']
        
        self.results['hourly'] = results_df
        
        # summary stats
        self.results['summary'] = {
            'total_cost': self.model.objVal,
            'total_import': results_df['P_import'].sum(),
            'total_export': results_df['P_export'].sum(),
            'total_load': results_df['P_load'].sum(),
            'total_pv_used': results_df['P_PV'].sum(),
            'total_pv_available': results_df['pv_available'].sum(),
            'total_pv_curtailed': results_df['pv_curtailment'].sum(),
            'total_import_cost': results_df['import_cost'].sum(),
            'total_export_revenue': results_df['export_revenue'].sum(),
        }
        
        # get dual variables
        self.results['duals'] = {}
        
        # energy balance duals (mu_t)
        energy_balance_constrs = self.model.getConstrByName("Energy_Balance[0]").index
        self.results['duals']['mu'] = []
        for t in self.hours:
            constr = self.model.getConstrs()[energy_balance_constrs + t]
            self.results['duals']['mu'].append(constr.Pi)
        
        # minimum energy dual (gamma)
        min_energy_constr = self.model.getConstrByName("Min_Daily_Energy")
        self.results['duals']['gamma'] = min_energy_constr.Pi
        
        print("Results extracted\n")
    
    def print_summary(self):
        if not self.results:
            print("No results yet")
            return
        
        summary = self.results['summary']
        
        print(f"\nRESULTS SUMMARY")
        print(f"="*50)
        print(f"\nCosts:")
        print(f"  Total Cost: {summary['total_cost']:.2f} DKK")
        print(f"  Import Cost: {summary['total_import_cost']:.2f} DKK")
        print(f"  Export Revenue: {summary['total_export_revenue']:.2f} DKK")
        
        print(f"\nEnergy:")
        print(f"  Total Load: {summary['total_load']:.2f} kWh")
        print(f"  Total Import: {summary['total_import']:.2f} kWh")
        print(f"  Total Export: {summary['total_export']:.2f} kWh")
        print(f"  PV Used: {summary['total_pv_used']:.2f} kWh")
        print(f"  PV Curtailed: {summary['total_pv_curtailed']:.2f} kWh")
        
        print(f"\nDual Variables:")
        print(f"  gamma: {self.results['duals']['gamma']:.4f} DKK/kWh")
        print(f"  mu_t range: [{min(self.results['duals']['mu']):.4f}, {max(self.results['duals']['mu']):.4f}]")
        print()
    
    def get_hourly_results(self):
        return self.results.get('hourly', pd.DataFrame())
    
    def get_summary(self):
        return self.results.get('summary', {})
    
    def create_scenario_analysis(self, base_params):
        Scenarios = {}

        base_prices = base_params['hourly_prices']['hourly_energy_price'].values
        base_tau_import = base_params['bus_info']['import_tariff_DKK/kWh']
        base_tau_export = base_params['bus_info']['export_tariff_DKK/kWh']

        # Scenario 0: Base Case
        Scenarios['Base Case'] = {
            'name': 'Base Case',
            'prices': base_prices,
            'tau_import': base_tau_import,
            'tau_export': base_tau_export
        }

        # Scenario 1: Constant Prices
        Scenarios['Constant Prices'] = {
            'name': 'Constant Prices',
            'prices': np.full(24, base_prices.mean()),
            'tau_import': base_tau_import,
            'tau_export': base_tau_export

        }

        # Scenario 2: No tarifs
        Scenarios['No Tariffs'] = {
            'name': 'No Tariffs',
            'prices': base_prices,
            'tau_import': 0.0,
            'tau_export': 0.0
        }

        #Scenario 3: No minimum energy requirement
        Scenarios['No Min Energy'] = {
            'name': 'No Min Energy',
            'prices': base_prices,
            'tau_import': base_tau_import,
            'tau_export': base_tau_export,
            'E_min': 0
        }

        # Scenario 4: High minimum energy requirement
        Scenarios['High Min Energy'] = {
            'name': 'High Min Energy',
            'prices': base_prices,
            'tau_import': base_tau_import,
            'tau_export': base_tau_export,
            'E_min': 36
        }

    def run_scenarios(self, base_params, Scenarios):
        """Solving optimization for all scenarios"""
        Scenario_results = {}

        for scenario in Scenarios.items():
            modified_data = base_params.copy()
            modified_data['hourly_prices']['price'] = pd.DataFrame({
                'hour': range(24),
                'price': scenario['prices']
            })
            modified_data['bus_info']['import_tariff_DKK/kWh'] = scenario['tau_import']
            modified_data['bus_info']['export_tariff_DKK/kWh'] = scenario['tau_export']

            opt_model = OptModel(modified_data, scenario['name'])
            opt_model.build_model()
            opt_model.solve()
            Scenario_results[scenario['name']] = {
                'summary': opt_model.get_summary(),
                'hourly': opt_model.get_hourly_results()
            }