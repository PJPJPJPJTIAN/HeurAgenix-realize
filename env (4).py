import os
import numpy as np
import ast
from src.problems.base.env import BaseEnv
from src.problems.psp.components import Solution


class Env(BaseEnv):
    """Port Scheduling env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "psp")
        self.construction_steps = self.instance_data["vessel_num"]
        self.key_item = "total_scheduling_cost"
        self.compare = lambda x, y: y - x  # Lower cost is better

    @property
    def is_complete_solution(self) -> bool:
        # A solution is complete when all vessels have been considered (assigned or unassigned)
        assigned_vessels = set(self.current_solution.vessel_assignments.keys())
        total_vessels = set(range(self.instance_data["vessel_num"]))
        return assigned_vessels == total_vessels

    def load_data(self, data_path: str) -> dict:
        """Load port scheduling problem instance data from text file."""
        instance_data = {}
        
        with open(data_path, 'r') as file:
            for line in file:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                
                # Parse parameter = value format
                if '=' in line:
                    param_name, param_value = line.split('=', 1)
                    param_name = param_name.strip()
                    param_value = param_value.strip()
                    
                    # Try to evaluate the value
                    try:
                        # Handle list format
                        if param_value.startswith('[') and param_value.endswith(']'):
                            value = ast.literal_eval(param_value)
                            instance_data[param_name] = np.array(value)
                        else:
                            # Handle single values
                            try:
                                # Try to parse as float first
                                value = float(param_value)
                                # Convert to int if it's a whole number
                                if value == int(value):
                                    value = int(value)
                                instance_data[param_name] = value
                            except ValueError:
                                # Keep as string if can't convert to number
                                instance_data[param_name] = param_value
                    except (ValueError, SyntaxError):
                        print(f"Warning: Could not parse value for {param_name}: {param_value}")
                        instance_data[param_name] = param_value
        
        return instance_data

    def init_solution(self) -> Solution:
        """Initialize an empty solution."""
        vessel_assignments = {i: None for i in range(self.instance_data["vessel_num"])}
        tugboat_inbound_assignments = {i: [] for i in range(self.instance_data["vessel_num"])}
        tugboat_outbound_assignments = {i: [] for i in range(self.instance_data["vessel_num"])}
        
        return Solution(
            vessel_assignments=vessel_assignments,
            tugboat_inbound_assignments=tugboat_inbound_assignments,
            tugboat_outbound_assignments=tugboat_outbound_assignments
        )

    def get_key_value(self, solution: Solution = None) -> float:
        """Calculate the total scheduling cost of the solution."""
        if solution is None:
            solution = self.current_solution
        
        total_cost = 0.0
        
        # Z1: Unserved vessel penalty
        unserved_penalty = 0.0
        for vessel_id in range(self.instance_data["vessel_num"]):
            if solution.vessel_assignments.get(vessel_id) is None:
                unserved_penalty += self.instance_data["M"] * self.instance_data["alpha"][vessel_id]
        
        # Z2: Total port time cost
        port_time_cost = 0.0
        for vessel_id in range(self.instance_data["vessel_num"]):
            assignment = solution.vessel_assignments.get(vessel_id)
            if assignment is not None:
                _, start_time = assignment
                # Calculate inbound and outbound service times
                inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
                outbound_services = solution.tugboat_outbound_assignments.get(vessel_id, [])
                
                if inbound_services and outbound_services:
                    inbound_start = min(service[1] for service in inbound_services)
                    outbound_start = max(service[1] for service in outbound_services)
                    
                    total_time = (outbound_start + self.instance_data["tau_out"][vessel_id]) - inbound_start
                    port_time_cost += (self.instance_data["alpha"][vessel_id] * 
                                     self.instance_data["beta"][vessel_id] * total_time)
        
        # Z3: ETA deviation cost
        eta_deviation_cost = 0.0
        for vessel_id in range(self.instance_data["vessel_num"]):
            assignment = solution.vessel_assignments.get(vessel_id)
            if assignment is not None:
                _, start_time = assignment
                eta = self.instance_data["vessel_etas"][vessel_id]
                
                early_deviation = max(0, eta - start_time)
                late_deviation = max(0, start_time - eta)
                
                eta_deviation_cost += (self.instance_data["alpha"][vessel_id] * 
                                     self.instance_data["gamma"][vessel_id] * 
                                     (early_deviation + late_deviation))
        
        # Z4: Tugboat utilization cost
        tugboat_cost = 0.0
        for vessel_id in range(self.instance_data["vessel_num"]):
            # Inbound tugboat services
            for tugboat_id, start_time in solution.tugboat_inbound_assignments.get(vessel_id, []):
                tugboat_cost += (self.instance_data["c_k"][tugboat_id] * 
                               self.instance_data["tau_in"][vessel_id])
            
            # Outbound tugboat services
            for tugboat_id, start_time in solution.tugboat_outbound_assignments.get(vessel_id, []):
                tugboat_cost += (self.instance_data["c_k"][tugboat_id] * 
                               self.instance_data["tau_out"][vessel_id])
        
        # Total weighted cost
        total_cost = (self.instance_data["lambda_1"] * unserved_penalty +
                     self.instance_data["lambda_2"] * port_time_cost +
                     self.instance_data["lambda_3"] * eta_deviation_cost +
                     self.instance_data["lambda_4"] * tugboat_cost)
        
        return total_cost

    def helper_function(self) -> dict:
        """Return helper functions for the problem."""
        return {
            "get_problem_state": self.get_problem_state,
            "validation_solution": self.validation_solution
        }

    def validation_solution(self, solution: Solution = None) -> bool:
        """
        Check the validation of the solution:
        1. Vessel-berth compatibility (vessel size <= berth capacity)
        2. Berth capacity constraints (no time conflicts)
        3. Tugboat horsepower requirements
        4. Tugboat availability constraints
        5. Time window constraints
        6. Service sequence constraints (inbound before outbound)
        """
        if solution is None:
            solution = self.current_solution

        if not isinstance(solution, Solution):
            return False

        # Check vessel-berth compatibility
        for vessel_id, assignment in solution.vessel_assignments.items():
            if assignment is not None:
                berth_id, start_time = assignment
                
                # Check if berth exists
                if not (0 <= berth_id < self.instance_data["berth_num"]):
                    return False
                
                # Check vessel-berth compatibility
                vessel_size = self.instance_data["vessel_sizes"][vessel_id]
                berth_capacity = self.instance_data["berth_capacities"][berth_id]
                if vessel_size > berth_capacity:
                    return False
                
                # Check time bounds
                if not (0 <= start_time < self.instance_data["time_periods"]):
                    return False
                
                # Check if vessel fits in time period
                duration = self.instance_data["vessel_durations"][vessel_id]
                if start_time + duration > self.instance_data["time_periods"]:
                    return False

        # Check berth capacity constraints (no overlapping assignments)
        berth_schedules = {}
        for vessel_id, assignment in solution.vessel_assignments.items():
            if assignment is not None:
                berth_id, start_time = assignment
                duration = self.instance_data["vessel_durations"][vessel_id]
                
                if berth_id not in berth_schedules:
                    berth_schedules[berth_id] = []
                
                # Check for overlapping time periods
                for existing_start, existing_end in berth_schedules[berth_id]:
                    if not (start_time + duration <= existing_start or start_time >= existing_end):
                        return False
                
                berth_schedules[berth_id].append((start_time, start_time + duration))

        # Check tugboat assignments
        for vessel_id in range(self.instance_data["vessel_num"]):
            assignment = solution.vessel_assignments.get(vessel_id)
            
            if assignment is not None:
                # Check tugboat horsepower requirements
                inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
                outbound_services = solution.tugboat_outbound_assignments.get(vessel_id, [])
                
                # Must have both inbound and outbound services for assigned vessels
                if not inbound_services or not outbound_services:
                    return False
                
                # Check horsepower requirements
                required_hp = self.instance_data["P_req"][vessel_id]
                
                inbound_total_hp = sum(self.instance_data["P_k"][tug_id] for tug_id, _ in inbound_services)
                if inbound_total_hp < required_hp:
                    return False
                
                outbound_total_hp = sum(self.instance_data["P_k"][tug_id] for tug_id, _ in outbound_services)
                if outbound_total_hp < required_hp:
                    return False
                
                # Check tugboat quantity limits
                if len(inbound_services) > self.instance_data["H_max"]:
                    return False
                if len(outbound_services) > self.instance_data["H_max"]:
                    return False
                
                # Check time window constraints
                _, berth_start = assignment
                eta = self.instance_data["vessel_etas"][vessel_id]
                early_limit = self.instance_data["Delta_early"][vessel_id]
                late_limit = self.instance_data["Delta_late"][vessel_id]
                
                if not (eta - early_limit <= berth_start <= eta + late_limit):
                    return False

        # Check tugboat availability (no overlapping services considering preparation time)
        tugboat_schedules = {k: [] for k in range(self.instance_data["tugboat_num"])}
        
        for vessel_id in range(self.instance_data["vessel_num"]):
            # Process inbound services
            for tugboat_id, start_time in solution.tugboat_inbound_assignments.get(vessel_id, []):
                service_duration = self.instance_data["tau_in"][vessel_id]
                prep_time = self.instance_data["rho_in"]
                end_time = start_time + service_duration + prep_time
                
                # Check for conflicts
                for existing_start, existing_end in tugboat_schedules[tugboat_id]:
                    if not (end_time <= existing_start or start_time >= existing_end):
                        return False
                
                tugboat_schedules[tugboat_id].append((start_time, end_time))
            
            # Process outbound services
            for tugboat_id, start_time in solution.tugboat_outbound_assignments.get(vessel_id, []):
                service_duration = self.instance_data["tau_out"][vessel_id]
                prep_time = self.instance_data["rho_out"]
                end_time = start_time + service_duration + prep_time
                
                # Check for conflicts
                for existing_start, existing_end in tugboat_schedules[tugboat_id]:
                    if not (end_time <= existing_start or start_time >= existing_end):
                        return False
                
                tugboat_schedules[tugboat_id].append((start_time, end_time))

        return True