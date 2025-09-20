from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution of Port Scheduling Problem.
    Contains vessel assignments, tugboat assignments, and timing information.
    
    Attributes:
        vessel_assignments (dict): Maps vessel_id to (berth_id, start_time) tuple. 
                                 None if vessel is unassigned.
        tugboat_inbound_assignments (dict): Maps vessel_id to list of (tugboat_id, start_time) tuples for inbound service.
        tugboat_outbound_assignments (dict): Maps vessel_id to list of (tugboat_id, start_time) tuples for outbound service.
    """
    def __init__(self, vessel_assignments: dict = None, 
                 tugboat_inbound_assignments: dict = None, 
                 tugboat_outbound_assignments: dict = None):
        self.vessel_assignments = vessel_assignments or {}
        self.tugboat_inbound_assignments = tugboat_inbound_assignments or {}
        self.tugboat_outbound_assignments = tugboat_outbound_assignments or {}

    def __str__(self) -> str:
        result = "Port Scheduling Solution:\n"
        result += "Vessel Assignments:\n"
        for vessel_id, assignment in self.vessel_assignments.items():
            if assignment is not None:
                berth_id, start_time = assignment
                result += f"  Vessel {vessel_id}: Berth {berth_id}, Start Time {start_time}\n"
            else:
                result += f"  Vessel {vessel_id}: Unassigned\n"
        
        result += "Inbound Tugboat Services:\n"
        for vessel_id, services in self.tugboat_inbound_assignments.items():
            if services:
                service_str = ", ".join([f"Tugboat {tug_id} at time {start_time}" 
                                       for tug_id, start_time in services])
                result += f"  Vessel {vessel_id}: {service_str}\n"
        
        result += "Outbound Tugboat Services:\n"
        for vessel_id, services in self.tugboat_outbound_assignments.items():
            if services:
                service_str = ", ".join([f"Tugboat {tug_id} at time {start_time}" 
                                       for tug_id, start_time in services])
                result += f"  Vessel {vessel_id}: {service_str}\n"
        
        return result


class AssignVesselOperator(BaseOperator):
    """Assign a vessel to a berth at a specific time."""
    def __init__(self, vessel_id: int, berth_id: int, start_time: int):
        self.vessel_id = vessel_id
        self.berth_id = berth_id
        self.start_time = start_time

    def run(self, solution: Solution) -> Solution:
        new_vessel_assignments = solution.vessel_assignments.copy()
        new_vessel_assignments[self.vessel_id] = (self.berth_id, self.start_time)
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=solution.tugboat_inbound_assignments.copy(),
            tugboat_outbound_assignments=solution.tugboat_outbound_assignments.copy()
        )


class UnassignVesselOperator(BaseOperator):
    """Remove a vessel assignment."""
    def __init__(self, vessel_id: int):
        self.vessel_id = vessel_id

    def run(self, solution: Solution) -> Solution:
        new_vessel_assignments = solution.vessel_assignments.copy()
        new_tugboat_inbound = solution.tugboat_inbound_assignments.copy()
        new_tugboat_outbound = solution.tugboat_outbound_assignments.copy()
        
        # Remove vessel assignment
        new_vessel_assignments[self.vessel_id] = None
        # Remove tugboat services for this vessel
        new_tugboat_inbound[self.vessel_id] = []
        new_tugboat_outbound[self.vessel_id] = []
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=new_tugboat_inbound,
            tugboat_outbound_assignments=new_tugboat_outbound
        )


class AssignInboundTugboatOperator(BaseOperator):
    """Assign tugboat(s) for inbound service to a vessel."""
    def __init__(self, vessel_id: int, tugboat_assignments: list[tuple[int, int]]):
        """
        Args:
            vessel_id: ID of the vessel
            tugboat_assignments: List of (tugboat_id, start_time) tuples
        """
        self.vessel_id = vessel_id
        self.tugboat_assignments = tugboat_assignments

    def run(self, solution: Solution) -> Solution:
        new_tugboat_inbound = solution.tugboat_inbound_assignments.copy()
        new_tugboat_inbound[self.vessel_id] = self.tugboat_assignments.copy()
        
        return Solution(
            vessel_assignments=solution.vessel_assignments.copy(),
            tugboat_inbound_assignments=new_tugboat_inbound,
            tugboat_outbound_assignments=solution.tugboat_outbound_assignments.copy()
        )


class AssignOutboundTugboatOperator(BaseOperator):
    """Assign tugboat(s) for outbound service to a vessel."""
    def __init__(self, vessel_id: int, tugboat_assignments: list[tuple[int, int]]):
        """
        Args:
            vessel_id: ID of the vessel
            tugboat_assignments: List of (tugboat_id, start_time) tuples
        """
        self.vessel_id = vessel_id
        self.tugboat_assignments = tugboat_assignments

    def run(self, solution: Solution) -> Solution:
        new_tugboat_outbound = solution.tugboat_outbound_assignments.copy()
        new_tugboat_outbound[self.vessel_id] = self.tugboat_assignments.copy()
        
        return Solution(
            vessel_assignments=solution.vessel_assignments.copy(),
            tugboat_inbound_assignments=solution.tugboat_inbound_assignments.copy(),
            tugboat_outbound_assignments=new_tugboat_outbound
        )


class SwapVesselAssignmentsOperator(BaseOperator):
    """Swap berth and time assignments between two vessels."""
    def __init__(self, vessel_id1: int, vessel_id2: int):
        self.vessel_id1 = vessel_id1
        self.vessel_id2 = vessel_id2

    def run(self, solution: Solution) -> Solution:
        new_vessel_assignments = solution.vessel_assignments.copy()
        
        # Swap assignments
        assignment1 = new_vessel_assignments.get(self.vessel_id1)
        assignment2 = new_vessel_assignments.get(self.vessel_id2)
        
        new_vessel_assignments[self.vessel_id1] = assignment2
        new_vessel_assignments[self.vessel_id2] = assignment1
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=solution.tugboat_inbound_assignments.copy(),
            tugboat_outbound_assignments=solution.tugboat_outbound_assignments.copy()
        )


class ModifyVesselTimeOperator(BaseOperator):
    """Modify the start time of a vessel's berth assignment."""
    def __init__(self, vessel_id: int, new_start_time: int):
        self.vessel_id = vessel_id
        self.new_start_time = new_start_time

    def run(self, solution: Solution) -> Solution:
        new_vessel_assignments = solution.vessel_assignments.copy()
        
        current_assignment = new_vessel_assignments.get(self.vessel_id)
        if current_assignment is not None:
            berth_id, _ = current_assignment
            new_vessel_assignments[self.vessel_id] = (berth_id, self.new_start_time)
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=solution.tugboat_inbound_assignments.copy(),
            tugboat_outbound_assignments=solution.tugboat_outbound_assignments.copy()
        )


class ReassignVesselBerthOperator(BaseOperator):
    """Reassign a vessel to a different berth while keeping the same start time."""
    def __init__(self, vessel_id: int, new_berth_id: int):
        self.vessel_id = vessel_id
        self.new_berth_id = new_berth_id

    def run(self, solution: Solution) -> Solution:
        new_vessel_assignments = solution.vessel_assignments.copy()
        
        current_assignment = new_vessel_assignments.get(self.vessel_id)
        if current_assignment is not None:
            _, start_time = current_assignment
            new_vessel_assignments[self.vessel_id] = (self.new_berth_id, start_time)
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=solution.tugboat_inbound_assignments.copy(),
            tugboat_outbound_assignments=solution.tugboat_outbound_assignments.copy()
        )


class SwapTugboatAssignmentOperator(BaseOperator):
    """Swap tugboat assignments between two vessels for a specific service type."""
    def __init__(self, vessel_id1: int, vessel_id2: int, service_type: str):
        """
        Args:
            vessel_id1, vessel_id2: Vessel IDs to swap assignments
            service_type: Either 'inbound' or 'outbound'
        """
        self.vessel_id1 = vessel_id1
        self.vessel_id2 = vessel_id2
        self.service_type = service_type

    def run(self, solution: Solution) -> Solution:
        new_tugboat_inbound = solution.tugboat_inbound_assignments.copy()
        new_tugboat_outbound = solution.tugboat_outbound_assignments.copy()
        
        if self.service_type == 'inbound':
            assignment1 = new_tugboat_inbound.get(self.vessel_id1, [])
            assignment2 = new_tugboat_inbound.get(self.vessel_id2, [])
            new_tugboat_inbound[self.vessel_id1] = assignment2.copy()
            new_tugboat_inbound[self.vessel_id2] = assignment1.copy()
        elif self.service_type == 'outbound':
            assignment1 = new_tugboat_outbound.get(self.vessel_id1, [])
            assignment2 = new_tugboat_outbound.get(self.vessel_id2, [])
            new_tugboat_outbound[self.vessel_id1] = assignment2.copy()
            new_tugboat_outbound[self.vessel_id2] = assignment1.copy()
        
        return Solution(
            vessel_assignments=solution.vessel_assignments.copy(),
            tugboat_inbound_assignments=new_tugboat_inbound,
            tugboat_outbound_assignments=new_tugboat_outbound
        )


class CompleteVesselAssignmentOperator(BaseOperator):
    """Assign a vessel to berth with complete tugboat services."""
    def __init__(self, vessel_id: int, berth_id: int, start_time: int, 
                 inbound_tugboats: list[tuple[int, int]], 
                 outbound_tugboats: list[tuple[int, int]]):
        self.vessel_id = vessel_id
        self.berth_id = berth_id
        self.start_time = start_time
        self.inbound_tugboats = inbound_tugboats
        self.outbound_tugboats = outbound_tugboats

    def run(self, solution: Solution) -> Solution:
        new_vessel_assignments = solution.vessel_assignments.copy()
        new_tugboat_inbound = solution.tugboat_inbound_assignments.copy()
        new_tugboat_outbound = solution.tugboat_outbound_assignments.copy()
        
        # Assign vessel to berth
        new_vessel_assignments[self.vessel_id] = (self.berth_id, self.start_time)
        # Assign tugboat services
        new_tugboat_inbound[self.vessel_id] = self.inbound_tugboats.copy()
        new_tugboat_outbound[self.vessel_id] = self.outbound_tugboats.copy()
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=new_tugboat_inbound,
            tugboat_outbound_assignments=new_tugboat_outbound
        )