from ...base_model import BaseModel
import logging
from typing import Optional, Dict, Any

class SpringMassDamper(BaseModel):
    """
    Spring-Mass-Damper system model implementation.
    
    This model simulates a classic spring-mass-damper system where:
    - A mass is connected to a spring and damper
    - External forces and disturbances can be applied
    - The system dynamics follow: m*a = F_external + F_disturbance - k*x - c*v
    
    Inputs:
        - force: External force applied to the mass (N)
        - disturbance: Disturbance force (N)
    
    Outputs:
        - position: Position of the mass (m)
        - velocity: Velocity of the mass (m/s)
        - acceleration: Acceleration of the mass (m/s²)
    
    Parameters:
        - mass: Mass of the object (kg)
        - stiffness: Spring stiffness coefficient (N/m)
        - damping: Damping coefficient (N⋅s/m)
    """
    MODEL_NAME = "example_model"

    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)
             

    def initialize(self):
        """
        Initialize the spring-mass-damper model.
        
        Sets up initial conditions and validates parameters.
        """
        # Physics simulation additional state variables
        self.previous_time = 0.0
        self.solver_method = None
        
        # self.logger.debug(f"Spring-Mass-Damper model '{self.name}' created")
        # Get solver method from user-defined config
        self.solver_method = self.config.user_defined.get('solver', 'euler')
        
        # Validate parameters
        if self.state.parameters['mass'] <= 0:
            raise ValueError("Mass must be positive")
        if self.state.parameters['stiffness'] < 0:
            raise ValueError("Stiffness must be non-negative")
        if self.state.parameters['damping'] < 0:
            raise ValueError("Damping must be non-negative")
        

    def step(self) -> None:
        """
        Execute one simulation step of the spring-mass-damper system.
        
        Implements the equation of motion: m*a = F_total - k*x - c*v
        where F_total = F_external + F_disturbance
        
        Args:
            current_time: Current simulation time
            time_step: Time step size (uses self.ts if not provided)
        """
        time_step = 0.1
            
        # Get previous outputs states
        x = self.state.outputs['position']        # position (m)
        v = self.state.outputs['velocity']        # velocity (m/s)
        
        # Get parameters
        m = self.state.parameters['mass']       # mass (kg)
        k = self.state.parameters['stiffness']  # stiffness (N/m)
        c = self.state.parameters['damping']    # damping (N⋅s/m)
        
        # Get inputs of this step
        F_ext = self.state.inputs.get('force', 0.0)       # external force (N)
        F_dist = self.state.inputs.get('disturbance', 0.0) # disturbance force (N)
        
        # Calculate total force
        F_total = F_ext + F_dist
        
        # Calculate acceleration using Newton's second law
        # F_net = F_total - F_spring - F_damping
        # m*a = F_total - k*x - c*v
        a = (F_total - k * x - c * v) / m
        
        # Numerical integration based on solver method
        if self.solver_method == 'euler':
            # Forward Euler integration
            x_new = x + v * time_step
            v_new = v + a * time_step
        elif self.solver_method == 'rk4':
            # Runge-Kutta 4th order integration
            x_new, v_new = self._rk4_step(x, v, F_total, m, k, c, time_step)
        else:
            # Default to Euler if unknown solver
            self.logger.warning(f"Unknown solver '{self.solver_method}', using Euler")
            x_new = x + v * time_step
            v_new = v + a * time_step
        
        # Update state outputs MANDATORY STEP!
        self.state.outputs['position'] = x_new
        self.state.outputs['velocity'] = v_new
        self.state.outputs['acceleration'] = a
        
       
        
       
    def _rk4_step(self, x: float, v: float, F_total: float, m: float, k: float, c: float, dt: float) -> tuple[float, float]:
        """
        Runge-Kutta 4th order integration step for the spring-mass-damper system.
        
        Args:
            x: Current position
            v: Current velocity  
            F_total: Total applied force
            m: Mass
            k: Spring stiffness
            c: Damping coefficient
            dt: Time step
            
        Returns:
            Tuple of (new_position, new_velocity)
        """
        def derivatives(pos: float, vel: float) -> tuple[float, float]:
            """Calculate derivatives (velocity, acceleration)"""
            acc = (F_total - k * pos - c * vel) / m
            return vel, acc
        
        # RK4 coefficients
        k1_x, k1_v = derivatives(x, v)
        k2_x, k2_v = derivatives(x + 0.5*dt*k1_x, v + 0.5*dt*k1_v)
        k3_x, k3_v = derivatives(x + 0.5*dt*k2_x, v + 0.5*dt*k2_v)
        k4_x, k4_v = derivatives(x + dt*k3_x, v + dt*k3_v)
        
        # Update using weighted average
        x_new = x + (dt/6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        v_new = v + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        return x_new, v_new

    def finalize(self) -> None:
        """
        Finalize the spring-mass-damper model simulation.
        
        Logs final state and performs cleanup.
        """
        self.logger.info(f"Spring-Mass-Damper model '{self.name}' finalized")
        self.logger.info(f"Final state - Position: {self.state['position']:.3f}m, "
                        f"Velocity: {self.state['velocity']:.3f}m/s, "
                        f"Acceleration: {self.state['acceleration']:.3f}m/s²")
        
        # Calculate final energy for analysis
        kinetic_energy = 0.5 * self.parameters['mass'] * self.state['velocity']**2
        potential_energy = 0.5 * self.parameters['stiffness'] * self.state['position']**2
        total_energy = kinetic_energy + potential_energy
        
        self.logger.info(f"Final energy - Kinetic: {kinetic_energy:.3f}J, "
                        f"Potential: {potential_energy:.3f}J, Total: {total_energy:.3f}J")


