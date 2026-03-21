from ...base_model import BaseModel


class SimplePIDController(BaseModel):
    """
    Discrete-time PID controller for building temperature regulation.

    Computes a modulation output (0–1) to drive a heat pump so that the
    indoor temperature tracks a setpoint.

    Control law (incremental / parallel form with anti-windup clamp):
        error    = setpoint - T_indoor
        P        = Kp * error
        I       += Ki * error * dt          (clamped to [0, 1])
        D        = Kd * (error - prev_error) / dt
        output   = clip(P + I + D, 0, 1)

    Inputs:
        - T_indoor : Measured indoor temperature [°C]

    Outputs:
        - modulation : Compressor modulation signal sent to heat pump [-] (0–1)

    Parameters:
        - T_setpoint : Desired indoor temperature [°C]
        - Kp         : Proportional gain [-/°C]
        - Ki         : Integral gain [-/(°C·s)]
        - Kd         : Derivative gain [-·s/°C]
    """

    MODEL_NAME = "simple_pid_controller"

    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)

    def initialize(self):
        """Reset integrator and previous-error memory."""
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self) -> None:
        """Compute one PID step and update the modulation output."""
        T_indoor = self.state.inputs.get("T_indoor", 20.0)
        setpoint = self.state.parameters["T_setpoint"]
        Kp = self.state.parameters["Kp"]
        Ki = self.state.parameters["Ki"]
        Kd = self.state.parameters["Kd"]

        dt = float(self.real_period)  # seconds per simulation step

        error = setpoint - T_indoor

        # Proportional term
        P = Kp * error

        # Integral term with anti-windup clamp
        self._integral += Ki * error * dt
        self._integral = max(0.0, min(1.0, self._integral))

        # Derivative term (backward difference)
        D = Kd * (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        # Output clamped to valid modulation range
        modulation = max(0.0, min(1.0, P + self._integral + D))

        self.state.outputs["modulation"] = modulation

    def finalize(self):
        self.logger.info(
            f"PID '{self.name}' finalized. "
            f"Last modulation: {self.state.outputs.get('modulation', 0):.3f}"
        )
