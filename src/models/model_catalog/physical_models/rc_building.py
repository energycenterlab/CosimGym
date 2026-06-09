"""
rc_building.py

Parametrizable 5R1C (ISO 13790 / EN 13790) single-zone building thermal model.

Ported from the RC_BuildingSimulator project's `Zone` thermal core
(https://github.com/architecture-building-systems/RC_BuildingSimulator,
rc_simulator/building_physics.py). The full RC_BuildingSimulator additionally
depends on `supply_system`/`emission_system` modules to convert the thermal
energy demand into fuel/electricity. To keep this integration self-contained and
dependency-free, this port:

  * keeps the 5R1C thermal node equations (t_m / t_s / t_air) verbatim,
  * uses an *ideal* emission system (the phi_*_plus correction terms are zero,
    matching RC_BuildingSimulator's `AirConditioning` / ideal emitter), so the
    computed energy demand is the ideal sensible load to hold the set-point,
  * replaces the supply-system layer with a simple constant-COP conversion to
    electrical power (cop_heating / cop_cooling parameters).

The original code assumes a 1-hour time step (the `c_m / 3600` term in ISO
eq. C.4). Here that 3600 is replaced by the federate `real_period` (seconds per
step) so the model is correct for any time step.

Control action  : t_set_heating / t_set_cooling  (set-points, °C) — drivable by
                  another federate (e.g. an RL agent or rule-based controller),
                  otherwise fall back to the parameter defaults.
Boundary inputs : T_ext (outdoor temp, °C), solar_gains (W), internal_gains (W).
Outputs (state) : T_indoor (air), T_mass, T_surface, T_operative (°C),
                  Q_heating, Q_cooling (W, >=0), energy_demand (signed W,
                  + heating / - cooling), P_elec (W).

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
"""

from ...base_model import BaseModel


class _Zone:
    """Self-contained 5R1C thermal core (ideal-emitter variant).

    Equation references are to Annex C of ISO 13790. State is carried between
    steps through the thermal-mass temperature `t_m_prev`.
    """

    def __init__(self, p, dt):
        # --- geometry / envelope (from parameters) -------------------------
        self.floor_area = p["floor_area"]
        self.mass_area = self.floor_area * 2.5
        self.room_vol = p["room_vol"]
        self.total_internal_area = p["total_internal_area"]
        self.A_t = self.total_internal_area

        # --- 5R1C lumped parameters ----------------------------------------
        self.c_m = p["thermal_capacitance_per_floor_area"] * self.floor_area
        self.h_tr_em = p["u_walls"] * p["walls_area"]      # mass <-> outside
        self.h_tr_w = p["u_windows"] * p["window_area"]    # window (no mass)

        ach_tot = p["ach_infl"] + p["ach_vent"]
        b_ek = 1.0 - (p["ach_vent"] / ach_tot) * p["ventilation_efficiency"]
        self.h_ve_adj = 1200.0 * b_ek * self.room_vol * (ach_tot / 3600.0)
        self.h_tr_ms = 9.1 * self.mass_area               # mass <-> surface
        self.h_tr_is = self.total_internal_area * 3.45    # surface <-> air

        # restrictions on deliverable power (W); large defaults ~ unrestricted
        self.max_heating_energy = p["max_heating_power_per_floor_area"] * self.floor_area
        self.max_cooling_energy = p["max_cooling_power_per_floor_area"] * self.floor_area

        # integration step (s); replaces the hard-coded 3600 of the hourly model
        self.dt = dt

        self.has_heating_demand = False
        self.has_cooling_demand = False

    # --- derived conductances (ISO 13790 C.6-C.8) --------------------------
    @property
    def h_tr_1(self):
        return 1.0 / (1.0 / self.h_ve_adj + 1.0 / self.h_tr_is)

    @property
    def h_tr_2(self):
        return self.h_tr_1 + self.h_tr_w

    @property
    def h_tr_3(self):
        return 1.0 / (1.0 / self.h_tr_2 + 1.0 / self.h_tr_ms)

    @property
    def t_operative(self):
        """C.12"""
        return 0.3 * self.t_air + 0.7 * self.t_s

    # --- heat flows (C.1-C.3); ideal emitter => no phi_*_plus terms --------
    def calc_heat_flow(self, internal_gains, solar_gains):
        self.phi_ia = 0.5 * internal_gains
        self.phi_st = (1.0 - (self.mass_area / self.A_t) -
                       (self.h_tr_w / (9.1 * self.A_t))) * (0.5 * internal_gains + solar_gains)
        self.phi_m = (self.mass_area / self.A_t) * (0.5 * internal_gains + solar_gains)

    def calc_phi_m_tot(self, t_out, energy_demand):
        """C.5 — energy_demand (W) injected to the air node via phi_ia."""
        phi_ia = self.phi_ia + energy_demand   # ideal emitter delivers all load to air
        t_supply = t_out
        self.phi_m_tot = self.phi_m + self.h_tr_em * t_out + \
            self.h_tr_3 * (self.phi_st + self.h_tr_w * t_out + self.h_tr_1 *
                           ((phi_ia / self.h_ve_adj) + t_supply)) / self.h_tr_2
        self._phi_ia_eff = phi_ia

    def calc_t_m_next(self, t_m_prev):
        """C.4 (3600 generalised to dt)."""
        cm_dt = self.c_m / self.dt
        self.t_m_next = ((t_m_prev * (cm_dt - 0.5 * (self.h_tr_3 + self.h_tr_em))) +
                         self.phi_m_tot) / (cm_dt + 0.5 * (self.h_tr_3 + self.h_tr_em))

    def calc_t_m(self, t_m_prev):
        """C.9"""
        self.t_m = (self.t_m_next + t_m_prev) / 2.0

    def calc_t_s(self, t_out):
        """C.10"""
        t_supply = t_out
        self.t_s = (self.h_tr_ms * self.t_m + self.phi_st + self.h_tr_w * t_out +
                    self.h_tr_1 * (t_supply + self._phi_ia_eff / self.h_ve_adj)) / \
                   (self.h_tr_ms + self.h_tr_w + self.h_tr_1)

    def calc_t_air(self, t_out):
        """C.11"""
        t_supply = t_out
        self.t_air = (self.h_tr_is * self.t_s + self.h_ve_adj * t_supply +
                      self._phi_ia_eff) / (self.h_tr_is + self.h_ve_adj)

    def temperatures(self, energy_demand, internal_gains, solar_gains, t_out, t_m_prev):
        self.calc_heat_flow(internal_gains, solar_gains)
        self.calc_phi_m_tot(t_out, energy_demand)
        self.calc_t_m_next(t_m_prev)
        self.calc_t_m(t_m_prev)
        self.calc_t_s(t_out)
        self.calc_t_air(t_out)
        return self.t_air

    # --- demand detection + sizing -----------------------------------------
    def has_demand(self, internal_gains, solar_gains, t_out, t_m_prev, t_set_h, t_set_c):
        t_air_free = self.temperatures(0.0, internal_gains, solar_gains, t_out, t_m_prev)
        if t_air_free < t_set_h:
            self.has_heating_demand, self.has_cooling_demand = True, False
        elif t_air_free > t_set_c:
            self.has_heating_demand, self.has_cooling_demand = False, True
        else:
            self.has_heating_demand = self.has_cooling_demand = False

    def calc_energy_demand(self, internal_gains, solar_gains, t_out, t_m_prev, t_set_h, t_set_c):
        """C.13 — linear interpolation between a free-float run and a 10 W/m2 run."""
        t_air_0 = self.temperatures(0.0, internal_gains, solar_gains, t_out, t_m_prev)
        t_set = t_set_h if self.has_heating_demand else t_set_c
        e_probe = 10.0 * self.floor_area
        t_air_10 = self.temperatures(e_probe, internal_gains, solar_gains, t_out, t_m_prev)

        unrestricted = e_probe * (t_set - t_air_0) / (t_air_10 - t_air_0)
        # clip to deliverable range
        if unrestricted > self.max_heating_energy:
            energy = self.max_heating_energy
        elif unrestricted < self.max_cooling_energy:
            energy = self.max_cooling_energy
        else:
            energy = unrestricted
        return energy

    def solve(self, internal_gains, solar_gains, t_out, t_m_prev, t_set_h, t_set_c):
        self.has_demand(internal_gains, solar_gains, t_out, t_m_prev, t_set_h, t_set_c)
        if not (self.has_heating_demand or self.has_cooling_demand):
            energy_demand = 0.0
        else:
            energy_demand = self.calc_energy_demand(
                internal_gains, solar_gains, t_out, t_m_prev, t_set_h, t_set_c)
        # final temperatures with the resolved load
        self.temperatures(energy_demand, internal_gains, solar_gains, t_out, t_m_prev)
        return energy_demand, self.t_m_next


class RCBuilding(BaseModel):
    """5R1C ISO 13790 single-zone building (ported from RC_BuildingSimulator).

    See module docstring for inputs / outputs / control semantics.
    """

    MODEL_NAME = "rc_building"

    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)

    def _param(self, key):
        return self.state.parameters[key]

    def initialize(self):
        p = self.state.parameters
        self.zone = _Zone(p, dt=self.real_period)

        # thermal-mass state carried between steps
        self.t_m_prev = p["t_m_initial"]

        # seed temperature outputs at the initial mass temperature
        for key in ("T_indoor", "T_mass", "T_surface", "T_operative"):
            if key in self.state.outputs:
                self.state.outputs[key] = self.t_m_prev
                self.init_state.outputs[key] = self.t_m_prev

    def _control(self, key):
        """Input value if connected/seeded, else the parameter default."""
        val = self.state.inputs.get(key)
        return val if val is not None else self.state.parameters[key]

    def step(self) -> None:
        t_out = self.state.inputs.get("T_ext", 10.0)
        solar = self.state.inputs.get("solar_gains")
        if solar is None:
            solar = self._param("default_solar_gains")
        internal = self.state.inputs.get("internal_gains")
        if internal is None:
            internal = self._param("default_internal_gains")

        t_set_h = self._control("t_set_heating")
        t_set_c = self._control("t_set_cooling")

        energy_demand, t_m_next = self.zone.solve(
            internal, solar, t_out, self.t_m_prev, t_set_h, t_set_c)

        # advance stored thermal-mass state
        self.t_m_prev = t_m_next

        q_heat = energy_demand if energy_demand > 0 else 0.0
        q_cool = -energy_demand if energy_demand < 0 else 0.0
        cop_h = self._param("cop_heating")
        cop_c = self._param("cop_cooling")
        p_elec = (q_heat / cop_h if cop_h > 0 else 0.0) + (q_cool / cop_c if cop_c > 0 else 0.0)

        out = self.state.outputs
        if "T_indoor" in out:
            out["T_indoor"] = self.zone.t_air
        if "T_mass" in out:
            out["T_mass"] = self.zone.t_m
        if "T_surface" in out:
            out["T_surface"] = self.zone.t_s
        if "T_operative" in out:
            out["T_operative"] = self.zone.t_operative
        if "Q_heating" in out:
            out["Q_heating"] = q_heat
        if "Q_cooling" in out:
            out["Q_cooling"] = q_cool
        if "energy_demand" in out:
            out["energy_demand"] = energy_demand
        if "P_elec" in out:
            out["P_elec"] = p_elec
        if "P_elec_mw" in out:
            out["P_elec_mw"] = p_elec / 1e6

    def reset(self, mode="full", ts=None, time=None) -> None:
        """Reset interfaces and the internal thermal-mass state."""
        super().reset(mode=mode, ts=ts, time=time)
        self.t_m_prev = self.state.parameters["t_m_initial"]

    def finalize(self):
        self.logger.info(
            f"RC building '{self.name}' finalized. "
            f"Final indoor temperature: {self.state.outputs.get('T_indoor', float('nan')):.2f} °C"
        )
