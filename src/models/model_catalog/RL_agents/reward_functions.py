"""
Reward Functions Library

Centralised collection of reward functions for use across different scenarios.
Each function must follow the signature:

    reward_fn(obs, action, prev_obs=None, **kwargs) -> float

where:
    obs       – current observation dict (keys = federation.federate.instance.variable)
    action    – action taken (int, float, or dict)
    prev_obs  – previous observation dict (may be None on first step)
    **kwargs  – reserved for future extensions

Add a new function here and reference it in the scenario YAML via:
    reinforcement_learning_config:
      agent:
        reward_function: "models.model_catalog.RL_agents.reward_functions.<function_name>"
"""


# ---------------------------------------------------------------------------
# Spring-mass-damper scenario
# ---------------------------------------------------------------------------

def spring_oscillation_reward(obs, action, prev_obs=None, **kwargs) -> float:
    """
    Encourages the spring-mass-damper to oscillate between +1 m and -1 m.

    Terms:
        amplitude_reward  – peaks at 1.0 when |position| == 1 m
        velocity_bonus    – small bonus for non-zero velocity (keeps motion alive)
        overshoot_penalty – extra quadratic cost when |position| > 1 m
    """
    try:
        position = float(obs['federation_1.spring_federate.0.position'])
        velocity = float(obs['federation_1.spring_federate.0.velocity']) if len(obs) > 1 else 0.0

        amplitude_reward  = 1.0 - (abs(position) - 1.0) ** 2
        velocity_bonus    = 0.1 * min(abs(velocity), 2.0)
        overshoot_penalty = max(0.0, abs(position) - 1.0) ** 2

        return amplitude_reward + velocity_bonus - overshoot_penalty
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Building + heat-pump scenario
# ---------------------------------------------------------------------------

def heatpump_comfort_efficiency_reward(obs, action, prev_obs=None, **kwargs) -> float:
    """
    Balances thermal comfort and energy efficiency for the building/heat-pump scenario.

    Terms:
        comfort_reward   – Gaussian penalty around the 20 °C setpoint
                           (max = 1.0 at T_indoor == T_setpoint, ~0 beyond ±5 °C)
        efficiency_penalty – penalises high modulation (energy use), scaled in [0, 0.3]

    Args:
        obs: must contain 'federation_1.building.0.T_indoor'
        action: modulation value in [0, 1] (int index for discrete action space)
    """
    T_SETPOINT = 20.0
    COMFORT_SIGMA = 3.0   # °C half-width of comfort zone
    EFFICIENCY_WEIGHT = 0.3

    try:
        T_indoor = float(obs['federation_1.building.0.T_indoor'])

        # Gaussian comfort reward: 1.0 at setpoint, decays with temperature deviation
        comfort_reward = float(
            __import__('math').exp(-0.5 * ((T_indoor - T_SETPOINT) / COMFORT_SIGMA) ** 2)
        )

        # Normalised modulation (action is a discrete bin index 0..N-1; treat as fraction)
        if isinstance(action, dict):
            mod = float(next(iter(action.values())))
        else:
            mod = float(action)
        # Normalise to [0,1] if the action is a large integer index
        # (assumes max discrete bins ~ 10; harmless for continuous)
        mod_norm = min(mod / 10.0, 1.0) if mod > 1.0 else mod
        efficiency_penalty = EFFICIENCY_WEIGHT * mod_norm

        return comfort_reward - efficiency_penalty
    except Exception:
        return 0.0



def building_heatpump_comfort(obs, action, prev_obs=None, **kwargs) -> float:
    """
    Comfort-only reward for the building/heat-pump scenario.

    Encourages maintaining indoor temperature close to 20 °C, ignoring energy use.
    """
    T_SETPOINT = 20.0
    COMFORT_SIGMA = 0.5    # °C half-width of comfort zone

    try:
        T_indoor = float(obs['federation_1.building.0.T_indoor'])
       
        # Negative quadratic penalty: 0 at setpoint, negative otherwise
        comfort_reward = -((T_indoor - T_SETPOINT) / COMFORT_SIGMA) ** 2
        return comfort_reward
    except Exception:
        return 0.0



# def soc_reward(obs, action, prev_obs=None, **kwargs) -> float:
#     try:
#         soc = float(obs['federation_1.battery_federate.0.SOC'])
#         p_clipped = float(obs['federation_1.battery_federate.0.P_clipped'])

#         # convert action to scalar
#         # a = float(action[0]) if hasattr(action, "__len__") else float(action)

#         # 1) SOC penalty: only outside [0.40, 0.60]
#         if soc < 0.40:
#             soc_penalty = (0.40 - soc) ** 2
#         elif soc > 0.60:
#             soc_penalty = (soc - 0.60) ** 2
#         else:
#             soc_penalty = 0.0

#         # 2) minimize absolute clipped power
#         clipped_penalty = abs(p_clipped)

#         # 3) discourage always choosing extreme action
#         # action_penalty = a ** 2

#         reward = -(
#             100000.0 * soc_penalty +
#             1.0 * clipped_penalty 
#         )

#         return reward

#     except Exception:
#         return -100.0

def soc_reward(obs, action, prev_obs=None, **kwargs) -> float:
   
    try:
      
        
        # Get battery power and SOC and p_clipped
        p_battery = float(obs.get('federation_1.battery_federate.0.P_net')) # Get >0 carico <0 scarico la batteria
        soc = float(obs['federation_1.battery_federate.0.SOC']) # Get SOC
        p_clipped = float(obs.get('federation_1.battery_federate.0.P_clipped'))  # Get >0 prendo dalla rete <0 immissione in rete

       
        
        # Battery power penalty: this penalize too much usage of the battery to avoid excessive cycling and degradation. The penalty is proportional to the absolute value of the battery power, encouraging the agent to use the battery more efficiently and avoid unnecessary charging/discharging. 
        power_penalty = abs(p_battery) 
        
        # minimize P_clipped
        w_imp = 1.0  # weight for grid import
        w_imm = 0.2 # weight for grid immission
        power_grid_penalty = w_imp * max(0.0, p_clipped) + w_imm * max(0.0, - p_clipped)
        
        # SOC penalty: φ(SOC_t) - piecewise function
        if soc < 0.40:
            soc_penalty = (0.40 - soc) ** 2
        elif soc > 0.60:
            soc_penalty = (soc - 0.60) ** 2
        else:
            soc_penalty = 0.0
        
        # Combined cost: J = P²_bat + λ * φ(SOC)
        alpha=0.5

        beta=1.5
        gamma=100000000.0
        # cost = gamma * soc_penalty + alpha * power_penalty + beta * power_grid_penalty  
        cost = gamma * soc_penalty + beta * power_grid_penalty  
        # Reward is negative cost (to maximize reward)
        reward = -cost
        
        return reward
    

    except Exception:
        return 0.0


def soc_band_clip_simple(obs, action, prev_obs=None, **kwargs) -> float:
    """
    Simple, well-scaled reward for battery case studies:
      1) keep SOC inside [0.40, 0.60]
      2) minimize absolute clipped power
      3) mildly discourage extreme actions

    Expected keys in obs:
      - federation_1.battery_federate.0.SOC
      - federation_1.battery_federate.0.P_clipped
    """
    try:
        soc = float(obs['federation_1.battery_federate.0.SOC'])
        p_clipped = float(obs.get('federation_1.battery_federate.0.P_clipped', 0.0))

        # Action can be scalar, array, or dict depending on wrappers.
        if isinstance(action, dict):
            a_raw = float(next(iter(action.values())))
        elif hasattr(action, "__len__") and not isinstance(action, (str, bytes)):
            a_raw = float(action[0])
        else:
            a_raw = float(action)

        a_norm = max(-1.0, min(1.0, a_raw / 8000.0))

        # Normalized band violation (linear): 0 in-band, increases outside.
        soc_low = max(0.0, 0.40 - soc)
        soc_high = max(0.0, soc - 0.60)
        soc_violation = (soc_low + soc_high) / 0.20

        # Keep SOC close to the center of the admissible range (0.50).
        center_cost = abs(soc - 0.50) / 0.10

        # Minimize clipped power (normalized by 8 kW scale).
        clipped_cost = abs(p_clipped) / 8000.0
        action_cost = abs(a_norm)

        # Directional shaping:
        # - below band: encourage charging (a>0), strongly penalize discharging (a<0)
        # - above band: encourage discharging (a<0), strongly penalize charging (a>0)
        direction_term = 0.0
        if soc < 0.45:
            direction_term = max(0.0, a_norm) - 2.0 * max(0.0, -a_norm)
        elif soc > 0.55:
            direction_term = max(0.0, -a_norm) - 2.0 * max(0.0, a_norm)

        in_band_bonus = 1.5 if 0.40 <= soc <= 0.60 else 0.0

        reward = (
            in_band_bonus
            - 5.0 * soc_violation
            - 0.10 * center_cost
            - 0.10 * clipped_cost
            - 0.05 * action_cost
            + 0.60 * direction_term
        )
        return float(reward)
    except Exception:
        return 0.0
