"""
RL_Federate.py

Implementation of Reinforcement Learning federates using Gymnasium within the Cosim_gym framework.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""
import os
from pathlib import Path
import numpy as np
import importlib
import gymnasium as gym
import helics as h
from datetime import timedelta
from typing import Any, Tuple, Optional, Dict
from core.BaseFederate import BaseFederate
import pprint as pp
pp = pp.PrettyPrinter(indent=4)


def build_space(spec: dict):
    """Recursively build a Gymnasium space from a dictionary specification."""

    space_type = spec["type"].lower()

    if space_type == "box":
        dtype = spec.get("dtype", "float32")
        shape = tuple(spec["shape"]) if spec.get("shape") is not None else None
        if shape is not None:
            low = np.full(shape, spec["low"], dtype=dtype)
            high = np.full(shape, spec["high"], dtype=dtype)
        else:
            low = np.array(spec["low"], dtype=dtype)
            high = np.array(spec["high"], dtype=dtype)
            shape = low.shape
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

    elif space_type == "discrete":
        return gym.spaces.Discrete(spec["n"])

    elif space_type == "multi_discrete":
        return gym.spaces.MultiDiscrete(np.array(spec["nvec"]))

    elif space_type == "multi_binary":
        return gym.spaces.MultiBinary(spec["n"])

    elif space_type == "tuple":
        return gym.spaces.Tuple([build_space(s) for s in spec["spaces"]])
     
    elif space_type == "dict":   # we only use this one that is a composition of all the others
        return gym.spaces.Dict({
            key: build_space(val) for key, val in spec["spaces"].items()
        })

    else:
        raise ValueError(f"Unknown space type '{spec['type']}'")

class HelicsGymEnv(gym.Env):
    """
    Gymnasium Env wrapper around a running RlFederate.

    Delegates reset/step to the federate.
    """

    metadata = {"render_modes": []}

    def __init__(self, rl_federate: Any, logger=None, obs_dict=None, act_dict=None):
        super().__init__()
        self.ts = 0
        self.federate = rl_federate
        self.logger = logger
        self.observation_space = self._create_observation_space(obs_dict) # always readable dict spaces
        self.action_space = self._create_action_space(act_dict) # always readable dict spaces
        self.action = None

    def _create_observation_space(self, obs_dict):
        # create observation space based on RL_Federate config
        self.logger.debug(f"Creating observation space from dict : {obs_dict}")
        return build_space(obs_dict)
        

    def _create_action_space(self, act_dict):
        # create action space based on RL_Federate config
        self.logger.debug(f"Creating action space from dict : {act_dict}")
        return build_space(act_dict)
    
    def agent_to_env_action(self, action):
        """
        Convert whatever the agent outputs into the OrderedDict that
        gym.spaces.Dict expects.

        Handles:
          - dict          → pass through
          - flat np.array → gym.spaces.utils.unflatten (canonical way)
          - list / tuple  → zip with dict keys in insertion order
          - scalar        → single-key dict shortcut
        """
        # 1. already a dict -> nothing to do
        if isinstance(action, dict):
            pass

        # 2. numpy array (flat representation) -> use gymnasium's unflatten
        elif isinstance(action, np.ndarray):
            action = gym.spaces.utils.unflatten(self.action_space, action)

        # 3. list or tuple -> one element per dict key, in order
        elif isinstance(action, (list, tuple)):
            keys = list(self.action_space.spaces.keys())
            if len(action) != len(keys):
                # might be a flat representation packed in a list
                action = gym.spaces.utils.unflatten(
                    self.action_space, np.asarray(action)
                )
            else:
                action = dict(zip(keys, action))

        # 4. scalar (int / float) -> only valid for single-key dicts
        elif np.isscalar(action):
            keys = list(self.action_space.spaces.keys())
            if len(keys) == 1:
                key = keys[0]
                subspace = self.action_space.spaces[key]
                if hasattr(subspace, "n"):          # Discrete
                    action = {key: int(action)}
                else:
                    action = {key: action}
            else:
                raise ValueError(
                    f"Received scalar action {action} but action_space has "
                    f"{len(keys)} keys: {keys}. Cannot map automatically."
                )
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

        # Unwrap single-element numpy arrays to plain Python scalars
        # so HELICS can publish them.
        action = {
            k: (v.item() if hasattr(v, "item") and getattr(v, "size", 1) == 1 else v)
            for k, v in action.items()
        }
        return action 

    def _ensure_obs_shape(self, obs):
        """
        Ensure observations have the correct shape as defined in observation_space.
        Scalar values are wrapped in (1,) arrays, arrays are reshaped as needed.
        
        This is critical because predict() requires strict shape compliance while learn() is more lenient.
        """
        if not isinstance(obs, dict):
            return obs
        
        shaped_obs = {}
        for key, value in obs.items():
            if key not in self.observation_space.spaces:
                shaped_obs[key] = value
                continue
            
            expected_shape = self.observation_space.spaces[key].shape
            
            # Convert to numpy array if needed
            if not isinstance(value, np.ndarray):
                value = np.array([value], dtype=np.float32)
            
            # Reshape to match observation space
            if value.shape != expected_shape:
                value = value.reshape(expected_shape)
            
            shaped_obs[key] = value
        
        return shaped_obs

    def step(self, action):
        # Ensure the action is always in the original Dict-space format that RL_Federate.step() expects.
        # gym.spaces.utils.unflatten reconstructs the OrderedDict from a flat array (if the agent
        # flattened the space) and is a no-op when the action is already a dict.
        self.logger.debug(f"Action received by HelicsGymEnv step: {action} and axtion space= {self.action_space}")
        action = self.agent_to_env_action(action)
        self.action = action

        # Unwrap single-element numpy arrays to plain Python scalars so HELICS can publish them.
        self.logger.debug(f"Action received by HelicsGymEnv step and unflattened: {action}")

        

        self.logger.debug(f"Action received by HelicsGymEnv step and prepared for FEDERATE: {action}")
        try:
            obs, terminated, truncated, info, reward = self.federate.step(action)
        except Exception as e:
            self.logger.error(f"Error occurred while stepping through federate: {e}")
            raise
        
        # Ensure observations have correct shape for model.predict()
        obs = self._ensure_obs_shape(obs)
        
        # reward = 1 if terminated else 0 
        self.ts += 1

        return obs, reward, terminated, truncated, info


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)
        _ = options
        obs , info = self.federate.reset_episode(old_action= self.action)
        
        # Ensure observations have correct shape for model.predict()
        obs = self._ensure_obs_shape(obs)
        
        return obs, info


class RL_Federate(BaseFederate):
    def __init__(self, name, config, logger, simid, federation_name=None):
        super().__init__(name, config, logger, simid, federation_name)
        self.agent_instances = None # Placeholder for RL agent instance
        self.agent = None
        self.env = None
        self.rl_task = config.rl_task

        # datastrcutures
        self.observations = None
        self.terminated = False
        self.truncated = False

        # utils
        # TODO: restructure the mapping features for both cases discrete with negative and choice of n of bins
        # TODO: add proper action_mapping_dict in this method _prepare_act_dict
        self.action_mapping = None # this used for discretize in more bin a continous action

        # rl_configs
        self.episode_length = self.rl_task.training.episode_length if self.rl_task.training is not None else None
        self.n_episodes = self.rl_task.training.n_episodes if self.rl_task.training is not None else None
        self.reset_observation_defaults = self.config.reset_observation_defaults or {}
        env_cfg = getattr(getattr(self.rl_task, "agent", None), "env", None)
        self.force_reset_observation_defaults = bool(
            getattr(env_cfg, "force_reset_observation_defaults", False)
        )

        # RL-specific episode tracking
        self._current_episode_reward = 0.0
        self._current_episode_steps = 0
        self._episode_count = 0
        # Observation that the policy used to pick the next action.
        self._obs_for_action = None
        
        self.logger.info(f"RL_Federate {self.name} initialized with config: {self.config}")

    def initialize_storage(self):
        """
        Initialize RL-specific storage partitioned by mode (train / test).

        Structure per partition:
            time:            [datetime, ...]
            granted_time:    [float, ...]
            observations:    {obs_key: [values...]}
            observations_before_action: {obs_key: [values...]}
            observations_after_action:  {obs_key: [values...]}
            actions:         {act_key: [values...]}
            rewards:         [float, ...]
            episode_ids:     [int, ...]          # which episode each step belongs to
            episode_rewards: [float, ...]        # total reward per episode (one entry per episode)
            episode_lengths: [int, ...]          # steps per episode
        """
        self.storage = {
            'train': self._create_rl_storage_partition(),
            'test': self._create_rl_storage_partition(),
        }

    def _create_rl_storage_partition(self):
        """Create one empty RL storage partition."""
        obs_keys = list(self.env.observation_space.spaces.keys()) if self.env else []
        act_keys = list(self.env.action_space.spaces.keys()) if self.env else []

        return {
            'time': [],
            'granted_time': [],
            'observations': {k: [] for k in obs_keys},
            'observations_before_action': {k: [] for k in obs_keys},
            'observations_after_action': {k: [] for k in obs_keys},
            'actions': {k: [] for k in act_keys},
            'rewards': [],
            'episode_ids': [],
            'episode_rewards': [],
            'episode_lengths': [],
        }

    def update_storage(self, obs=None, action=None, reward=None, obs_before=None, obs_after=None):
        """
        Append one transition to the current mode's storage partition.

        Args:
            obs: backward-compatible alias for obs_after
            action: dict of action values (keys = act space keys)
            reward: float reward for this step
            obs_before: observation seen by the policy before applying `action`
            obs_after: observation resulting after time advancement
        """
        mode = getattr(self, 'mode', 'test') or 'test'
        partition = self.storage.get(mode, self.storage['test'])
        obs_after = obs if obs_after is None else obs_after

        partition['time'].append(self.date_time)
        partition['granted_time'].append(float(self.time_granted))
        partition['episode_ids'].append(self._episode_count)

        # observations after action (legacy key kept for compatibility + explicit key)
        for key in partition['observations'].keys():
            after_val = obs_after.get(key, None) if obs_after is not None else None
            if hasattr(after_val, 'tolist'):
                after_val = after_val.tolist()
            partition['observations'][key].append(after_val)
            partition['observations_after_action'][key].append(after_val)

            before_val = obs_before.get(key, None) if obs_before is not None else None
            if hasattr(before_val, 'tolist'):
                before_val = before_val.tolist()
            partition['observations_before_action'][key].append(before_val)

        # actions
        if action is not None:
            for key in partition['actions'].keys():
                val = action.get(key, None)
                if hasattr(val, 'tolist'):
                    val = val.tolist()
                partition['actions'][key].append(val)

        # reward
        if reward is not None:
            partition['rewards'].append(float(reward))
            self._current_episode_reward += float(reward)
            self._current_episode_steps += 1

    def on_episode_end(self):
        """
        Called when an episode ends (terminated or truncated).
        Records episode-level aggregates and resets episode accumulators.
        """
        mode = getattr(self, 'mode', 'test') or 'test'
        partition = self.storage.get(mode, self.storage['test'])

        partition['episode_rewards'].append(self._current_episode_reward)
        partition['episode_lengths'].append(self._current_episode_steps)

        self.logger.info(
            f"[{mode.upper()}] Episode {self._episode_count} ended — "
            f"steps: {self._current_episode_steps}, "
            f"total_reward: {self._current_episode_reward:.4f}"
        )

        # reset accumulators
        self._current_episode_reward = 0.0
        self._current_episode_steps = 0
        self._episode_count += 1

    def store_local_file(self):
        """Store each mode partition as a separate JSON file."""
        import json
        scenario_name = self.simulation_id[:-16]
        sim_id = self.simulation_id[-15:]
        repo_root = Path(__file__).resolve().parents[2]
        base_dir = os.path.join(
            str(repo_root / "results"),
            scenario_name, sim_id, self.federation_name,
        )
        os.makedirs(base_dir, exist_ok=True)

        for mode_key, partition in self.storage.items():
            # Skip empty partitions (no timesteps recorded)
            if not partition['time']:
                continue
            filename = f"{self.name}_{mode_key}_rl_storage.json"
            file_path = os.path.join(base_dir, filename)
            with open(file_path, 'w') as f:
                json.dump(partition, f, default=str, indent=4)
            self.logger.info(f"RL storage data ({mode_key}) saved to {file_path}")
    
    def _register_entities(self):
        agent_conf = self.rl_task.agent

       
        #  instantiate the environment
        obs_dict = self._prepare_obs_dict()
        act_dict = self._prepare_act_dict()
        self.env = HelicsGymEnv(self, logger=self.logger, obs_dict=obs_dict, act_dict=act_dict)

        # instantiate the agent only single model for now (the model could be multiagent but we keep it simple for now
        model_metadata = self.catalog.get_model_metadata(agent_conf.model_name)
        script = model_metadata.module_path
        class_name = model_metadata.class_name
        module = importlib.import_module(script.replace('/', '.').rstrip('.py'))
        model_class = getattr(module, class_name)
        self.agent = model_class(self.env, logger= self.logger, rl_task=self.rl_task)

        entities = [{'id':self.name,'object': self.agent}]
        return entities

    def _prepare_obs_dict(self):
        obs_list = self.rl_task.agent.env.observations
        prev_obs = self.rl_task.agent.env.include_prev_obs
        obs_dict = {"type": "dict", "spaces": {}}
        params_specs = {}
        for _obs , model_name in self.config.observed_models.items():
            # we need to get the space specification for each observed variable from the model catalog
            # for now we assume all observations are Box spaces, we can extend this later to support different types of spaces
            params_specs[_obs] = self._get_io_specs(_obs, model_name, io_section="outputs")
        
        for i, obs in enumerate(obs_list): 
            # for now we assume all observations are Box spaces, we can extend this later to support different types of spaces
            type= None
            var_name = obs.split('.')[-1]
            if var_name not in params_specs[obs]:
                raise ValueError(
                    f"Missing outputs spec for '{obs}' (model '{self.config.observed_models.get(obs)}'). "
                    f"Ensure catalog override is loaded before federate startup."
                )
            low = getattr(params_specs[obs][var_name], 'min_value', -np.inf)
            high = getattr(params_specs[obs][var_name], 'max_value', np.inf)
            raw_type = params_specs[obs][var_name].type.value

            # TODO: we have to understand space from observation so for now only box spaces, but we can extend this later to support different types of spaces
            if raw_type == 'int':
                type = 'int32' # int should be with discrete space but for ints with negative values is a problem

            elif raw_type == 'float':
                type = 'float32'
              
            else:
                raise ValueError(f"Unsupported type {raw_type} for observation {obs}")

            shape = (1, prev_obs[i]) if prev_obs is not None and prev_obs[i]!=0 else (1,)
            obs_dict["spaces"][obs] = {"type": "box",
                                        "low": low, 
                                        "high": high, 
                                        "shape": shape, 
                                        "dtype": type}
        return obs_dict

    
    def _prepare_act_dict(self):
        # TODO: add a complet remapping logic for discrete actions
        act_list = self.rl_task.agent.env.actions
        act_spaces_type = self.rl_task.agent.env.action_spaces_type
        action_boundaries = self.rl_task.agent.env.action_boundaries
        n_bins = self.rl_task.agent.env.action_bins
        
        act_dict = {"type": "dict", "spaces": {}}
        params_specs = {}
        for _act , model_name in self.config.controlled_models.items():
            # we need to get the space specification for each action variable from the model catalog
            # for now we assume all actions are Box spaces, we can extend this later to support different types of spaces
            params_specs[_act] = self._get_io_specs(_act, model_name, io_section="inputs")
        
        for i, act in enumerate(act_list): 
            type_of_space = act_spaces_type[i]
            type = None
            var_name = act.split('.')[-1]
            if var_name not in params_specs[act]:
                raise ValueError(
                    f"Missing inputs spec for '{act}' (model '{self.config.controlled_models.get(act)}'). "
                    f"Ensure catalog override is loaded before federate startup."
                )
            # getting boundaries or bins or mappings
            low = getattr(params_specs[act][var_name], 'min_value', -np.inf)  # from model catalog 
            high = getattr(params_specs[act][var_name], 'max_value', np.inf)  # from model catalog
            raw_type = params_specs[act][var_name].type.value

            if action_boundaries is not None and len(action_boundaries) > i and action_boundaries[i] is not None:
                low, high = action_boundaries[i]
            
            if n_bins is not None and len(n_bins) > i and n_bins[i] is not None:
                n = n_bins[i]
            else:
                n = int(high - low + 1) 

            # TODO: we have to understand space from action so for now only box spaces, but we can extend this later to support different types of spaces
            if raw_type == 'int' and type_of_space == 'discrete':
                type = 'int32'
                act_dict["spaces"][act]= {"type": "discrete", "n": n}
                # TODO possibly add remapping for discrete with negative values

            elif raw_type == 'int' and type_of_space == 'box':
                self.logger.warning(f"Model input for Action {act} is of type int but action space type is box, we will treat it as a box of int-- NOT optimal")
                type = 'int32'
                shape = (1,)
                act_dict["spaces"][act] = {"type": "box",
                                            "low": low, 
                                            "high": high, 
                                            "shape": shape, 
                                            "dtype": type}

            elif raw_type == 'float' and type_of_space == 'box':
                type = 'float32'
                shape = (1,)
                act_dict["spaces"][act] = {"type": "box",
                                            "low": low, 
                                            "high": high, 
                                            "shape": shape, 
                                            "dtype": type}
            elif raw_type == 'float' and type_of_space == 'discrete':
                self.logger.warning(f"Model input for Action {act} is of type float but action space type is discrete. DISCRETIZE THE ACTION SPACE  and adding a remapping in case of negative min values")
                type = 'int32'
                if self.rl_task.agent.env.action_bins is not None and len(self.rl_task.agent.env.action_bins) > i:    
                    self.action_mapping = {b:round(v, 2) for b,v in zip(range(n), np.linspace(low, high, n))}
                
                act_dict["spaces"][act]= {"type": "discrete", "n": n}
                # if low < 0:
                #     self.action_remapping = (low, high) if low < 0 else (0, high) # if low is negative we need to remap the action space to start from 0 for the agent and then remap back to the original space in the step function
            else:
                raise ValueError(f"Unsupported type {raw_type} for action {act}")

           
        return act_dict

    def _parse_attr_context(self, attr_key):
        parts = attr_key.split('.')
        if len(parts) < 4:
            raise ValueError(
                f"Invalid attribute key '{attr_key}'. Expected format: federation.federate.instance.attr"
            )
        return {
            "federation": parts[0],
            "federate": parts[1],
            "instance": int(parts[2]),
            "var": parts[-1],
        }

    def _get_io_specs(self, attr_key, model_name, io_section):
        ctx = self._parse_attr_context(attr_key)
        specs = self.catalog.get_inputs_outputs(
            model_name,
            simulation_id=self.simulation_id,
            instance_ctx=ctx,
        )
        if specs is None:
            raise ValueError(
                f"No catalog specs found for model '{model_name}' and attribute '{attr_key}'."
            )
        return specs.get(io_section, {})

    
    def run(self):
        h.helicsFederateEnterExecutingMode(self.federate)
        self.logger.info(f'Federate {self.name} entered executing mode. Start simulation loop.')
        # self.logger.info(f'Federate will perform a total number of steps: {self.stop_time}, with a realWolrd frequency of: {self.real_period} s and a simulation frequency of {self.time_period}')
        self.time_granted = 0.0
        self.ts = 0
        self.mode = None #TODO should passs the mode from unique point ScenarioManager
        self._enforce_startup_input_sync()
        try:
            # online training loop
            if self.rl_task.training is not None and self.rl_task.training.mode != 'offline':
                self.logger.info("===================================================================================================================")
                self.logger.info(f"Starting online training for RL_Federate {self.name}")
                self.mode = 'train'
                # self.agent.train(mode='online', max_steps=self.rl_task.training.total_steps, update_every=self.rl_task.training.train_frequency, updates_per_step=self.rl_task.training.gradient_steps, eval_every=self.rl_task.training.eval_frequency, eval_episodes=self.rl_task.training.n_eval_episodes, log_every=self.rl_task.training.log_interval)
                self.agent.online_training_loop()
            # Testing loop 
            if self.rl_task.test is not None :
                self.logger.info("===================================================================================================================")
                self.logger.info(f"Starting testing for RL_Federate {self.name}")
                self.mode = 'test'
                self._episode_count = 0  # reset episode counter for test mode
                self._current_episode_reward = 0.0
                self._current_episode_steps = 0
                # self.agent.train(mode='testing', max_steps=self.rl_task.testing.max_steps, log_every=self.rl_task.testing.log_every)
                self.agent.testing_loop()

            self.logger.info(f"RL_Federate {self.name} run completed.")
        except Exception as e:
            self.logger.error(f"Error during run of RL_Federate {self.name}: {e}")
            raise
        finally:
            # Always persist storage even if an error occurred
            self.store_local_file()

    # def _compute_terminated(self, obs: Dict[str, np.ndarray], action: Any) -> bool:
    #     if self._terminated_fn is None:
    #         return False
    #     return bool(self._terminated_fn(obs=obs, action=action, t=self.granted_time, cfg=self.rl_task))

    def _compute_truncated(self, obs: Dict[str, np.ndarray], action: Any) -> bool:
        if self._truncated_fn is not None:
            return bool(self._truncated_fn(obs=obs, action=action, t=self.granted_time, cfg=self.rl_task))
        return self.granted_time >= self.stop_time
    
    # def remap(x, old_min, old_max, new_min, new_max):
    #     return new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)
    
    def _action_to_publish(self, action):
        self.outputs={}
        self.outputs[self.name] ={} # we need is a dict for multiple model instances in this case is not used but respected
        for act_key, act_value in action.items():
            if self.action_mapping is not None:
                act_value = self.action_mapping[act_value]

            act_key = act_key.split('.')[-1] # we need only the variable name to publish, we can get it from the topic of the subscription
            self.outputs[self.name][act_key] =  act_value # TODO: check if it comes out as a single value that can be published
            
    def _inputs_to_observations(self, use_staged_next_step=False):
        # we need to map the inputs we receive from the federate to the observations we want to give to the agent based on the config
        # obs_key is the full path (e.g. 'federation_1.spring_federate.0.position')
        # inputs are keyed by short variable name (e.g. 'position') extracted from the subscription topic
        obs = {}
        entity_inputs = self.inputs.get(self.name, {})
        entity_meta = self._last_input_meta.get(self.name, {})
        for obs_key in self.env.observation_space.spaces.keys():
            var_name = obs_key.split('.')[-1]
            selected_value = None

            if use_staged_next_step:
                meta = entity_meta.get(var_name, {})
                if meta.get("causality") == "next_step" and meta.get("staged_value") is not None:
                    # Use the newest just-received next_step value for transition alignment.
                    selected_value = meta.get("staged_value")
                elif var_name in entity_inputs:
                    selected_value = entity_inputs[var_name]
            elif var_name in entity_inputs:
                selected_value = entity_inputs[var_name]

            if selected_value is None:
                self.logger.error(f"Observation key {obs_key} (var: {var_name}) not found in inputs for federate {self.name}")
                obs[obs_key] = 0.0 # default value if not found
            else:
                obs[obs_key] = selected_value
        return obs

    def step(self, action=None):
        self.logger.debug(f"RL_Federate {self.name} stepping in mode {self.mode}.")
        self.logger.debug('=======================================================================================================================')
        # self.logger.debug(f'-------------------- Step {self.ts} out of {self.stop_time} in current mode: {self.mode} -------------------------')
        # self.logger.debug(f'Current realworld datetime: {self.date_time} - Current granted time for HELICS: {self.time_granted}')
        try:
        # action to pubs, apply rescale if present in config
            self._action_to_publish(action)

            # publish actions to the corresponding topics
            self._publish_outputs()

            # request time  TODO this blcok decide where it should be...
            
            self.request_time_advance()

            # Apply delayed (next_step) inputs from previous step before reading new updates.
            self._apply_deferred_inputs()

            # get inputs from subscription
            self._receive_inputs()

            # compute observations based on inputs and observed variables.
            # For next_step subscriptions, consume just-read staged values so the
            # transition seen by the RL env is not delayed by an extra step.
            obs = self._inputs_to_observations(use_staged_next_step=True)

            # compute reward based on reward function and observations
            # compute terminated and truncated flags
            terminated = self._compute_terminated() #TODO: self._compute_terminated(obs, action)
            truncated=False #TODO:  this will always be false becuse i canno change episode lenghts to do so i should dinamiccally update information for all federates now it is imposisble
            info = {}
            reward = self.agent.compute_reward(obs, action) if hasattr(self.agent, 'compute_reward') else 0.0
            self.logger.debug(f"Step {self.ts} - Observations: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            # --- STORAGE: record this transition ---
            obs_before_action = dict(self._obs_for_action) if isinstance(self._obs_for_action, dict) else None
            self.update_storage(
                action=action,
                reward=reward,
                obs_before=obs_before_action,
                obs_after=obs,
            )
            self._obs_for_action = dict(obs)

            if terminated or truncated:
                self.on_episode_end()

        except Exception as e:
            self.logger.error(f"Error during step of RL_Federate {self.name}: {e}")
            raise e
        self.logger.info(f"RL_Federate {self.name} step completed.")
        return obs, terminated, truncated, info, reward
    
        
    def _compute_terminated(self):
        # TODO: Check termination logic based episode lenght
        if self.mode == 'train' and self.episode_length is not None and self.ts > 0 and self.ts % self.episode_length == 0:
            return True
        else:            
            return False
    
    def reset_episode(self, old_action=None):
        self.logger.info(f"RL_Federate {self.name} resetting episode.")
        # Episode boundary should not carry staged next_step values from the previous episode.
        self._clear_deferred_inputs()
        # Keep reset at current granted time to preserve global step-count alignment.
        # We read latest values and then override with reset defaults (if available)
        # so the next action is chosen from the reset start state, not terminal state.
        self._receive_inputs(force_read_all=True)
        # compute observations based on inputs and observed variables
        obs = self._inputs_to_observations()
        if self.reset_observation_defaults:
            applied_defaults = {}
            for obs_key, reset_value in self.reset_observation_defaults.items():
                var_name = obs_key.split('.')[-1]
                current_val = obs.get(obs_key, None)
                is_missing = var_name not in self.inputs.get(self.name, {})
                is_invalid = (
                    current_val is None
                    or self._has_non_finite_numeric(current_val)
                    or self._contains_invalid_numeric_sentinel(current_val)
                )
                should_apply_default = self.force_reset_observation_defaults or is_missing or is_invalid
                # Default behavior patches missing/invalid values only.
                # Optionally force defaults at episode boundaries for controlled
                # reset baselines in RL studies.
                if should_apply_default:
                    obs[obs_key] = reset_value
                    self.inputs[self.name][var_name] = reset_value
                    applied_defaults[obs_key] = reset_value
            if applied_defaults:
                self.logger.debug(f"Applied reset observation defaults for {self.name}: {applied_defaults}")
        self._obs_for_action = dict(obs)
        self.logger.info(f"RL_Federate {self.name} episode reset completed.")
        return obs, {}
    
    def finalize(self):
        self.logger.info(f"RL_Federate {self.name} finalizing.")
        # Persist any remaining storage data
        self.store_local_file()
        super().finalize()
        self.logger.info(f"RL_Federate {self.name} finalized.")
