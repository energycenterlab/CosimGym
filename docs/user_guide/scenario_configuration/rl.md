# Reinforcement Learning Configuration








### Action Spaces:
Action spaces are configurable following a specific logic. First of all the RL federate only deals with Gym.Dict spaces (this hold for every spaces). so in the agent itself wrappers will be needed to convert to desired format (e.g. stablebaseline3 only accept box or discrete, flatten spaces).

For each action in the scenario it is possible to specify the following keys:

* actions : (required) List of action variables

    * rules: the action must be expressed with its complete id following a dot notation : <federation_name>.<federate_name>.<model_instance_number>.<attr_name>
        * example:
        
        ```yaml
        action_name: [federation_1.battery_federate.0.Battery_power] 
      
        ```
* action_spaces_type : (required) List of corresponding types for each action specified in the previous list
