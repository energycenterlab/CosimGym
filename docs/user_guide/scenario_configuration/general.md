# Scenario General Configuration

Generic Scenario information in the yaml files they will appear at first hierarchy level. 

full example:
```yaml
version: "1.0.0"
scenario_name: "pv_batt_test_base"
scenario_description: "Simple test case with PV, Battery, Weather, Rule-based controller, and electric Load"
start_time: "2024-01-01T00:00:00"
end_time: "2024-01-03T00:00:00"
log_level: DEBUG

memory_config: <MemoryConfig>

reinforcement_learning_config: <ReinforcementLearningConfig>

federations: <Dict of FederationConfig per federation>
```


**`scenario_name`** :
- **Type:** String
- **Required:** Yes
- **Meaning:** Name of the scenario
- **Accepted Values:** Any string
- **Rules:** Must be unique; used as identifier in logs and results


**`scenario_description`** :
- **Type:** String
- **Required:** No
- **Meaning:** Description of the scenario
- **Accepted Values:** Any string
- **Rules:** For documentation purposes


**`start_time`** :
- **Type:** string
- **Required:** Yes
- **Meaning:** Random seed for reproducibility
- **Accepted Values:** Any correct datetime string formatted as in rules
- **Rules:** accepted formats: isoformat( "YYYY-MM-DDTHH:MM:SS")


**`end_time`** :
- **Type:** string
- **Required:** Yes
- **Meaning:** Random seed for reproducibility
- **Accepted Values:** Any correct datetime string formatted as in rules
- **Rules:** accepted formats: isoformat( "YYYY-MM-DDTHH:MM:SS")


**`log_level`** :
- **Type:** string
- **Required:** No
- **Meaning:** global logging level
- **Accepted Values:** *[ 'INFO', 'DEBUG', 'WARNING', 'ERROR' ]*
- **Rules:** 


**`seed`** :
- **Type:** Integer
- **Required:** No
- **Meaning:** Random seed for reproducibility
- **Accepted Values:** Any integer number
- **Rules:** For documentation purposes


**`memory_config`** : 

structure config for storage of results: refer to MemoryConfiguration specs

```yaml
    memory_config:
        batch_size: 1000
        attrs: "all"
 ```

- **`batch_size`** :
    - **Type:** Integer
    - **Required:** No
    - **Meaning:** Random seed for reproducibility
    - **Accepted Values:** Any integer number
    - **Rules:** For documentation purposes

- **`attrs`** :
    - **Type:** String | List of strings
    - **Required:** No
    - **Meaning:** Random seed for reproducibility
    - **Accepted Values:** Any integer number
    - **Rules:** For documentation purposes



**`federations`** : 

Dict of structured FederationConfiguration for each federation look at [FederationConfiguration specs](#Federation-Configuration). 

example:

 ```yaml
    federations: 
        federation_name1 : FederationConfiguration
        federation_name2 : FederationConfiguration 
 ```


**`reinforcement_learning_config`** : 

Structured configuration for reinforcement learning task look at [ReinforcementlearningConfiguration specs](#Reinforcement-Learning-Configuration)

example:

 ```yaml
    reinforcement_learning_config: ReinforcementlearningConfiguration
 ```


