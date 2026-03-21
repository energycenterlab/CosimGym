# Federate Configuration
This Configuration structure concern all the data for settin up a Federate along with co-simulation related aspects, model instances aspects, Interfaces specification.

The main compolex datastructures are :

- [Timing Configurations](#Timings-Configurations)
- [Flag Configurations](#Flags-Configurations)
- [List of Interfaces](#Interfaces-List) (Enpoints/Subscriptions/Publications)
- [Model Configurations](#Model-Configurations)


```yaml
    federate_name:
        name: "weather_federate"
        log_level: DEBUG
        type: "base"
        core_name: "fed1"
        core_type: "zmq"

        timing_configs: <TimingConfigurations>

        flags: <FlagConfigurations>

        connections:
          endpoints: <ListofEndpoints>
          subscribes: <ListofSubscriptions> (Inputs)
          publishes: <ListofPublications> (Outputs)

        model_configs: <ModelConfigurations>
```

- **`name`** :
    - **Type:** Integer
    - **Required:** No
    - **Meaning:** Random seed for reproducibility
    - **Accepted Values:** Any integer number
    - **Rules:** For documentation purposes

- **`log_level`** :
    - **Type:** String | List of strings
    - **Required:** No
    - **Meaning:** Random seed for reproducibility
    - **Accepted Values:** Any integer number
    - **Rules:** For documentation purposes

- **`type`** :
    - **Type:** String | List of strings
    - **Required:** YES
    - **Meaning:** Random seed for reproducibility
    - **Accepted Values:** Any integer number
    - **Rules:** For documentation purposes

- **`core_name`** :
    - **Type:** Int
    - **Required:** Yes
    - **Meaning:** number of federates that will connect to this broker (will be automatized from number of federates partecipating to this federation)
    - **Accepted Values:** Any integer number
    - **Rules:** For documentation purposes

- **`core_type`** :
    - **Type:** Int
    - **Required:** Yes
    - **Meaning:** number of federates that will connect to this broker (will be automatized from number of federates partecipating to this federation)
    - **Accepted Values:** Any integer number
    - **Rules:** For documentation purposes

### Timings Configuration
The timing configurations are related to all the aspect that concern synchronization, time clock allignment between simulation time and real-world time

### Flags Configuration
This structured set of information is optional and refers to *HELICS* specific federate flags and other flagging option on time and execution aspects. Refer to the official *HELICS* documentation:
- timing flags [documentation](#https://docs.helics.org/en/helics2/configuration/Timing.html#timing-flags)
- federate flags [documentation](#https://docs.helics.org/en/helics2/configuration/FederateFlags.html)

**Timing Flags**
- **`uninterruptible`** : Bool (True|False)

- **`source_only`** : Bool (True|False)

- **`observer`** : Bool (True|False)

- **`only_update_on_change`** : Bool (True|False)

- **`only_transmit_on_change`** : Bool (True|False)

- **`wait_for_current_time_update`** : Bool (True|False)

⚠️ **Warning** All the others could be imposed but not tested with the actual implementation of this repository v 1.0.0

**Federate Flags**
- **`single_thread_federate`** : Bool (True|False)

- **`ignore_time_mismatch_warnings`** : Bool (True|False)

- **`connections_required`** : Bool (True|False)

- **`connections_optional`** : Bool (True|False)

- **`strict_input_type_checking`** : Bool (True|False)

- **`slow_responding`** : Bool (True|False)

- **`debugging`** : Bool (True|False)

- **`terminate_on_error`** : Bool (True|False)

#### Interfaces List
⚠️ **Warning** Endpoints not implemented

The connections field contain a dict with as keys the three main interfaces type supported by HELICS co-simulation library: Endpoints, Subscriptions and Publications. For all of them the configuration accepts a list of similarly structured items. The only difference is withing some additional fields for the subscription items.

example:

```yaml
        connections:
          endpoints: <ListofEndpoints>
          subscribes: <ListofSubscriptions> (Inputs)
          publishes: <ListofPublications> (Outputs)

```
fields for an Interface item:

**common fields for publishes and subscribes items**

- **`key`** :
    - **Type:** String
    - **Required:** YES
    - **Meaning:** this is the name that will be used by federate and model
    - **Accepted Values:** Any string
    - **Rules:** You can choice any name without considering the other coupled interfaces, but this name must be consistent with the name used inside the model

- **`type`** :
    - **Type:** Int
    - **Required:** YES
    - **Meaning:** number of federates that will connect to this broker (will be automatized from number of federates partecipating to this federation)
    - **Accepted Values:** Any integer number
    - **Rules:** For documentation purposes

- **`units`** :
    - **Type:** String
    - **Required:** YES
    - **Meaning:** unit of measurement
    - **Accepted Values:** Any String
    - **Rules:** units must be cosistent across I/O (publish/subscribe)

**only subscribes items fields**

- **`targets`** :
    - **Type:** Dict | String
    - **Required:** YES
    - **Meaning:** the publication interface to which connect to get the value during co-simulation
    - **Accepted Values:** Any integer number
    - **Rules:** It can be filled with both dict specifying targets for each model instance or as a string that will be applied to all model instances. In case of RL task it must be omitted it will be automatically configured by Scenario Manager

- **`multi_input_handling`** :
    - **Type:** Int
    - **Required:** YES
    - **Meaning:** number of federates that will connect to this broker (will be automatized from number of federates partecipating to this federation)
    - **Accepted Values:** Any integer number
    - **Rules:** For documentation purposes



### Model Configurations






