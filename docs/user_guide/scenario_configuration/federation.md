# Federation Configuration

This is the config structure to define all the need information to build a single federation. There are two main Configuration blocks:
- [FederateConfiguration](#Federate-Configuration)
- [BrokerConfiguration](#Broker-Configuration)

A full examples is reported in the following and then each section is described later.

```yaml
    broker_config: BrokerConfiguration

    federate_configs: 
        fed_1: FederateConfiguration
        fed_2: FederateConfiguration
```

### Broker Configuration

Broker configuration file will be automatized leaving the possibility to reduce user inputs in the config file!

```yaml
broker_config:
      core_type: "zmq"
      port: 23404
      host: "localhost"
      federates: 5    
```
In the following all the fields for broker configuration:

- **`core_type`** :
    - **Type:** Integer
    - **Required:** No
    - **Meaning:** Random seed for reproducibility
    - **Accepted Values:** Any integer number
    - **Rules:** For documentation purposes

- **`port`** :
    - **Type:** String | List of strings
    - **Required:** No
    - **Meaning:** Random seed for reproducibility
    - **Accepted Values:** Any integer number
    - **Rules:** For documentation purposes

- **`host`** :
    - **Type:** String | List of strings
    - **Required:** No
    - **Meaning:** Random seed for reproducibility
    - **Accepted Values:** Any integer number
    - **Rules:** For documentation purposes

- **`federates`** :
    - **Type:** Int
    - **Required:** Yes
    - **Meaning:** number of federates that will connect to this broker (will be automatized from number of federates partecipating to this federation)
    - **Accepted Values:** Any integer number
    - **Rules:** For documentation purposes

