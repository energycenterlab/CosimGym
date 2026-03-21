import numpy as np
class BESS:
    def __init__(self, real_period, **kwargs):
        """

        :param kwargs:
        """
        # load parameters
        kwargs = {k.lower(): v for k, v in kwargs.items()}

        if 'rated_capacity' in kwargs:
            self.rated_capacity = kwargs['rated_capacity']
        else:
            self.rated_capacity = 300

        if 'maximum_charge_power' in kwargs:
            self.maximum_charge_power = kwargs['maximum_charge_power']
        else:
            self.maximum_charge_power = 65

        if 'maximum_discharge_power' in kwargs:
            self.maximum_discharge_power = kwargs['maximum_discharge_power']
        else:
            self.maximum_discharge_power = 65

        if 'soc_upper_limit' in kwargs:
            self.SOC_upper_limit = kwargs['soc_upper_limit']
        else:
            self.SOC_upper_limit = 0.95

        if 'soc_lower_limit' in kwargs:
            self.SOC_lower_limit = kwargs['soc_lower_limit']
        else:
            self.SOC_lower_limit = 0.25

        if 'charge_efficiency' in kwargs:
            self.charge_efficiency = kwargs['charge_efficiency']
        else:
            self.charge_efficiency = 1

        if 'discharge_efficiency' in kwargs:
            self.discharge_efficiency = kwargs['discharge_efficiency']
        else:
            self.discharge_efficiency = 1

        if 'self_discharge_rate' in kwargs:
            self.self_discharge_rate = kwargs['self_discharge_rate']
        else:
            self.self_discharge_rate = 0

        self.P_net=0.0
        self.P_clipped=0.0
        self.SOC = 0.75
        self.real_period = real_period

    def setSOC(self, SOC):
        """
        set SOC, especially when initializing the BESS
        :param SOC:
        :return:
        """
        if SOC >= self.SOC_lower_limit and SOC <= self.SOC_upper_limit:
            self.SOC = SOC
        elif SOC < self.SOC_lower_limit:
            self.SOC = self.SOC_lower_limit
        else:
            self.SOC = self.SOC_upper_limit

    def getlegalpower(self, power):
        """
        get the legal value of charging/discharging power
        :param power: the expected charging/discharging power; (+) means charging, (-) means discharging
        :return: the legal value of power, which is below the maximum of power
        """
        if power > 0:  # charging
            power_legal = min(power, self.maximum_charge_power)
        elif power < 0:  # discharging
            power_legal = max(power, -self.maximum_discharge_power)
        else:
            power_legal = 0
        return power_legal

    def calculatepower(self, power_command, P_load, PV_power, dt):
        """
        calculate the charging/discharging process of BESS by a timestep; after charging/discharging, SOC is updated
        :param power: the expected charging/discharging power; (+) means charging, (-) means discharging
        :param dt: timestep (s)
        :return: the actual energy received or supplied by BESS within a timestep
        """
        net_power =  PV_power - P_load  # net load power; (+) means net load, (-) means net surplus
        if net_power >= 0 and power_command > 0:
            power_command = min(power_command, net_power)  # the battery cannot charge more than the net load
        elif net_power < 0 and power_command < 0:
            power_command = max(power_command, net_power)  # the battery cannot discharge more than the net surplus
        
        # elif net_power > 0 and power_command < 0:
        #     power_command = max(power_command, - net_power)  # 
        # elif net_power < 0 and power_command > 0:
        #     power_command = min(power_command, - net_power)  # the battery cannot charge more than the net surplus
        
        self.P_net = self.getlegalpower(power_command) # P_net>0 means charging, P_net<0 means discharging

        # Enforce SOC-feasible power so the battery cannot keep charging/discharging
        # once the SOC bound has been reached.
        step_scale = dt / self.real_period if self.real_period else 0.0
        if step_scale > 0:
            if self.P_net >= 0:
                eff_charge = self.charge_efficiency
                if eff_charge <= 0:
                    eff_charge = 1e-12
                p_max_soc = (
                    (self.SOC_upper_limit - self.SOC) * self.rated_capacity
                ) / (step_scale * eff_charge)
                self.P_net = min(self.P_net, max(0.0, p_max_soc))
            else:
                eff_discharge = 1.0 / self.discharge_efficiency if self.discharge_efficiency > 0 else 1e12
                p_min_soc = (
                    (self.SOC_lower_limit - self.SOC) * self.rated_capacity
                ) / (step_scale * eff_discharge)
                self.P_net = max(self.P_net, min(0.0, p_min_soc))
        
        

        self.P_clipped = net_power - self.P_net # the power that is clipped due to the maximum charge/discharge power limit; P_clipped>0 means grid import, P_clipped<0 means grid immission

        if self.P_net >= 0:
            efficiency = self.charge_efficiency
        else:
            efficiency = 1 / self.discharge_efficiency

        SOC = (self.SOC * self.rated_capacity + self.P_net * dt / self.real_period * efficiency) / self.rated_capacity
        
        Energy_last = self.getE()
        self.setSOC(SOC)
        Energy_now = self.getE()
        Energy_out = (Energy_now - Energy_last) / efficiency

        self.Energy_out = Energy_out



    def getE(self):
        """
        :return: the current remaining energy in BESS
        """
        return self.SOC * self.rated_capacity

    def getSOC(self):
        """
        :return: the current SOC of the BESS
        """
        return self.SOC
