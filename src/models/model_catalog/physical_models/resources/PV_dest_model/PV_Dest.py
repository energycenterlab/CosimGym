import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('src/models/model_catalog/physical_models/resources/PV_dest_model/')
from sun import *
import math


class PV_model:
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)
        #TODO parametrized all the following
        #self.SVF_hori = 1
        #self.Reflectance = 0.2
        self.sun = site(self.std_long, self.long, self.lat)  # 默认为北京时间

        # self.eta_pv_stc = 0.21
        # self.NOCT = 45
        # self.Power_rated_pv = 410  # W，没有倾角
        # self.length = 2.05
        # self.width = 1.02
        self.S_PV = self.length * self.width
        self.N_PV = int(self.calc_area * self.area_ratio / self.S_PV)
        #self.solar_constant = 1353

    def step(self, t, G_H_R, D_H_R, Ambient_temperature):
        """Step of PV model.

    Args:
        t (int): Time step, an integer representing the hour of the day.
        G_H_R (float): Global horizontal radiation [W/m2].
        D_H_R (float): Diffuse horizontal radiation [W/m2].
        Ambient_temperature (float): External temperature [C].

    Returns:
        Power_PV (float): Power produced by the PV panel [W].
    """
        B_H_R = G_H_R - D_H_R

        self.sun.set_time_std(t * 3600)
        if (self.sun.sun_rise_time_std > t * 3600) and (self.sun.sun_rise_time_std < (t + 1) * 3600):
            cal_time = (self.sun.sun_rise_time_std + (t + 1) * 3600) / 2
        elif (self.sun.sun_set_time_std > t * 3600) and (self.sun.sun_set_time_std < (t + 1) * 3600):
            cal_time = (self.sun.sun_set_time_std + t * 3600) / 2

        else:
            cal_time = (t + 0.5) * 3600

        self.sun.set_time_std(cal_time)
        alt = self.sun.sun_altitude
        az = self.sun.sun_azimuth

        if (az > 2 * math.pi):
            az_new = az - 2 * math.pi
        elif (az < 0):
            az_new = az + 2 * math.pi
        else:
            az_new = az
        # 根据方位角和高度角计算得到一个固定的顺序号： ix
        ix = int(az_new * 180 / math.pi / 10) * 19 + int(alt * 180 / math.pi / 5)
        #self.ix_list.append(ix)

        # 计算法向直射
        if (alt * 180 / math.pi <= 0):
            B_N_R = 0
        elif (alt * 180 / math.pi <= 10):
            B_N_R = B_H_R * (5.6156 - (alt - 0.1745) * 30.2541)
        elif (alt * 180 / math.pi <= 30):
            B_N_R = B_H_R * (math.sqrt(1229 + math.pow(614 * math.sin(alt), 2)) - 614 * math.sin(alt))
        elif (math.cos(math.pi / 2 - alt) > 0):
            B_N_R = B_H_R / math.cos(math.pi / 2 - alt)
        else:
            B_N_R = 0

        light_coef = 1

        Angle_incidence = math.acos(
            math.cos(alt) * math.sin(self.Tilt_angle) * math.cos(az - self.Azimuth_angle) + math.sin(alt) * math.cos(
                self.Tilt_angle)) * 180 / math.pi
        if (math.cos(Angle_incidence * math.pi / 180) <= 0):
            B_T_R = 0
        else:
            B_T_R = B_N_R * math.cos(Angle_incidence * math.pi / 180) * light_coef
        temp = math.cos(math.pi / 2 - alt) + 0.45665 * math.pow(math.pi / 2 - alt, 0.07) * math.pow(
            96.4936 - (math.pi / 2 - alt), -1.697)
        m = min(38.136, math.pow(temp, -1))
        sky_brightness_coefficient = D_H_R * m / self.solar_constant  # 太阳常数1353

        if (D_H_R == 0) or (alt < 0):
            D_T_R = 0
        else:
            sky_clearness_coefficient = ((D_H_R + B_N_R) / D_H_R + 1.041 * math.pow(math.pi / 2 - alt, 3)) / (
                        1 + 1.041 * math.pow(math.pi / 2 - alt, 3))
            F1, F2 = F1F2(sky_clearness_coefficient, sky_brightness_coefficient, math.pi / 2 - alt)
            a_any_direction = max(0.0, math.cos(Angle_incidence * math.pi / 180))
            b = max(0.087, math.cos(math.pi / 2 - alt))
            I_horizon_any = D_H_R * F2 * math.sin(self.Tilt_angle)
            I_circumsolar_any = D_H_R * F1 * a_any_direction / b
            I_dome_any = D_H_R * (1 - F1) * (1 + math.cos(self.Tilt_angle)) / 2

            D_T_R = self.SVF_hori * (max(0, I_horizon_any) + max(0, I_circumsolar_any) + max(0, I_dome_any))
        if alt >= 0:
            G_R_R = self.Reflectance * G_H_R * (1 - math.cos(self.Tilt_angle)) / 2
        else:
            G_R_R = 0
        G_T_R = B_T_R + D_T_R + G_R_R

        """新算法"""
        gamma_pv = -0.0035  # PVWatts的Temperature Coefficient
        T_inoct = self.NOCT - 0  # 修正NOCT为INOCT，根据安装形式不同。参考：Operating temperature of photovoltaic modules: A survey of pertinent correlations
        T_ref = 25
        T_cell = Ambient_temperature + G_T_R / 800 * (T_inoct - 20)
        P_dc = G_T_R / 1000 * self.Power_rated_pv * (1 + gamma_pv * (T_cell - T_ref))
        Power_PV= self.N_PV * P_dc



        return Power_PV


def F1F2(sky_clearness_coefficient,sky_brightness,zenith):

    matrix = [[-0.0083117, 0.5877285,-0.0620636,-0.0596012, 0.0721249,-0.0220216],
              [ 0.1299457, 0.6825954,-0.1513752,-0.0189325, 0.0659650,-0.0288748],
              [ 0.3296958, 0.4868735,-0.2210958, 0.0554140,-0.0639588,-0.0260542],
              [ 0.5682053, 0.1874525,-0.2951290, 0.1088631,-0.1519229,-0.0139754],
              [ 0.8730280,-0.3920403,-0.3616149, 0.2255647,-0.4620442,0.00124480],
              [ 1.1326077,-1.2367284,-0.4118494, 0.2877813,-0.8230357,0.05586510],
              [ 1.0601591,-1.5999137,-0.3589221, 0.2642124,-1.1272340,0.13106940],
              [ 0.6777470,-0.3272588,-0.2504286, 0.1561313,-1.3765031,0.25062120]]
    if sky_clearness_coefficient<=1.065:
        num = 0
    elif sky_clearness_coefficient <= 1.230:
        num = 1
    elif sky_clearness_coefficient <= 1.500:
        num = 2
    elif sky_clearness_coefficient <= 1.950:
        num = 3
    elif sky_clearness_coefficient <= 2.800:
        num = 4
    elif sky_clearness_coefficient <= 4.500:
        num = 5
    elif sky_clearness_coefficient <= 6.200:
        num = 6
    else:
        num = 7

    F11 = matrix[num][0]
    F12 = matrix[num][1]
    F13 = matrix[num][2]
    F21 = matrix[num][3]
    F22 = matrix[num][4]
    F23 = matrix[num][5]

    F1 = F11 + F12 * sky_brightness + F13 * zenith
    F2 = F21 + F22 * sky_brightness + F23 * zenith

    return F1,F2


if __name__ == '__main__':
    # todo ask zahoru for these params they do not seem correct calc_area
    params={
        'lat':30.867,
        'long': 120.102,
        'calc_area': 100,
        'Tilt_angle': 2.05*1.02*27.25/180*math.pi ,
        'Azimuth_angle': 44.24/180*math.pi ,
        'area_ratio': 1

    }

    meteo = pd.read_csv('/data/ITA_Torino-CaselleEPW.csv')
    #testing
    pv = PV_model(**params)
    t=0
    P_record = []
    while t<8760:
        G_hor = meteo.iloc[t]['GloHorzRad']
        G_diff = meteo.iloc[t]['DifHorzRad']
        T_ext = meteo.iloc[t]['DryBulb']
        P_out = pv.step(t, G_hor, G_diff, T_ext)
        P_record.append(P_out)
        t+=1

    res_df = pd.DataFrame(columns=['datetime', 'PV(kWh/h)'])
    res_df['datetime'] = pd.date_range(start='20230101', end='20240101', freq='1h')
    res_df.set_index('datetime')
    res_df['PV(kWh/h)'] = P_record

    plt.plot(P_record)
    plt.show()