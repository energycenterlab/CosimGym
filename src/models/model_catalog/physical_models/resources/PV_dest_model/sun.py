#!/usr/bin/env python

import math

class site:
    # 计算各种太阳参数，以当地太阳时计算
    def __init__(self, std_longitude, longitude, latitude):
        # 初始化经纬度使用角度，赋值给内部变量时转换为弧度
        self._std_longitude = std_longitude * math.pi / 180
        self._longitude = longitude * math.pi / 180
        self._latitude = latitude * math.pi / 180
        self._timezone_offset = 12 * 3600 *(self._longitude - self._std_longitude) / math.pi
        self._local_time = 0
        self._day_in_year_local = 0
        self._day_in_year_std = 0
        self._is_all_day = False
        self._is_all_night = False
        self._cal_sun()

    def _cal_sun(self):
        nday = self._day_in_year_std
        w = 2*(nday+1)*math.pi/360
        # et 时差 单位 hour
        et = -0.0002786409+0.1227715*math.cos(w+1.498311)-0.1654575*math.cos(2*w-1.261546)-0.005353830*math.cos(3*w-1.157100)
        # decl 赤纬角 rad
        decl = (0.3622133-23.24763*math.cos(w+0.1532310)-0.3368908*math.cos(2*w+0.2070988)-0.1852646*math.cos(3*w+0.6201293))*math.pi/180
        # ws 日落时的太阳时角 rad
        x = -math.tan(decl)*math.tan(self._latitude)
        if x > 1 :
            x = 1
            self._is_all_night = True
            self._is_all_day = False
        elif x < -1:
            x = -1
            self._is_all_night = False
            self._is_all_day = True
        else:
            self._is_all_night = False
            self._is_all_day = False
        ws = math.acos(x)
        # 日出时间 单位 hour
        tsr = 12.*(1-ws/math.pi)-et
        # 日落时间 单位 hour
        tss = 12.*(1+ws/math.pi)-et
        khr = self._local_time/3600. - nday * 24 # => 当天时间 hour
        wh = 15.*(khr+et-12.)*math.pi/180.
        # 太阳高度角 rad
        solalt=math.asin(math.sin(self._latitude)*math.sin(decl)+math.cos(self._latitude)*math.cos(decl)*math.cos(wh))
        caz=(math.sin(self._latitude)*math.sin(solalt)-math.sin(decl))/(math.cos(self._latitude)*math.cos(solalt))
        saz=math.cos(decl)*math.sin(wh)/math.cos(solalt)
        if caz>1.0: caz=1.0
        # 太阳方位角
        solaz=0
        if saz<0.:
            solaz=-math.acos(caz)
        elif saz>0.:
            solaz=math.acos(caz)
        else: solaz=0.
        # 记入缓存
        self._altitude = solalt
        self._azimuth = solaz
        self._azimuth_sun_set = ws
        if self._is_all_day:
            self._time_sun_rise = 0
            self._time_sun_set = 0
        elif self._is_all_night:
            self._time_sun_rise = 86400
            self._time_sun_set = 86400
        else:
            self._time_sun_rise = tsr * 3600
            self._time_sun_set = tss * 3600

    def set_time_local(self, local_time) :
        if(self._local_time==local_time): return
        self._local_time = local_time
        self._day_in_year_local = int(self._local_time / 86400)
        self._day_in_year_std = int((self._local_time - self._timezone_offset) / 86400)
        self._cal_sun()
    def set_time_std(self, std_time):
        if self._local_time==(std_time+self._timezone_offset): return
        # 标准时区时间转为当地太阳时间
        self._local_time = std_time + self._timezone_offset
        self._day_in_year_local = int(self._local_time / 86400)
        self._day_in_year_std = int(std_time / 86400)
        self._cal_sun()
    # 以下为计算结果
    @property
    def sun_altitude(self):
        return self._altitude
    @property
    def sun_azimuth(self):
        return self._azimuth
    @property
    def sun_set_azimuth(self):
        return self._azimuth_sun_set
    @property
    def sun_rise_time(self):
        return self._time_sun_rise
    @property
    def sun_set_time(self):
        return self._time_sun_set
    @property
    def sun_rise_time_local(self):
        if self._is_all_day:
            return self._day_in_year_local * 86400
        elif self._is_all_night:
            return (self._day_in_year_local + 1) * 86400
        else:
            return self._time_sun_rise + self._day_in_year_local * 86400
    @property
    def sun_set_time_local(self):
        if self._is_all_day:
            return self._day_in_year_local * 86400
        elif self._is_all_night:
            return (self._day_in_year_local + 1) * 86400
        else:
            return self._time_sun_set + self._day_in_year_local * 86400
    @property
    def sun_rise_time_std(self):
        if self._is_all_day:
            return self._day_in_year_local * 86400
        elif self._is_all_night:
            return (self._day_in_year_local + 1) * 86400
        else:
            tmx = self._time_sun_rise - self._timezone_offset
            return tmx + self._day_in_year_std * 86400
    @property
    def sun_set_time_std(self):
        if self._is_all_day:
            return self._day_in_year_local * 86400
        elif self._is_all_night:
            return (self._day_in_year_local + 1) * 86400
        else:
            tmx = self._time_sun_set - self._timezone_offset
            return tmx + self._day_in_year_std * 86400
    @property
    def day_of_year_local(self):
        return self._day_in_year_local

# 功能测试
# if __name__ == "__main__":
#     st = site(120, 116.467, 39.8)
#     t = 0
#     for t in range(72):
#         st.set_time_std(t * 3600)
#         print("day =", st.day_of_year_local, "hour = ", t, "alt = ", st.sun_altitude, "azi = ", st.sun_azimuth, "rise = ", st.sun_rise_time_std, "set = ", st.sun_set_time_std)
