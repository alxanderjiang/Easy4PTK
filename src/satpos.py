#文件名:satpos.py
#Source File Name: satpos.py
#时间、坐标、模型改正函数库
#A pure Python Core Source File for Assistants
#作者: 蒋卓君, 杨泽恩, 黄文静, 钱闯, 武汉理工大学
#Copyright 2025-, by Zhuojun Jiang, Zeen Yang, Wenjing Huang, Chuang Qian, Wuhan University of Technology, China



#以下常数参量基于WGS-84坐标系
#the following constant parameters are based on WGS-84 coordinate system
ea=6378137.0
pi=3.1415926535897932384626433832795
es2=0.00669437999013#第一偏心率的平方
                    #square of the first eccentricity
e22=0.006739496742227#第二偏心率的平方
                     #square of the second eccentricity
#define es2 0.0818191910428*0.0818191910428//第一偏心率的平方
                                           #square of the first eccentricity
#define e22 0.0820944381519*0.0820944381519//第二偏心率的平方
                                           #square of the second eccentricity
clight=299792458.0

from math import sqrt,cos,sin,atan,tan,acos,asin,atan2,factorial,fmod
import numpy as np
import datetime



def blh2xyz(b,l,h):
    #函数:ECEF-BLH转ECEF-XYZ(直角坐标系转大地坐标系)
    # function:ECEF-BLH to ECEF-XYZ (rectangular coordinate system to geodetic coordinate system)
    #输入:坐标b,l,h
    #input:coordinates b,l,h
    #输出:坐标x,y,z
    #output:coordinates x,y,z
    x=0.0
    y=0.0
    z=0.0

    b=b/180.0*pi
    l=l/180.0*pi
    
    n=ea/sqrt(1-es2*pow(sin(b),2))

    x=(n+h)*cos(b)*cos(l)
    y=(n+h)*cos(b)*sin(l)
    z=(n*(1-es2)+h)*sin(b)
    
    return x,y,z


def xyz2blh(x,y,z):
    #函数:ECEF-XYZ转ECEF-BLH(直角坐标系转大地坐标系)
    # function:ECEF-XYZ to ECEF-BLH (rectangular coordinate system to geodetic coordinate system)
    #输入:坐标x,y,z
    #input:coordinates x,y,z
    #输出:坐标b,l,h
    #output:coordinates b,l,h
    
    l = acos(x / sqrt(x**2 +y**2))
    m1 = 1 / sqrt(x**2 + y**2)
    m2 = ea * es2
    m3 = 1 - es2
    temp1 = z / sqrt(x**2 + y**2)
    #以下为迭代
    # The following is the iteration
    while(1):
        temp2 = temp1
        temp1 = m1 * (z + m2 * temp1 / sqrt(1 + m3 * (temp1**2)))
        if(abs(temp1-temp2<1e-12)):
            break
    
    tb = temp1; b = atan(tb)
    n = ea / sqrt(1 - es2 * (sin(b)**2))
    h = sqrt(pow(x, 2) + pow(y, 2)) / cos(b) - ea / sqrt(1 - es2 * pow(sin(b), 2));
    if (y < 0): 
        l = -l
    return b*180/pi,l*180/pi,h

def getazel(rs,rr):
    #函数:卫星高度角、方位角计算
    #Function: Calculation of satellite altitude angle and azimuth angle
    #输入:卫星位置、测站位置(列表)
    #Input: satellite position and station position (list)
    #输出:卫星方位角、高度角
    #output:satellite azimuth and altitude angle

    
    rsb,rsl,rsh=xyz2blh(rs[0],rs[1],rs[2])
    rrb,rrl,rrh=xyz2blh(rr[0],rr[1],rr[2])
    rsblh=[rsb,rsl,rsh]
    rrblh=[rrb,rrl,rrh]

    e=[0.0,0.0,0.0]
    d = sqrt(pow(rs[0] - rr[0], 2) + pow(rs[1] - rr[1], 2) + pow(rs[2] - rr[2], 2))
    e1 = (rs[0] - rr[0]) / d; e2 = (rs[1] - rr[1]) / d; e3 = (rs[2] - rr[2]) / d
    e[0] = rs[0] - rr[0]
    e[1] = rs[1] - rr[1]
    e[2] = rs[2] - rr[2]
    B = rrblh[0] * pi / 180
    L = rrblh[1] * pi / 180
    H= np.array([[-sin(L),cos(L),0],
        [-sin(B) * cos(L),-sin(B) * sin(L),cos(B)],
        [cos(B) * cos(L),cos(B) * sin(L),sin(B)] ])
    e=np.array(e)
    ER=np.dot(H,e)
    E = ER[0]
    N = ER[1]
    U = ER[2]
    az = atan2(E, N)
    el = asin(U / d)
    return az,el


##时间转换函数
##time conversion function
# 通用时构造函数
#universal time constructor
def COMMTIME(year,month,day,hour,minute,second):
    Ct={"year":int(year),
        "month":int(month),
        "day":int(day),
        "hour":int(hour),
        "minute":int(minute),
        "second":float(second)}
    return Ct

# 闰年判断
# leap year judgment
def isYear(year):
    # 输入: 待判断的年份
    # input: year to be judged
    # 返回: 是否闰年的bool变量
    # return: bool variable of leap year or not
    if((year%4==0 and year%100!=0) or year%400==0):
        return True
    else:
        return False

# UNIX时转UTC
# UNIX time to UTC
def time2epoch(unix_timestamp):
    # 输入: UNIX时间戳
    # input: UNIX timestamp
    # 输出: 通用时间系统(年月日)时间(字符串)
    # output: universal time system (year, month and day) time (string)
    # 将Unix时间戳转换为datetime对象
    # Convert Unix timestamps to datetime objects
    dt = datetime.datetime.fromtimestamp(unix_timestamp)
    # 将datetime对象转换为UTC时间
    # Convert a datetime object to UTC time
    utc_time = dt.astimezone(datetime.timezone.utc)

    return str(utc_time)[:-6]

def time2COMMONTIME(UNIXtime):
    # 输入: UNIX时间戳
    # input: UNIX timestamp
    # 输出: 通用时间系统(年月日)时间(字符串)
    # output: universal time system (year, month and day) time (string)
    Days=int(UNIXtime/86400)
    sec=UNIXtime-Days*86400
    mday=[31,28,31,30,31,30,31,31,30,31,30,31,
          31,28,31,30,31,30,31,31,30,31,30,31,
          31,29,31,30,31,30,31,31,30,31,30,31,
          31,28,31,30,31,30,31,31,30,31,30,31 ]
    d=Days%1461
    for mon in range(0,48):
        if(d>=mday[mon]):
            d-=mday[mon]
        else:
            break
    year=1970+int(Days/1461)*4+int(mon/12)
    month=mon%12+1
    day=d+1
    hour=int(sec/3600)
    minute=int( (sec-hour*3600)/60 )
    second=sec-hour*3600-minute*60

    comtime=COMMTIME(year,month,day,hour,minute,second)
    return comtime


# UTC转UNIX
#UTC to UNIX
def epoch2time(epoch):
    # 输入: 通用时字典
    # Input: Universal Time Dictionary
    # 输出: UNIX时间戳
    # output: UNIX timestamp
    year=epoch["year"]
    month=epoch["month"]
    day=epoch["day"]
    hour=epoch["hour"]
    minute=epoch["minute"]
    second=epoch["second"]
    # 通用时距起点经过的Days
    # Days after the start of universal time interval
    Days=0
    for i in range(1970,year+1):
        # 整年数
        # integer year
        if(i!=year):
            Days+=365
            if(isYear(i)):
                Days+=1
        # 不整年数
        # fractional year
        else:
            for j in range(1,month):
                # 大月
                # month with 31 days
                if(j==1 or j==3 or j==5 or j==7 or j==8 or j==10 or j==12):
                    Days+=31
                if(j==4 or j==6 or j==9 or j==11):
                    Days+=30
                if(j==2 and isYear(year)):
                    Days+=29
                if(j==2 and (not isYear(year))):
                    Days+=28
            Days+=day
            Days-=1 #减去最后一天
                    # Subtract the last day
    # 计算UNIX时间戳
    # Calculate UNIX timestamp
    unixtime=Days*86400
    unixtime+=hour*3600
    unixtime+=minute*60
    unixtime+=second
    return unixtime

# GPS时转UNIX时
# GPS time to UNIX time
def gpst2time(week,second):
    # 输入: GPS周, GPS秒
    # input: GPS week, GPS second 
    # 输出: 计算机UNIX时
    # output: computer UNIX time
    unixtime=86400*7*week+second+315964800
    return unixtime

# UNIX时转GPS时
# UNIX time to GPS time
def time2gpst(unixtime):
    # 输入: UNIX时间
    # inpuut:UNIX time
    # 输出: GPS周, GPS秒
    # output: GPS week, GPS second 
    sec=unixtime-315964800
    week=int(sec/(86400*7))
    second=sec-week*86400*7
    return week,second

# GPS时转UTC跳秒
# GPS to UTC leap second correction
def gpst2utc(gpst_unix):
    leap=[1483228800.0 ,1435708800.0 ,1341100800.0 ,1230768000.0 ,1136073600.0 ,
          915148800.0 , 867715200.0 , 820454400.0 , 773020800.0 , 741484800.0 ,
          709948800.0 ,662688000.0 ,631152000.0 ,567993600.0 ,489024000.0 ,
          425865600.0 ,394329600.0 ,362793600.0]
    leap_sec=[-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]
    gps_utc=leap_sec[0]
    for i in range(0,len(leap)):
        if(gpst_unix<leap[i]):
            gps_utc=leap_sec[i]
    return gps_utc


#GPS Klobuchar模型计算电离层延迟
#calculation of ionospheric delay by GPS klobuchar model

def get_ion_GPS(rt,pos,s,ion):
    
    az,el=getazel(s,pos)
    GPSweek,GPSsec=time2gpst(rt)
    
    rrb,rrl,rrh=xyz2blh(pos[0], pos[1],pos[2])
    rrblh=[rrb,rrl,rrh]
    #计算地球中心角
    # Calculate the central angle of the earth
    Phi=0.0137/((el/pi)+0.11)-0.022
    #计算电离层穿刺点的纬度phi1
    #calculate the latitude phi1 of the ionospheric puncture point 

    phiu=rrblh[0]/180.0
    phi1=phiu+Phi*cos(az)
    if(phi1>0.416):
        phi1=0.416
    elif (phi1<-0.416):
        phi1=-0.416
    #计算电离层穿刺点经度lamda1
    #calculate the longitude lamda1 of the ionospheric puncture point 
    lamdau= rrblh[1]/180.0
    lamda1= lamdau+Phi*sin(az)/cos(phi1*pi)
    #计算电离层穿刺点的地磁纬度phim
    # Calculate the geomagnetic latitude phim of the ionospheric puncture point

    phim=phi1+0.064*cos((lamda1-1.617)*pi)
    
    localtime = 43200 * lamda1 + GPSsec #计算当地时（以 GPS 周内秒为基准）
                                        #Calculate local time (based on GPS seconds in a week)
    localtime = localtime- np.floor(localtime / 86400.0) * 86400 #扣除整数天数，得到一天内的地方时秒数
                                                                 # Deduct the integer days to get the local hours and seconds in a day
    #计算电离层延迟的幅度 A1
    # Calculate the amplitude A1 of ionospheric delay
    A1 = ion[0] + phim * (ion[1] + phim * (ion[2] + phim * ion[3]))
    if (A1 < 0):
        A1 = 0
    #计算电离层延迟的周期 P1
    # Calculate the period P1 of ionospheric delay
    P1 = ion[4] + phim * (ion[5] + phim * (ion[6] + phim * ion[7]))
    if (P1 < 72000):
        P1 = 72000
    #计算电离层延迟相位 X1
    # Calculate ionospheric delay phase X1
    X1 = 2 * pi * (localtime- 50400) / P1
    #计算倾斜因子 F
    # Calculate the tilt factor F
    F = 1.0 + 16.0 * pow((0.53- el / pi), 3)
    #模型参数计算完毕，下面根据模型计算电离层延迟 IL1GPS
    # After the model parameters are calculated, the ionospheric delay IL1GPS is calculated according to the model
    if (abs(X1) <= 1.57):
        IL1GPS = clight * (5 * (1e-9) + A1 * (1- 0.5 * X1 * X1 + pow(X1, 4) / 24.0)) *F
    else:
        IL1GPS = 5 * (1e-9) * clight * F
    return IL1GPS

#GPS  Saastamonien 模型计算对流层延迟
#calculation of tropospheric delay by GPS  Saastamonien model
def get_Tropdelay(pos,s):
    _,el=getazel(s,pos)
    #将直角坐标转换为大地坐标
    #xyz to BLH
    rrb,rrl,rrh=xyz2blh(pos[0],pos[1],pos[2])
    posblh=[rrb,rrl,rrh]
    #不在地球上，对流层延迟归零
    #Not on Earth, tropospheric delay returns to zero
    if(posblh[2]<-100.0 or 1E4< posblh[2] or el<=0): 
        return 0.0
    humi=0.7
    h=posblh[2]
    b=posblh[0]*pi/180.0#因为在头文件中坐标转换函数定义为角度值，所以计算前需要复原
                        # Because the coordinate transformation function is defined as an angle value in the header file, it needs to be restored before calculation
    if(posblh[2]<0.0):
        h=0.0 #地面高程归零处理
              # Ground elevation zeroing treatment
    T=15.0-6.5*1e-3*h+273.16
    e=6.108*humi*np.exp((17.15*T-4684.0)/(T-38.45))
    p=1013.25*pow((1-2.2557e-5*h),5.2568)
    z=pi/2.0-el
    trph=0.0022768*p/(cos(z)*(1.0-0.00266*cos(2*b)-0.00028*h/1000.0))
    trpw=0.002277*(1255.0/T+0.05)*e/cos(z)
    trp=trph+trpw
    return trp

# 大地坐标系与站心地平坐标系互转
# xyz to neu
def xyz2neu(xyz,xyz0):
    #原点笛卡尔坐标
    # Cartesian coordinates of origin
    x0=float(xyz0[0])
    y0=float(xyz0[1])
    z0=float(xyz0[2])

    #待转点笛卡尔坐标
    # Cartesian coordinates of the point to be turned
    x=float(xyz[0])
    y=float(xyz[1])
    z=float(xyz[2])

    #笛卡尔坐标差值
    # Cartesian coordinate difference
    dx=x-x0
    dy=y-y0
    dz=z-z0

    #原点地理系坐标
    # origin geographic system coordinates
    b0,l0,h0=xyz2blh(x0,y0,z0)

    #东北天求解
    #ENU solution
    R=[[-sin(b0/180*pi)*cos(l0/180*pi), -sin(b0/180*pi)*sin(l0/180*pi), cos(b0/180*pi)],
       [-sin(l0/180*pi),cos(l0/180*pi),0],
       [cos(b0/180*pi)*cos(l0/180*pi),cos(b0/180*pi)*sin(l0/180*pi),sin(b0/180*pi)]]
    R=np.array(R)
    X0=np.array([dx,dy,dz]).reshape(3,1)

    return list(R.dot(X0).reshape(3,))


# COMMTIME转新儒略日
#CoMMTIME to MJD
def COMMTIME2MJD(ct):
    #数据读取
    #data reading
    Y=ct['year']
    M=ct['month']
    D=ct['day']
    UT=ct['hour']+ct['minute']/60+ct['second']/3600
    #处理闰月
    # Deal with leap month
    y=0.0
    m=0.0
    if(M<=2):
        y=Y-1
        m=M+12
    else:
        y=Y
        m=M
    #计算MJD
    #calculate MJD
    MJD=int(365.25*y)+int(30.6001*(m+1))+D+UT/24-679019.0
    return MJD

#新儒略日转COMMTIME
#MJD to COMMTIME
def MJD2COMMONTIME(MJD):
    JD=MJD+2400000.5
    F=(JD+0.5)-int(JD+0.5)
    a=int(JD+0.5)
    b=a+1537
    c=int((b-122.1)/365.25)
    d=int(365.25*c)
    e = int((b - d) / 30.6001)
    #重建COMMTIME
    #rebuild COMMTIME
    day=b - d - int(30.6001*e)
    month = e - 1 - 12 * int(e / 14)
    year = c - 4715 - int((7 + month) / 10)
    hour = int(F * 24)
    Minute = int((F * 24 - hour) * 60)
    Second = ((F * 24 - hour) * 60 - Minute) * 60

    return COMMTIME(year,month,day,hour,Minute,Second)


#电离层相关
# Ionospheric correlation
def IMF_ion(rr,rs,MF_mode=2,H_ion=400e3):
    #电离层投影
    # Ionospheric projection
                       #圆周率
                       #PI
    Re=6378000              #地球平均半径
                            #Average radius of the earth
    
    _,el=getazel(rs,rr)
    
    if(MF_mode==0):         #不投影, 直接输出VTEC
                            #output VTEC directly without projection
        MF=1
    elif(MF_mode==1):       #普通球体假设, 来源: BDS官方ICD协议
                            #Ordinary Sphere Hypothesis, Source: BDS Official ICD Protocol
        MF=1/sqrt( 1- ( Re*cos(el)/(Re+H_ion) )**2 )
    elif(MF_mode==2):       #MSLM投影
                            #MSLM Projection
        MF=1/sqrt( 1- ( Re*0.9782*sin(pi/2-el)/(Re+H_ion) )**2  )
    elif(MF_mode==3):       #Klobuchar投影
                            #Klobuchar Projection
        MF=1+16*( (0.53-el/pi)**3 )
    else:
        print('No IMF mode, MF_ion will be set as 1.0')
        MF=1.0
    return MF


def get_double_satge(n):
    # 计算奇数项二阶乘
    # Calculate the second factorial of odd terms
    r=1
    if(n==0):
        return 1
    for i in range(1,2*n+1,2):
       r*=i
    return r

def get_Nnm(n,m):
    #计算规划勒让德函数的正则化函数
    # Regularization function of calculation programming Legendre function
    n=abs(n)
    m=abs(m)
    if(m==0):
        r=sqrt( factorial(n-m)*(2*n+1)/factorial(n+m) )
    else:
        r=sqrt( factorial(n-m)*(2*n+1)*2/factorial(n+m) )
    return r

def get_Pnm(n,m,lat_up):
    #计算标准勒让德函数
    #Calculate the standard Legendre function
    n=abs(n)
    m=abs(m)
    if(m==n):
        return get_double_satge(n)* ( (1-sin(lat_up)*sin(lat_up)) )**(n/2)
    elif(n==(m+1)):
        return sin(lat_up)*(2*m+1)*get_Pnm(m,m,lat_up)
    else:
        return ( (2*n-1)*sin(lat_up)*get_Pnm(n-1,m,lat_up)-(n+m-1)*get_Pnm(n-2,m,lat_up) )/( n-m )
    
def get_ion_A0(rt_MJD,lat_up,lng_up):
    
    #首先确定距离观测值最近的MJD奇数整点tp
    #First, determine the odd integer tp of MJD nearest to the observed value
    rt_MJD_float=rt_MJD-int(rt_MJD)
    min=1.1
    tp=0
    for i in range(1,24,2):
        dt=abs(rt_MJD_float-i/24)
        if(dt<min):
            min=dt
            tp=i
    tp=int(rt_MJD)+tp/24
    
    #计算w_k列表
    #calculate w_k list
    omg_k=[2*pi/1, 2*pi/0.5, 2*pi/0.33, 2*pi/14.6, 2*pi/27.0, 2*pi/121.6, 2*pi/182.51, 2*pi/365.25, 2*pi/4028.71, 2*pi/2014.35, 2*pi/1342.90, 2*pi/1007.18]

    
    #非发播系数矩阵
    #Non-broadcast coefficient matrix
    a=[
    [ -0.61, -1.31, -2.00, -0.03, 0.15, -0.48, -0.40, 2.28, -0.16, -0.21, -0.10, -0.13, 0.21, 0.68, 1.06, 0, -0.12],
    [ -0.51, -0.43,  0.34, -0.01, 0.17,  0.02, -0.06, 0.30,  0.44, -0.28, -0.31, -0.17, 0.04, 0.39, -0.12, 0.12, 0 ],
    [ -0.06, -0.05,  0.06,  0.17, 0.15,  0,     0.11, -0.05,-0.16,  0.02,  0.11,  0.04, 0.12, 0.07,  0.02, -0.14, -0.14],
    [  0.01, -0.03,  0.01, -0.01, 0.05, -0.03,  0.05, -0.03,-0.01,  0,    -0.08, -0.04, 0,   -0.02, -0.03,  0,    -0.03],
    [ -0.01, 0, 0.01, 0, 0.01, 0, -0.01,-0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [  0, 0, 0.03, 0.01, 0.02, 0.01, 0, -0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ -0.19, -0.02, 0.12, -0.10, 0.06, 0, -0.02, -0.08, -0.02, -0.07, 0.01, 0.03, 0.15, 0.06, -0.05, -0.03, -0.10],
    [ -0.18, 0.06, -0.55, -0.02, 0.09, -0.08, 0, 0.86, -0.18, -0.05, -0.07, 0.04, 0.14, -0.03, 0.37, -0.11, -0.12],
    [ 1.09, -0.14, -0.21,  0.52, 0.27, 0, 0.11, 0.17, 0.23, 0.35, -0.05, 0.02, -0.60, 0.02, 0.01, 0.27, 0.32 ],
    [ -0.34, -0.09, -1.22, 0.05, 0.15, -0.29, -0.17, 1.58, -0.06, -0.15, 0.00, 0.13, 0.28, -0.08, 0.62, -0.01, -0.04],
    [ -0.13, 0.07, -0.37, 0.05, 0.06, -0.11, -0.07, 0.46, 0.00, -0.04, 0.01, 0.07, 0.09, -0.05, 0.15, -0.01, 0.01], 
    [ -0.06, 0.13, -0.07, 0.03, 0.02, -0.05, -0.05, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ -0.03, 0.08, -0.01, 0.04, 0.01, -0.02, -0.02, -0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    ]
    b=[
    [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ],
    [ 0.23, -0.20, -0.31, 0.16, -0.03, 0.02, 0.04, 0.18, 0.34, 0.45, 0.19, -0.25, -0.12, 0.18, 0.40, -0.09, 0.21 ],
    [ 0.02, -0.08, -0.06, -0.11, 0.15, -0.14, 0.01, 0.01, 0.04, -0.14, -0.05, 0.08, 0.08, -0.01, 0.01, 0.11, -0.12],
    [ 0, -0.02, -0.03, -0.05, -0.01, -0.07, -0.03,-0.01, 0.02, -0.01, 0.03, -0.10, 0.01, 0.05, -0.01, 0.04, 0.00 ],
    [ 0, -0.02, 0.01, 0, -0.01, 0.01, 0, -0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0.01, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ -0.09, 0.07, 0.03, 0.06, 0.09, 0.01, 0.02, 0, -0.04, -0.02, -0.01, 0.01, -0.10, 0, -0.01, 0.02, 0.05 ],
    [  0.15, -0.31, 0.13, 0.05, -0.09, -0.03, 0.06, -0.36, 0.08, 0.05, 0.06, -0.02, -0.05, 0.06, -0.20, 0.04, 0.07 ],
    [  0.50, -0.08, -0.38, 0.36, 0.14, 0.04, 0, 0.25, 0.17, 0.27, -0.03, -0.03, -0.32, -0.10, 0.20, 0.10, 0.30],
    [  0, -0.11, -0.22, 0.01, 0.02, -0.03, -0.01, 0.49, -0.03, -0.02, 0.01, 0.02, 0.04, -0.04, 0.16, -0.02, -0.01],
    [  0.05, 0.03, 0.07, 0.02, -0.01, 0.03, 0.02, -0.04,-0.01, -0.01, 0.02, 0.03, 0.02, -0.04, -0.04, -0.01, 0 ],
    [  0.03, -0.02, 0.04, -0.01, -0.03, 0.02, 0.01, 0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [  0.04, -0.02, -0.04, 0.00, -0.01, 0, 0.01, 0.07, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    #计算beta_j
    #calculate beta_j
    beta_j=[]
    for j in range(0,17):
        t_beta_j=a[0][j]
        for k in range(1,12+1):
            t_beta_j=t_beta_j+a[k][j]*cos(omg_k[k-1]*tp)+b[k][j]*sin(omg_k[k-1]*tp)
        beta_j.append(t_beta_j)
    
    #计算B_j
    #calculate B_j
    n=[3,3,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5]
    m=[0,1,-1,2,-2,3,-3,0,1,-1,2,-2,0,1,-1,2,-2]
    B_j=[]
    for j in range(0,17):
        t_B_j=0.0
        if(m[j]>=0):
            t_B_j=get_Nnm(n[j],m[j])*get_Pnm(n[j],m[j],lat_up)*cos(m[j]*lng_up)
        else:
            t_B_j=get_Nnm(n[j],m[j])*get_Pnm(n[j],m[j],lat_up)*sin(-m[j]*lng_up)
        B_j.append(t_B_j)
    
    #计算电离层预报值A0
    # Calculate ionospheric prediction value A0
    A0=0.0
    for i in range(0,17):
        A0=A0+beta_j[i]*B_j[i]
    return A0



#B2b信号电离层模型改正值计算
# B2b Signal Ionospheric Model Correction Calculation
def get_BDSGIM(rt,ion_param,rr,rs,MF_mode=1):
    
    #B2b信号推荐参数设置
    #B2b signal recommended parameter setting
    H_ion=400000            #电离层薄层高度
                            #Ionospheric thin layer height
    Re=6378000              #地球平均半径
                            #Average radius of the earth
    lng_M=-72.58/180*pi     #地磁北极的地理经度
                            #Geographic longitude of geomagnetic North Pole
    lat_M=80.27/180*pi      #地磁北极的地理纬度
                            #Geographic latitude of geomagnetic North Pole
    
    #站星射线与高度角计算
    #Calculation of star ray and height angle at station 

    az,el=getazel(rs,rr)

    #测站经纬度
    #latitude and longitude of the observation station
    rrb,rrl,rrh=xyz2blh(rr[0],rr[1],rr[2])
    rrb=rrb/180*pi
    rrl=rrl/180*pi

    #电离层穿刺点地心张角计算
    #Calculation of geocentric angle of ionospheric puncture point
    PHI=pi/2-el-asin(Re/(Re+H_ion)*cos(el))

    #电离层穿刺点在地球表面投影的地理经纬度
    #Geographical latitude and longitude of ionospheric puncture point projected on the earth's surface
    lat_g=asin( sin(rrb)*cos(PHI)+cos(rrb)*sin(PHI)*cos(az) )
    lng_g=rrl+atan2(sin(PHI)*sin(az)*cos(rrb),cos(PHI)-sin(rrb)*sin(lat_g))
    # print(PHI,lat_g,lng_g,el)
    #电离层延迟在地球表面投影的地磁经纬度
    #Geomagnetic Latitude and Latitude of Ionospheric Delay Projected on the Earth's Surface
    lat_m=asin(sin(lat_M)*sin(lat_g) + cos(lat_M)*cos(lat_g)*cos(lng_g-lng_M) )
    lng_m=atan2(cos(lat_g)*sin(lng_g-lng_M)*cos(lat_M) , sin(lat_M)*sin(lat_m)-sin(lat_g) )
    # print(lat_m,lng_m)
    #日固坐标系下电离层穿刺点的地磁经纬度
    #Geomagnetic Coordinates of Ionospheric Pierce Points in Sun-Fixed Frame
    rt_c=time2COMMONTIME(rt)
    rt_MJD=COMMTIME2MJD(rt_c)
    S_ion=pi*(1-2*(rt_MJD-int(rt_MJD)))  #平太阳地理经度
                                         #Geographic Precision of Mean Sun
    lat_up=lat_m
    lng_up=lng_m-atan2( sin(S_ion-lng_M), sin(lat_M)*cos(S_ion-lng_M) )
    # print(lat_up,lng_up,rt_c,rt_MJD)
    #归化勒让德函数计算
    #Calculation of normalized Legendre function
    Ai=[]
    n=[0,1,1,1,2,2,2,2,2]
    m=[0,0,1,-1,0,1,-1,2,-2]
    for i in range(9):
        Pnm=get_Pnm(n[i],m[i],lat_up)*get_Nnm(n[i],m[i])
        if(m[i]>=0):
            Ai.append(Pnm*cos(m[i]*lng_up))
        else:
            Ai.append(Pnm*sin(-m[i]*lng_up))

    #电离层非发播预报值A0计算
    #Calculation of ionospheric non-broadcast forecast value A0
    A0=get_ion_A0(rt_MJD,lat_up,lng_up)
    # print(A0)
    #穿刺点垂直电子总量VTEC计算
    #VTEC calculation of vertical electron total at puncture point
    vtec=A0
    for i in range(9):
        vtec+=ion_param[i]*Ai[i]
    
    #VTEC数值重整(来源: CAS官方示例程序bdgim.c)
    #VTEC numerical reconstruction (source: CAS official example program bdgim. c)
    if(ion_param[0]>35.0):
        vtec=max(ion_param[0]/10.0,vtec)
    elif(ion_param[0]>20.0):
        vtec=max(ion_param[0]/8.0,vtec)
    elif(ion_param[0]>12.0):
        vtec=max(ion_param[0]/6.0,vtec)
    else:
        vtec=max(ion_param[0]/4.0,vtec)
    
    # print(vtec)
    
    #投影函数计算
    #Projection function calculation
    
    if(MF_mode==0):         #不投影, 直接输出VTEC
                            #No projection, direct output VTEC
        MF=1
    elif(MF_mode==1):       #普通球体假设, 来源: BDS官方ICD协议
                            #Ordinary sphere assumption, source: BDS official ICD protocol
        MF=1/sqrt( 1- ( Re*cos(el)/(Re+H_ion) )**2 )
    elif(MF_mode==2):       #MSLM投影
                            #MSLM projection
        MF=1/sqrt( 1- ( Re*0.9782*sin(pi/2-el)/(Re+H_ion) )**2  )
    elif(MF_mode==3):       #Klobuchar投影
                            #Klobuchar projection
        MF=1+16*( (0.53-el/pi)**3 )
    else:
        print('No IMF mode, MF will be set as 1.0')
        MF=1
    #print(MF)
    
    #STEC计算
    #STEC calculation
    stec=MF*vtec    
    return stec


#测站天顶对流层干延迟计算
#Zenith tropospheric dry delay calculation at monitoring station
def get_Trop_delay_dry(rr):
    #参数
    #Parameters
    pi=3.1415926535897932384626433832795
    #mode=1 Saastamoninen模型计算对流层干延迟
    #Calculation of tropospheric dry delay using the Saastamoninen model with mode=1
    #将直角坐标转换为大地坐标
    #xyz to BLH
    rrb,rrl,rrh=xyz2blh(rr[0],rr[1],rr[2])
    rrb=rrb/180*pi
    posblh=[rrb,rrl,rrh]
    h=posblh[2]
    if(posblh[2]<0.0):
        h=0.0 #地面高程归零处理
              # Ground elevation zeroing treatment
    p=1013.25*pow((1-2.2557e-5*h),5.2568)
    return 0.002277*p/(1-0.0026*cos(2*rrb)-0.00028*h)

def NMF_insert(rrb,lat_id,avgi,avgi_1,ampi,ampi_1,DOY):
    #参数
    #Parameters
    pi=3.1415926535897932384626433832795
    #计算内插值(二次线性内插)
    #Computational interpolation (quadratic linear interpolation)
    r=0.0
    r=avgi+(avgi_1-avgi)*(rrb-15*lat_id)/15
    r=r + ( ampi+(ampi_1-ampi)*((rrb-15*lat_id)/15)*cos(2*pi*(DOY-28)/365.25) )
    return r

#天顶对流层延迟投影函数(干分量、湿分量)
#Zenith tropospheric delay projection function (dry and wet components)
def NMF(rr,rs,rt_unix):
    #参数
    #Parameters
    pi=3.1415926535897932384626433832795
    #首先计算卫星高度角和测站纬度
    #Firstly, calculate the satellite altitude angle and station latitude

    _,el=getazel(rs,rr)
    rrb,rrl,rrh=xyz2blh(rr[0],rr[1],rr[2])#注意, 本函数所有测站纬度项:rrb均以角度制出现
                                          #Note that all station latitude terms: rrb in this function appear in angular format
    #计算年积日
    #calculate DOY
    rt_c=time2COMMONTIME(rt_unix)
    rt_y=COMMTIME(rt_c['year'],1,1,0,0,0.0)
    rt_y_unix=epoch2time(rt_y)
    DOY=int((rt_unix-rt_y_unix)/86400)+1
    #内插计算投影系数ah,bh,ch;aw,bw,cw
    #Interpolation calculation projection coefficient
    if(rrb<15):
        ah=1.2769934e-3+1.2769934e-3*(cos(2*pi*(DOY-28)/365.25))
        bh=2.9153695e-3+2.9153695e-3*(cos(2*pi*(DOY-28)/365.25))
        ch=62.610505e-3+62.610505e-3*(cos(2*pi*(DOY-28)/365.25))
        aw=5.8021897e-4
        bw=1.4275268e-3
        cw=4.3472961e-2
    elif(rrb>75):
        ah=1.2045996e-3+1.2045996e-3*(cos(2*pi*(DOY-28)/365.25))
        bh=2.9024912e-3+2.9024912e-3*(cos(2*pi*(DOY-28)/365.25))
        ch=64.258455e-3+64.258455e-3*(cos(2*pi*(DOY-28)/365.25))
        aw=6.1641693e-4
        bw=1.7599082e-3
        cw=5.4736038e-2
    else:
        #首先确定数据位于哪两个位置之间
        # First determine where the data is located
        lat_id=int(rrb/15)
        #构建参数表
        #build list of parameters
        ah_l=[[1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3],
              [0,	         1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5]]
        bh_l=[[2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3],
              [0,            2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5]]
        ch_l=[[62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3],
              [0,            9.0128400e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5]]
        
        aw_l=[5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4]
        bw_l=[1.4275268e-3, 1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3]
        cw_l=[4.3472961e-2, 4.6729510e-2, 4.3908931e-2, 4.4626982e-2, 5.4736038e-2]
        
        #参数内插
        #Parameter interpolation
        ah=NMF_insert(rrb,lat_id,ah_l[0][lat_id-1],ah_l[0][lat_id],ah_l[1][lat_id-1],ah_l[1][lat_id],DOY)
        bh=NMF_insert(rrb,lat_id,bh_l[0][lat_id-1],bh_l[0][lat_id],bh_l[1][lat_id-1],bh_l[1][lat_id],DOY)
        ch=NMF_insert(rrb,lat_id,ch_l[0][lat_id-1],ch_l[0][lat_id],ch_l[1][lat_id-1],ch_l[1][lat_id],DOY)

        aw=NMF_insert(rrb,lat_id,aw_l[lat_id-1],aw_l[lat_id],0,0,DOY)
        bw=NMF_insert(rrb,lat_id,bw_l[lat_id-1],bw_l[lat_id],0,0,DOY)
        cw=NMF_insert(rrb,lat_id,cw_l[lat_id-1],cw_l[lat_id],0,0,DOY)
    
    #参数内插完成,计算投影函数
    #Parameter interpolation completed, calculate projection function
    mh_h=( 1+( ah/( 1+bh/( 1+ch ) ) )  )/(  sin(el)+( ah / ( sin(el) + bh / ( sin(el)+ch ) ) ) )
    mh_t=1/sin(el) - ( 1+2.53e-5/(1+5.49e-3/(1+1.14e-3)) ) / ( sin(el)+2.53e-5/(sin(el)+5.49e-3/(sin(el)+1.14e-3)) )
    mh=mh_h+mh_t*rrh/1000

    mw=(1+aw/(1+bw/(1+cw)))/(sin(el)+aw/(sin(el)+bw/(sin(el)+cw)))

    return mh,mw   


def sun_moon_pos_eci(rt_unix):
    #输入: 观测时间
    #Input: Observation time

    #输出: 太阳位置(ECI), 月亮位置(ECI)
    #Output: Sun position (ECI), Moon position (ECI)
    
    #1.计算太阳和月亮在太阳ECI天球系中的位置
    #1. Calculate the position of the sun and moon in the solar ECI celestial system
    ep2000=COMMTIME(2000,1,1,12,0,0.0)
    ep2000=epoch2time(ep2000)
    t=(rt_unix-ep2000)/86400.0/36525.0  #计算观测时间与2000协议地极时间的差值
                                        # Calculate the difference between observation time and 2000 protocol polar time

    #计算天文参数f[5]
    #Calculate astronomical parameter f[5]
    fc=[[134.96340251, 1717915923.2178,  31.8792,  0.051635, -0.00024470],
        [357.52910918,  129596581.0481,  -0.5532,  0.000136, -0.00001149],
        [93.27209062, 1739527262.8478, -12.7512, -0.001037,  0.00000417],
        [297.85019547, 1602961601.2090,  -6.3706,  0.006593, -0.00003169],
        [125.04455501,   -6962890.2665,   7.4722,  0.007702, -0.00005939]]#coefficients for iau 1980 nutation
    
    tt=[t,0.0,0.0,0.0]
    for i in range(1,4):
        tt[i]=tt[i-1]*t
    f=[0.0,0.0,0.0,0.0,0.0]#天文参数结果列表
                           #Astronomical Parameter Result List
                           
    for i in range(5):
        f[i]=fc[i][0]*3600.0
        for j in range(4):
            f[i]=f[i]+fc[i][j+1]*tt[j]
        f[i]=fmod(f[i]*(pi/180.0/3600.0),2.0*pi)
    
    #黄赤夹角
    #obliquity of the ecliptic
    eps=(23.439291-0.0130042*t)/180.0*pi
    
    #计算太阳位置(ECEF)
    #calculate position of the sun
    Ms=(357.5277233+35999.05034*t)/180.0*pi
    ls=( 280.460+36000.770*t+1.914666471*sin(Ms)+0.019994643*sin(2.0*Ms) )/180.0*pi
    rs= 149597870691.0*(1.000140612-0.016708617*cos(Ms)-0.000139589*cos(2.0*Ms)) 
    rsun=[rs*cos(ls),rs*cos(eps)*sin(ls),rs*sin(eps)*sin(ls)]
    
    #计算月亮位置(ECEF)
    #calculate position of the moon
    lm=218.32+481267.883*t+6.29*sin(f[0])-1.27*sin(f[0]-2.0*f[3])+0.66*sin(2.0*f[3])+0.21*sin(2.0*f[0])-0.19*sin(f[1])-0.11*sin(2.0*f[2])
    pm=5.13*sin(f[2])+0.28*sin(f[0]+f[2])-0.28*sin(f[2]-f[0])-0.17*sin(f[2]-2.0*f[3])
    rm=6378137.0/sin((0.9508+0.0518*cos(f[0])+0.0095*cos(f[0]-2.0*f[3])+0.0078*cos(2.0*f[3])+0.0028*cos(2.0*f[0]))*(pi/180.0))
    lm=lm/180.0*pi
    pm=pm/180.0*pi    
    rmoon=[rm*cos(pm)*cos(lm) ,rm*(cos(eps)*cos(pm)*sin(lm)-sin(eps)*sin(pm)) ,rm*( sin(eps)*cos(pm)*sin(lm)+cos(eps)*sin(pm)) ]

    return rsun,rmoon

def eci2ecef(rt_unix):
    ep2000=COMMTIME(2000,1,1,12,0,0.0)
    t=(rt_unix-gpst2utc(rt_unix)-epoch2time(ep2000)+19.0+32.184)/86400.0/36525.0
    t2=t*t
    t3=t*t*t

    #计算天文参数f[5]
    #Calculate astronomical parameter f[5]
    fc=[[134.96340251, 1717915923.2178,  31.8792,  0.051635, -0.00024470],
        [357.52910918,  129596581.0481,  -0.5532,  0.000136, -0.00001149],
        [93.27209062, 1739527262.8478, -12.7512, -0.001037,  0.00000417],
        [297.85019547, 1602961601.2090,  -6.3706,  0.006593, -0.00003169],
        [125.04455501,   -6962890.2665,   7.4722,  0.007702, -0.00005939]]#coefficients for iau 1980 nutation
    
    tt=[t,0.0,0.0,0.0]
    for i in range(1,4):
        tt[i]=tt[i-1]*t
    f=[0.0,0.0,0.0,0.0,0.0]#天文参数结果列表
                           #Astronomical Parameter Result List
    for i in range(5):
        f[i]=fc[i][0]*3600.0
        for j in range(4):
            f[i]=f[i]+fc[i][j+1]*tt[j]
        f[i]=fmod(f[i]*(pi/180.0/3600.0),2.0*pi)

    # iau 1976 precession
    ze=(2306.2181*t+0.30188*t2+0.017998*t3)*(pi/180.0/3600.0)
    th=(2004.3109*t-0.42665*t2-0.041833*t3)*(pi/180.0/3600.0)
    z =(2306.2181*t+1.09468*t2+0.018203*t3)*(pi/180.0/3600.0)
    eps=(84381.448-46.8150*t-0.00059*t2+0.001813*t3)*(pi/180.0/3600.0)

    R1=np.array([[cos(-z),-sin(-z),0.0],
        [sin(-z), cos(z) ,0.0],
        [0.0    , 0.0    ,1.0]])
    R2=np.array([[cos(th),0.0 ,sin(th)],
        [0.0    ,1.0     ,0.0],
        [-sin(th),0.0,cos(th)]])
    R3=np.array([[cos(-ze)    ,-sin(-ze)     ,0.0],
        [sin(-ze),cos(-ze),0.0],
        [0.0,0.0,1.0]])
    
    #P=(R1.dot(R2)).dot(R3)
    P=R3.dot(R2.dot(R1))

    #iau 1980 nutation
    nut=[[   0,   0,   0,   0,   1, -6798.4, -171996, -174.2, 92025,   8.9],
        [   0,   0,   2,  -2,   2,   182.6,  -13187,   -1.6,  5736,  -3.1],
        [   0,   0,   2,   0,   2,    13.7,   -2274,   -0.2,   977,  -0.5],
        [   0,   0,   0,   0,   2, -3399.2,    2062,    0.2,  -895,   0.5],
        [   0,  -1,   0,   0,   0,  -365.3,   -1426,    3.4,    54,  -0.1],
        [   1,   0,   0,   0,   0,    27.6,     712,    0.1,    -7,   0.0],
        [   0,   1,   2,  -2,   2,   121.7,    -517,    1.2,   224,  -0.6],
        [   0,   0,   2,   0,   1,    13.6,    -386,   -0.4,   200,   0.0],
        [   1,   0,   2,   0,   2,     9.1,    -301,    0.0,   129,  -0.1],
        [   0,  -1,   2,  -2,   2,   365.2,     217,   -0.5,   -95,   0.3],
        [  -1,   0,   0,   2,   0,    31.8,     158,    0.0,    -1,   0.0],
        [   0,   0,   2,  -2,   1,   177.8,     129,    0.1,   -70,   0.0],
        [  -1,   0,   2,   0,   2,    27.1,     123,    0.0,   -53,   0.0],
        [   1,   0,   0,   0,   1,    27.7,      63,    0.1,   -33,   0.0],
        [   0,   0,   0,   2,   0,    14.8,      63,    0.0,    -2,   0.0],
        [  -1,   0,   2,   2,   2,     9.6,     -59,    0.0,    26,   0.0],
        [  -1,   0,   0,   0,   1,   -27.4,     -58,   -0.1,    32,   0.0],
        [   1,   0,   2,   0,   1,     9.1,     -51,    0.0,    27,   0.0],
        [  -2,   0,   0,   2,   0,  -205.9,     -48,    0.0,     1,   0.0],
        [  -2,   0,   2,   0,   1,  1305.5,      46,    0.0,   -24,   0.0],
        [   0,   0,   2,   2,   2,     7.1,     -38,    0.0,    16,   0.0],
        [   2,   0,   2,   0,   2,     6.9,     -31,    0.0,    13,   0.0],
        [   2,   0,   0,   0,   0,    13.8,      29,    0.0,    -1,   0.0],
        [   1,   0,   2,  -2,   2,    23.9,      29,    0.0,   -12,   0.0],
        [   0,   0,   2,   0,   0,    13.6,      26,    0.0,    -1,   0.0],
        [   0,   0,   2,  -2,   0,   173.3,     -22,    0.0,     0,   0.0],
        [  -1,   0,   2,   0,   1,    27.0,      21,    0.0,   -10,   0.0],
        [   0,   2,   0,   0,   0,   182.6,      17,   -0.1,     0,   0.0],
        [   0,   2,   2,  -2,   2,    91.3,     -16,    0.1,     7,   0.0],
        [  -1,   0,   0,   2,   1,    32.0,      16,    0.0,    -8,   0.0],
        [   0,   1,   0,   0,   1,   386.0,     -15,    0.0,     9,   0.0],
        [   1,   0,   0,  -2,   1,   -31.7,     -13,    0.0,     7,   0.0],
        [   0,  -1,   0,   0,   1,  -346.6,     -12,    0.0,     6,   0.0],
        [   2,   0,  -2,   0,   0, -1095.2,      11,    0.0,     0,   0.0],
        [  -1,   0,   2,   2,   1,     9.5,     -10,    0.0,     5,   0.0],
        [   1,   0,   2,   2,   2,     5.6,      -8,    0.0,     3,   0.0],
        [   0,  -1,   2,   0,   2,    14.2,      -7,    0.0,     3,   0.0],
        [   0,   0,   2,   2,   1,     7.1,      -7,    0.0,     3,   0.0],
        [   1,   1,   0,  -2,   0,   -34.8,      -7,    0.0,     0,   0.0],
        [   0,   1,   2,   0,   2,    13.2,       7,    0.0,    -3,   0.0],
        [  -2,   0,   0,   2,   1,  -199.8,      -6,    0.0,     3,   0.0],
        [   0,   0,   0,   2,   1,    14.8,      -6,    0.0,     3,   0.0],
        [   2,   0,   2,  -2,   2,    12.8,       6,    0.0,    -3,   0.0],
        [   1,   0,   0,   2,   0,     9.6,       6,    0.0,     0,   0.0],
        [   1,   0,   2,  -2,   1,    23.9,       6,    0.0,    -3,   0.0],
        [   0,   0,   0,  -2,   1,   -14.7,      -5,    0.0,     3,   0.0],
        [   0,  -1,   2,  -2,   1,   346.6,      -5,    0.0,     3,   0.0],
        [   2,   0,   2,   0,   1,     6.9,      -5,    0.0,     3,   0.0],
        [   1,  -1,   0,   0,   0,    29.8,       5,    0.0,     0,   0.0],
        [   1,   0,   0,  -1,   0,   411.8,      -4,    0.0,     0,   0.0],
        [   0,   0,   0,   1,   0,    29.5,      -4,    0.0,     0,   0.0],
        [   0,   1,   0,  -2,   0,   -15.4,      -4,    0.0,     0,   0.0],
        [   1,   0,  -2,   0,   0,   -26.9,       4,    0.0,     0,   0.0],
        [   2,   0,   0,  -2,   1,   212.3,       4,    0.0,    -2,   0.0],
        [   0,   1,   2,  -2,   1,   119.6,       4,    0.0,    -2,   0.0],
        [   1,   1,   0,   0,   0,    25.6,      -3,    0.0,     0,   0.0],
        [   1,  -1,   0,  -1,   0, -3232.9,      -3,    0.0,     0,   0.0],
        [  -1,  -1,   2,   2,   2,     9.8,      -3,    0.0,     1,   0.0],
        [   0,  -1,   2,   2,   2,     7.2,      -3,    0.0,     1,   0.0],
        [   1,  -1,   2,   0,   2,     9.4,      -3,    0.0,     1,   0.0],
        [   3,   0,   2,   0,   2,     5.5,      -3,    0.0,     1,   0.0],
        [  -2,   0,   2,   0,   2,  1615.7,      -3,    0.0,     1,   0.0],
        [   1,   0,   2,   0,   0,     9.1,       3,    0.0,     0,   0.0],
        [  -1,   0,   2,   4,   2,     5.8,      -2,    0.0,     1,   0.0],
        [   1,   0,   0,   0,   2,    27.8,      -2,    0.0,     1,   0.0],
        [  -1,   0,   2,  -2,   1,   -32.6,      -2,    0.0,     1,   0.0],
        [   0,  -2,   2,  -2,   1,  6786.3,      -2,    0.0,     1,   0.0],
        [  -2,   0,   0,   0,   1,   -13.7,      -2,    0.0,     1,   0.0],
        [   2,   0,   0,   0,   1,    13.8,       2,    0.0,    -1,   0.0],
        [   3,   0,   0,   0,   0,     9.2,       2,    0.0,     0,   0.0],
        [   1,   1,   2,   0,   2,     8.9,       2,    0.0,    -1,   0.0],
        [   0,   0,   2,   1,   2,     9.3,       2,    0.0,    -1,   0.0],
        [   1,   0,   0,   2,   1,     9.6,      -1,    0.0,     0,   0.0],
        [   1,   0,   2,   2,   1,     5.6,      -1,    0.0,     1,   0.0],
        [   1,   1,   0,  -2,   1,   -34.7,      -1,    0.0,     0,   0.0],
        [   0,   1,   0,   2,   0,    14.2,      -1,    0.0,     0,   0.0],
        [   0,   1,   2,  -2,   0,   117.5,      -1,    0.0,     0,   0.0],
        [   0,   1,  -2,   2,   0,  -329.8,      -1,    0.0,     0,   0.0],
        [   1,   0,  -2,   2,   0,    23.8,      -1,    0.0,     0,   0.0],
        [   1,   0,  -2,  -2,   0,    -9.5,      -1,    0.0,     0,   0.0],
        [   1,   0,   2,  -2,   0,    32.8,      -1,    0.0,     0,   0.0],
        [   1,   0,   0,  -4,   0,   -10.1,      -1,    0.0,     0,   0.0],
        [   2,   0,   0,  -4,   0,   -15.9,      -1,    0.0,     0,   0.0],
        [   0,   0,   2,   4,   2,     4.8,      -1,    0.0,     0,   0.0],
        [   0,   0,   2,  -1,   2,    25.4,      -1,    0.0,     0,   0.0],
        [  -2,   0,   2,   4,   2,     7.3,      -1,    0.0,     1,   0.0],
        [   2,   0,   2,   2,   2,     4.7,      -1,    0.0,     0,   0.0],
        [   0,  -1,   2,   0,   1,    14.2,      -1,    0.0,     0,   0.0],
        [   0,   0,  -2,   0,   1,   -13.6,      -1,    0.0,     0,   0.0],
        [   0,   0,   4,  -2,   2,    12.7,       1,    0.0,     0,   0.0],
        [   0,   1,   0,   0,   2,   409.2,       1,    0.0,     0,   0.0],
        [   1,   1,   2,  -2,   2,    22.5,       1,    0.0,    -1,   0.0],
        [   3,   0,   2,  -2,   2,     8.7,       1,    0.0,     0,   0.0],
        [  -2,   0,   2,   2,   2,    14.6,       1,    0.0,    -1,   0.0],
        [  -1,   0,   0,   0,   2,   -27.3,       1,    0.0,    -1,   0.0],
        [   0,   0,  -2,   2,   1,  -169.0,       1,    0.0,     0,   0.0],
        [   0,   1,   2,   0,   1,    13.1,       1,    0.0,     0,   0.0],
        [  -1,   0,   4,   0,   2,     9.1,       1,    0.0,     0,   0.0],
        [   2,   1,   0,  -2,   0,   131.7,       1,    0.0,     0,   0.0],
        [   2,   0,   0,   2,   0,     7.1,       1,    0.0,     0,   0.0],
        [   2,   0,   2,  -2,   1,    12.8,       1,    0.0,    -1,   0.0],
        [   2,   0,  -2,   0,   1,  -943.2,       1,    0.0,     0,   0.0],
        [   1,  -1,   0,  -2,   0,   -29.3,       1,    0.0,     0,   0.0],
        [  -1,   0,   0,   1,   1,  -388.3,       1,    0.0,     0,   0.0],
        [  -1,  -1,   0,   2,   1,    35.0,       1,    0.0,     0,   0.0],
        [   0,   1,   0,   1,   0,    27.3,       1,    0.0,     0,   0.0]]

    ang=0.0
    dpsi=0.0
    deps=0.0
    for i in range(106):
        ang=0.0
        for j in range(5):
            ang+=nut[i][j]*f[j]
        dpsi=dpsi+(nut[i][6]+nut[i][7]*t)*sin(ang)
        deps=deps+(nut[i][8]+nut[i][9]*t)*cos(ang)
    dpsi=dpsi*1e-4*pi/180.0/3600.0
    deps=deps*1e-4*pi/180.0/3600.0

    R1=np.array([[1.0,0.0,0.0],
        [0.0, cos(-eps-deps),-sin(-eps-deps)],
        [0.0, sin(-eps-deps), cos(-eps-deps)]])
    R2=np.array([[cos(-dpsi) ,-sin(-dpsi), 0.0],
        [sin(-dpsi), cos(-dpsi)     ,0.0],
        [0.0,0.0,1.0]])
    R3=np.array([[1.0    ,0.0     ,0.0],
        [0.0,cos(eps),-sin(eps)],
        [0.0,sin(eps),cos(eps)]])
    
    #N=(R1.dot(R2)).dot(R3)
    N=R3.dot(R2.dot(R1))

    #utc2gmst
    tut0=COMMTIME(  time2COMMONTIME(rt_unix)['year'],
                    time2COMMONTIME(rt_unix)['month'],
                    time2COMMONTIME(rt_unix)['day'],
                       0,0,0)
    ut=time2COMMONTIME(rt_unix)['hour']*3600+time2COMMONTIME(rt_unix)['minute']*60+time2COMMONTIME(rt_unix)['second']
    t1=(epoch2time(tut0)-epoch2time(ep2000))/86400.0/36525.0
    t2=t1*t1
    t3=t2*t1
    gmst0=24110.54841+8640184.812866*t1+0.093104*t2-6.2E-6*t3
    gmst=gmst0+1.002737909350795*ut

    gmst=fmod(gmst,86400.0)*pi/43200.0

    # greenwich aparent sidereal time (rad)
    gast=gmst+dpsi*cos(eps)
    gast+=(0.00264*sin(f[4])+0.000063*sin(2.0*f[4]))*(pi/180.0/3600.0)

    # eci to ecef transformation matrix
    R1=np.array([[1.0,0.0,0.0],
                 [0.0,1.0,0.0],
                 [0.0,0.0,1.0]])
    R2=np.array([[1.0,0.0,0.0],
                 [0.0,1.0,0.0],
                 [0.0,0.0,1.0]])
    R3=np.array([[cos(gast),-sin(gast),0.0],
                 [sin(gast), cos(gast),0.0],
                 [0.0,0.0,1.0]])
    
    #R=(R1.dot(R2)).dot(R3)
    R=(R3).dot(R2.dot(R1))
    #U=R.dot(N.dot(P))
    U=(P.dot(N)).dot(R)

    return U,gmst

def sun_moon_pos(rt_utc):
    rsun_t,rmoon_t=sun_moon_pos_eci(rt_utc)
    U,gmst=eci2ecef(rt_utc)
    rsun=np.array([0.0,0.0,0.0])
    rsun[0]=U[0][0]*rsun_t[0]+U[1][0]*rsun_t[1]+U[2][0]*rsun_t[2]
    rsun[1]=U[0][1]*rsun_t[0]+U[1][1]*rsun_t[1]+U[2][1]*rsun_t[2]
    rsun[2]=U[0][2]*rsun_t[0]+U[1][2]*rsun_t[1]+U[2][2]*rsun_t[2]

    rmoon=np.array([0.0,0.0,0.0])
    rmoon[0]=U[0][0]*rmoon_t[0]+U[1][0]*rmoon_t[1]+U[2][0]*rmoon_t[2]
    rmoon[1]=U[0][1]*rmoon_t[0]+U[1][1]*rmoon_t[1]+U[2][1]*rmoon_t[2]
    rmoon[2]=U[0][2]*rmoon_t[0]+U[1][2]*rmoon_t[1]+U[2][2]*rmoon_t[2]

    return rsun,rmoon,gmst



#固体潮相关
#Solid Earth Tide Corrections
def tide_pl(eu,rp,GMp,pos):
    H3=0.292
    L3=0.015
    GME=3.98604415E14
    RE_WGS84=6378137.0

    r=np.linalg.norm(rp)
    if(r<=0.0):
        return np.array([0,0,0])
    
    ep=np.array(rp)/r

    K2=GMp/GME*(RE_WGS84**2)*(RE_WGS84**2)/(r*r*r)
    K3=K2*RE_WGS84/r

    latp=asin(ep[2])
    lonp=atan2(ep[1],ep[0])
    cosp=cos(latp)
    sinl=sin(pos[0])
    cosl=cos(pos[0])

    #/* step1 in phase (degree 2) */
    p=(3.0*sinl*sinl-1.0)/2.0
    H2=0.6078-0.0006*p
    L2=0.0847+0.0002*p
    a=ep[0]*eu[0]+ep[1]*eu[1]+ep[2]*eu[2]
    dp=K2*3.0*L2*a
    du=K2*(H2*(1.5*a*a-0.5)-3.0*L2*a*a)

    #/* step1 in phase (degree 3) */
    dp+=K3*L3*(7.5*a*a-1.5)
    du+=K3*(H3*(2.5*a*a*a-1.5*a)-L3*(7.5*a*a-1.5)*a)

    #/* step1 out-of-phase (only radial) */
    du+=3.0/4.0*0.0025*K2*sin(2.0*latp)*sin(2.0*pos[0])*sin(pos[1]-lonp)
    du+=3.0/4.0*0.0022*K2*cosp*cosp*cosl*cosl*sin(2.0*(pos[1]-lonp))

    dr=np.array([0.0,0.0,0.0])
    dr[0]=dp*ep[0]+du*eu[0]
    dr[1]=dp*ep[1]+du*eu[1]
    dr[2]=dp*ep[2]+du*eu[2]

    return dr

#地球固体潮改正(直接加到测站坐标位置上)
#Solid Earth Tide Correction (Applied Directly to Station Coordinates)
def solid_tides(rt_unix,X):
    #1.测站位置
    #1.station position
    rr=[X[0][0],X[1][0],X[2][0]]
    pos=[asin(rr[2]/np.linalg.norm(rr)),atan2(rr[1],rr[0])]
    sinp=sin(pos[0])
    cosp=cos(pos[0])
    sinl=sin(pos[1])
    cosl=cos(pos[1])

    E=[-sinl,-sinp*cosl,cosp*cosl,cosl,-sinp*sinl,cosp*sinl,0.0,cosp,sinp]

    #2.太阳、月亮位置
    #2.positions of the sun and the moon
    rsun,rmoon,gmst=sun_moon_pos(rt_unix+gpst2utc(rt_unix))

    #3.地心地固坐标系(ecef)下太阳位置
    #3.The position of the sun in ecef
    # U,gmst=eci2ecef(rt_unix)
    # rsun=U.dot(rsun)
    # rmoon=U.dot(rmoon)

    #4.displacement by solid earth tide
    # /* step1: time domain */
    eu=np.array([0.0,0.0,0.0])
    eu[0]=E[2]
    eu[1]=E[5]
    eu[2]=E[8]

    GMS=1.327124E20#太阳万有引力常数
                   #Constant of universal gravitational force of the sun
    GMM=4.902801E12#地球万有引力常数
                   #Constant of universal gravitational force of the earth
    dr1=tide_pl(eu,rsun,GMS,pos)
    dr2=tide_pl(eu,rmoon,GMM,pos)

    #step2: frequency domain, only K1 radial
    sin2l=sin(2.0*pos[0])
    du=-0.012*sin2l*sin(gmst+pos[1])

    dr=np.array([0.0,0.0,0.0])
    dr[0]=dr1[0]+dr2[0]+du*E[2]
    dr[1]=dr1[1]+dr2[1]+du*E[5]
    dr[2]=dr1[2]+dr2[2]+du*E[8]

    #/* eliminate permanent deformation */
    sinl=sin(pos[0]); 
    du=0.1196*(1.5*sinl*sinl-0.5)
    dn=0.0247*sin2l
    dr[0]=dr[0]+du*E[2]+dn*E[1]
    dr[1]=dr[1]+du*E[5]+dn*E[4]
    dr[2]=dr[2]+du*E[8]+dn*E[7]

    return dr



