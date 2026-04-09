#文件名:RINEX.py
#Source File Name: RINEX.py
#数据文件与产品读取函数库
#A pure Python Core Source File of Data Loader and Products Pre-processor
#作者: 蒋卓君, 杨泽恩, 黄文静, 钱闯, 武汉理工大学
#Copyright 2025-, by Zhuojun Jiang, Zeen Yang, Wenjing Huang, Chuang Qian, Wuhan University of Technology, China

import numpy as np
import satpos as satpos

from math import sin,cos

def getsp3(filename):
    #函数:从精密星历中读取星历数据 
    #Function: Read ephemeris data from precise ephemeris
    #输入：文件名
    #Input:File name
    #输出:全部精密星历数据列表
    #Output:List of all precise ephemeris data
    
    IGS=[]#精密星历数据列表 
          #Precise ephemeris data list
    data={}
    with open(filename,"r") as f:
        lines=f.readlines()
    for line in lines:
        #逐历元记录精密星历数据
        #Record precise ephemeris data epoch by epoch
        if(line[0]=="*"):
            IGS.append(data.copy())
            ct_str=line.split()
            data={}
            ct=satpos.COMMTIME(ct_str[1],ct_str[2],ct_str[3],ct_str[4],ct_str[5],ct_str[6])
            unixt=satpos.epoch2time(ct)
            gweek,gsec=satpos.time2gpst(unixt)
            data["GPSweek"]=gweek
            data["GPSsec"]=gsec
        elif(line[0]=="P"):
            navd=line.split()
            #质量控制: 去除非有效值
            if(float(navd[1])==0.0 or float(navd[2])==0.0 or float(navd[3])==0.0):
                continue
            data[navd[0].replace("P","")]=[float(navd[1])*1e3,float(navd[2])*1e3,float(navd[3])*1e3,float(navd[4])*1e-6]
    IGS.append(data)
    return IGS[1:]

def getclk(filename):
    #函数: 从精密钟差中读取钟差数据
    #Function:Read clock offset data from precise clock offset
    #输入: 文件名    
    #Input:File name
    #输出: 全部精密钟差列表数据
    #Output:List of all precise clock offset data
    CLK=[]#精密星历数据列表 
          #Precise ephemeris data list  
    data={}
    with open(filename,"r") as f:
        lines=f.readlines()
    ishead=1
    for line in lines:
        #逐历元记录精密星历数据
        #Record precise ephemeris data epoch by epoch
        if('END OF HEADER' in line):
            ishead=0
        if(ishead):
            old_ct=satpos.COMMTIME(1970,1,1,0,0,0)
            continue

        if(line[0:2]=="AS"):
            ct_str=line.split()
            # print(ct_str)
            # data={}
            ct=satpos.COMMTIME(ct_str[2],ct_str[3],ct_str[4],ct_str[5],ct_str[6],ct_str[7])
            if(ct!=old_ct):
                gweek,gsec=satpos.time2gpst(satpos.epoch2time(old_ct))
                data['GPSweek']=gweek
                data['GPSsec']=gsec
                CLK.append(data)
                data={}
                old_ct=ct
            si_PRN=ct_str[1]
            data[si_PRN]=float(ct_str[9])
    gweek,gsec=satpos.time2gpst(satpos.epoch2time(old_ct))
    data['GPSweek']=gweek
    data['GPSsec']=gsec
    CLK.append(data)
    return CLK[1:]


def lagrange_interpolation(x_points, y_points, x):
    #函数:对单颗卫星的精密星历应用拉格朗日内插
    #Function:Application of Lagrange Interpolation to precise ephemeris for a single satellite
    #输入:拉格朗日插值下标,拉格朗日插值上标,x
    #Input:Lagrange interpolation subscript,Lagrange interpolation superscript,x
    n = len(x_points)
    P = 0.0
    for i in range(n):
        L_i = 1.0
        for j in range(n):
            if i != j:
                L_i *= (x - x_points[j]) / (x_points[i] - x_points[j])
        P += y_points[i] * L_i
    return P

def sp3_earth_roll(t,insert_time,indextime):
    #gps时间系统
    #gps time system
    clight=2.99792458e8
    OMGE=7.2921151467E-5

    t_in=[t[0],t[1],t[2],t[3]]
    
    #计算内插时刻与星历标称时刻的时间差
    #Calculate the time difference between the interpolation epoch and the nominal ephemeris epoch
    dtt=indextime-insert_time
    alpha=OMGE*dtt#计算自转角
                  #Calculate the rotation angle

    sx=t_in[0]
    sy=t_in[1]

    #地球自转改正
    #Earth Rotation Correction
    t_in[0]=cos(alpha)*sx-sin(alpha)*sy
    t_in[1]=sin(alpha)*sx+cos(alpha)*sy

    return t_in

def insert_satpos_froom_sp3(IGS,inserttime,PRN,sp3_interval=300):
    #函数:对精密星历应用滑动窗口的拉格朗日内插(17阶)
    #Function:Apply 17th-Order Sliding Window Lagrange Interpolation to Precise Ephemeris
    #输入:原始精密星历,待插值的时间(unixtime), 待插值卫星PRN或PRN列表
    #Input:Original precise ephemeris, time to be interpolated (unixtime), PRN or PRN list of satellites to be interpolated
    #输出:插值后,插值历元的卫星精密星历位置字典
    #Output:Dictionary of satellite precise ephemeris positions at interpolation epochs after interpolation 
    #特别注意:本函数严格遵守拖尾效应抑制策略(即不对星历前8历元内和后9历元内的位置进行内插)
    #Special Note: This function strictly adheres to the trailing effect suppression strategy (i.e., no interpolation is performed on positions within the first 8 epochs or the last 9 epochs of the ephemeris data).

    
    #GPS时间
    #GPS time
    GPSweek,GPSsec=satpos.time2gpst(inserttime)
    
    #将待插卫星PRN号转为列表格式
    #Convert PRN number of satellite to be inserted into list format
    
    if(type(PRN)==type('G01')):
        PRNs=[PRN]
    else:
        PRNs=PRN
    IGS_start=satpos.gpst2time(IGS[0]['GPSweek'],IGS[0]['GPSsec'])
    IGS_end=satpos.gpst2time(IGS[-1]['GPSweek'],IGS[-1]['GPSsec'])
    #判断待插值历元是否在插值空间内
    #Judge whether the epoch to be interpolated is in the interpolation space
    # if inserttime< IGS_start+8*sp3_interval or inserttime>IGS_end-9*sp3_interval:
    #     print("No IGSnav in the inserttime")
    #     return False
    if inserttime< IGS_start-sp3_interval or inserttime>IGS_end+sp3_interval:
        print("No IGSnav in the inserttime")
        return False
    
    #逐颗卫星插值运算
    #Satellite-by-Satellite Interpolation Processing
    index=int((np.floor(inserttime)-IGS_start)/sp3_interval)     #中心历元向下取整计算
                                                                 #Central epoch floor calculation
    if(index<8):
        #前8历元
        index=8
    if(index>len(IGS)-10):
        #后九历元
        index=len(IGS)-10
    indextime=satpos.gpst2time(IGS[index]['GPSweek'],IGS[index]['GPSsec'])
    tIGS={'GPSweek':GPSweek,'GPSsec':GPSsec}
    for PRN in PRNs:
        #前8历元
        #The first 8 epochs
        t_8=sp3_earth_roll(IGS[index-8][PRN].copy(),inserttime,indextime-8*sp3_interval)
        t_7=sp3_earth_roll(IGS[index-7][PRN].copy(),inserttime,indextime-7*sp3_interval)
        t_6=sp3_earth_roll(IGS[index-6][PRN].copy(),inserttime,indextime-6*sp3_interval)
        t_5=sp3_earth_roll(IGS[index-5][PRN].copy(),inserttime,indextime-5*sp3_interval)
        t_4=sp3_earth_roll(IGS[index-4][PRN].copy(),inserttime,indextime-4*sp3_interval)
        t_3=sp3_earth_roll(IGS[index-3][PRN].copy(),inserttime,indextime-3*sp3_interval)
        t_2=sp3_earth_roll(IGS[index-2][PRN].copy(),inserttime,indextime-2*sp3_interval)
        t_1=sp3_earth_roll(IGS[index-1][PRN].copy(),inserttime,indextime-1*sp3_interval)
        #中心历元
        #The central epoch
        t=sp3_earth_roll(IGS[index][PRN],inserttime,indextime)
        #后9历元
        #The following 9 epochs
        tu1=sp3_earth_roll(IGS[index+1][PRN].copy(),inserttime,indextime+1*sp3_interval)
        tu2=sp3_earth_roll(IGS[index+2][PRN].copy(),inserttime,indextime+2*sp3_interval)
        tu3=sp3_earth_roll(IGS[index+3][PRN].copy(),inserttime,indextime+3*sp3_interval)
        tu4=sp3_earth_roll(IGS[index+4][PRN].copy(),inserttime,indextime+4*sp3_interval)
        tu5=sp3_earth_roll(IGS[index+5][PRN].copy(),inserttime,indextime+5*sp3_interval)
        tu6=sp3_earth_roll(IGS[index+6][PRN].copy(),inserttime,indextime+6*sp3_interval)
        tu7=sp3_earth_roll(IGS[index+7][PRN].copy(),inserttime,indextime+7*sp3_interval)
        tu8=sp3_earth_roll(IGS[index+8][PRN].copy(),inserttime,indextime+8*sp3_interval)
        tu9=sp3_earth_roll(IGS[index+9][PRN].copy(),inserttime,indextime+9*sp3_interval)
        #将历元点重整为列表
        #Reorganize epoch points into a list
        x_points=[t_8[0],t_7[0],t_6[0],t_5[0],t_4[0],t_3[0],t_2[0],t_1[0],t[0],tu1[0],tu2[0],tu3[0],tu4[0],tu5[0],tu6[0],tu7[0],tu8[0],tu9[0]]
        y_points=[t_8[1],t_7[1],t_6[1],t_5[1],t_4[1],t_3[1],t_2[1],t_1[1],t[1],tu1[1],tu2[1],tu3[1],tu4[1],tu5[1],tu6[1],tu7[1],tu8[1],tu9[1]]
        z_points=[t_8[2],t_7[2],t_6[2],t_5[2],t_4[2],t_3[2],t_2[2],t_1[2],t[2],tu1[2],tu2[2],tu3[2],tu4[2],tu5[2],tu6[2],tu7[2],tu8[2],tu9[2]]
        dt_points=[t_8[3],t_7[3],t_6[3],t_5[3],t_4[3],t_3[3],t_2[3],t_1[3],t[3],tu1[3],tu2[3],tu3[3],tu4[3],tu5[3],tu6[3],tu7[3],tu8[3],tu9[3]]
        
        t_points=[-8*sp3_interval,-7*sp3_interval,-6*sp3_interval,-5*sp3_interval,-4*sp3_interval,-3*sp3_interval,-2*sp3_interval,-1*sp3_interval, 0*sp3_interval,
                   1*sp3_interval, 2*sp3_interval, 3*sp3_interval, 4*sp3_interval, 5*sp3_interval, 6*sp3_interval, 7*sp3_interval, 8*sp3_interval, 9*sp3_interval]
        x_point=lagrange_interpolation(t_points,x_points,inserttime-indextime)
        y_point=lagrange_interpolation(t_points,y_points,inserttime-indextime)
        z_point=lagrange_interpolation(t_points,z_points,inserttime-indextime)
        dt_point=lagrange_interpolation(t_points,dt_points,inserttime-indextime)
        tIGS[PRN]=[x_point,y_point,z_point,dt_point]
    return tIGS

def insert_clk_from_sp3(CLK,inserttime,PRN,interval=30):
    #GPS时间
    #GPS time
    GPSweek,GPSsec=satpos.time2gpst(inserttime)
    
    #将待插卫星PRN号转为列表格式
    #Convert PRN number of satellite to be inserted into list format
    if(type(PRN)==type('G01')):
        PRNs=[PRN]
    else:
        PRNs=PRN
    IGS_start=satpos.gpst2time(CLK[0]['GPSweek'],CLK[0]['GPSsec'])
    IGS_end=satpos.gpst2time(CLK[-1]['GPSweek'],CLK[-1]['GPSsec'])
    #判断待插值历元是否在插值空间内
    #Judge whether the epoch to be interpolated is in the interpolation space
    if inserttime< IGS_start-interval or inserttime>IGS_end+interval:
        print("No CLK in the inserttime")
        return False

    
    #卫星钟差线性插值运算
    #Linear interpolation operation of satellite clock offset
    index=int((np.floor(inserttime)-IGS_start)/interval)     #中心历元向下取整计算
                                                             #Central epoch floor calculation
    if(index<0):
        index=0
    
    indextime=satpos.gpst2time(CLK[index]['GPSweek'],CLK[index]['GPSsec'])
    tCLK={'GPSweek':GPSweek,'GPSsec':GPSsec}
    for PRN in PRNs:
        #末历元直接返回
        #Return the last epoch directly
        if(index==len(CLK)):
            tCLK[PRN]=CLK[-1][PRN]
        #非末历元插值
        #Non-final epoch interpolation
        clk=CLK[index][PRN]
        clk_up=CLK[index+1][PRN]
        tCLK[PRN]=clk+(clk_up-clk)*(inserttime-indextime)/interval
    return tCLK


#RINEX观测文件解析子函数
#Sub-function for Parsing RINEX Observation Files
def decode_epoch_record(str):
    
    #函数: RINEX3 观测文件(.o) 行处理 观测历元首行字段
    #Function: RINEX3 observation file (.o) line processing observation epoch header row field 
    #参数类型检查(字符串)
    #parameter type check (string)
    if type(str) != type("aaa"):
        return ValueError
    #接收机钟时间, 年字段
    #Receiver clock time,year
    year=int(str[2:6])
    #接收机钟时间, 月字段
    #Receiver clock time,month
    month=int(str[7:9])
    #接收机钟时间, 日字段
    #Receiver clock time,day
    day=int(str[10:12])
    #接收机钟时间, 小时字段
    #Receiver clock time,hour
    hour=int(str[13:15])
    #接收机钟时间, 分钟字段
    #Receiver clock time,minute
    min=int(str[16:18])
    #接收机钟时间, 秒字段
    #Receiver clock time,second
    sec=float(str[18:29])
    #历元观测值状态字段
    #Epoch observation status field
    epoch_OK=int(str[31])
    #历元观测数量字段
    #Epoch observation count field
    s_num=int(str[32:35])
    #时间系统转换(统一为GPS时间, 便利存储)
    #Time system conversion (unified as GPS time for convenient storage)
    comt=satpos.COMMTIME(year,month,day,hour,min,sec)
    u_t=satpos.epoch2time(comt)
    GPSweek,GPSsec=satpos.time2gpst(u_t)
    
    #返回历元观测值基本信息字典
    #Return to the dictionary of basic information of epoch observations
    return {'type':'Observation','GPSweek':GPSweek,'GPSsec':GPSsec,'s_num':s_num,'Epoch_OK':epoch_OK}

def decode_epoch_GPS(str,id=range(8),cbia_0=0.0,cbia_1=0.0,cbia_2=0.0,f1=1575.42*1e6,f2=1227.60*1e6):
    
    # 函数: RINEX3 观测文件(.o) 行处理 GPS双频观测值(L1,L2)
    #Function: Process RINEX3 Observation File Lines (.o) for GPS Dual-Frequency Observations (L1, L2)
    # 频间DCB改正系数(非电离层组合系数)
    #Inter-frequency DCB Correction Coefficients (Ionospheric-free Combination Coefficients)
    C_1=f1*f1/(f1*f1-f2*f2)
    C_2=-f2*f2/(f1*f1-f2*f2)
    
    #参数类型检查(字符串)
    #parameter type check (string)
    if type(str) != type("aaa"):
        return ValueError
    
    #卫星PRN号
    #satellite PRB number
    PRN=str[0:3]
    try:
        PRN=PRN[:1]+"{:02d}".format(int(PRN[1:]))
    except:
        PRN=str[0:3]
    #L1 C/A码 观测值
    #L1 C/A code  Observations 
    try:
        C1C=float(str[3+id[0]*16:3+id[0]*16+14])+cbia_1-C_2*cbia_0
    except:
        C1C=0.0
    try:
        L1C=float(str[3+id[1]*16:3+id[1]*16+14])
    except:
        L1C=0.0
    try:
        L1C_LLI=int(str[3+id[1]*16+14:3+(id[1]+1)*16])
    except:
        L1C_LLI=0
    try:
        D1C=float(str[3+id[2]*16:3+id[2]*16+14])
    except:
        D1C=0.0
    try:
        S1C=float(str[3+id[3]*16:3+id[3]*16+14])
    except:
        S1C=0.0
    
    #L2 P码   观测值
    #L2 P code  Observations
    try:
        C2L=float(str[3+id[4]*16:3+id[4]*16+14])+cbia_2+C_1*cbia_0
    except:
        C2L=0.0
    try:
        L2L=float(str[3+id[5]*16:3+id[5]*16+14])
    except:
        L2L=0.0
    try:
        L2L_LLI=int(str[3+id[5]*16+14:3+(id[5]+1)*16])
    except:
        L2L_LLI=0.0
    try:
        D2L=float(str[3+id[6]*16:3+id[6]*16+14])
    except:
        D2L=0.0
    try:
        S2L=float(str[3+id[7]*16:3+id[7]*16+14])
    except:
        S2L=0.0
    return{'PRN':PRN,'OBS':[C1C,L1C,L1C_LLI,D1C,S1C,C2L,L2L,L2L_LLI,D2L,S2L]}

def RINEX3_to_obsmat(obsfile,obs_type=[],sys='G',dcb_correction=0,dcb_file_0="",dcb_file_1="",dcb_file_2="",f1=1575.42e6,f2=1227.60*1e6):
    obs_mat=[]
    if(not len(obs_type)):
        #如果无双频频点标识符输入, 则预置为L1, L2P码
        #If no dual-frequency signal identifiers' input, preset to L1P and L2P codes
        obs_type=['C1W','L1W','D1W','S1W','C2W','L2W','D2W','S2W']
    
    double_f_index=[]#信号行位置指示
                     #Signal line position indicator
    obs_type_num=len(obs_type)

    #1.DCB字典读取
    #1.Read DCB dictionary. Attention: The DCB Correction Only Support GPS L1/L2 & BDS B1/B2 combinations 
    if(dcb_correction):
        #频间偏差读取P1-P2
        #Inter-frequency DCB retrieval P1-P2
        try:
            cbias_0=eval(str(np.load(dcb_file_0,allow_pickle=True)))
        except:
            cbias_0=RINEX3_to_DCB(dcb_file_0)
        #判断信号类型
        #Determine signal type
        if obs_type[0][1]=='1':
            if(obs_type[0][2]=='C'):
                try:
                    cbias_1=eval(str(np.load(dcb_file_1,allow_pickle=True)))
                except:
                    cbias_1=RINEX3_to_DCB(dcb_file_1)
        else:
            cbias_1=False
        if obs_type[4][1]=='2':
            if(obs_type[4][2]=='C'):
                try:
                    cbias_2=eval(str(np.load(dcb_file_2,allow_pickle=True)))
                except:
                    cbias_2=RINEX3_to_DCB(dcb_file_2)
        else:
            cbias_2=False
    #2.RINEX观测文件读取
    #2.RINEX observation files reading
    with open(obsfile,'r') as f:
        lines=f.readlines()#数据读取
                           #read data
        flag='header'      #文件指针指示
                           #File pointer indication
        e_flag='epoch_out'
    
        obs_epoch=[]

        all_sys_type={}
    
        for line in lines:
            # 首先判断当前数据读取位置(文件头/数据体)
            #Determine the current parsing position (file header/data body）
            if("END OF HEADER" in line):
                flag='data_frame'
                continue
            
            #当前文件指针位于文件头
            #The current file pointer is at the file header
            if(flag=='header'):
                #处理观测数据标识
                #Processing observation data identification
                if('SYS / # / OBS TYPES' in line):
                    ls=line.replace('SYS / # / OBS TYPES','')
                    ls=ls.split()
                    if(ls[0] in ['G','R','E','C','J','S','I']):
                        obs_sys=ls[0]
                        all_sys_type[obs_sys]=[]
                    for lss in ls:
                        #3字符串填入
                        #3string filling
                        if(len(lss)==3):
                            all_sys_type[obs_sys].append(lss)
            
            
            #当前文件指针处于数据体
            #The current file pointer is at the data body
            if(flag=='data_frame'):
                #首先求取信号指示位置
                #First, find the position indicated by the signal
                if(not len(double_f_index)):
                    try:
                        ids=all_sys_type[sys]
                    except:
                        ids=[0,1,2,3,4,5,6,7]
                    for j in range(obs_type_num):
                        try:
                            double_f_index.append(ids.index(obs_type[j]))
                        except:
                            print("No "+obs_type[j])
                            double_f_index.append(999)
                    #print(double_f_index)
                # 新历元进入, 重置历元观测数组, 创建新历元观测值
                # Enter the new epoch, reset the epoch observation array, and create a new epoch observation value
                if('>' in line and line[2:6]!="    "):
                
                    #保存前一历元全部观测值
                    #Save all observations from the previous epoch
                    #if(len(obs_epoch)!=0):
                    if(True):
                        #print(line)
                        try:
                            obs_mat.append([obs_format.copy(), obs_epoch.copy()]) #数据存储
                        except:                                                   #store data
                            pass                                                  
                        obs_epoch=[]                                          #观测数组置零
                                                                              #set observed array to zero
                
                    #创建新历元
                    #create new epoch
                    obs_format=decode_epoch_record(line)
                    obs_format['obstype']=obs_type
                    #print(obs_format)
            
                # 观测值读取
                #read observations
                elif( sys in line):
                    cbia_1=0.0
                    cbia_2=0.0
                    cbia_0=0.0
                    try:
                        cbia_1=cbias_1[line[:3]][1]
                    except:
                        pass
                    try:
                        cbia_2=cbias_2[line[:3]][1]
                    except:
                        pass
                    try:
                        cbia_0=cbias_0[line[:3]][1]
                    except:
                        pass
                    obs_epoch.append(decode_epoch_GPS(line,double_f_index,cbia_0,cbia_1,cbia_2,f1=f1,f2=f2))
        obs_mat.append([obs_format.copy(), obs_epoch.copy()])        
        
        #返回观测矩阵
        #return observation matrix
        return obs_mat
    
def RINEX3_to_ATX(filename):
    with open(filename) as f:
        lines=f.readlines()
        for header in range(len(lines)):
            if "END OF HEADER" in lines[header]:
                break
        sat_pcos=[]
        sat_pco={}
        PRNS=[]
        for i in range(header+1,len(lines)):
        
            if "SFART OF ANTENNA" in lines[i]:
                sat_pco={}#重置PCO列表
                          #reset PCO list
            if "TYPE / SERIAL NO" in lines[i]:
                sat_pco['PRN']=lines[i][20:].split()[0]
                sat_pco['sat_type']=lines[i][:20].replace(" ","")
                if(lines[i][20:].split()[0] not in PRNS):
                    PRNS.append(lines[i][20:].split()[0])
        
            if "VALID FROM" in lines[i]:
                ls=lines[i].split()
                sat_pco['Stime']=satpos.epoch2time(satpos.COMMTIME(ls[0],ls[1],ls[2],ls[3],ls[4],ls[5]))
        
            if "VALID UNTIL" in lines[i]:
                ls=lines[i].split()
                sat_pco['Etime']=satpos.epoch2time(satpos.COMMTIME(ls[0],ls[1],ls[2],ls[3],ls[4],ls[5]))
        
            if "START OF FREQUENCY" in lines[i]:
                ls=lines[i].split()
                ls_1=lines[i+1].split()
                ls_2=lines[i+2].split()
                sat_pco["L{}".format(int(ls[0][1:]))]=[float(ls_1[0])*0.001,float(ls_1[1])*0.001,float(ls_1[2])*0.001]
                sat_pco["L{}_pcv".format(int(ls[0][1:]))]=[]
                for pcv in ls_2[1:]:
                    sat_pco["L{}_pcv".format(int(ls[0][1:]))].append(float(pcv)*0.001)
        
            if "END OF ANTENNA" in lines[i]:
                sat_pcos.append(sat_pco.copy())
                sat_pco={}

    #重整为按PRN号编排的字典
    #Reorganize into a dictionary sorted by PRN number
    sat_pco_recombine={}
    for prn in PRNS:
        if(prn=="TYPE"):
            continue
        sat_pco_recombine[prn]=[]
        for sat_pco in sat_pcos:
            if(sat_pco['PRN']==prn):
                sat_pco_recombine[prn].append(sat_pco.copy())

    return sat_pco_recombine


def RINEX3_to_DCB(filename):
    
    try:
        with open(filename,'r') as f:
            print('解析DCB文件: ',filename)
    except:
        return {}
    
    
    with open(filename,'r') as f:
        lines=f.readlines()
        dcb_type=0
        header=0
        #CODE DCB产品
        #CODE DCB Products
        for i in range(len(lines)):
            if "DIFFERENTIAL (P1-P2) CODE BIASES" in lines[i]:
                dcb_type=1
            if "DIFFERENTIAL (P1-C1) CODE BIASES" in lines[i]:
                dcb_type=2
            if "DIFFERENTIAL (P2-C2) CODE BIASES" in lines[i]:
                dcb_type=3
            if "PRN / STATION NAME        VALUE (NS)  RMS (NS)" in lines[i]:
                header=i+1
                break
        cbias={}
        dcb_typename=["None","P1-P2","P1-C1","P2-C2"]
        for i in range(header+1,len(lines)):
            prn=0
            try:
                prn=int(lines[i][1:3])
            except:
                continue
            if(prn):
                PRN=lines[i][:3]
                ls=lines[i].split()
                cbias[PRN]=[dcb_typename[dcb_type],float(ls[1])*1e-9*satpos.clight]

    return cbias

def CAS_DCB(filename,osignal='C1W',tsignal='C2W'):
    #CAS DCB文件读取
    #CAS DCB file reading
    with open(filename,"r") as f:
        lines=f.readlines()
        header=0
        cbias={}
        for line in lines:
            if("+BIAS/SOLUTION" in line):
                header=1
                continue
            if("-BIAS/SOLUTION" in line):
                break

            if(header==1):
                #目标DCB
                #Target DCB
                if(osignal in line and tsignal in line):
                    ls=line.split()
                    if(len(ls[2])==3):
                        PRN=ls[2]
                        cbias[PRN]=[osignal+'_'+tsignal,float(ls[8])*1e-9*satpos.clight]
    np.save("{}_{}.npy".format(osignal,tsignal),cbias)
    return "{}_{}.npy".format(osignal,tsignal),cbias

def RINEX2ion_params(filename):
    ion_params=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    #首先尝试RENEX2版本广播星历电离层参数读取
    #First,attempt to read the ionosphere parameters from the RINEX version2 broadcast ephemeris
    with open(filename,'r') as f:
        lines=f.readlines()
        for line in lines:
            if('ION ALPHA' in line):
                ion_params[0]=float(line.split()[0].replace('D','E'))
                ion_params[1]=float(line.split()[1].replace('D','E'))
                ion_params[2]=float(line.split()[2].replace('D','E'))
                ion_params[3]=float(line.split()[3].replace('D','E'))
            if('ION BETA'  in line):
                ion_params[4]=float(line.split()[0].replace('D','E'))
                ion_params[5]=float(line.split()[1].replace('D','E'))
                ion_params[6]=float(line.split()[2].replace('D','E'))
                ion_params[7]=float(line.split()[3].replace('D','E'))
    
    #以下为RINEX3版本广播星历
    #Broadcast ephemeris in RINEX version3
    if(ion_params==[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]):
        with open(filename,'r') as f:
            lines=f.readlines()
            for line in lines:
                if('IONOSPHERIC CORR' in line and 'GPSA' in line):
                    ion_params[0]=float(line.split()[1].replace('D','E'))
                    ion_params[1]=float(line.split()[2].replace('D','E'))
                    ion_params[2]=float(line.split()[3].replace('D','E'))
                    ion_params[3]=float(line.split()[4].replace('D','E'))
                if('IONOSPHERIC CORR' in line and 'GPSB' in line):
                    ion_params[4]=float(line.split()[1].replace('D','E'))
                    ion_params[5]=float(line.split()[2].replace('D','E'))
                    ion_params[6]=float(line.split()[3].replace('D','E'))
                    ion_params[7]=float(line.split()[4].replace('D','E'))
    return ion_params