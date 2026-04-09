import yaml
import numpy as np
import os
from tqdm import tqdm
import csv
import sys
sys.path.append("src")
from satpos import *
from sppp import *
#from sppp_multiGNSS import *
from lambda_common import *

@numba.jit(nopython=True)
def numba_inv(A):
    return inv(A)

def reconstruct_obs_mat(obs_mat):
    #函数: 重整有效观测数据字典(根据Epoch_OK标识)
    #输入: 有效观测数据字典
    #输出: 重整后的观测数据字典
    r_obsmat=[]
    for i in range(len(obs_mat)):
        if(obs_mat[i][0]['Epoch_OK'])!=0:
            continue
        else:
            r_obsmat.append(obs_mat[i])
    #返回观测数据字典
    return r_obsmat

def check_obs_mats(obs_mats):
    #函数: 校验多系统观测值匹配性
    #输入: 多系统观测值列表
    #输出: 观测值数据
    lens=[]
    for i in range(len(obs_mats)):
        lens.append(len(obs_mats[i]))
    #列表仅单值
    if(lens==1):
        print("Only one system observed")
        return True
    #列表有多系统观测
    for i in range(len(lens)-1):
        if(lens[i]!=lens[i+1]):
            print("Observations among systems not equal")
            return False
    #长度校验通过, 开始时间校验
    for i in range(len(obs_mats[0])):
        GPS_week=obs_mats[0][i][0]["GPSweek"]
        GPS_sec=obs_mats[0][i][0]["GPSsec"]
        for j in range(len(obs_mats)):
            check_week=obs_mats[j][i][0]['GPSweek']
            check_sec=obs_mats[j][i][0]['GPSsec']
            if(check_week!=GPS_week or check_sec!=GPS_sec):
                print("Observations among systems not in the same time period")
                return False
    
    #全部校验通过, 返回True
    return True

def CAS_DCB_SR(filename,osignal='C1W',tsignal='C2W',sta=""):
    #读取卫星DCB
    dcb_file_0,DCB_mat=CAS_DCB(filename,osignal,tsignal)
    with open(filename,"r") as f:
        lines=f.readlines()
        header=0
        cbias_receiver=0.0
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
                    if(len(ls[3])==4):
                        if(sta==ls[3]):
                            #print(ls)
                            cbias_receiver=float(ls[9])*1e-9*satpos.clight
                            break
                        # PRN=ls[2]
                        # cbias[PRN]=[osignal+'_'+tsignal,float(ls[8])*1e-9*satpos.clight]
    #读取到测站DCB, 叠加到卫星DCB中
    if(cbias_receiver!=0.0):
        print("Receiver DCB of {}: {}->{} {}m".format(sta,osignal,tsignal,cbias_receiver))
    for key in DCB_mat.keys():
        DCB_mat[key][1]=DCB_mat[key][1]+cbias_receiver
    np.save("{}_{}.npy".format(osignal,tsignal),DCB_mat)
    return "{}_{}.npy".format(osignal,tsignal),DCB_mat


#多系统基于精密星历的单点定位
def SPP_from_IGS_M(obs_mats,obs_index,IGS,CLK,sat_out,ion_param,sat_pcos,freqs,sol_mode='SF',el_threthod=7.0,obslist=[],pre_rr=[]):
    rr=[100,100,100]
    #观测值列表构建(异常值剔除选星)
    if(not len(obslist)):
        obslist=[]
        #从各分系统中读取数据
        for obs_mat in obs_mats:
            for i in range(len(obs_mat[obs_index][1])):
                obsdata=obs_mat[obs_index][1][i]['OBS']
                obshealth=1
                if(obsdata[0]==0.0 or obsdata[1]==0.0 or obsdata[5]==0.0 or obsdata[6]==0.0):
                    obshealth=0
                #检验星历内插有效性
                #观测时间&观测值
                rt_week=obs_mat[obs_index][0]['GPSweek']
                rt_sec=obs_mat[obs_index][0]['GPSsec']
                rt_unix=satpos.gpst2time(rt_week,rt_sec)
                p1=obsdata[0]
                insert_time=rt_unix-p1/satpos.clight
                IGS_interval=IGS[1]['GPSsec']-IGS[0]['GPSsec']
                if(IGS_interval<0):
                    IGS_interval=IGS[2]['GPSsec']-IGS[1]['GPSsec']
                CLK_interval=CLK[1]['GPSsec']-CLK[0]['GPSsec']
                if(CLK_interval<0):
                    CLK_interval=CLK[2]['GPSsec']-CLK[1]['GPSsec']
                try:
                    h_is=insert_satpos_froom_sp3(IGS,insert_time,obs_mat[obs_index][1][i]['PRN'],sp3_interval=IGS_interval)
                    if(h_is==False):
                        obshealth=0
                    h_is=insert_clk_from_sp3(CLK,insert_time,obs_mat[obs_index][1][i]['PRN'],CLK_interval)
                    if(h_is==False):
                        obshealth=0
                except:
                    obshealth=0
                #排除星历无效
                if(obshealth):
                    if obs_mat[obs_index][1][i]['PRN'] not in sat_out:
                        obslist.append(obs_mat[obs_index][1][i])
    
    obslist_new=obslist.copy()#高度角截至列表
    sat_num=len(obslist)
    sat_prns=[t['PRN'] for t in obslist]
    sat_num_G=0
    sat_num_C=0
    sat_num_E=0
    for p in sat_prns:
        if "G" in p:
            sat_num_G=sat_num_G+1
        if "C" in p:
            sat_num_C=sat_num_C+1
        if "E" in p:
            sat_num_E=sat_num_E+1
    
    ex_index=np.zeros(sat_num,dtype=int)
    
    #方程满秩校验(三系统观测模型要求观测数量大于未知数数量)
    #首先确定未知数数量(即共有多少个有效观测系统)
    state_used=3
    for n in [sat_num_G,sat_num_E,sat_num_C]:
        if(n>0):
            state_used+=1
    #然后判断方程数量是否足够
    if(sat_num_G+sat_num_C+sat_num_E<state_used):
        print("The number of Observations are not enough, GPS: {}, BDS: {}, GAL: {}, pass epoch".format(sat_num_G,sat_num_C,sat_num_E))
        return [0,0,0,0],[],[]
    
    # if(sat_num_G<4):
    #     print("The number of GPS < 4, pass epoch.")
    #     return [0,0,0,0],[],[]
    # if(sat_num_C<1):
    #     pass
    #     #print("The number of BDS < 1, ISB_BDS no observations, set ISB_BDS=0.")
    #     #return [0,0,0,0],[],[]
    # if(sat_num_E<1):
    #     pass
        #print("The number of GAL < 1, ISB_GAL no observations, set ISB_GAL=0.")
        #return [0,0,0,0],[],[]
    
    #卫星列表构建
    peph_sat_pos={}
    for i in range(0,sat_num):
        #光速
        clight=2.99792458e8
        #观测时间&观测值
        rt_week=obs_mats[0][obs_index][0]['GPSweek']
        rt_sec=obs_mats[0][obs_index][0]['GPSsec']
        rt_unix=satpos.gpst2time(rt_week,rt_sec)
        
        #计算卫星速度的间隔时间
        dt=0.001

        #计算精密星历历元间隔
        IGS_interval=IGS[1]['GPSsec']-IGS[0]['GPSsec']
        if(IGS_interval<0):
            IGS_interval=IGS[2]['GPSsec']-IGS[1]['GPSsec']
        
        #计算精密钟差历元间隔
        CLK_interval=CLK[1]['GPSsec']-CLK[0]['GPSsec']
        if(CLK_interval<0):
            CLK_interval=CLK[2]['GPSsec']-CLK[1]['GPSsec']
        IGS_interval=round(IGS_interval)
        CLK_interval=round(CLK_interval)
        
        #原始伪距
        p1=obslist[i]['OBS'][0]
        s1=obslist[i]['OBS'][4]
        p2=obslist[i]['OBS'][5]
        s2=obslist[i]['OBS'][9]
        #print(p1,p2,phi1,phi2)
            
        #卫星位置内插
        si_PRN=obslist[i]['PRN']#此处为卫星PRN
        #频率分发
        if("G" in si_PRN):
            f1=freqs[0][0]
            f2=freqs[0][1]
        if("C" in si_PRN):
            f1=freqs[1][0]
            f2=freqs[1][1]
        if("E" in si_PRN):
            f1=freqs[2][0]
            f2=freqs[2][1] 
        
        rs1=insert_satpos_froom_sp3(IGS,rt_unix-p1/clight,si_PRN,sp3_interval=IGS_interval)    #观测历元卫星位置
        rs2=insert_satpos_froom_sp3(IGS,rt_unix-p1/clight+dt,si_PRN,sp3_interval=IGS_interval) #插值求解卫星速度矢量
        rs=[rs1[si_PRN][0],rs1[si_PRN][1],rs1[si_PRN][2]]
        dts=insert_clk_from_sp3(CLK,rt_unix-p1/clight,si_PRN,CLK_interval)[si_PRN]
        drs=[(rs2[si_PRN][0]-rs[0])/dt,(rs2[si_PRN][1]-rs[1])/dt,(rs2[si_PRN][2]-rs[2])/dt]
        dF=-2/clight/clight*( rs[0]*drs[0]+rs[1]*drs[1]+rs[2]*drs[2] )      #利用精密星历进行相对论效应改正
        dts=dts+dF

        rs1=insert_satpos_froom_sp3(IGS,rt_unix-p1/clight-dts,si_PRN,sp3_interval=IGS_interval)    #观测历元卫星位置
        rs2=insert_satpos_froom_sp3(IGS,rt_unix-p1/clight-dts+dt,si_PRN,sp3_interval=IGS_interval) #插值求解卫星速度矢量
        rs=[rs1[si_PRN][0],rs1[si_PRN][1],rs1[si_PRN][2]]
        dts=insert_clk_from_sp3(CLK,rt_unix-p1/clight-dts,si_PRN,CLK_interval)[si_PRN]
        drts=insert_clk_from_sp3(CLK,rt_unix-p1/clight-dts,si_PRN,CLK_interval)[si_PRN]
        drs=[(rs2[si_PRN][0]-rs[0])/dt,(rs2[si_PRN][1]-rs[1])/dt,(rs2[si_PRN][2]-rs[2])/dt]
        dF=-2/clight/clight*( rs[0]*drs[0]+rs[1]*drs[1]+rs[2]*drs[2] )      #利用精密星历进行相对论效应改正
        dts=dts+dF

        # #太阳位置
        rsun,_,_=sun_moon_pos(rt_unix-p1/clight-dts+gpst2utc(rt_unix-p1/clight-dts))

        #/* unit vectors of satellite fixed coordinates */
        r=np.array([-rs[0],-rs[1],-rs[2]])
        ez=r/np.linalg.norm(r)
        r=np.array([rsun[0]-rs[0],rsun[1]-rs[1],rsun[2]-rs[2]])
        es=r/np.linalg.norm(r)
        r=np.cross(ez,es)
        ey=r/np.linalg.norm(r)
        ex=np.cross(ey,ez)

        gamma=f1*f1/f2/f2
        C1=gamma/(gamma-1.0)
        C2=-1.0 /(gamma-1.0)

        #选择卫星PCO参数
        if("G" in si_PRN):
            obs_mat=obs_mats[0]
        if("C" in si_PRN):
            obs_mat=obs_mats[1]
        if("E" in si_PRN):
            obs_mat=obs_mats[2]
        PCO_F1='L'+obs_mat[obs_index][0]['obstype'][0][1]
        PCO_F2='L'+obs_mat[obs_index][0]['obstype'][5][1]
        pco_params=sat_pcos[si_PRN]
        for param in pco_params:
            if(rt_unix-p1/clight-dts> param['Stime']):
                try:
                    off1=param[PCO_F1]
                    off2=param[PCO_F2]
                except:
                    off1=[0.0,0.0,0.0]
                    off2=[0.0,0.0,0.0]
        dant=[0.0,0.0,0.0]
        for k in range(3):
            dant1=off1[0]*ex[k]+off1[1]*ey[k]+off1[2]*ez[k]
            dant2=off2[0]*ex[k]+off2[1]*ey[k]+off2[2]*ez[k]
            dant[k]=C1*dant1+C2*dant2
        rs[0]=rs[0]+dant[0]
        rs[1]=rs[1]+dant[1]
        rs[2]=rs[2]+dant[2]
        peph_sat_pos[si_PRN]=[rs[0],rs[1],rs[2],dts,drs[0],drs[1],drs[2],(drts-dts)/dt]
    
    if(sol_mode=="Sat only"):
        return peph_sat_pos
        
    
    #伪距单点定位
    if(len(pre_rr)):
        #有先验位置
        rr[0]=pre_rr[0]
        rr[1]=pre_rr[1]
        rr[2]=pre_rr[2]
    result=np.zeros((4+2),dtype=np.float64) #结果维数4+2(X Y Z GPS钟差 BDS钟差 GAL钟差)
    result[0:3]=rr
    result[3]=1.0   #GPS钟差
    result[4]=1.0   #BDS钟差
    result[5]=1.0   #ISB钟差
    if(len(pre_rr)):
        result[3]=pre_rr[3]
    
    #print("标准单点定位求解滤波状态初值")
    #最小二乘求解滤波初值
    ls_count=0
    while(1):
        #光速, GPS系统维持的地球自转角速度(弧度制)
        clight=2.99792458e8
        OMGE=7.2921151467E-5

        #观测值矩阵初始化
        Z=np.zeros(sat_num,dtype=np.float64)
        #设计矩阵初始化
        H=np.zeros((sat_num,4+2),dtype=np.float64)
        #单位权中误差向量初始化
        var=[]
    
        #观测值、设计矩阵构建
        for i in range(0,sat_num):
        
            #观测时间&观测值
            rt_week=obs_mat[obs_index][0]['GPSweek']
            rt_sec=obs_mat[obs_index][0]['GPSsec']
            rt_unix=satpos.gpst2time(rt_week,rt_sec)
            #print(rt_week,rt_sec,rt_unix)
        
            #伪距
            p1=obslist[i]['OBS'][0]
            s1=obslist[i]['OBS'][4]
            p2=obslist[i]['OBS'][5]
            s2=obslist[i]['OBS'][6]
            #print(p1,p2,phi1,phi2)
            
            #卫星位置
            si_PRN=obslist[i]['PRN']
            rs=[peph_sat_pos[si_PRN][0],peph_sat_pos[si_PRN][1],peph_sat_pos[si_PRN][2]]
            dts=peph_sat_pos[si_PRN][3]
            
            r0=sqrt( (rs[0]-rr[0])*(rs[0]-rr[0])+(rs[1]-rr[1])*(rs[1]-rr[1])+(rs[2]-rr[2])*(rs[2]-rr[2]) )
            #线性化的站星单位向量
            urs_x=(rr[0]-rs[0])/r0
            urs_y=(rr[1]-rs[1])/r0
            urs_z=(rr[2]-rs[2])/r0
            
            #单卫星设计矩阵赋值与ISB赋值
            isb=0.0
            if("G" in si_PRN):
                H[i]=[urs_x,urs_y,urs_z,1.0,0.0,0.0]
                isb=result[3]
                f1=freqs[0][0]
                f2=freqs[0][1]
            if("C" in si_PRN):
                H[i]=[urs_x,urs_y,urs_z,0.0,1.0,0.0]
                isb=result[4]
                f1=freqs[1][0]
                f2=freqs[1][1]
            if("E" in si_PRN):
                H[i]=[urs_x,urs_y,urs_z,0.0,0.0,1.0]
                isb=result[5]
                f1=freqs[2][0]
                f2=freqs[2][1]
            
            #地球自转改正到卫地距上
            r0=r0+OMGE*(rs[0]*rr[1]-rs[1]*rr[0])/clight
            
            #观测矩阵
            if(sol_mode=='SF'):
                Z[i]=p1-r0-isb-satpos.get_Tropdelay(rr,rs)-satpos.get_ion_GPS(rt_unix,rr,rs,ion_param)+clight*dts
            
            #双频无电离层延迟组合
            elif(sol_mode=='IF'):
                f12=f1*f1
                f22=f2*f2
                p_IF=f12/(f12-f22)*p1-f22/(f12-f22)*p2
                Z[i]=p_IF-r0-isb-satpos.get_Tropdelay(rr,rs)+clight*dts

            #随机模型
            #var[i][i]= 0.00224*10**(-s1 / 10) 
            _,el=satpos.getazel(rs,rr)
            var.append(0.3*0.3+0.3*0.3/sin(el)/sin(el))
            if(el*180.0/satpos.pi<el_threthod):
                var[i]=var[i]*100#低高度角拒止
                ex_index[i]=1
            if(ex_index[i]==1 and el*180.0/satpos.pi>=el_threthod):
                ex_index[i]=0
            
            if(sol_mode=='IF'):
                var[i]=var[i]*9
        
        #最小二乘求解:
        # if(sat_num_C<1 and sat_num_E>0):
        #     H_sub_ISBBDS=np.zeros((1,4+2),dtype=np.float64)
        #     var_sub_ISBBDS=np.zeros((1,sat_num+1),dtype=np.float64)
        #     H_sub_ISBBDS[0][4]=1.0      #ISB_BDS秩亏系数
        #     Z_sub_ISBBDS=0.0            #ISB_BDS秩亏观测
        #     var_sub_ISBBDS[0][-1]=1/0.01         #ISB_BDS秩亏方差
        #     H=np.concatenate((H,H_sub_ISBBDS)) #设计矩阵处理
        #     Z=np.append(Z,Z_sub_ISBBDS)        #观测矩阵处理
        #     W = np.hstack((W, np.zeros((sat_num,1)))) #方差阵增加一列
        #     W=np.concatenate((W,var_sub_ISBBDS))
        # if(sat_num_E<1 and sat_num_C>0):
        #     H_sub_ISBGAL=np.zeros((1,4+2),dtype=np.float64)
        #     var_sub_ISBGAL=np.zeros((1,sat_num+1),dtype=np.float64)
        #     H_sub_ISBGAL[0][5]=1.0      #ISB_GAL秩亏系数
        #     Z_sub_ISBGAL=0.0            #ISB_GAL秩亏观测
        #     var_sub_ISBGAL[0][-1]=1/0.01         #ISB_GAL秩亏方差
        #     H=np.concatenate((H,H_sub_ISBGAL)) #设计矩阵处理
        #     Z=np.append(Z,Z_sub_ISBGAL)        #观测矩阵处理
        #     W = np.hstack((W, np.zeros((sat_num,1)))) #方差阵增加一列
        #     W=np.concatenate((W,var_sub_ISBGAL))
        # if(sat_num_E<1 and sat_num_C<1):
        #     H_sub_ISBBDS=np.zeros((1,4+2),dtype=np.float64)
        #     var_sub_ISBBDS=np.zeros((1,sat_num+1),dtype=np.float64)
        #     H_sub_ISBBDS[0][4]=1.0      #ISB_BDS秩亏系数
        #     Z_sub_ISBBDS=0.0            #ISB_BDS秩亏观测
        #     var_sub_ISBBDS[0][-1]=1.0/0.01         #ISB_BDS秩亏方差
        #     H=np.concatenate((H,H_sub_ISBBDS)) #设计矩阵处理
        #     Z=np.append(Z,Z_sub_ISBBDS)        #观测矩阵处理
        #     W = np.hstack((W, np.zeros((sat_num,1))))
        #     W=np.concatenate((W,var_sub_ISBBDS))    #方差阵增加一列
            
        #     H_sub_ISBGAL=np.zeros((1,4+2),dtype=np.float64)
        #     var_sub_ISBGAL=np.zeros((1,sat_num+2),dtype=np.float64)
        #     H_sub_ISBGAL[0][5]=1.0      #ISB_GAL秩亏系数
        #     Z_sub_ISBGAL=0.0            #ISB_GAL秩亏观测
        #     var_sub_ISBGAL[0][-1]=1/0.01         #ISB_GAL秩亏方差
        #     H=np.concatenate((H,H_sub_ISBGAL)) #设计矩阵处理
        #     Z=np.append(Z,Z_sub_ISBGAL)        #观测矩阵处理
        #     W = np.hstack((W, np.zeros((sat_num+1,1)))) #方差阵增加一列
        #     W=np.concatenate((W,var_sub_ISBGAL))      #方差阵增加一行
         #各系统钟差失配,添加虚拟零观测确保H满秩
        if(sat_num_G==0):
            H_sub_G=np.array([0,0,0,1,0,0]).reshape(1,6)
            var.append(0.01)#虚拟方差
            H=np.concatenate((H,H_sub_G)) #设计矩阵处理
            Z=np.append(Z,0.0)
        if(sat_num_C==0):
            H_sub_C=np.array([0,0,0,0,1,0]).reshape(1,6)
            var.append(0.01)#虚拟方差
            H=np.concatenate((H,H_sub_C)) #设计矩阵处理
            Z=np.append(Z,0.0)
        if(sat_num_E==0):
            H_sub_E=np.array([0,0,0,0,0,1]).reshape(1,6)
            var.append(0.01)#虚拟方差
            H=np.concatenate((H,H_sub_E)) #设计矩阵处理
            Z=np.append(Z,0.0)
        #权重矩阵构建
        W=np.zeros((len(var),len(var)),dtype=np.float64)
        for i in range(len(var)):
            W[i][i]=1.0/var[i]
        
        dresult=getLSQ_solution(H,Z,W=W,weighting_mode='S')
        
        #迭代值更新
        result[0]+=dresult[0]
        result[1]+=dresult[1]
        result[2]+=dresult[2]
        result[3]+=dresult[3] #GPS_CLK
        result[4]+=dresult[4] #BDS_CLK
        result[5]+=dresult[5] #BDS_CLK

        #更新测站位置
        rr[0]=result[0]
        rr[1]=result[1]
        rr[2]=result[2]
        #print(dresult)
        ls_count+=1
        if(abs(dresult[0])<1e-4 and abs(dresult[1])<1e-4 and abs(dresult[2])<1e-4):
            #估计先验精度因子
            break
        
        if(ls_count>200):
            #最小二乘迭代次数
            break
    
    #排除低高度角卫星
    for i in range(sat_num):
        if(ex_index[i]):
            obslist_new.remove(obslist[i])
    return result,obslist_new,peph_sat_pos


#多系统非差非组合PPP状态合并初始化
def init_UCPPP_M(X_G,X_C,X_E,
                 Pk_G,Pk_C,Pk_E,
                 Qk_G,Qk_C,Qk_E,
                 GF_sign_G,GF_sign_C,GF_sign_E,
                 Mw_sign_G,Mw_sign_C,Mw_sign_E,
                 slip_sign_G,slip_sign_C,slip_sign_E,
                 dN_sign_G,dN_sign_C,dN_sign_E,
                 X_time_G,X_time_C,X_time_E,
                 phase_bias_G,phase_bias_C,phase_bias_E):
    #函数: 多系统非差非组合PPP状态合并初始化
    #输入: sppp.py中单系统初始化结果
    #输出: sppp_multiGNSS.py所需PPP输入
    #各系统星座卫星数量
    sat_num_G=int((X_G.shape[0]-5)/3)
    sat_num_C=int((X_C.shape[0]-5)/3)
    sat_num_E=int((X_E.shape[0]-5)/3)
    #状态、方差、过程噪声向量矩阵生成
    X=np.zeros((5+3*(sat_num_G+sat_num_C+sat_num_E)+2,1))#将ISB置于状态向量最末尾
    X_time=np.zeros((5+3*(sat_num_G+sat_num_C+sat_num_E)+2,1))
    Pk=np.zeros((5+3*(sat_num_G+sat_num_C+sat_num_E)+2,5+3*(sat_num_G+sat_num_C+sat_num_E)+2),dtype=np.float64)#滤波方差阵生成
    Qk=np.zeros((5+3*(sat_num_G+sat_num_C+sat_num_E)+2,5+3*(sat_num_G+sat_num_C+sat_num_E)+2),dtype=np.float64)#滤波过程噪声阵生成
    
    X[0][0]=X_G[0][0]                                    #以GPS系统位置、钟差、ZWD为初始状态
    X[1][0]=X_G[1][0]
    X[2][0]=X_G[2][0]
    X[3][0]=X_G[3][0]
    X[4][0]=X_G[4][0]
    #如果位置初值为零, 调整赋值
    if(X[0][0]==0):
        X[0][0],X[1][0],X[2][0]=X_C[0][0],X_C[1][0],X_C[2][0]
    if(X[0][0]==0):
        X[0][0],X[1][0],X[2][0]=X_E[0][0],X_E[1][0],X_E[2][0]
    for i in range(5):
        Pk[i][i]=Pk_G[i][i]
        Qk[i][i]=Qk_G[i][i]
        X_time[i][0]=X_time_G[i][0]
    
    for i in range(0,3*sat_num_G):                       #GPS电离层、模糊度
        X[5+i][0]=X_G[5+i][0]
        X_time[5+i][0]=X_time_G[5+i][0]
        Pk[5+i][5+i]=Pk_G[5+i][5+i]
        Qk[5+i][5+i]=Qk_G[5+i][5+i]
    for i in range(0,3*sat_num_C):                       #BDS电离层、模糊度
        X[5+3*sat_num_G+i][0]=X_C[5+i][0]
        X_time[5+3*sat_num_G+i][0]=X_time_C[5+i][0]
        Pk[5+3*sat_num_G+i][5+3*sat_num_G+i]=Pk_C[5+i][5+i]                
        Qk[5+3*sat_num_G+i][5+3*sat_num_G+i]=Qk_C[5+i][5+i]                
    for i in range(0,3*sat_num_E):                       #GAL电离层、模糊度
        X[5+3*sat_num_G+3*sat_num_C+i][0]=X_E[5+i][0]
        X_time[5+3*sat_num_G+3*sat_num_C+i][0]=X_time_E[5+i][0]
        Pk[5+3*sat_num_G+3*sat_num_C+i][5+3*sat_num_G+3*sat_num_C+i]=Pk_E[5+i][5+i]
        Qk[5+3*sat_num_G+3*sat_num_C+i][5+3*sat_num_G+3*sat_num_C+i]=Qk_E[5+i][5+i]
           
    #ISB区块合并
    X[5+3*sat_num_G+3*sat_num_C+3*sat_num_E][0]=X_C[3][0]        #CLK_BDS
    X_time[5+3*sat_num_G+3*sat_num_C+3*sat_num_E][0]=X_time[3][0]          
    Pk[5+3*sat_num_G+3*sat_num_C+3*sat_num_E][5+3*sat_num_G+3*sat_num_C+3*sat_num_E]=Pk[3][3]
    Qk[5+3*sat_num_G+3*sat_num_C+3*sat_num_E][5+3*sat_num_G+3*sat_num_C+3*sat_num_E]=Qk[3][3]
    X[5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1][0]=X_E[3][0]      #CLK_GAL
    X_time[5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1][0]=X_time[3][0]
    Pk[5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1][5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1]=Pk[3][3]
    Qk[5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1][5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1]=Qk[3][3]

    #历元间差分检验量合并
    GF_sign=np.concatenate((GF_sign_G,GF_sign_C,GF_sign_E))
    Mw_sign=np.concatenate((Mw_sign_G,Mw_sign_C,Mw_sign_E))
    slip_sign=np.concatenate((slip_sign_G,slip_sign_C,slip_sign_E))
    dN_sign=np.concatenate((dN_sign_G,dN_sign_C,dN_sign_E))

    #相位误差字典合并
    phase_bias={}
    for key in phase_bias_G.keys():
        phase_bias[key]={}
        phase_bias[key]=phase_bias_G[key]
    for key in phase_bias_C.keys():
        phase_bias[key]={}
        phase_bias[key]=phase_bias_C[key]
    for key in phase_bias_E.keys():
        phase_bias[key]={}
        phase_bias[key]=phase_bias_E[key]
    
    return X,Pk,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,X_time,phase_bias

#多系统PPP观测模型构建
def createKF_HRZ_M(obslist,rt_unix,X,X_time,Pk,Qk,ion_param,phase_bias,peph_sat_pos,freqs,ex_threshold_v=30,exthreshold_v_sigma=4,post=True):
    
    #初始化卫星数量
    sat_num=len(obslist)
    sys_sat_sum=round((X.shape[0]-7)/3)#GCE三系统
    sat_out=[]
    sat_out_post=[]
    t_phase_bias=phase_bias.copy()
    #光速, GPS系统维持的地球自转角速度(弧度制)
    clight=2.99792458e8
    OMGE=7.2921151467E-5
    #dt=0.001    #计算卫星速度用于相对论效应改正(改正到钟差, 由SPP_from_IGS完成)
    rr=np.array([X[0],X[1],X[2],X[3]]).reshape(4)

    dr=solid_tides(rt_unix,X)

    rr[0]=rr[0]+dr[0]
    rr[1]=rr[1]+dr[1]
    rr[2]=rr[2]+dr[2]
    
    #创建设计矩阵和观测值矩阵(观测模型)
    H=np.zeros((4*sat_num,3*sat_num+7),dtype=np.float64)
    #Z=np.zeros((4*sat_num,1),dtype=np.float64)
    R=np.eye(4*sat_num,dtype=np.float64)
    v=np.zeros((4*sat_num,1),dtype=np.float64)
    #print("H,Z,R",H.shape,Z.shape,R.shape)
    for i in range(sat_num): #逐卫星按行创建设计矩阵
        #状态索引求解
        si_PRN=obslist[i]['PRN']
        sys_shift=0
        f1=freqs[0][0]
        f2=freqs[0][1]
        sys_sat_num=32      #GPS系统星座卫星数量
        if('C' in si_PRN):
            sys_shift=32   #多系统索引偏置(GPS)
            sys_sat_num=65
            f1=freqs[1][0]
            f2=freqs[1][1]
        if('E' in si_PRN):
            sys_shift=32+65   #多系统索引偏置(GPS+BDS)
            sys_sat_num=37
            f1=freqs[2][0]
            f2=freqs[2][1] 
        PRN_index=int(si_PRN[1:])-1
        ion_index=5+3*sys_shift+PRN_index
        N1_index=5+3*sys_shift+sys_sat_num+PRN_index
        N2_index=5+3*sys_shift+sys_sat_num*2+PRN_index
        #观测时间&观测值
        rt_unix=rt_unix
        ##伪距&相位&CNo
        p1=obslist[i]['OBS'][0]
        l1=obslist[i]['OBS'][1]
        s1=obslist[i]['OBS'][4]
        p2=obslist[i]['OBS'][5]
        l2=obslist[i]['OBS'][6]
        s2=obslist[i]['OBS'][9]
        #print(p1,p2,phi1,phi2)
        
        #卫星位置
        si_PRN=obslist[i]['PRN']
        rs=[peph_sat_pos[si_PRN][0],peph_sat_pos[si_PRN][1],peph_sat_pos[si_PRN][2]]
        dts=peph_sat_pos[si_PRN][3]
        drs=[peph_sat_pos[si_PRN][4],peph_sat_pos[si_PRN][5],peph_sat_pos[si_PRN][6]]

        #线性化的站星向量
        r0=sqrt( (rs[0]-rr[0])*(rs[0]-rr[0])+(rs[1]-rr[1])*(rs[1]-rr[1])+(rs[2]-rr[2])*(rs[2]-rr[2]) )

        #线性化的站星单位向量
        urs_x=(rr[0]-rs[0])/r0
        urs_y=(rr[1]-rs[1])/r0
        urs_z=(rr[2]-rs[2])/r0

        #对流层延迟投影函数
        Mh,Mw=NMF(rr,rs,rt_unix)
        #电离层延迟投影函数
        Mi=IMF_ion(rr,rs)

        #单卫星四行设计矩阵分量构建
        #p1
        H_sub1=np.zeros(3*sat_num+7,dtype=np.float64)   #初始化频1伪距行
        H_sub1[0:5]=[urs_x,urs_y,urs_z,0,Mw]            #基础项
        H_sub1[5+i]=1                                   #频1伪距STEC系数
        if("G" in si_PRN):
            H_sub1[3]=1                                 #GPS钟差
        if("C" in si_PRN):
            H_sub1[-2]=1                                #BDS钟差
        if("E" in si_PRN):
            H_sub1[-1]=1                                #GAL钟差
        #l1
        H_sub2=np.zeros(3*sat_num+7,dtype=np.float64)   #初始化频1相位行
        H_sub2[0:5]=[urs_x,urs_y,urs_z,0,Mw]            #基础项
        H_sub2[5+i]=-1                                  #频1相位STEC系数
        H_sub2[5+sat_num+i]=1                           #频1模糊度
        if("G" in si_PRN):
            H_sub2[3]=1                                 #GPS钟差
        if("C" in si_PRN):
            H_sub2[-2]=1                                #BDS钟差
        if("E" in si_PRN):
            H_sub2[-1]=1                                #GAL钟差
        #p2
        H_sub3=np.zeros(3*sat_num+7,dtype=np.float64)   #初始化频2伪距行
        H_sub3[0:5]=[urs_x,urs_y,urs_z,0,Mw]            #基础项
        H_sub3[5+i]=f1*f1/f2/f2                         #频2伪距STEC系数
        if("G" in si_PRN):
            H_sub3[3]=1                                 #GPS钟差
        if("C" in si_PRN):
            H_sub3[-2]=1                                #BDS钟差
        if("E" in si_PRN):
            H_sub3[-1]=1                                #GAL钟差
        #l2
        H_sub4=np.zeros(3*sat_num+7,dtype=np.float64)   #初始化频1相位行
        H_sub4[0:5]=[urs_x,urs_y,urs_z,0,Mw]            #基础项
        H_sub4[5+i]=-f1*f1/f2/f2                        #频2相位STEC系数
        H_sub4[5+2*sat_num+i]=1                         #频2模糊度
        if("G" in si_PRN):
            H_sub4[3]=1                                 #GPS钟差
        if("C" in si_PRN):
            H_sub4[-2]=1                                #BDS钟差
        if("E" in si_PRN):
            H_sub4[-1]=1                                #GAL钟差

        #设计矩阵
        H[i*4]=H_sub1
        H[i*4+1]=H_sub2
        H[i*4+2]=H_sub3
        H[i*4+3]=H_sub4

        #相位改正
        phw=sat_phw(rt_unix+rr[3]/clight,si_PRN,1,rr,rs,drs,t_phase_bias)
        l1=l1-phw
        l2=l2-phw
        t_phase_bias[si_PRN]={}
        t_phase_bias[si_PRN]['phw']=phw
        
        #伪距自转改正
        r0=r0+OMGE*(rs[0]*rr[1]-rs[1]*rr[0])/clight
        
        #残差向量
        
        #钟差量
        if('G' in si_PRN):
            isb=X[3][0]
        if("C" in si_PRN):
            isb=X[-2][0]
        if("E" in si_PRN):
            isb=X[-1][0]

        v[i*4]=  p1 -           (r0 + isb - (clight*dts) + (Mh*get_Trop_delay_dry(rr)+Mw*X[4][0]) + (1*X[ion_index][0]) )
        v[i*4+1]=l1*clight/f1 - (r0 + isb - (clight*dts) + (Mh*get_Trop_delay_dry(rr)+Mw*X[4][0]) + ( -1*X[ion_index][0]) + (1*X[N1_index][0]) )
        v[i*4+2]=p2 -           (r0 + isb - (clight*dts) + (Mh*get_Trop_delay_dry(rr)+Mw*X[4][0]) + ( f1*f1/f2/f2*X[ion_index][0]) )
        v[i*4+3]=l2*clight/f2 - (r0 + isb - (clight*dts) + (Mh*get_Trop_delay_dry(rr)+Mw*X[4][0]) + ( -f1*f1/f2/f2*X[ion_index][0]) + (1*X[N2_index][0]) )
        #观测噪声(随机模型)
        
        _,el=satpos.getazel(rs,rr)
        var=0.003*0.003+0.003*0.003/sin(el)/sin(el)
        
        # var=0.00224*10**(-s1 / 10)
        # var_1=1.0
        # var_2=1.0
        var_ion=Qk[ion_index][ion_index]
        var_N1=Qk[N1_index][N1_index]
        var_N2=Qk[N2_index][N2_index]
        var_trop=0.01*0.01
        var_ion=0.0
        var_N1=0.0
        var_N2=0.0
        
        R[i*4][i*4]=100*100*(var)+var_ion+var_trop#伪距/相位标准差倍数
        R[i*4+1][i*4+1]=var+var_ion+var_N1+var_trop
        R[i*4+2][i*4+2]=100*100*(var)+var_ion+var_trop
        R[i*4+3][i*4+3]=var+var_ion+var_N2+var_trop
        
        #验前残差粗差识别
        if(post==False):
            if(abs(v[i*4])>ex_threshold_v or abs(v[i*4+1])>ex_threshold_v or abs(v[i*4+2])>ex_threshold_v or abs(v[i*4+3])>ex_threshold_v):
                #非首历元粗差剔除
                #print("去除粗差前观测列表: ", obslist)
                sat_out.append(i)
                #H,R,phase_bias,v,obslist=createKF_HRZ_new(obslist,rt_unix,X,ion_param,phase_bias)
                #print(si_PRN,'验前残差检验不通过',v[i*4],v[i*4+1],v[i*4+2],v[i*4+3])
                #print("去除粗差后观测列表: ",obslist)
        #验后方差校验
        if(post==True):
            out_v=[]
            if abs(v[i*4])>exthreshold_v_sigma*sqrt(R[i*4][i*4]): 
                #print(si_PRN," 验后方差校验不通过",v[i*4],4*sqrt(R[i*4][i*4]))
                out_v.append(v[i*4])
            if abs(v[i*4+1])>exthreshold_v_sigma*sqrt(R[i*4+1][i*4+1]):
                #print(si_PRN," 验后方差校验不通过",v[i*4+1],4*sqrt(R[i*4+1][i*4+1]))
                out_v.append(v[i*4+1])
            if abs(v[i*4+2])>exthreshold_v_sigma*sqrt(R[i*4+2][i*4+2]): 
                #print(si_PRN," 验后方差校验不通过",v[i*4],v[i*4+2],4*sqrt(R[i*4+2][i*4+2]))
                out_v.append(v[i*4+2])
            if abs(v[i*4+3])>exthreshold_v_sigma*sqrt(R[i*4+3][i*4+3]):
                #print(si_PRN," 验后方差校验不通过",v[i*4+3],4*sqrt(R[i*4+3][i*4+3]))
                out_v.append(v[i*4+3])
            if(len(out_v)):
                out_v.append(i)
                sat_out_post.append(out_v)

    #循环结束, 处理验前粗差
    obslist_new=obslist.copy()
    for s in sat_out:
        obslist_new.remove(obslist[s])
    #处理验后残差
    if(post==True):
        #全部校验通过
        if(len(sat_out_post)==0):
            return "KF fixed", obslist_new, t_phase_bias, v
        
        #找到最大残差值
        vmax=0.0
        v_out=0
        for s in sat_out_post:
            for v_i in range(0,len(s)-1):
                if(abs(s[v_i])>vmax):
                    v_out=s[-1]
                    vmax=s[v_i]
        #print("验后残差排除", obslist[v_out]['PRN'])
        obslist_new.remove(obslist[v_out])
        return "KF fixing", obslist_new, phase_bias,v

    return X,X_time,H,R,t_phase_bias,v,obslist_new

def createKF_XkPkQk_M(obslist,X,Pk,Qk):
    #系统模型构建
    #输入: 观测字典列表, 全局状态X, 全局方差Pk, 全局过程噪声Qk
    #输出: 滤波状态t_Xk, 滤波方差t_Pk, 滤波过程噪声t_Qk
    sat_num=len(obslist)#本历元有效观测卫星数量
    sys_sat_sum=round((X.shape[0]-7)/3)#全局状态卫星数量
    
    #本历元更新状态所用系统临时变量(依据: 在观测列表内的卫星数量)
    t_Xk=np.zeros((3*sat_num+7,1),dtype=np.float64)
    t_Pk=np.zeros((3*sat_num+7,3*sat_num+7),dtype=np.float64)
    t_Qk=np.zeros((3*sat_num+7,3*sat_num+7),dtype=np.float64)
    
    #有效卫星索引(计算本历元有效卫星各状态量在总状态中的索引)
    sat_use=[]#首先保证不变状态量保存在内
    for i in range(len(obslist)):
        si_PRN=obslist[i]['PRN']
        PRN_index=int(si_PRN[1:])-1
        sat_use.append([si_PRN[0],PRN_index])#系统标识符和PRN位置索引
    index_use=[0,1,2,3,4]#基础导航状态(X Y Z Dt ZWD)
    sys_shift=0
    for s in sat_use:
        if(s[0]=="C"):
            sys_shift=3*32
        if(s[0]=="E"):
            sys_shift=3*32+3*65
        index_use.append(5+sys_shift+s[1])               #电离层状态导入
    for s in sat_use:
        if(s[0]=="G"):
            sys_shift=32
        if(s[0]=="C"):
            sys_shift=3*32+65
        if(s[0]=="E"):
            sys_shift=3*32+3*65+37
        index_use.append(5+sys_shift+s[1])   #L1模糊度状态导入
    for s in sat_use:
        if(s[0]=="G"):
            sys_shift=2*32
        if(s[0]=="C"):
            sys_shift=3*32+2*65
        if(s[0]=="E"):
            sys_shift=3*32+3*65+2*37
        index_use.append(5+sys_shift+s[1]) #L2模糊度状态导入
    #ISB导入
    index_use.append(5+3*sys_sat_sum)
    index_use.append(5+3*sys_sat_sum+1)
    
    #系统状态,方差,过程噪声赋值
    for i in range(5+3*sat_num+2):
        t_Xk[i]=X[index_use[i]]                         #系统状态
        t_Qk[i][i]=Qk[index_use[i]][index_use[i]]       #系统过程噪声
        for j in range(5+3*sat_num+2):
            t_Pk[i][j]=Pk[index_use[i]][index_use[j]]   #系统方差
    #ISB赋值
    
    #返回系统模型各临时矩阵
    return t_Xk,t_Pk,t_Qk


#多系统非差非组合PPP状态合并初始化
def init_UCPPP_M(X_G,X_C,X_E,
                 Pk_G,Pk_C,Pk_E,
                 Qk_G,Qk_C,Qk_E,
                 GF_sign_G,GF_sign_C,GF_sign_E,
                 Mw_sign_G,Mw_sign_C,Mw_sign_E,
                 slip_sign_G,slip_sign_C,slip_sign_E,
                 dN_sign_G,dN_sign_C,dN_sign_E,
                 X_time_G,X_time_C,X_time_E,
                 phase_bias_G,phase_bias_C,phase_bias_E):
    #函数: 多系统非差非组合PPP状态合并初始化
    #输入: sppp.py中单系统初始化结果
    #输出: sppp_multiGNSS.py所需PPP输入
    #各系统星座卫星数量
    sat_num_G=int((X_G.shape[0]-5)/3)
    sat_num_C=int((X_C.shape[0]-5)/3)
    sat_num_E=int((X_E.shape[0]-5)/3)
    #状态、方差、过程噪声向量矩阵生成
    X=np.zeros((5+3*(sat_num_G+sat_num_C+sat_num_E)+2,1))#将ISB置于状态向量最末尾
    X_time=np.zeros((5+3*(sat_num_G+sat_num_C+sat_num_E)+2,1))
    Pk=np.zeros((5+3*(sat_num_G+sat_num_C+sat_num_E)+2,5+3*(sat_num_G+sat_num_C+sat_num_E)+2),dtype=np.float64)#滤波方差阵生成
    Qk=np.zeros((5+3*(sat_num_G+sat_num_C+sat_num_E)+2,5+3*(sat_num_G+sat_num_C+sat_num_E)+2),dtype=np.float64)#滤波过程噪声阵生成
    
    X[0][0]=X_G[0][0]                                    #以GPS系统位置、钟差、ZWD为初始状态
    X[1][0]=X_G[1][0]
    X[2][0]=X_G[2][0]
    X[3][0]=X_G[3][0]
    X[4][0]=X_G[4][0]
    #如果位置初值为零, 调整赋值
    if(X[0][0]==0):
        X[0][0],X[1][0],X[2][0]=X_C[0][0],X_C[1][0],X_C[2][0]
    if(X[0][0]==0):
        X[0][0],X[1][0],X[2][0]=X_E[0][0],X_E[1][0],X_E[2][0]
    for i in range(5):
        Pk[i][i]=Pk_G[i][i]
        Qk[i][i]=Qk_G[i][i]
        X_time[i][0]=X_time_G[i][0]
    
    for i in range(0,3*sat_num_G):                       #GPS电离层、模糊度
        X[5+i][0]=X_G[5+i][0]
        X_time[5+i][0]=X_time_G[5+i][0]
        Pk[5+i][5+i]=Pk_G[5+i][5+i]
        Qk[5+i][5+i]=Qk_G[5+i][5+i]
    for i in range(0,3*sat_num_C):                       #BDS电离层、模糊度
        X[5+3*sat_num_G+i][0]=X_C[5+i][0]
        X_time[5+3*sat_num_G+i][0]=X_time_C[5+i][0]
        Pk[5+3*sat_num_G+i][5+3*sat_num_G+i]=Pk_C[5+i][5+i]                
        Qk[5+3*sat_num_G+i][5+3*sat_num_G+i]=Qk_C[5+i][5+i]                
    for i in range(0,3*sat_num_E):                       #GAL电离层、模糊度
        X[5+3*sat_num_G+3*sat_num_C+i][0]=X_E[5+i][0]
        X_time[5+3*sat_num_G+3*sat_num_C+i][0]=X_time_E[5+i][0]
        Pk[5+3*sat_num_G+3*sat_num_C+i][5+3*sat_num_G+3*sat_num_C+i]=Pk_E[5+i][5+i]
        Qk[5+3*sat_num_G+3*sat_num_C+i][5+3*sat_num_G+3*sat_num_C+i]=Qk_E[5+i][5+i]
           
    #ISB区块合并
    X[5+3*sat_num_G+3*sat_num_C+3*sat_num_E][0]=X_C[3][0]        #CLK_BDS
    X_time[5+3*sat_num_G+3*sat_num_C+3*sat_num_E][0]=X_time[3][0]          
    Pk[5+3*sat_num_G+3*sat_num_C+3*sat_num_E][5+3*sat_num_G+3*sat_num_C+3*sat_num_E]=Pk[3][3]
    Qk[5+3*sat_num_G+3*sat_num_C+3*sat_num_E][5+3*sat_num_G+3*sat_num_C+3*sat_num_E]=Qk[3][3]
    X[5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1][0]=X_E[3][0]      #CLK_GAL
    X_time[5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1][0]=X_time[3][0]
    Pk[5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1][5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1]=Pk[3][3]
    Qk[5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1][5+3*sat_num_G+3*sat_num_C+3*sat_num_E+1]=Qk[3][3]

    #历元间差分检验量合并
    GF_sign=np.concatenate((GF_sign_G,GF_sign_C,GF_sign_E))
    Mw_sign=np.concatenate((Mw_sign_G,Mw_sign_C,Mw_sign_E))
    slip_sign=np.concatenate((slip_sign_G,slip_sign_C,slip_sign_E))
    dN_sign=np.concatenate((dN_sign_G,dN_sign_C,dN_sign_E))

    #相位误差字典合并
    phase_bias={}
    for key in phase_bias_G.keys():
        phase_bias[key]={}
        phase_bias[key]=phase_bias_G[key]
    for key in phase_bias_C.keys():
        phase_bias[key]={}
        phase_bias[key]=phase_bias_C[key]
    for key in phase_bias_E.keys():
        phase_bias[key]={}
        phase_bias[key]=phase_bias_E[key]
    
    return X,Pk,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,X_time,phase_bias

def upstateKF_XkPkQk_M(obslist,rt_unix,t_Xk,t_Pk,t_Qk,X,Pk,Qk,X_time):
    #系统模型恢复与更新
    #输入: 观测字典列表, 滤波状态t_Xk, 滤波方差t_Pk, 滤波过程噪声t_Qk, 全局状态X, 全局方差Pk, 全局过程噪声Qk
    #输出: 恢复并更新后的全局状态X, 全局方差Pk, 全局过程噪声Qk
    sat_num=len(obslist)#本历元有效观测卫星数量
    sys_sat_sum=round((X.shape[0]-7)/3)#全局状态卫星数量
    
    #本历元更新状态所用系统临时变量(不能占用全局状态储存空间)
    t_X=np.zeros((3*sys_sat_sum+7,1),dtype=np.float64)
    t_X_time=np.zeros(3*sys_sat_sum+7,dtype=np.float64)
    t_P=np.zeros((3*sys_sat_sum+7,3*sys_sat_sum+7),dtype=np.float64)
    t_Q=np.zeros((3*sys_sat_sum+7,3*sys_sat_sum+7),dtype=np.float64)
    #拷贝原值
    t_X=X.copy()
    t_X_time=X_time.copy()
    t_P=Pk.copy()
    t_Q=Qk.copy()

    #有效卫星索引(计算本历元有效卫星各状态量在总状态中的索引)
    sat_use=[]#首先保证不变状态量保存在内
    for i in range(len(obslist)):
        si_PRN=obslist[i]['PRN']
        PRN_index=int(si_PRN[1:])-1
        sat_use.append([si_PRN[0],PRN_index])
    
    index_use=[0,1,2,3,4]
    sys_shift=0
    for s in sat_use:
        if(s[0]=="C"):
            sys_shift=3*32
        if(s[0]=="E"):
            sys_shift=3*32+3*65
        index_use.append(5+sys_shift+s[1])               #电离层状态导入
    for s in sat_use:
        if(s[0]=="G"):
            sys_shift=32
        if(s[0]=="C"):
            sys_shift=3*32+65
        if(s[0]=="E"):
            sys_shift=3*32+3*65+37
        index_use.append(5+sys_shift+s[1])   #L1模糊度状态导入
    for s in sat_use:
        if(s[0]=="G"):
            sys_shift=2*32
        if(s[0]=="C"):
            sys_shift=3*32+2*65
        if(s[0]=="E"):
            sys_shift=3*32+3*65+2*37
        index_use.append(5+sys_shift+s[1]) #L2模糊度状态导入
    #ISB导入
    index_use.append(5+3*sys_sat_sum)
    index_use.append(5+3*sys_sat_sum+1)
    
    #系统状态,方差,过程噪声恢复更新
    for i in range(5+3*sat_num+2):
        t_X[index_use[i]]=t_Xk[i]                         #系统状态
        t_X_time[index_use[i]]=rt_unix                    #系统状态时标
        t_Q[index_use[i]][index_use[i]]=t_Qk[i][i]        #系统过程噪声
        for j in range(5+3*sat_num+2):
            #滤波方差异常值处理
            if(i==j and t_Pk[i][j]<0.0):
                t_Pk[i][j]=60*60
            t_P[index_use[i]][index_use[j]]=t_Pk[i][j]   #系统方差

    #返回系统模型各全局矩阵
    return t_X,t_P,t_Q,t_X_time

def update_phase_slip_M(obslist,GF_sign,Mw_sign,slip_sign,Mw_threshold,GF_threshold,freqs,dN=[],dN_fix_mode=0):
    #首先清空周跳标志
    for i in range(len(slip_sign)):
        slip_sign[i]=0
    
    #清空无观测值的周跳检测量
    prns=[t['PRN'] for t in obslist]    #字符型PRN列表
    for i in range(len(GF_sign)):       #遍历各系统各卫星周跳标志
        if(i<32):
            in_PRN="G{:02d}".format(i+1)
        elif(i<32+65):
            in_PRN="C{:02d}".format(i-32+1)
        elif(i<32+65+37):
            in_PRN="E{:02d}".format(i-32-65+1)
        if(in_PRN not in prns):
            GF_sign[i]=0.0
            Mw_sign[i]=0.0
    
    #周跳检测
    sat_num=len(obslist)
    
    for i in range(sat_num):
        si_PRN=obslist[i]['PRN']
        PRN_index=int(si_PRN[1:])-1
        f_ids=0
        if("C" in si_PRN):
            PRN_index=PRN_index+32
            f_ids=1
        if("E" in si_PRN):
            PRN_index=PRN_index+32+65
            f_ids=2
        
        p1=obslist[i]['OBS'][0]
        l1=obslist[i]['OBS'][1]
        p2=obslist[i]['OBS'][5]
        l2=obslist[i]['OBS'][6]

        GF,Mw,slip,dN1,dN2=get_phase_jump(p1,p2,l1,l2,GF_sign[PRN_index],Mw_sign[PRN_index],Mw_threshold,GF_threshold,f1=freqs[f_ids][0],f2=freqs[f_ids][1])
        if(slip):
            #print('{} 发生周跳 GF:{}->{} Mw:{}->{} p1:{} l1:{} p2:{} l2:{} dN1:{} dN2:{}'.format(si_PRN,GF_sign[PRN_index],GF,Mw_sign[PRN_index],Mw,p1,l1,p2,l2,dN1,dN2))
            pass
            #print('{} phase jump occurred, GF:{}->{} Mw:{}->{} p1:{} l1:{} p2:{} l2:{} dN1:{} dN2:{}'.format(si_PRN,GF_sign[PRN_index],GF,Mw_sign[PRN_index],Mw,p1,l1,p2,l2,dN1,dN2))
        GF_sign[PRN_index]=GF
        Mw_sign[PRN_index]=Mw
        slip_sign[PRN_index]=slip
        
        if(dN_fix_mode):
            dN[PRN_index][0]=dN1
            dN[PRN_index][1]=dN2
    
    return GF_sign,Mw_sign,slip_sign,dN

def updata_PPP_state_M(X,Pk,spp_rr,epoch,rt_unix,X_time,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,GF_threshold,Mw_threshold,
                     sat_num,rnx_obs,out_age,freqs,dy_mode):
    #状态递推
    X[3]=spp_rr[3]#接收机钟差数值更新
    X[-2]=spp_rr[4]#系统间偏差数值更新(ISB_BDS)
    X[-1]=spp_rr[5]#系统间偏差数值更新(ISB_GAL)
    
    if(X[0][0]==100.0):
        #上历元先验重置
        print("上历元无解,重置")
        X[0][0]=spp_rr[0]
        X[1][0]=spp_rr[1]
        X[2][0]=spp_rr[2]
        Pk[0][0]=30*30
        Pk[1][1]=30*30
        Pk[2][2]=30*30
    
    
    if(dy_mode!='static'):
        #非静态观测
        X[0][0]=spp_rr[0]    
        X[1][0]=spp_rr[1]    
        X[2][0]=spp_rr[2]    
    
    #非首历元, 状态重置
    if(epoch):
        #计算位置/钟差/对流层状态更新时间差
        dt=rt_unix-X_time[0][0]
        #位置/钟差/对流层状态过程噪声
        if(dy_mode=='static'):
            Qk[0][0]=0.0#3600.0#坐标改正数
            Qk[1][1]=0.0#3600.0#坐标改正数
            Qk[2][2]=0.0#3600.0#坐标改正数
            Qk[3][3]=30*30#接收机钟差(白噪声)
            Qk[4][4]=1e-8*dt#对流层延迟(缓慢变化)
            Qk[-2][-2]=30*30#接收机钟差(ISB_BDS, 白噪声)
            Qk[-1][-1]=30*30#接收机钟差(ISB_GAL, 白噪声)
        else:
            Qk[0][0]=3600.0#坐标改正数
            Qk[1][1]=3600.0#坐标改正数
            Qk[2][2]=3600.0#坐标改正数
            Qk[3][3]=30*30#接收机钟差(白噪声)
            Qk[4][4]=1e-8*dt#对流层延迟(缓慢变化)
            Qk[-2][-2]=30*30#接收机钟差(ISB_BDS, 白噪声)
            Qk[-1][-1]=30*30#接收机钟差(ISB_GAL, 白噪声)

        #部分更新的状态量
        #GPS电离层状态范围[5:5+32]
        for j in range(5,5+32):
            dt=rt_unix-X_time[j][0]
            si_PRN="G{:02d}".format(j-5+1)                  #待更新状态的卫星PRN
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]   #观测列表中的PRNs
            if(si_PRN in rnx_obs_prns and dt<=out_age ): 
                Qk[j][j]=0.0016*dt  #电离层()
            if(si_PRN in rnx_obs_prns and dt>out_age):
                # if(dt>out_age):
                #     GF_sign[int(si_PRN[1:])-1]=0.0
                #     Mw_sign[int(si_PRN[1:])-1]=0.0
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                ion=update_ion(p1,p2,f1=freqs[0][0],f2=freqs[0][1])
                X[j]=ion        #重置垂直电离层估计
                Qk[j][j]=30*30  #重置过程噪声
                Pk[j,:]=0.0 
                Pk[:,j]=0.0 
        #BDS电离层状态范围[5+3*32,5+3*32+65]
        for j in range(5+3*32,5+3*32+65):
            dt=rt_unix-X_time[j][0]
            si_PRN="C{:02d}".format(j-5-3*32+1)#整型PRN序号(PRN-1)
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age ): 
                Qk[j][j]=0.0016*dt  #电离层()
            if(si_PRN in rnx_obs_prns and dt>out_age):
                # if(dt>out_age):
                    # GF_sign[32+int(si_PRN[1:])-1]=0.0
                    # Mw_sign[32+int(si_PRN[1:])-1]=0.0
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                ion=update_ion(p1,p2,f1=freqs[1][0],f2=freqs[1][1])
                X[j]=ion       #重置垂直电离层估计
                Qk[j][j]=30*30  #重置过程噪声
                Pk[j,:]=0.0 
                Pk[:,j]=0.0 
        #GAL电离层状态范围[5+3*32+3*65,5+3*32+3*65+37]
        for j in range(5+3*32+3*65,5+3*32+3*65+37):
            dt=rt_unix-X_time[j][0]
            si_PRN="E{:02d}".format(j-5-3*32-3*65+1)#整型PRN序号(PRN-1)
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age ): 
                Qk[j][j]=0.0016*dt  #电离层()
            if(si_PRN in rnx_obs_prns and dt>out_age):
                # if(dt>out_age):
                    # GF_sign[32+65+int(si_PRN[1:])-1]=0.0
                    # Mw_sign[32+65+int(si_PRN[1:])-1]=0.0
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                ion=update_ion(p1,p2,f1=freqs[2][0],f2=freqs[2][1])
                X[j]=ion       #重置垂直电离层估计
                Qk[j][j]=30*30  #重置过程噪声
                Pk[j,:]=0.0 
                Pk[:,j]=0.0 
        
        #GPS 第一频率模糊度[5+32,5+2*32]
        for j in range(5+32,5+2*32):
            dt=rt_unix-X_time[j][0]
            si_PRN="G{:02d}".format(j-(5+32)+1)              #PRN码
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=0e-14*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p1,l1,freqs[0][0],p1,p2,f1=freqs[0][0],f2=freqs[0][1])
                Qk[j][j]=30*30  #重置过程噪声
                Pk[j,:]=0.0 
                Pk[:,j]=0.0 
        #BDS 第一频率模糊度[5+3*32+65,5+3*32+2*65]
        for j in range(5+3*32+65,5+3*32+2*65):
            dt=rt_unix-X_time[j][0]
            si_PRN="C{:02d}".format(j-(5+3*32+65)+1)              #PRN码
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=0e-14*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p1,l1,freqs[1][0],p1,p2,f1=freqs[1][0],f2=freqs[1][1])
                Qk[j][j]=30*30  #重置过程噪声
                Pk[j,:]=0.0 
                Pk[:,j]=0.0 
        #GAL 第一频率模糊度[5+3*32+3*65+37,5+3*32+3*65+2*37]
        for j in range(5+3*32+3*65+37,5+3*32+3*65+2*37):
            dt=rt_unix-X_time[j][0]
            si_PRN="E{:02d}".format(j-(5+3*32+3*65+37)+1)              #PRN码
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=0e-14*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p1,l1,freqs[2][0],p1,p2,f1=freqs[2][0],f2=freqs[2][1])
                Qk[j][j]=30*30  #重置过程噪声
                Pk[j,:]=0.0 
                Pk[:,j]=0.0       
        
        #GPS 第二频率模糊度[5+2*32,5+3*32]
        for j in range(5+2*32,5+3*32):
            dt=rt_unix-X_time[j][0]
            si_PRN="G{:02d}".format(j-(5+2*32)+1)   #PRN
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的PRN
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=0e-14*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p2,l2,freqs[0][1],p1,p2,f1=freqs[0][0],f2=freqs[0][1])
                Qk[j][j]=30*30  #重置过程噪声
                Pk[j,:]=0.0 
                Pk[:,j]=0.0 
        #BDS 第二频率模糊度[5+3*32+2*65,5+3*32+3*65]
        for j in range(5+3*32+2*65,5+3*32+3*65):
            dt=rt_unix-X_time[j][0]
            si_PRN="C{:02d}".format(j-(5+3*32+2*65)+1)#整型PRN序号(PRN-1)
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=0e-14*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p2,l2,freqs[1][1],p1,p2,f1=freqs[1][0],f2=freqs[1][1])
                Qk[j][j]=30*30  #重置过程噪声
                Pk[j,:]=0.0 
                Pk[:,j]=0.0 
        #GAL 第二频率模糊度[5+3*32+3*65+2*37,5+3*32]
        for j in range(5+3*32+3*65+2*37,5+3*32+3*65+3*37):
            dt=rt_unix-X_time[j][0]
            si_PRN="E{:02d}".format(j-(5+3*32+3*65+2*37))#整型PRN序号(PRN-1)
            rnx_obs_prns=[t['PRN'] for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=0e-14*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p2,l2,freqs[2][1],p1,p2,f1=freqs[2][0],f2=freqs[2][1])
                Qk[j][j]=30*30  #重置过程噪声
                Pk[j,:]=0.0 
                Pk[:,j]=0.0     
        #周跳探测
        #GF_sign,Mw_sign,slip_sign,dN_sign=
        update_phase_slip_M(rnx_obs,GF_sign,Mw_sign,slip_sign,Mw_threshold,GF_threshold,freqs,dN_sign,dN_fix_mode=1)
        #小周跳修复/大周跳重置
        for j in range(len(slip_sign)):
            # if(slip_sign[j] and (abs(dN_sign[j][0])<500 or abs(dN_sign[j][1]<500))):
            #     #print('{} G{:02d} 周跳修复 GF: {} Mw:{} dN1:{} dN2:{}'.format(epoch,j+1,GF_sign[j],Mw_sign[j],dN_sign[j][0],dN_sign[j][1]))
            #     X[5+sat_num+j]=X[5+sat_num+j]+dN_sign[j][0]*clight/f1                
            #     X[5+2*sat_num+j]=X[5+2*sat_num+j]+dN_sign[j][1]*clight/f2
            #     Qk[5+sat_num+j][5+sat_num+j]=1e2                
            #     Qk[5+2*sat_num+j][5+2*sat_num+j]=1e2
            if(slip_sign[j] and (abs(dN_sign[j][0])>=0.0 or abs(dN_sign[j][1]>=0.0))):
                sys_shift=0
                sys_in_num=32
                sys_in_id=0
                freqs_id=0
                if(j<32):
                    si_PRN="G{:02d}".format(j+1)
                    sys_in_id=j
                elif(j<32+65):
                    si_PRN="C{:02d}".format(j-32+1)
                    sys_shift=32
                    sys_in_num=65
                    sys_in_id=j-32
                    freqs_id=1
                elif(j<32+65+37):
                    si_PRN="E{:02d}".format(j-32-65+1)
                    sys_shift=32+65
                    sys_in_num=37
                    sys_in_id=j-32-65
                    freqs_id=2
                rnx_obs_prns=[t['PRN'] for t in rnx_obs]
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                #周跳状态量与过程噪声更新
                X[5+3*sys_shift+sys_in_num+sys_in_id]=update_phase_amb(p1,l1,freqs[freqs_id][0],p1,p2,f1=freqs[freqs_id][0],f2=freqs[freqs_id][1])              
                X[5+3*sys_shift+2*sys_in_num+sys_in_id]=update_phase_amb(p2,l2,freqs[freqs_id][1],p1,p2,f1=freqs[freqs_id][0],f2=freqs[freqs_id][1])
                Qk[5+3*sys_shift+sys_in_num+sys_in_id][5+3*sys_shift+sys_in_num+sys_in_id]=60*60                
                Pk[5+3*sys_shift+sys_in_num+sys_in_id,:]=0.0       
                Pk[:,5+3*sys_shift+sys_in_num+sys_in_id]=0.0         
                Qk[5+3*sys_shift+2*sys_in_num+sys_in_id][5+3*sys_shift+2*sys_in_num+sys_in_id]=60*60
                Pk[5+3*sys_shift+2*sys_in_num+sys_in_id,:]=0.0
                Pk[:,5+3*sys_shift+2*sys_in_num+sys_in_id]=0.0

def log2out_M(rt_unix,v,obslist,X,X_time,Pk,peph_sat_pos,freqs,ratio=0.0):
    #历元数据整备
    out={}
    for i in range(len(obslist)):
        si_PRN=obslist[i]['PRN']            #提取卫星PRN码
        PRN_index=int(si_PRN[1:])-1         #卫星PRN码对应的系统内偏置
        sys_sat_num=int((X.shape[0]-5)/3)   #总卫星数

        p1=obslist[i]['OBS'][0]
        l1=obslist[i]['OBS'][1]
        s1=obslist[i]['OBS'][4]
        p2=obslist[i]['OBS'][5]
        l2=obslist[i]['OBS'][6]
        s2=obslist[i]['OBS'][9]
        #print(p1,p2,phi1,phi2)
        
        #卫星位置
        rs=[peph_sat_pos[si_PRN][0],peph_sat_pos[si_PRN][1],peph_sat_pos[si_PRN][2],peph_sat_pos[si_PRN][3]]

        out[si_PRN]={}
        out[si_PRN]['GPSweek'],out[si_PRN]['GPSsec']=satpos.time2gpst(rt_unix)
        
        out[si_PRN]['sat_x']=rs[0]
        out[si_PRN]['sat_y']=rs[1]
        out[si_PRN]['sat_z']=rs[2]
        out[si_PRN]['sat_cdt']=satpos.clight*rs[3]

        #测站坐标
        out[si_PRN]['sta_x']=X[0][0]
        out[si_PRN]['std_sta_x']=Pk[0][0]
        out[si_PRN]['sta_y']=X[1][0]
        out[si_PRN]['std_sta_y']=Pk[1][1]
        out[si_PRN]['sta_z']=X[2][0]
        out[si_PRN]['std_sta_z']=Pk[2][2]
        
        #GPS钟差
        out[si_PRN]['GPSsec_dt']=X[3][0]
        out[si_PRN]['std_GPSsec_dt']=Pk[3][3]
        
        #BDS钟差改正数
        out[si_PRN]['CLK_BDS']=X[5+3*sys_sat_num][0]
        out[si_PRN]['std_CLK_BDS']=Pk[5+3*sys_sat_num][5+3*sys_sat_num]
        #GAL钟差改正数
        out[si_PRN]['CLK_GAL']=X[5+3*sys_sat_num+1][0]
        out[si_PRN]['std_CLK_GAL']=Pk[5+3*sys_sat_num+1][5+3*sys_sat_num+1]
        #天顶对流层湿延迟
        out[si_PRN]['ztd_w']=X[4][0]
        out[si_PRN]['std_ztd_w']=Pk[4][4]

        rr=[X[0][0],X[1][0],X[2][0]]
        
        #天顶对流层干延迟
        out[si_PRN]['ztd_h']=get_Trop_delay_dry(rr)
        
        #滤波后验残差
        out[si_PRN]['res_p1']=v[4*i][0]
        out[si_PRN]['res_l1']=v[4*i+1][0]
        out[si_PRN]['res_p2']=v[4*i+2][0]
        out[si_PRN]['res_l2']=v[4*i+3][0]

        #站星几何关系
        az,el=getazel(rs,rr)
        out[si_PRN]['azel']=[az/pi*180.0,el/pi*180.0]

        #电离层状态更新
        sys_shift=0
        f_id=0
        if("C" in si_PRN):
            sys_shift=3*32
            f_id=1
        if("E" in si_PRN):
            sys_shift=3*32+3*65
            f_id=2
        if(X_time[5+sys_shift+PRN_index]==rt_unix):
            Mi=IMF_ion(rr,rs,MF_mode=1,H_ion=350e3)
            out[si_PRN]['STEC']=X[5+sys_shift+PRN_index][0]*(freqs[f_id][0]/1e8)*(freqs[f_id][0]/1e8)/40.28
            out[si_PRN]['std_STEC']=Pk[5+sys_shift+PRN_index][5+sys_shift+PRN_index]*((freqs[f_id][0]/1e8)*(freqs[f_id][0]/1e8)/40.28)**2
        
        #模糊度状态更新
        sys_shift=32    #N1基础状态
        if("C" in si_PRN):
            sys_shift=3*32+65
        if("E" in si_PRN):
            sys_shift=3*32+3*65+37
        if(X_time[5+sys_shift+PRN_index]==rt_unix):
            out[si_PRN]['N1']=X[5+sys_shift+PRN_index][0]
            out[si_PRN]['std_N1']=Pk[5+sys_shift+PRN_index][5+sys_shift+PRN_index]
        sys_shift=2*32  #N2基础状态
        if("C" in si_PRN):
            sys_shift=3*32+2*65
        if("E" in si_PRN):
            sys_shift=3*32+3*65+2*37
        if(X_time[5+sys_shift+PRN_index]==rt_unix):
            out[si_PRN]['N2']=X[5+sys_shift+PRN_index][0]
            out[si_PRN]['std_N2']=Pk[5+sys_shift+PRN_index][5+sys_shift+PRN_index]
        #记录ratio值
        out[si_PRN]['ratio']=ratio
    
    return out



#测试多系统PPP-AR
def UCPPP_M(obs_mats,obs_start,obs_epoch,IGS,clk,
          sat_out,ion_param,sat_pcos,el_threthod,ex_threshold_v,ex_threshold_v_sigma,Mw_threshold,GF_threshold,dy_mode,
          X,Pk,Qk,phase_bias,X_time,GF_sign,Mw_sign,slip_sign,dN_sign,sat_num,out_age,freqs,AMB_FIX=0,sta_mode='None',RTK_Info={},AMB_FIX_Info={}):
    
    Out_log,Out_log_fix=[],[]

    obs_index=obs_start
    for epoch in tqdm(range(obs_epoch)):
    
        #观测时间
        rt_week=obs_mats[0][obs_index+epoch][0]['GPSweek']
        rt_sec=obs_mats[0][obs_index+epoch][0]['GPSsec']
        rt_unix=satpos.gpst2time(rt_week,rt_sec)
    
        #进行单点定位
        spp_rr,rnx_obs,peph_sat_pos=SPP_from_IGS_M(obs_mats,obs_index+epoch,IGS,clk,sat_out,ion_param,sat_pcos,freqs,el_threthod=el_threthod,sol_mode="IF",pre_rr=[X[0][0],X[1][0],X[2][0],X[3][0]])
        #无单点定位解
        if(not len(rnx_obs)):
            print("No valid observations, Pass epoch: Week: {}, sec: {}.".format(rt_week,rt_sec))
            continue
        #流动站重收敛
        if(RTK_Info['reinitial_sec'] and rt_sec%(RTK_Info['reinitial_sec'])==0):
            # 分别初始化各系统PPP子滤波状态与协方差
            X_G,Pk_G,Qk_G,GF_sign_G,Mw_sign_G,slip_sign_G,dN_sign_G,X_time_G,phase_bias_G=init_UCPPP(obs_mats[0],obs_index+epoch,IGS,clk,sat_out,ion_param,sat_pcos,sys_sat_num=32,f1=freqs[0][0],f2=freqs[0][1])
            X_C,Pk_C,Qk_C,GF_sign_C,Mw_sign_C,slip_sign_C,dN_sign_C,X_time_C,phase_bias_C=init_UCPPP(obs_mats[1],obs_index+epoch,IGS,clk,sat_out,ion_param,sat_pcos,sys_sat_num=65,f1=freqs[1][0],f2=freqs[1][1])
            X_E,Pk_E,Qk_E,GF_sign_E,Mw_sign_E,slip_sign_E,dN_sign_E,X_time_E,phase_bias_E=init_UCPPP(obs_mats[2],obs_index+epoch,IGS,clk,sat_out,ion_param,sat_pcos,sys_sat_num=37,f1=freqs[2][0],f2=freqs[2][1])
    
            #多系统非差PPP滤波状态初始化
            X,Pk,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,X_time,phase_bias=init_UCPPP_M(X_G,X_C,X_E,
                 Pk_G,Pk_C,Pk_E,
                 Qk_G,Qk_C,Qk_E,
                 GF_sign_G,GF_sign_C,GF_sign_E,
                 Mw_sign_G,Mw_sign_C,Mw_sign_E,
                 slip_sign_G,slip_sign_C,slip_sign_E,
                 dN_sign_G,dN_sign_C,dN_sign_E,
                 X_time_G,X_time_C,X_time_E,
                 phase_bias_G,phase_bias_C,phase_bias_E)

        #PPP状态更新
        updata_PPP_state_M(X,Pk,spp_rr,epoch,rt_unix,X_time,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,GF_threshold,Mw_threshold,sat_num,rnx_obs,out_age,freqs,dy_mode)

        #PPP时间更新
        X,Pk,Qk,X_time,v,phase_bias,rnx_obs=KF_UCPPP_M(X,X_time,Pk,Qk,ion_param,peph_sat_pos,rnx_obs,ex_threshold_v,ex_threshold_v_sigma,rt_unix,phase_bias,freqs,AMB_FIX,mode=sta_mode,RTK_Info=RTK_Info,AMB_FIX_Info=AMB_FIX_Info,Out_log_fix=Out_log_fix)

        #结果保存
        #Out_log.append([X[0][0],X[1][0],X[2][0]])
        Out_log.append(log2out_M(rt_unix,v,rnx_obs,X,X_time,Pk,peph_sat_pos,freqs))
    
    return Out_log,Out_log_fix

def find_elmax(rnx_obs,peph_sat_pos,Xk_dot,t_Pk,sys='G'):
    el_max=0.0
    el_max_id=999
    for i in range(len(rnx_obs)):
        si_PRN=rnx_obs[i]['PRN']
        if(si_PRN[0]!=sys):
            continue
        #ion_index=5+3*sys_shift+PRN_index
        N1_index=5+len(rnx_obs)+i
        N2_index=5+2*len(rnx_obs)+i
        #计算高度角
        sat_xyzt=peph_sat_pos[rnx_obs[i]['PRN']]
        #读取模糊度状态的方差
        tk_N1=t_Pk[N1_index][N1_index]
        tk_N2=t_Pk[N2_index][N2_index]
        _,el=getazel(sat_xyzt,[Xk_dot[0][0],Xk_dot[1][0],Xk_dot[2][0]])
        # if(tk_N1>2.25 or tk_N2>2.25):
        #     #模糊度状态在本历元重置, 不能作为参考星
        #     continue
        if(el>el_max):
            el_max=el
            el_max_id=i
    if(el_max_id!=999):
        el_max_prn=rnx_obs[el_max_id]['PRN']
    else:
        el_max_prn='X00'
    return el_max_prn,el_max_id


def PPP_AR_M(rnx_obs,peph_sat_pos,Xk_dot,t_Pk,freqs,ratio_threshold=2.0,amb_float_threshold=1.0,P_float_threshold=1.5**2,min_AR_num=3):
    #从Pk和X中恢复整数部分
    Xfloat_N12=np.zeros((2*len(rnx_obs),1))
    Pfloat_N12=np.zeros((2*len(rnx_obs),2*len(rnx_obs)))
    lambda_trans=np.zeros((2*len(rnx_obs),2*len(rnx_obs)))
    el_max_prns,el_max_ids=['X00','X00','X00'],[999,999,999]
    #逐系统查找高度角最高卫星
    el_max_prns[0],el_max_ids[0]=find_elmax(rnx_obs,peph_sat_pos,Xk_dot,t_Pk,'G')
    el_max_prns[1],el_max_ids[1]=find_elmax(rnx_obs,peph_sat_pos,Xk_dot,t_Pk,'C')
    el_max_prns[2],el_max_ids[2]=find_elmax(rnx_obs,peph_sat_pos,Xk_dot,t_Pk,'E')
    #星间单差算子
    for id in range(len(rnx_obs)):
        #卫星PRN
        si_prn=rnx_obs[id]['PRN']
        el_max_id=el_max_ids[['G','C','E'].index(si_prn[0])]#设置系统对应参考星
        f1,f2=freqs[['G','C','E'].index(si_prn[0])][0],freqs[['G','C','E'].index(si_prn[0])][1]
        #状态和方差信息获取
        Xfloat_N12[id][0]=Xk_dot[5+len(rnx_obs)+id][0]
        Pfloat_N12[id]=t_Pk[5+len(rnx_obs)+id][5+len(rnx_obs):-2]#注意模糊度状态范围
        Xfloat_N12[len(rnx_obs)+id]=Xk_dot[5+2*len(rnx_obs)+id][0]
        Pfloat_N12[len(rnx_obs)+id]=t_Pk[5+2*len(rnx_obs)+id][5+len(rnx_obs):-2]
        
        #星间单差算子
        lambda_trans[id][id]=f1/clight
        lambda_trans[len(rnx_obs)+id][len(rnx_obs)+id]=f2/clight
        lambda_trans[id][el_max_id]=-f1/clight
        lambda_trans[len(rnx_obs)+id][len(rnx_obs)+el_max_id]=-f2/clight#星间单差算子
    #星间单差
    Xfloat_N12=lambda_trans.dot(Xfloat_N12)
    Pfloat_N12=lambda_trans.dot(Pfloat_N12).dot(lambda_trans.T)
    #构建星间单差模糊度矩阵
    X_float_N12_SD=[]
    id_use=[]
    #首先循环选择标准差小于1.5周的单差模糊度
    for id in range(len(Xfloat_N12)):
        t=np.array(el_max_ids)+len(rnx_obs)
        if(id not in el_max_ids and id not in t and Pfloat_N12[id][id]<P_float_threshold):
            #大于0.15周的模糊度不固定
            N_float_part=abs(round(Xfloat_N12[id][0])-Xfloat_N12[id][0])
            if(N_float_part>amb_float_threshold):
                continue
            X_float_N12_SD.append(Xfloat_N12[id])
            id_use.append(id)
    #如果无候选模糊度或候选模糊度过少, 则部分模糊度固定失败, 返回失败标志
    if(len(X_float_N12_SD)<min_AR_num):
        return False
    P_float_N12_SD=[]
    for id in id_use:
        P_temp=[]
        for j_id in id_use:
            P_temp.append(Pfloat_N12[id][j_id])
        P_float_N12_SD.append(P_temp.copy())
    #如果位置内符合精度太差, 跳过模糊度固定
    # if(t_Pk[0][0]>0.25 or t_Pk[1][1]>0.25 or t_Pk[2][2]>0.25):
    #     return False
    X_float_N12_SD=np.array(X_float_N12_SD).reshape((len(X_float_N12_SD),1))
    try:
        #ratios,ds,N12_FIX=LAMBDA_FIX(Xfloat_N12,Pfloat_N12)
        ratios,ds,N12_FIX=LAMBDA_FIX(X_float_N12_SD,P_float_N12_SD,loopmax=50000)
        #N12_FIX_res=(N12_FIX-X_float_N12_SD.T)
    except:
        ratios=[0.0,0.0]
        ds=[9999,9999]
    if(ratios[0]>ratio_threshold or ratios[1]>ratio_threshold):
        #单次模糊度固定ration校验通过
        #返回模糊度固定信息
        N12_SD_FIX_info={'ratios':[ratios[0],ratios[1]]}
        N12_SD_FIX_info['Ref_sat']=el_max_prns
        N12_SD_FIX_info['N12_SD_sat']=[]
        N12_SD_FIX_info['N12_SD_value']=[]
        for id in id_use:
            #L1频率上的星间单差模糊度
            if (id<len(rnx_obs)):
                N12_SD_FIX_info['N12_SD_sat'].append([rnx_obs[id]['PRN'],'N1'])
            #L2频率上的星间单差模糊度
            else:
                N12_SD_FIX_info['N12_SD_sat'].append([rnx_obs[id-len(rnx_obs)]['PRN'],'N2'])
            N12_SD_FIX_info['N12_SD_value'].append(N12_FIX[id_use.index(id)])
        return N12_SD_FIX_info
    else:
        #单次模糊度校验不通过, 进行部分模糊度固定(PAR)
        #方法一、依次剔除方差较大的模糊度
        if(len(X_float_N12_SD)==min_AR_num):
            return False#最小模糊度固定数量固定失败, 返回False
        sub_id_use,t_ratios=PAR_Search(Xfloat_N12,Pfloat_N12,id_use)
        if(sub_id_use):
            #返回部分模糊度固定信息
            N12_SD_FIX_info={"ratios":[t_ratios[0],t_ratios[1]]}
            N12_SD_FIX_info['Ref_sat']=el_max_prns
            N12_SD_FIX_info['N12_SD_sat']=[]
            N12_SD_FIX_info['N12_SD_value']=[]
            for id in sub_id_use:
                #L1频率上的星间单差模糊度
                if (id<len(rnx_obs)):
                    N12_SD_FIX_info['N12_SD_sat'].append([rnx_obs[id]['PRN'],'N1'])
                #L2频率上的星间单差模糊度
                else:
                    N12_SD_FIX_info['N12_SD_sat'].append([rnx_obs[id-len(rnx_obs)]['PRN'],'N2'])
                N12_SD_FIX_info['N12_SD_value'].append(N12_FIX[id_use.index(id)])
            return N12_SD_FIX_info
        return False

def PPP_AR_FIX_HOLD_M(t_Xk,t_Pk,rnx_obs,Fix_info,X,freqs):
    #利用虚拟观测更新固定解
    #计算单系统卫星数量
    sys_sat_num=round((X.shape[0]-5)/3)
    prns=[t['PRN'] for t in rnx_obs]#获取本历元观测卫星的PRN顺序
    #添加固定解虚拟约束
    fix_value_count=0
    H=np.zeros( (len(Fix_info['N12_SD_sat']),len(t_Xk)) )   #虚拟设计矩阵维数: (固定模糊度行, 本历元状态量列)
    v=np.zeros( (len(Fix_info['N12_SD_sat']),1) )           #虚拟观测值维数: (固定模糊度行,1)
    R=np.zeros( (len(Fix_info['N12_SD_sat']),len(Fix_info['N12_SD_sat'])) )
    for fix in Fix_info['N12_SD_sat']:
        fix_prn=fix[0]
        fix_freq=int(fix[1][1:])                                      #计算模糊度频率所属
        ref_prn=Fix_info['Ref_sat'][['G','C','E'].index(fix_prn[0])]     #查找参考星
        sat_id=prns.index(fix_prn)                                      #在本历元观测序列中查找目标卫星位置
        ref_sat_id=prns.index(ref_prn)                                  #在本历元观测序列中查找参考卫星位置

        freq=freqs[['G','C','E'].index(fix_prn[0])][fix_freq-1]#频率分发
        
        H[fix_value_count][5+fix_freq*len(rnx_obs)+sat_id]=1.0
        H[fix_value_count][5+fix_freq*len(rnx_obs)+ref_sat_id]=-1.0
        
        #计算模糊度单差的残差, 对残差向量添加一行
        sys_shift=[0,3*32,3*(32+65)][['G','C','E'].index(fix_prn[0])]
        sys_snum=[32,65,37][['G','C','E'].index(fix_prn[0])]
        ta=X[5+sys_shift+fix_freq*sys_snum+int(fix_prn[1:])-1][0]
        tb=X[5+sys_shift+fix_freq*sys_snum+int(ref_prn[1:])-1][0]
        float_v=ta-tb
        fix_v=round(Fix_info['N12_SD_value'][fix_value_count])*clight/freq-(float_v)
        v[fix_value_count][0]=fix_v #固定解约束
        R[fix_value_count][fix_value_count]=0.001 #虚拟方差
        fix_value_count+=1      #虚拟观测序号+1
    #1.状态一步预测
    X_up=t_Xk
    #2.状态一步预测误差
    Pk_1_k=t_Pk
    #3.滤波增益计算
    #Kk=(Pk_1_k.dot(H.T)).dot(inv((H.dot(Pk_1_k)).dot(H.T)+R))
    Kk=(Pk_1_k.dot(H.T)).dot(numba_inv((H.dot(Pk_1_k)).dot(H.T)+R))
    #滤波结果
    Xk_dot=X_up+Kk.dot(v)
    #滤波方差更新
    t_Pk=((np.eye(t_Xk.shape[0],dtype=np.float64)-Kk.dot(H))).dot(Pk_1_k)  
    t_Pk=t_Pk.dot((np.eye(t_Xk.shape[0],dtype=np.float64)-Kk.dot(H)).T)+Kk.dot(R).dot(Kk.T)
    t_Xk=Xk_dot

    return Xk_dot,t_Pk  #返回滤波更新后的值

def add_PPP_RTK_corr(H,R,v,corr_v,corr_sstd,ref_ids,rnx_obs):
    #函数: PPP-RTK约束函数
    #输入: 非约束观测模型HRv, 约束向量corr_v, 约束向量的方差corr_sstd
    #输出: 有约束观测模型H_corr, R_corr, v_corr
    H_corr=H.copy()
    R_corr=R.copy()
    v_corr=v.copy()
    prns=[t['PRN'] for t in rnx_obs]
    for i in range(len(corr_v)):        #顺序为卫星列表顺序

        if(corr_v[i]==0.0):
            continue                    #该状态量无改正数
        H_corr=np.append(H_corr,np.zeros((1,H_corr.shape[1])),axis=0)#设计矩阵添加一行
        H_corr[-1][i]=1.0                                            #约束对应状态的系数置1(基准站位置约束)
        
        if(i>=5):                                                    #电离层约束
            prn=prns[i-5]                     #卫星PRN码
            sys_id=['G','C','E'].index(prn[0])
            H_corr[-1][5+ref_ids[sys_id]]=-1.0                       #参考星
        v_corr=np.append(v_corr,np.array([[corr_v[i]]]),axis=0)      #残差向量添加一行
        
        #观测噪声阵向右下添加一块
        R_corr=np.append(R_corr,np.zeros((1,R_corr.shape[1])),axis=0)   #先添加一行
        R_corr=np.append(R_corr,np.zeros((R_corr.shape[0],1)),axis=1)   #再添加一列
        R_corr[-1][-1]=corr_sstd[i]                                #虚拟观测值赋权重
    
    #返回添加PPP-RTK改正数约束后的观测模型
    return H_corr,R_corr,v_corr

#大气约束
def rtkinfo2SIONTRO(rtk_info,freqs,Qi_init=2.0):
    SION,TRO={},{}
    for prn in rtk_info.keys():
        #1. 剔除低高度角改正数
        try:
            el=rtk_info[prn]['azel'][1]
            if(el<10.0):
                continue
        except:
            pass
        #2. 分发频率
        f1=freqs[['G','C','E'].index(prn[0])][0]
        #3. 构建电离层延迟改正数群组
        SION[prn]=[rtk_info[prn]['STEC']*40.28/(f1/1e8)/(f1/1e8),Qi_init**2]
        try:
            SION[prn][1]=(rtk_info[prn]['std_STEC']+Qi_init**2)*(40.28/(f1/1e8)/(f1/1e8))**2
        except:
            pass
    #3. 分发基准站位置 (用于定权)
    base_pos=[0,0,0]
    if(rtk_info!={}):
        base_x=rtk_info[list(rtk_info.keys())[0]]['sta_x']
        base_y=rtk_info[list(rtk_info.keys())[0]]['sta_y']
        base_z=rtk_info[list(rtk_info.keys())[0]]['sta_z']
        base_pos=[base_x,base_y,base_z]
        try:
            TRO['ZTD']=rtk_info[list(rtk_info.keys())[0]]['ztd_w']+rtk_info[list(rtk_info.keys())[0]]['ztd_h']    #含模型值的ZTD必须传递原始值
            TRO['ZTD-Q']=rtk_info[list(rtk_info.keys())[0]]['std_ztd_w']    #利用估计量的代替模型值
        except:
            TRO['ZTD']=get_Trop_delay_dry(base_pos)+0.15
            TRO['ZTD-W-Q']=1.0
    
    return SION,TRO,base_pos

def get_IPP_rad(rr,rs):
    #B2b信号推荐参数设置
    H_ion=350000            #电离层薄层高度
    Re=6378000              #地球平均半径
    lng_M=-72.58/180*pi     #地磁北极的地理经度
    lat_M=80.27/180*pi      #地磁北极的地理纬度
    
    #站星射线与高度角计算
    #Calculation of star ray and height angle at station 
    az,el=getazel(rs,rr)
    #测站经纬度
    rrb,rrl,rrh=xyz2blh(rr[0],rr[1],rr[2])
    rrb=rrb/180*pi
    rrl=rrl/180*pi
    #电离层穿刺点地心张角计算
    PHI=pi/2-el-asin(Re/(Re+H_ion)*cos(el))
    #电离层穿刺点在地球表面投影的地理经纬度
    lat_g=asin( sin(rrb)*cos(PHI)+cos(rrb)*sin(PHI)*cos(az) )
    lng_g=rrl+atan2(sin(PHI)*sin(az)*cos(rrb),cos(PHI)-sin(rrb)*sin(lat_g))
    #电离层延迟在地球表面投影的地磁经纬度
    lat_m=asin(sin(lat_M)*sin(lat_g) + cos(lat_M)*cos(lat_g)*cos(lng_g-lng_M) )
    lng_m=atan2(cos(lat_g)*sin(lng_g-lng_M)*cos(lat_M) , sin(lat_M)*sin(lat_m)-sin(lat_g) )
    return lat_g,lng_g

def caculate_PPP_RTK_corr_M(rnx_obs, X, pos=[], TRO={}, SION={}, peph_sat_pos={}, base_pos=None, rove_pos=[], Qi_scale=10e3, Qi_ele_threshold=10,Qt_scale=10e6):
    # 函数: 计算PPP_RTK约束向量与方差向量
    # 输入: 有效观测列表rnx_obs, 位置约束(初始化为空), 大气约束(初始化为空字典), 电离层约束(初始化为空字典),  
    # 输出: PPP_RTK约束向量corr_v, 方差向量corr_sstd
    corr_v=[]
    corr_sstd=[]
    sys_sat_num=round((len(X)-5)/3)
    #位置约束
    if(len(pos)!=3):
        corr_v.append(0.0)
        corr_v.append(0.0)
        corr_v.append(0.0)
        corr_sstd.append(0.001)
        corr_sstd.append(0.001)
        corr_sstd.append(0.001)
    else:
        corr_v.append(pos[0]-X[0][0])
        corr_v.append(pos[1]-X[1][0])
        corr_v.append(pos[2]-X[2][0])
        corr_sstd.append(0.001)
        corr_sstd.append(0.001)
        corr_sstd.append(0.001)
    
    #钟差约束预留位置
    corr_v.append(0.0)
    corr_sstd.append(0.0)

    #对流层约束(当前仅支持ZWD加入约束)
    if(TRO!={} and Qt_scale!=-1):
        baseline=sqrt( (base_pos[0]-X[0][0])**2+(base_pos[1]-X[1][0])**2+(base_pos[2]-X[2][0])**2 )
        corr_v.append((TRO['ZTD']-get_Trop_delay_dry(rove_pos))-X[4][0])
        #corr_v.append(0.0)
        corr_sstd.append(TRO['ZTD-Q']+(baseline/Qt_scale)**2)#以10cm为方差
        #corr_sstd.append(1.0*1.0)
    else:
        corr_v.append(0.0)
        corr_sstd.append(0.0)
    
    #电离层约束(星间单差模式)
    #首先选择参考星
    ref_prns=['X00','X00','X00']
    el_maxs=[0,0,0]
    ref_ids=[999,999,999]
    count_id=0
    for sat in rnx_obs:
        prn=sat["PRN"]
        _,el=getazel(peph_sat_pos[prn],[X[0][0],X[1][0],X[2][0]])   #获取高度角
        if(prn not in SION.keys()):                            #无SSR的卫星跳过
            count_id+=1
            continue
        if(el>el_maxs[0] and prn[0]=='G'):
            el_maxs[0]=el
            ref_prns[0],ref_ids[0]=prn,count_id
        if(el>el_maxs[1] and prn[0]=='C'):
            el_maxs[1]=el
            ref_prns[1],ref_ids[1]=prn,count_id
        if(el>el_maxs[2] and prn[0]=='E'):
            el_maxs[2]=el
            ref_prns[2],ref_ids[2]=prn,count_id
        count_id+=1
    
    for sat in rnx_obs:
        prn=sat['PRN']
        sys_id=['G','C','E'].index(prn[0])
        sys_shift=[0,3*32,3*(32+65)]
        sat_count=int(prn[1:])-1
        ref_count=int(ref_prns[sys_id][1:])-1
        _,el=getazel(peph_sat_pos[prn],[X[0][0],X[1][0],X[2][0]])
        try:
            sion,sion_r=SION[prn][0],SION[ref_prns[sys_id]][0]              #基准站站解ION(单位为m)
            sion_q,sion_r_q=SION[prn][1],SION[ref_prns[sys_id]][1]          #基准站电离层质量因子(单位为m)
            ion,ion_r=X[5+sys_shift[sys_id]+sat_count][0],X[5+sys_shift[sys_id]+ref_count][0]
            #电离层状态方差信息
            Qi0=sion_q#+sion_r_q                                #基准站质量因子
            lat_b,lng_b=get_IPP_rad(base_pos,peph_sat_pos[prn])
            lat_r,lng_r=get_IPP_rad(rove_pos,peph_sat_pos[prn])
            Qid=(float(Qi_scale)**2)*((lat_b-lat_r)**2+(lng_b-lng_r)**2)        #基线质量衰减因子
            Qi1=Qid/sin(el)/sin(el)                             #高度角加权因子
            #所有约束信息校验通过, 添加约束和约束方差
            if(Qi_scale!=-1 and el/pi*180.0>=Qi_ele_threshold):
                corr_v.append((sion-sion_r)-(ion-ion_r))                        #虚拟观测约束添加
                corr_sstd.append(Qi0+Qi1)                           #电离层SSR约束方差
            else:
                corr_sstd.append(0.0)
                corr_v.append(0.0)
        except:
            corr_v.append(0.0)
            corr_sstd.append(0.0)
    return corr_v,corr_sstd,ref_ids

def KF_UCPPP_M(X,X_time,Pk,Qk,ion_param,peph_sat_pos,rnx_obs,ex_threshold_v,ex_threshold_sigma,rt_unix,phase_bias,freqs,AMB_FIX=1,mode="None",RTK_Info={},AMB_FIX_Info={},Out_log_fix=[]):
    #扩展卡尔曼滤波
    for KF_times in range(8):
        #相位改正值拷贝
        t_phase_bias=phase_bias.copy()
        
        #观测模型(两次构建, 验前粗差剔除)
        #print(rnx_obs)
        X,X_time,H,R,_,v,rnx_obs=createKF_HRZ_M(rnx_obs,rt_unix,X,X_time,Pk,Qk,ion_param,t_phase_bias,peph_sat_pos,freqs=freqs,exthreshold_v_sigma=ex_threshold_sigma,post=False,ex_threshold_v=ex_threshold_v)
        if(not len(rnx_obs)):
            #无先验通过数据
            #全部状态重置
            X[0]=100.0
            X[1]=100.0
            X[2]=100.0
            # for i in range(len(X)):
            #     X_time[i]=0.0
            #     break
        X,X_time,H,R,_,v,rnx_obs=createKF_HRZ_M(rnx_obs,rt_unix,X,X_time,Pk,Qk,ion_param,t_phase_bias,peph_sat_pos,freqs=freqs,exthreshold_v_sigma=ex_threshold_sigma,post=False,ex_threshold_v=ex_threshold_v)

        #基准站约束
        if(mode=='Base'):
            STA_P=RTK_Info['STA_P']
            STA_Q=RTK_Info['STA_Q']
            corr_v=[STA_P[0]-X[0][0], STA_P[1]-X[1][0], STA_P[2]-X[2][0]]
            corr_sstd=[STA_Q[0],STA_Q[1],STA_Q[2]]
            ref_id=0
            H,R,v=add_PPP_RTK_corr(H,R,v,corr_v,corr_sstd,[],rnx_obs)
        #流动站约束
        if(mode=='Rove'):
            _,GPSsec=time2gpst(rt_unix)
            t_interval=RTK_Info['t_interval']
            GPSsec=round(GPSsec/t_interval)*t_interval      #SSR时间戳
            try:
                rtk_info_id=RTK_Info['rtk_corr_info_time'].index(GPSsec)
                rtk_info=RTK_Info['rtk_info'][rtk_info_id]#查找目标时间对应的SSR组
            except:
                rtk_info={}
            SION,TRO,base_pos=rtkinfo2SIONTRO(rtk_info,freqs,Qi_init=RTK_Info['Qi_init'])
            rove_pos=[X[0][0],X[1][0],X[2][0]]
            corr_v,corr_sstd,ref_ids=caculate_PPP_RTK_corr_M(rnx_obs,X,TRO=TRO,SION=SION,peph_sat_pos=peph_sat_pos,base_pos=base_pos,rove_pos=rove_pos,Qi_scale=RTK_Info['Qi_scale'],Qi_ele_threshold=RTK_Info['Qi_ele_threshold'],Qt_scale=RTK_Info['Qt_scale'])
            H,R,v=add_PPP_RTK_corr(H,R,v,corr_v,corr_sstd,ref_ids=ref_ids,rnx_obs=rnx_obs)
            

        #系统模型(根据先验抗差得到的数据)
        t_Xk,t_Pk,t_Qk=createKF_XkPkQk_M(rnx_obs,X,Pk,Qk)

        #抗差滤波准备
        #R=IGGIII(v,R)
        #扩展卡尔曼滤波
        #1.状态一步预测
        PHIk_1_k=np.eye(t_Xk.shape[0],dtype=np.float64)
        X_up=PHIk_1_k.dot(t_Xk)
        #2.状态一步预测误差
        Pk_1_k=(PHIk_1_k.dot(t_Pk)).dot(PHIk_1_k.T)+t_Qk
        #3.滤波增益计算
        #Kk=(Pk_1_k.dot(H.T)).dot(inv((H.dot(Pk_1_k)).dot(H.T)+R))
        Kk=(Pk_1_k.dot(H.T)).dot(numba_inv((H.dot(Pk_1_k)).dot(H.T)+R))
        #滤波结果
        Xk_dot=X_up+Kk.dot(v)
        #滤波方差更新
        t_Pk=((np.eye(t_Xk.shape[0],dtype=np.float64)-Kk.dot(H))).dot(Pk_1_k)  
        t_Pk=t_Pk.dot((np.eye(t_Xk.shape[0],dtype=np.float64)-Kk.dot(H)).T)+Kk.dot(R).dot(Kk.T)
        #滤波暂态更新
        t_Xk_f,t_Pk_f,t_Qk_f,t_X_time=upstateKF_XkPkQk_M(rnx_obs,rt_unix,Xk_dot,t_Pk,t_Qk,X,Pk,Qk,X_time)
        
        #验后方差
        info='KF fixed'
        info,rnx_obs,tt_phase_bias,v=createKF_HRZ_M(rnx_obs,rt_unix,t_Xk_f,t_X_time,t_Pk_f,t_Qk_f,ion_param,t_phase_bias,peph_sat_pos,freqs=freqs,exthreshold_v_sigma=ex_threshold_sigma,post=True)
        #_,info=get_post_v(rnx_obs,rt_unix,Xk_dot,ion_param,phase_bias)
        
        if(info=='KF fixed'):    
            #验后校验通过
            X,Pk,Qk,X_time=upstateKF_XkPkQk_M(rnx_obs,rt_unix,Xk_dot,t_Pk,t_Qk,X,Pk,Qk,X_time)
            phase_bias=tt_phase_bias.copy()
            #不固定模糊度
            if(not AMB_FIX):
                break
            #固定模糊度
            try:
                Fix_info=PPP_AR_M(rnx_obs,peph_sat_pos,Xk_dot,t_Pk,freqs,ratio_threshold=AMB_FIX_Info['ratio_threshold'],
                                  amb_float_threshold=AMB_FIX_Info['amb_float_threshold'],
                                  P_float_threshold=AMB_FIX_Info['P_float_threshold'],
                                  min_AR_num=AMB_FIX_Info['min_AR_num'])
                if(Fix_info!=False):
                    #滤波法更新模糊度
                    Xk_dot,t_Pk=PPP_AR_FIX_HOLD_M(Xk_dot,t_Pk,rnx_obs,Fix_info,X,freqs)
                    #验后校验通过
                    t_X,t_P,Qk,X_time=upstateKF_XkPkQk_M(rnx_obs,rt_unix,Xk_dot,t_Pk,t_Qk,X,Pk,Qk,X_time)
                    #print(xyz2neu(X_SD_fix,[ -2267750.307108253, 5009154.576877178, 3221294.4087844505]))
                    Out_log_fix.append(log2out_M(rt_unix,v,rnx_obs,t_X,t_X_time,t_P,peph_sat_pos,freqs,Fix_info['ratios'][0][0][0]))
                    pass
                else:
                #固定失败, 保存浮点解
                     Out_log_fix.append(log2out_M(rt_unix,v,rnx_obs,X,X_time,Pk,peph_sat_pos,freqs))
            except:
                Out_log_fix.append(log2out_M(rt_unix,v,rnx_obs,X,X_time,Pk,peph_sat_pos,freqs))
                pass
            break
    if(info!='KF fixed'):
        print("Warning: KF itertion overflow")

    return X,Pk,Qk,X_time,v,phase_bias,rnx_obs

def find_OSB_index(filename):
    targets=["BIAS","SVN","PRN","STATION","OBS1","OBS2","BIAS_START","BIAS_END","UNIT","ESTIMATED_VALUE","STD_DEV"]
    target_ids=[-1 for t in range(len(targets))]
    target_ide=[-1 for t in range(len(targets))]
    with open(filename,'r',encoding='gbk') as f:
        lines=f.readlines()
        bias_in=0
        for line in lines:
            if ("+BIAS/SOLUTION" in line):
                bias_in=1
            if(bias_in and line[0]=='*'):
                ls=line.split()
                for t in range(len(targets)):
                    target=targets[t]
                    #循环字符查找
                    for i in range(len(ls)):
                        #找到目标索引
                        if(target in ls[i]):
                            for j in range(len(line)-len(ls[i])):
                                #字符切片匹配成功
                                if(line[j:j+len(ls[i])]==ls[i]):
                                    target_ids[t]=j
                                    target_ide[t]=j+len(ls[i])
                                    break
                            break
            if("-BIAS/SOLUTION" in line):
                bias_in=0
                break
    return targets,target_ids,target_ide


def OSBcorr2obsmat(filename,obs_mat,obs_type,f1,f2):
    
    #首先确定索引
    targets,target_ids,target_ide=find_OSB_index(filename)
    id_osb=targets.index('BIAS')
    id_obs=targets.index('OBS1')
    id_prn=targets.index('PRN')
    id_value=targets.index('ESTIMATED_VALUE')
    
    with open(filename,'r',encoding='gbk') as f:
        data={}    
        lines=f.readlines()
        bias_in=0
        for line in lines:
            #数据段判断
            if ("+BIAS/SOLUTION" in line):
                bias_in=1
            ls=line.split()
            if(not bias_in):
                continue
            #记录OSB字典
            if('OSB' in line[target_ids[id_osb]:target_ide[id_osb]]):#找到OSB行
                prn_c=line[target_ids[id_prn]:target_ide[id_prn]]
                if(prn_c not in data.keys()):
                    data[prn_c]={}
                obs_sign=line[target_ids[id_obs]:target_ide[id_obs]].replace(" ","")
                osb_value=line[target_ids[id_value]:target_ide[id_value]]
                data[prn_c][obs_sign]=float(osb_value)*clight*1e-9
            
            if ("-BIAS/SOLUTION" in line):
                bias_in=0
                break

    for i in range(len(obs_mat)):
        for sat in range(len(obs_mat[i][1])):
            prn=obs_mat[i][1][sat]['PRN']
            if(obs_mat[i][1][sat]['OBS'][0]!=0.0):
                try:
                    obs_mat[i][1][sat]['OBS'][0]-=data[prn][obs_type[0]]
                except:
                    pass
            if(obs_mat[i][1][sat]['OBS'][1]!=0.0):
                try:
                        obs_mat[i][1][sat]['OBS'][1]-=data[prn][obs_type[1]]*f1/clight
                except:
                    pass
            if(obs_mat[i][1][sat]['OBS'][5]!=0.0):
                try:
                    obs_mat[i][1][sat]['OBS'][5]-=data[prn][obs_type[4]]
                except:
                    pass
            if(obs_mat[i][1][sat]['OBS'][6]!=0.0):
                try:
                    obs_mat[i][1][sat]['OBS'][6]-=data[prn][obs_type[5]]*f2/clight
                except:
                    pass
    return obs_mat


def PTK_YAML_GCE(PPP_cfg):
    #首先可视化配置
    print("Easy4PPP Configurations:")
    for key in PPP_cfg.keys():
        print(key,PPP_cfg[key])
    
    #多系统双频非差非组合PPP, 解算文件最小配置(观测值/观测值类型/精密星历文件/精密钟差文件/天线文件/结果输出路径)
    obs_file=PPP_cfg['obs_file']

    sys_indexs=PPP_cfg['sys_indexs']
    sys_select_num=len(sys_indexs)
    sys_select_ids=sys_indexs.copy()                               #用户选择的原始系统标识
    sys_indexs=['G','C','E']                                       #重置系统标识

    obs_type=PPP_cfg['obs_type']                                   #混合观测值类型, 以RINEX协议为准
    freqs=PPP_cfg['freqs']                                         #各频点观测值中央频率
    
    #系统与信号标识符校验
    try:
        if(len(sys_indexs)==len(obs_type)):
            print("Systems set as: ",sys_indexs)
        else:
            print("Systems set error for: ")
            print("sys_indexs: ", sys_indexs)
            print("obs_type: ",   obs_type)
    except:
        ValueError("Systems not set correctly")

    #重整观测值列表和频率数组
    if(sys_select_num):
        try:
            freqs_G=freqs[sys_select_ids.index("G")]
            obs_type_G=obs_type[sys_select_ids.index("G")]
        except:
            freqs_G=[1575.42E+6, 1227.60E+6]#默认L1/L2
            obs_type_G=['C1C','L1C','D1C','S1C','C2W','L2W','D2W','S2W']
        try:
            freqs_C=freqs[sys_select_ids.index("C")]
            obs_type_C=obs_type[sys_select_ids.index("C")]
        except:
            freqs_C=[1561.098E+6,1268.52E+6]#默认B1/B3
            obs_type_C=['C2I','L2I','D2I','S2I','C6I','L6I','D6I','S6I']
        try:
            freqs_E=freqs[sys_select_ids.index("E")]
            obs_type_E=obs_type[sys_select_ids.index("E")]
        except:
            freqs_E=[1575.42E+6, 1176.45E+6]#默认E1/E5
            obs_type_E=['C1X','L1X','D1X','S1X','C5X','L5X','D5X','S5X']
        freqs=[freqs_G,freqs_C,freqs_E]
        obs_type=[obs_type_G,obs_type_C,obs_type_E]
    
    SP3_file=PPP_cfg['SP3_file']                                #精密星历文件路径
    CLK_file=PPP_cfg['CLK_file']                                #精密钟差文件路径
    ATX_file=PPP_cfg['ATX_file']                                #天线文件, 支持格式转换后的npy文件和IGS天线文件

    out_path=PPP_cfg['out_path']                                        #导航结果输出文件路径
    print("Navigation Results saved in path: ",out_path)
    ion_param=PPP_cfg['ion_param']                                                 #自定义Klobuchar电离层参数
    
    if(len(ion_param)):
        print("ion_params set as: ",ion_param)
    else:
        print("No ion_param, read from file. ")
    
    #可选配置(广播星历文件/DCB改正与产品选项/)
    BRDC_file=PPP_cfg['BRDC_file']                             #广播星历文件, 支持BRDC/RINEX3混合星历
    print("Broadcast eph file: ",BRDC_file)
    dcb_correction=PPP_cfg['dcb_correction']                   #DCB修正选项
    dcb_products=PPP_cfg['dcb_products']                       #DCB产品来源, 支持CODE月解文件/CAS日解文件
    dcb_file_0=PPP_cfg['dcb_file_0']                           #频间偏差文件, 支持预先转换后的.npy格式
    dcb_file_1=PPP_cfg['dcb_file_1']
    dcb_file_2=PPP_cfg['dcb_file_2']

    obs_start=PPP_cfg['obs_start']                             #解算初始时刻索引
    obs_epoch=PPP_cfg['obs_epoch']                             #解算总历元数量
    out_age=PPP_cfg['out_age']                                 #最大容忍失锁阈值时间(单位: s, 用于电离层、模糊度状态重置)
    dy_mode=PPP_cfg['dy_mode']                                 #PPP动态模式配置, 支持static, dynamic
    el_threthod=PPP_cfg['el_threthod']                         #设置截止高度角
    ex_threshold_v=PPP_cfg['ex_threshold_v']                   #设置先验残差阈值
    ex_threshold_v_sigma=PPP_cfg['ex_threshold_v_sigma']       #设置后验残差阈值
    Mw_threshold=PPP_cfg['Mw_threshold']                       #设置Mw组合周跳检验阈值
    GF_threshold=PPP_cfg['GF_threshold']                       #设置GF组合周跳检验阈值
    sat_out=PPP_cfg['sat_out']

    #模糊度固定
    try:
        AMB_FIX=PPP_cfg['AMB_FIX']
        OSB_path=PPP_cfg['OSB_path']
        ratio_threshold=PPP_cfg['ratio_threshold']
        amb_float_threshold=PPP_cfg['amb_float_threshold']
        P_float_threshold=PPP_cfg['P_float_threshold']
        min_AR_num=PPP_cfg['min_AR_num']
    except:
        AMB_FIX=0
        OSB_path=""
        ratio_threshold=2.0
        amb_float_threshold=1.0
        P_float_threshold=2.25
        min_AR_num=3
    AMB_FIX_Info={'ratio_threshold':ratio_threshold,'amb_float_threshold':amb_float_threshold,
                      'P_float_threshold':P_float_threshold,'min_AR_num':min_AR_num}    
    
    #OSB校正
    try:
        OSB_YES=PPP_cfg['OSB_YES']
        ATX_YES=PPP_cfg['PCO_YES']
    except:
        OSB_YES=0
        ATX_YES=1   #默认改正PCO

    #基准站信息
    try:
        sta_mode=PPP_cfg['sta_mode']
        STA_P=PPP_cfg['STA_P']
        STA_Q=PPP_cfg['STA_Q']
    except:
        sta_mode='None'
        STA_P=None
        STA_Q=None

    #整理PPP-RTK信息
    RTK_Info={}
    try:
        RTK_Info['reinitial_sec']=PPP_cfg['reinitial_sec']  #记录重收敛信息
    except:
        RTK_Info['reinitial_sec']=0                         #没有重收敛信息
    
    if(sta_mode=='Base'):
        RTK_Info['STA_P']=STA_P
        RTK_Info['STA_Q']=STA_Q
    if(sta_mode=='Rove'):
        RTK_Info['t_interval']=PPP_cfg['t_interval']
        RTK_Info['rtk_info']=np.load(PPP_cfg["rtk_info_mat"],allow_pickle=True)
        
        rtk_corr_info_time=[]
        for log in RTK_Info['rtk_info']:
            try:
                rtk_corr_info_time.append(log[list(log.keys())[0]]['GPSsec'])
            except:
                rtk_corr_info_time.append(9999999)
        RTK_Info['rtk_corr_info_time']=rtk_corr_info_time
        RTK_Info['Qi_init']=PPP_cfg['Qi_init']
        RTK_Info['Qi_scale']=PPP_cfg['Qi_scale']
        RTK_Info['Qi_ele_threshold']=PPP_cfg['Qi_ele_threshold']
        RTK_Info['Qt_scale']=PPP_cfg['Qt_scale']
    
    #处理非GCE系统情况:
    if("G" not in sys_select_ids):
        for i in range(1,33):
            sat_out.append("G{:02d}".format(i))
    if("C" not in sys_select_ids):
        for i in range(1,66):
            sat_out.append("C{:02d}".format(i))
    if("E" not in sys_select_ids):
        for i in range(1,37):
            sat_out.append("E{:02d}".format(i))
    
    #
    #多系统观测值分别读取
    STA_name=obs_file.split('/')[-1][:4].upper()
    print("The name of station (RINEX observation format): ",STA_name)
    dcb_file_0_=""
    if "G" in sys_indexs:
        #CAS DCB产品数据读取
        if(dcb_correction==1 and dcb_products=='CAS'):
            dcb_file_0_,_=CAS_DCB_SR(dcb_file_0,obs_type[0][0],obs_type[0][4],STA_name)
            dcb_file_1=''       #CAS产品同时包含码间和频间偏差
            dcb_file_2=''
        obs_mat_GPS=RINEX3_to_obsmat(obs_file,obs_type[0],sys="G",dcb_correction=dcb_correction,dcb_file_0=dcb_file_0_,dcb_file_1=dcb_file_1,dcb_file_2=dcb_file_2)
        obs_mat_GPS=reconstruct_obs_mat(obs_mat_GPS)
        if(AMB_FIX or OSB_YES):
            obs_mat_GPS=OSBcorr2obsmat(OSB_path,obs_mat_GPS,obs_type[0],freqs_G[0],freqs_G[1])
        #删除CAS-DCB产品辅助文件
        if(dcb_file_0_!=""):
            os.unlink(dcb_file_0_)
    if "C" in sys_indexs:
        if(dcb_correction==1 and dcb_products=='CAS'):
            dcb_file_0_,_=CAS_DCB_SR(dcb_file_0,obs_type[1][0],obs_type[1][4],STA_name)
            dcb_file_1=''       #CAS产品同时包含码间和频间偏差
            dcb_file_2=''   
        obs_mat_BDS=RINEX3_to_obsmat(obs_file,obs_type[1],sys="C",dcb_correction=dcb_correction,dcb_file_0=dcb_file_0_,dcb_file_1=dcb_file_1,dcb_file_2=dcb_file_2)
        obs_mat_BDS=reconstruct_obs_mat(obs_mat_BDS)
        if(AMB_FIX or OSB_YES):
            obs_mat_BDS=OSBcorr2obsmat(OSB_path,obs_mat_BDS,obs_type[1],freqs_C[0],freqs_C[1])
        if(dcb_file_0_!=""):
            os.unlink(dcb_file_0_)
    if "E" in sys_indexs:
        if(dcb_correction==1 and dcb_products=='CAS'):
            dcb_file_0_,_=CAS_DCB_SR(dcb_file_0,obs_type[2][0],obs_type[2][4],STA_name)
            dcb_file_1=''       #CAS产品同时包含码间和频间偏差
            dcb_file_2=''
        obs_mat_GAL=RINEX3_to_obsmat(obs_file,obs_type[2],sys="E",dcb_correction=dcb_correction,dcb_file_0=dcb_file_0_,dcb_file_1=dcb_file_1,dcb_file_2=dcb_file_2)
        obs_mat_GAL=reconstruct_obs_mat(obs_mat_GAL)
        if(AMB_FIX or OSB_YES):
            obs_mat_GAL=OSBcorr2obsmat(OSB_path,obs_mat_GAL,obs_type[2],freqs_E[0],freqs_E[1])
        if(dcb_file_0_!=""):
            os.unlink(dcb_file_0_)
    #读取观测文件
    if (not check_obs_mats([obs_mat_GPS,obs_mat_BDS,obs_mat_GAL])):
        ValueError()

    if(not obs_epoch):
        obs_epoch=len(obs_mat_GPS)
        print("End index not set, solve all the observations. Total: {}".format(obs_epoch))
    
    #读取精密轨道和钟差文件
    IGS=getsp3(SP3_file)
    clk=getclk(CLK_file)
    
    #读取天线文件
    try:
        #npy格式
        sat_pcos=np.load(ATX_file,allow_pickle=True)
        sat_pcos=eval(str(sat_pcos))
    except:
        #ATX格式
        sat_pcos=RINEX3_to_ATX(ATX_file)
    if(ATX_YES==0):
        sat_pcos={}
    
    #读取广播星历电离层参数
    if(not len(ion_param)):
        ion_param=RINEX2ion_params(BRDC_file)

    
    #根据配置设置卫星数量
    sat_num=0
    sat_num_G=0
    sat_num_C=0
    sat_num_E=0
    if('G' in sys_indexs):
        sat_num_G=32
    if('C' in sys_indexs):
        sat_num_C=65
    if('E' in sys_indexs):
        sat_num_E=37
    sat_num=sat_num_G+sat_num_C+sat_num_E
    print("Total satellite number of selected systems: ", sat_num," GPS: ",sat_num_G," BDS: ",sat_num_C," GAL",sat_num_E)
    
    #排除卫星列表
    print("Satellites outside for user config",sat_out)
    
    # 分别初始化各系统PPP子滤波状态与协方差
    X_G,Pk_G,Qk_G,GF_sign_G,Mw_sign_G,slip_sign_G,dN_sign_G,X_time_G,phase_bias_G=init_UCPPP(obs_mat_GPS,obs_start,IGS,clk,sat_out,ion_param,sat_pcos,sys_sat_num=sat_num_G,f1=freqs[0][0],f2=freqs[0][1])
    X_C,Pk_C,Qk_C,GF_sign_C,Mw_sign_C,slip_sign_C,dN_sign_C,X_time_C,phase_bias_C=init_UCPPP(obs_mat_BDS,obs_start,IGS,clk,sat_out,ion_param,sat_pcos,sys_sat_num=sat_num_C,f1=freqs[1][0],f2=freqs[1][1])
    X_E,Pk_E,Qk_E,GF_sign_E,Mw_sign_E,slip_sign_E,dN_sign_E,X_time_E,phase_bias_E=init_UCPPP(obs_mat_GAL,obs_start,IGS,clk,sat_out,ion_param,sat_pcos,sys_sat_num=sat_num_E,f1=freqs[2][0],f2=freqs[2][1])
    
    #多系统非差PPP滤波状态初始化
    X,Pk,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,X_time,phase_bias=init_UCPPP_M(X_G,X_C,X_E,
                 Pk_G,Pk_C,Pk_E,
                 Qk_G,Qk_C,Qk_E,
                 GF_sign_G,GF_sign_C,GF_sign_E,
                 Mw_sign_G,Mw_sign_C,Mw_sign_E,
                 slip_sign_G,slip_sign_C,slip_sign_E,
                 dN_sign_G,dN_sign_C,dN_sign_E,
                 X_time_G,X_time_C,X_time_E,
                 phase_bias_G,phase_bias_C,phase_bias_E)

    Out_log,Out_log_fix=UCPPP_M([obs_mat_GPS,obs_mat_BDS,obs_mat_GAL],obs_start,obs_epoch,IGS,clk,
          sat_out,ion_param,sat_pcos,el_threthod,ex_threshold_v,ex_threshold_v_sigma,Mw_threshold,GF_threshold,dy_mode,
          X,Pk,Qk,phase_bias,X_time,GF_sign,Mw_sign,slip_sign,dN_sign,sat_num,out_age,freqs,AMB_FIX,sta_mode,RTK_Info=RTK_Info,AMB_FIX_Info=AMB_FIX_Info)

    #结果以numpy数组格式保存在指定输出目录下, 若输出目录为空, 则存于nav_result
    try:
        np.save(out_path+'/{}.out'.format(os.path.basename(obs_file)),Out_log,allow_pickle=True)
        print("Navigation results saved at ",out_path+'/{}.out'.format(os.path.basename(obs_file)))
        if(AMB_FIX):
            np.save(out_path+'/{}.fixout'.format(os.path.basename(obs_file)),Out_log_fix,allow_pickle=True)
            print("Navigation results (AR) saved at ",out_path+'/{}.fixout'.format(os.path.basename(obs_file)))
    except:
        np.save('nav_result/{}.out'.format(os.path.basename(obs_file)),Out_log,allow_pickle=True)
        print("Navigation results saved at ",'nav_result/{}.out'.format(os.path.basename(obs_file)))
        if(AMB_FIX):
            np.save('nav_result/{}.fixout'.format(os.path.basename(obs_file)),Out_log_fix,allow_pickle=True)
            print("Navigation results (AR) saved at ",out_path+'/{}.fixout'.format(os.path.basename(obs_file)))
    
    return True



if __name__=='__main__':
    #读取yaml
    PPP_YAML="Easy4PPP.yaml"
    with open(PPP_YAML,"r",encoding='utf-8') as f:
        cfg=yaml.safe_load(f)
    print("Easy4PTK: PPP-RTK Toolbox")
    PTK_YAML_GCE(cfg)
