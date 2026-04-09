#文件名:sppp.py
#Source File Name: sppp.py
#PPP核心运算库
#A pure Python Core Source File for Single Point Position& Precise Point Position Computation
#作者: 蒋卓君, 杨泽恩, 黄文静, 钱闯, 武汉理工大学
#Copyright 2025-, by Zhuojun Jiang, Zeen Yang, Wenjing Huang, Chuang Qian, Wuhan University of Technology, China

import numpy as np
import os
from tqdm import tqdm
import csv
from satpos import *
from math import sqrt
from numpy.linalg import inv
from math import sin,cos,tan,asin,acos,atan2,fmod
from RINEX import *
import numba

@numba.jit(nopython=True)
def numba_inv(A):
    return inv(A)

def getLSQ_solution(H,Z,W,weighting_mode='E'):
    
    #输入: 最小二乘设计矩阵, 观测矩阵, 加权权重矩阵: 默认单位阵模式, 待估参数向量维数: 默认为8 
    #输出: 最小二乘结果
    
    #检查数组维数匹配
    m=len(H)
    n=len(H[0])
    
    #非加权模式设置权重矩阵为单位阵
    if(weighting_mode=='E'):
        W=np.eye(m,dtype=int)
    
    #np数组化
    H=np.array(H)
    Z=np.array(Z)
    W=np.array(W)
    
    #观测向量维度变换(非行非列转列向量)
    Z=Z.reshape(m,1)
    
    #加权最小二乘
    HWH=(H.T.dot(W)).dot(H)
    #print(HWH)
    iHWH=inv(HWH)
    #iHWH=numba_inv(HWH)
    #print(iHWH)

    iHWHH=iHWH.dot(H.T)
    iHWHHW=iHWHH.dot(W)
    X=iHWHHW.dot(Z)

    #返回求解结果(展开到一维)
    return(X.reshape(n,))

#从精密星历中计算卫星位置并进行单点定位
def SPP_from_IGS(obs_mat,obs_index,IGS,CLK,sat_out,ion_param,sat_pcos,sol_mode='SF',f1=1575.42*1e6,f2=1227.60*1e6,el_threthod=7.0,obslist=[],pre_rr=[]):
    rr=[100,100,100]
    #观测值列表构建(异常值剔除选星)
    if(not len(obslist)):
        obslist=[]
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
            if(obshealth):
                if obs_mat[obs_index][1][i]['PRN'] not in sat_out:
                    obslist.append(obs_mat[obs_index][1][i])
    
    obslist_new=obslist.copy()#高度角截至列表
    sat_num=len(obslist)
    ex_index=np.zeros(sat_num,dtype=int)
    
    if(sat_num<4):
        print("sat_num<4, pass epoch.")
        return [0,0,0,0],[],[]
    
    #卫星列表构建
    peph_sat_pos={}
    for i in range(0,sat_num):
        #光速
        clight=2.99792458e8
        #观测时间&观测值
        rt_week=obs_mat[obs_index][0]['GPSweek']
        rt_sec=obs_mat[obs_index][0]['GPSsec']
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
        si_PRN=obslist[i]['PRN'] 
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
    result=np.zeros((4),dtype=np.float64)
    result[0:3]=rr
    result[3]=1.0
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
        H=np.zeros((sat_num,4),dtype=np.float64)
        #单位权中误差矩阵初始化
        var=np.zeros((sat_num,sat_num),dtype=np.float64)
        #权重矩阵初始化
        W=np.zeros((sat_num,sat_num),dtype=np.float64)
    
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
            
            #单卫星设计矩阵赋值
            H[i]=[urs_x,urs_y,urs_z,1]
            #地球自转改正到卫地距上
            r0=r0+OMGE*(rs[0]*rr[1]-rs[1]*rr[0])/clight
            
            #观测矩阵
            if(sol_mode=='SF'):
                Z[i]=p1-r0-result[3]-satpos.get_Tropdelay(rr,rs)-satpos.get_ion_GPS(rt_unix,rr,rs,ion_param)+clight*dts
            
            #双频无电离层延迟组合
            elif(sol_mode=='IF'):
                f12=f1*f1
                f22=f2*f2
                p_IF=f12/(f12-f22)*p1-f22/(f12-f22)*p2
                Z[i]=p_IF-r0-result[3]-satpos.get_Tropdelay(rr,rs)+clight*dts

            #随机模型
            #var[i][i]= 0.00224*10**(-s1 / 10) 
            _,el=satpos.getazel(rs,rr)
            var[i][i]=0.3*0.3+0.3*0.3/sin(el)/sin(el)
            if(el*180.0/satpos.pi<el_threthod):
                var[i][i]=var[i][i]*100#低高度角拒止
                ex_index[i]=1
            if(ex_index[i]==1 and el*180.0/satpos.pi>=el_threthod):
                ex_index[i]=0
            
            if(sol_mode=='IF'):
                var[i][i]=var[i][i]*9
            W[i][i]=1.0/var[i][i]
        
        #最小二乘求解:
        dresult=getLSQ_solution(H,Z,W=W,weighting_mode='S')
        
        #迭代值更新
        result[0]+=dresult[0]
        result[1]+=dresult[1]
        result[2]+=dresult[2]
        result[3]+=dresult[3]

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
            break
    
    #排除低高度角卫星
    for i in range(sat_num):
        if(ex_index[i]):
            obslist_new.remove(obslist[i])
    return result,obslist_new,peph_sat_pos

#IGGIII等价权抗差卡尔曼滤波
def IGGIII(v,R,k0=1.5,k1=3.0):
    Rv=np.zeros(v.shape[0],dtype=np.float64)
    #求残差:Rvi=vi/sqrt(Ri)
    for i in range(len(v)):
        Rv[i]=abs(v[i]/sqrt(R[i][i]))
    #求标准化残差中位数: 
    Rv_median=np.median(Rv)    
    #中位方差
    delta=Rv_median*1.4826
    #标准化
    for i in range(len(v)):
        Rv[i]=Rv[i]/delta
    #迭代IGGIII最大抗差
    imax=np.argmax(Rv)
    #最大残差小于阈值
    if(Rv[imax]<=k0):
        return R
    #最大残差位于等价权阈值区间
    elif(Rv[imax]<=k1 and Rv[imax]>k0):
        R[imax][imax]=R[imax][imax] / (k0/Rv[imax] * ((k1-Rv[imax])/(k1-k0)) * ((k1-Rv[imax])/(k1-k0)))
    #最大残差位于等价权阈值外
    else:
        R[imax][imax]=R[imax][imax]*1e8
    return R


def createKF_XkPkQk(obslist,X,Pk,Qk):
    #系统模型构建
    #输入: 观测字典列表, 全局状态X, 全局方差Pk, 全局过程噪声Qk
    #输出: 滤波状态t_Xk, 滤波方差t_Pk, 滤波过程噪声t_Qk
    sat_num=len(obslist)#本历元有效观测卫星数量
    sys_sat_sum=round((X.shape[0]-5)/3)#全局状态卫星数量
    
    #本历元更新状态所用系统临时变量(依据: 在观测列表内的卫星数量)
    t_Xk=np.zeros((3*sat_num+5,1),dtype=np.float64)
    t_Pk=np.zeros((3*sat_num+5,3*sat_num+5),dtype=np.float64)
    t_Qk=np.zeros((3*sat_num+5,3*sat_num+5),dtype=np.float64)
    
    #有效卫星索引(计算本历元有效卫星各状态量在总状态中的索引)
    sat_use=[]#首先保证不变状态量保存在内
    for i in range(len(obslist)):
        si_PRN=obslist[i]['PRN']
        PRN_index=int(si_PRN[1:])-1
        sat_use.append(PRN_index)
    index_use=[0,1,2,3,4]#基础导航状态
    for s in sat_use:
        index_use.append(5+s)               #电离层状态导入
    for s in sat_use:
        index_use.append(5+sys_sat_sum+s)   #L1模糊度状态导入
    for s in sat_use:
        index_use.append(5+2*sys_sat_sum+s) #L2模糊度状态导入
    
    #系统状态,方差,过程噪声赋值
    for i in range(5+3*sat_num):
        t_Xk[i]=X[index_use[i]]                         #系统状态
        t_Qk[i][i]=Qk[index_use[i]][index_use[i]]       #系统过程噪声
        for j in range(5+3*sat_num):
            t_Pk[i][j]=Pk[index_use[i]][index_use[j]]   #系统方差
    
    #返回系统模型各临时矩阵
    return t_Xk,t_Pk,t_Qk

def upstateKF_XkPkQk(obslist,rt_unix,t_Xk,t_Pk,t_Qk,X,Pk,Qk,X_time):
    #系统模型恢复与更新
    #输入: 观测字典列表, 滤波状态t_Xk, 滤波方差t_Pk, 滤波过程噪声t_Qk, 全局状态X, 全局方差Pk, 全局过程噪声Qk
    #输出: 恢复并更新后的全局状态X, 全局方差Pk, 全局过程噪声Qk
    sat_num=len(obslist)#本历元有效观测卫星数量
    sys_sat_sum=round((X.shape[0]-5)/3)#全局状态卫星数量
    
    #本历元更新状态所用系统临时变量(不能占用全局状态储存空间)
    t_X=np.zeros((3*sys_sat_sum+5,1),dtype=np.float64)
    t_X_time=np.zeros(3*sys_sat_sum+5,dtype=np.float64)
    t_P=np.zeros((3*sys_sat_sum+5,3*sys_sat_sum+5),dtype=np.float64)
    t_Q=np.zeros((3*sys_sat_sum+5,3*sys_sat_sum+5),dtype=np.float64)
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
        sat_use.append(PRN_index)
    index_use=[0,1,2,3,4]#基础导航状态
    for s in sat_use:
        index_use.append(5+s)               #电离层状态导入
    for s in sat_use:
        index_use.append(5+sys_sat_sum+s)   #L1模糊度状态导入
    for s in sat_use:
        index_use.append(5+2*sys_sat_sum+s) #L2模糊度状态导入
    
    #系统状态,方差,过程噪声恢复更新
    for i in range(5+3*sat_num):
        t_X[index_use[i]]=t_Xk[i]                         #系统状态
        t_X_time[index_use[i]]=rt_unix                    #系统状态时标
        t_Q[index_use[i]][index_use[i]]=t_Qk[i][i]        #系统过程噪声
        for j in range(5+3*sat_num):
            if(i==j and t_Pk[i][j]<0.0):
                t_Pk[i][j]=60*60
            t_P[index_use[i]][index_use[j]]=t_Pk[i][j]   #系统方差

    #返回系统模型各全局矩阵
    return t_X,t_P,t_Q,t_X_time

def update_ion(p1,p2,f1=1575.42*1e6,f2=1227.60*1e6):
    #输入: 双频伪距观测值p1,p2, 双频频率f1,f2
    #输出: f1基准频率上的斜延迟    
    return  (p1-p2)/(1-f1*f1/f2/f2)

def update_phase_amb(p,l,f,p1,p2,f1=1575.42*1e6,f2=1227.60*1e6):
    #输入: 双频观测值, 待计算初值的观测值p, l及频率f, 频点频率  
    #输出: 模糊度重置结果
    #重置模糊度AMB

    #计算f1频率上电离层延迟
    s_dion=(p1-p2)/(1-f1*f1/f2/f2)
    l=l*satpos.clight/f
    
    return l-p+2*s_dion*f1*f1/f/f

#周跳探测
def get_phase_jump(p1,p2,l1,l2,GF_sign,Mw_sign,Mw_threshold,GF_threshold,f1=1575.42*1e6,f2=1227.60*1e6):    
    clight=satpos.clight
    slip=0
    
    #如果无观测值, 则返回0
    if(p1==0.0 or p2==0.0 or l1==0.0 or l2==0.0):
        return 0.0,0.0,0,0.0,0.0
    
    #GF组合探测大周跳
    g1=clight*l1/f1-clight*l2/f2#本历元GF组合观测值
    g0=GF_sign
    GF_sign=g1#更新GF组合观测值
    
    #Mw组合探测小周跳
    lambda_1=clight/f1
    lambda_2=clight/f2
    m1=((f1-f2)/(f1+f2))*(p1/lambda_1+p2/lambda_2)-(l1-l2)
    m0=Mw_sign
    Mw_sign=m1#更新Mw组合观测值    
    
    if(g0!=0.0 and abs(g1-g0)>GF_threshold):
        slip=1
    
    if(m0!=0.0 and abs(m1-m0)>Mw_threshold):
        slip=1
    #测试:修复周跳
    dN1=0.0
    dN2=0.0
    if(slip):
        dMw=m1-m0
        dGF=g1-g0
        dN1=(dMw+(dGF-dMw)/(lambda_1-lambda_2))
        dN2=((dGF-dMw)/(lambda_1-lambda_2))
    

    return GF_sign,Mw_sign,slip,dN1,dN2

def update_phase_slip(obslist,GF_sign,Mw_sign,slip_sign,Mw_threshold,GF_threshold,f1,f2,dN=[],dN_fix_mode=0):
    #首先清空周跳标志
    for i in range(len(slip_sign)):
        slip_sign[i]=0
    
    
    #清空无观测值的周跳检测量
    prns=[int(t['PRN'][1:]) for t in obslist]
    for i in range(len(GF_sign)):
        in_PRN=i+1
        if(in_PRN not in prns):
            GF_sign[i]=0.0
            Mw_sign[i]=0.0
    
    #周跳检测
    sat_num=len(obslist)
    
    for i in range(sat_num):
        si_PRN=obslist[i]['PRN']
        PRN_index=int(si_PRN[1:])-1
        
        p1=obslist[i]['OBS'][0]
        l1=obslist[i]['OBS'][1]
        p2=obslist[i]['OBS'][5]
        l2=obslist[i]['OBS'][6]

        GF,Mw,slip,dN1,dN2=get_phase_jump(p1,p2,l1,l2,GF_sign[PRN_index],Mw_sign[PRN_index],Mw_threshold,GF_threshold,f1=f1,f2=f2)
        if(slip):
            print('{} 发生周跳 GF:{}->{} Mw:{}->{} p1:{} l1:{} p2:{} l2:{} dN1:{} dN2:{}'.format(si_PRN,GF_sign[PRN_index],GF,Mw_sign[PRN_index],Mw,p1,l1,p2,l2,dN1,dN2))
        GF_sign[PRN_index]=GF
        Mw_sign[PRN_index]=Mw
        slip_sign[PRN_index]=slip
        
        if(dN_fix_mode):
            dN[PRN_index][0]=dN1
            dN[PRN_index][1]=dN2
    
    return GF_sign,Mw_sign,slip_sign,dN

#天线相位缠绕改正
def sat_phw(rt_unix,si_PRN,opt,rr,rs,rs_speed,phase_bias):
    #输入:观测时间,待计算相位缠绕的卫星PRN,类型,选项,接收机坐标,卫星位置
    #输出:天线相位缠绕改正数
    
    #首先判断是否改正卫星天线相位缠绕
    if(opt!=1):
        return 0
    #上一历元phw继承
    try:
        phw_old=phase_bias[si_PRN]['phw']
    except:
        phw_old=0.0
    #改正天线相位缠绕
    #1.卫星指向接收机单位矢量
    urs_rr_1=rr[0]-rs[0]
    urs_rr_2=rr[1]-rs[1]
    urs_rr_3=rr[2]-rs[2]
    urs_rr_morm=sqrt(urs_rr_1*urs_rr_1+urs_rr_2*urs_rr_2+urs_rr_3*urs_rr_3)
    up_k=[urs_rr_1/urs_rr_morm,urs_rr_2/urs_rr_morm,urs_rr_3/urs_rr_morm]#卫星指向接收机的单位矢量
    up_k=np.array(up_k).reshape((3,))

    #2.测站当地地平坐标系下的单位矢量(ECEF转ENU的旋转矩阵)
    rr_b,rr_l,rr_h=satpos.xyz2blh(rr[0],rr[1],rr[2])#测站地理坐标
    rrb=rr_b/180*satpos.pi
    rrl=rr_l/180*satpos.pi
    up_x_rr=[-sin(rrb)*cos(rrl),-sin(rrb)*sin(rrl),cos(rrb)]#对应教材中x^, 北方向单位矢量
    up_y_rr=[sin(rrl),-cos(rrl),0.0]                        #对应教材中y^, 西方向单位矢量

    #3.卫星星固坐标系下单位矢量(卫星载体系/姿态, Z指向地心, XY构建天球坐标系中航向)
    OMGE=7.2921151467E-5
    rsun,rmoon,_=sun_moon_pos(rt_unix+gpst2utc(rt_unix))

    #4.地心地固坐标系(ecef)下太阳位置
    # U,gmst=eci2ecef(rt_unix)
    # rsun=U.dot(rsun)
    # rmoon=U.dot(rmoon)
    
    
    #太阳轨道倾角及beta参数计算
    ri=np.array([rs[0],rs[1],rs[2],rs_speed[0]-OMGE*rs[1],rs_speed[1]+OMGE*rs[0],rs_speed[2]])
    n=np.cross(ri[:3],ri[3:])
    p=np.cross(rsun,n)

    es=ri[:3]/np.linalg.norm(ri[:3])
    esun=np.array(rsun)/np.linalg.norm(rsun)
    en=n/np.linalg.norm(n)
    ep=p/np.linalg.norm(p)

    beta=satpos.pi/2.0-acos(esun.dot(en))
    E=acos(es.dot(ep))
    mu=satpos.pi/2.0
    if(es.dot(esun)<=0):
        mu=mu-E
    else:
        mu=mu+E
    if(mu<-satpos.pi/2.0):
        mu=mu+2.0*satpos.pi
    elif(mu>=satpos.pi/2.0):
        mu=mu-2.0*satpos.pi
    
    #卫星航向角
    yaw=0.0
    if (abs(beta)<1E-12 and abs(mu)<1E-12): 
        yaw=satpos.pi
    else:
        yaw=atan2(-tan(beta),sin(mu))+satpos.pi
    ex=np.cross(en,es)

    up_x_rs=[ -sin(yaw)*en[0]+cos(yaw)*ex[0], -sin(yaw)*en[1]+cos(yaw)*ex[1], -sin(yaw)*en[2]+cos(yaw)*ex[2] ]#对应教材中x', 北方向单位矢量
    up_y_rs=[ -cos(yaw)*en[0]-sin(yaw)*ex[0], -cos(yaw)*en[1]-sin(yaw)*ex[1], -cos(yaw)*en[2]-sin(yaw)*ex[2] ]#对应教材中y', 西方向单位矢量

    #4.准备工作结束, 计算相位缠绕
    up_ks=np.cross(up_k,up_y_rs)
    up_kr=np.cross(up_k,up_y_rr)
    ds=up_x_rs-up_k*(up_k.dot(up_x_rs))-up_ks
    dr=up_x_rr-up_k*(up_k.dot(up_x_rr))+up_kr

    cosp=(ds.dot(dr))/np.linalg.norm(ds)/np.linalg.norm(dr)
    if(cosp<-1.0):
        cosp=-1.0
    elif(cosp>1.0):
        cosp=1.0
    if(abs(abs(cosp)-1.0)<1e-10):
        return 0.0
    #相位缠绕改正计算, 单位: 周
    ph=acos(cosp)/2.0/satpos.pi

    #sign函数
    drs=np.cross(ds,dr)
    if(up_k.dot(drs)<0.0):
        ph=-ph
    #前一历元PHW继承
    if(phw_old==0.0):
        phw=ph
    else:
        phw=ph+np.floor(phw_old-ph+0.5)
    return phw


def createKF_HRZ_new(obslist,rt_unix,X,X_time,Pk,Qk,ion_param,phase_bias,peph_sat_pos,f1=1575.42*1e6,f2=1227.60*1e6,ex_threshold_v=30,exthreshold_v_sigma=4,post=True):
    
    #初始化卫星数量
    sat_num=len(obslist)
    sys_sat_sum=round((X.shape[0]-5)/3)
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
    H=np.zeros((4*sat_num,3*sat_num+5),dtype=np.float64)
    #Z=np.zeros((4*sat_num,1),dtype=np.float64)
    R=np.eye(4*sat_num,dtype=np.float64)
    v=np.zeros((4*sat_num,1),dtype=np.float64)
    #print("H,Z,R",H.shape,Z.shape,R.shape)
    for i in range(sat_num): #逐卫星按行创建设计矩阵
        #状态索引求解
        si_PRN=obslist[i]['PRN']
        PRN_index=int(si_PRN[1:])-1
        ion_index=5+PRN_index
        N1_index=5+sys_sat_sum+PRN_index
        N2_index=5+sys_sat_sum*2+PRN_index
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
        H_sub1=np.zeros(3*sat_num+5,dtype=np.float64)   #初始化频1伪距行
        H_sub1[0:5]=[urs_x,urs_y,urs_z,1,Mw]            #基础项
        H_sub1[5+i]=1                                   #频1伪距STEC系数
        #l1
        H_sub2=np.zeros(3*sat_num+5,dtype=np.float64)   #初始化频1相位行
        H_sub2[0:5]=[urs_x,urs_y,urs_z,1,Mw]            #基础项
        H_sub2[5+i]=-1                                  #频1相位STEC系数
        H_sub2[5+sat_num+i]=1                           #频1模糊度
        #p2
        H_sub3=np.zeros(3*sat_num+5,dtype=np.float64)   #初始化频2伪距行
        H_sub3[0:5]=[urs_x,urs_y,urs_z,1,Mw]            #基础项
        H_sub3[5+i]=f1*f1/f2/f2                         #频2伪距STEC系数
        #l2
        H_sub4=np.zeros(3*sat_num+5,dtype=np.float64)   #初始化频1相位行
        H_sub4[0:5]=[urs_x,urs_y,urs_z,1,Mw]            #基础项
        H_sub4[5+i]=-f1*f1/f2/f2                        #频2相位STEC系数
        H_sub4[5+2*sat_num+i]=1                         #频2模糊度

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
        v[i*4]=  p1 -           (r0 + rr[3]- (clight*dts) + (Mh*get_Trop_delay_dry(rr)+Mw*X[4][0]) + (1*X[ion_index][0]) )
        v[i*4+1]=l1*clight/f1 - (r0 + rr[3]- (clight*dts) + (Mh*get_Trop_delay_dry(rr)+Mw*X[4][0]) + ( -1*X[ion_index][0]) + (1*X[N1_index][0]) )
        v[i*4+2]=p2 -           (r0 + rr[3]- (clight*dts) + (Mh*get_Trop_delay_dry(rr)+Mw*X[4][0]) + ( f1*f1/f2/f2*X[ion_index][0]) )
        v[i*4+3]=l2*clight/f2 - (r0 + rr[3]- (clight*dts) + (Mh*get_Trop_delay_dry(rr)+Mw*X[4][0]) + ( -f1*f1/f2/f2*X[ion_index][0]) + (1*X[N2_index][0]) )
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

def log2out(rt_unix,v,obslist,X,X_time,Pk,peph_sat_pos,f1=1575.42*1e6):
    #历元数据整备
    out={}
    for i in range(len(obslist)):
        si_PRN=obslist[i]['PRN']
        PRN_index=int(si_PRN[1:])-1
        sys_sat_num=int((X.shape[0]-5)/3)

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

        out[si_PRN]['sta_x']=X[0][0]
        out[si_PRN]['std_sta_x']=Pk[0][0]
        out[si_PRN]['sta_y']=X[1][0]
        out[si_PRN]['std_sta_y']=Pk[1][1]
        out[si_PRN]['sta_z']=X[2][0]
        out[si_PRN]['std_sta_z']=Pk[2][2]
        out[si_PRN]['GPSsec_dt']=X[3][0]
        out[si_PRN]['std_GPSsec_dt']=Pk[3][3]

        out[si_PRN]['ztd_w']=X[4][0]
        out[si_PRN]['std_ztd_w']=Pk[4][4]

        rr=[X[0][0],X[1][0],X[2][0]]
        out[si_PRN]['ztd_h']=get_Trop_delay_dry(rr)
        out[si_PRN]['res_p1']=v[4*i][0]
        out[si_PRN]['res_l1']=v[4*i+1][0]
        out[si_PRN]['res_p2']=v[4*i+2][0]
        out[si_PRN]['res_l2']=v[4*i+3][0]

        az,el=getazel(rs,rr)
        out[si_PRN]['azel']=[az/pi*180.0,el/pi*180.0]

        #电离层状态更新
        if(X_time[5+PRN_index]==rt_unix):
            Mi=IMF_ion(rr,rs,MF_mode=1,H_ion=350e3)
            out[si_PRN]['STEC']=X[5+PRN_index][0]*(f1/1e8)*(f1/1e8)/40.28
            out[si_PRN]['std_STEC']=Pk[5+PRN_index][5+PRN_index]*((f1/1e8)*(f1/1e8)/40.28)**2
        
        #模糊度状态更新
        if(X_time[5+2*sys_sat_num+PRN_index]==rt_unix):
            out[si_PRN]['N1']=X[5+sys_sat_num+PRN_index][0]
            out[si_PRN]['std_N1']=Pk[5+sys_sat_num+PRN_index][5+sys_sat_num+PRN_index]
        
        if(X_time[5+2*sys_sat_num+PRN_index]==rt_unix):
            out[si_PRN]['N2']=X[5+2*sys_sat_num+PRN_index][0]
            out[si_PRN]['std_N2']=Pk[5+2*sys_sat_num+PRN_index][5+2*sys_sat_num+PRN_index]
    return out

#PPP状态初始化
def init_UCPPP(obs_mat,obs_start,IGS,clk,sat_out,ion_param,sat_pcos,sys_sat_num,f1,f2):
    
    #准备
    
    #1.单点定位最小二乘求解滤波初值
    obs_index=obs_start #设置初始化时间索引
    
    rr,obslist,peph_sat_pos=SPP_from_IGS(obs_mat,obs_index,IGS,clk,sat_out,ion_param,sat_pcos,sol_mode='SF',el_threthod=0.0,f1=f1,f2=f2)

    #位置、接收机钟差、天顶对流层延迟初值向量
    #观测时间&观测值
    rt_week=obs_mat[obs_index][0]['GPSweek']
    rt_sec=obs_mat[obs_index][0]['GPSsec']
    rt_unix=satpos.gpst2time(rt_week,rt_sec)
    X0_xyztm=np.array([rr[0],rr[1],rr[2],rr[3],0.15])                   #初始化位置/钟差/对流层
    X0_xyztm_time=np.array([rt_unix,rt_unix,rt_unix,rt_unix,rt_unix])   #初始化位置/钟差/电离层时标

    #观测列表卫星数量
    s_num=len(obslist)

    #电离层延迟改正数初值向量()
    X0_I=np.zeros(sys_sat_num,dtype=np.float64)
    X0_I_time=np.zeros(sys_sat_num,dtype=np.float64)#所有电离层状态量初始化为不可靠
    for i in range(s_num):
        si_PRN=obslist[i]['PRN']
        PRN_index=int(si_PRN[1:])-1#计算卫星在系统中的序号
        #观测时间&观测值
        rt_week=obs_mat[obs_index][0]['GPSweek']
        rt_sec=obs_mat[obs_index][0]['GPSsec']
        rt_unix=satpos.gpst2time(rt_week,rt_sec)
        #print(rt_week,rt_sec,rt_unix)
        
        #双频伪距计算电离层初值
        p1=obslist[i]['OBS'][0]
        s1=obslist[i]['OBS'][4]
        p2=obslist[i]['OBS'][5]
    
        #卫星位置
        si_PRN=obslist[i]['PRN'] 
        rs=[peph_sat_pos[si_PRN][0],peph_sat_pos[si_PRN][1],peph_sat_pos[si_PRN][2]]
        #X0_I[i]=satpos.get_ion_GPS(rt_unix,rr,rs,ion_param)
        ion=(p1-p2)/((f2*f2-f1*f1)/f2/f2)                              #伪距双差获取电离层延迟初值
        # if(ion<0.0):
        #     ion=satpos.get_ion_GPS(rt_unix,rr,rs,ion_param)
        X0_I[PRN_index]=ion                                          #估计电离层延迟
        X0_I_time[PRN_index]=rt_unix

    #L1载波上各星模糊度初值
    X0_N1=np.zeros(sys_sat_num,dtype=np.float64)
    X0_N1_time=np.zeros(sys_sat_num,dtype=np.float64)
    for i in range(s_num):
        si_PRN=obslist[i]['PRN']
        PRN_index=int(si_PRN[1:])-1#计算卫星在系统中的序号
    
        #观测时间&观测值
        rt_week=obs_mat[obs_index][0]['GPSweek']
        rt_sec=obs_mat[obs_index][0]['GPSsec']
        rt_unix=satpos.gpst2time(rt_week,rt_sec)
    
        p1=obslist[i]['OBS'][0]
        p2=obslist[i]['OBS'][5]
        l1=obslist[i]['OBS'][1]
    
        #伪距IF组合与相位互差求解模糊度初值
        # P_Ion_free=f1*f1/(f1*f1-f2*f2)*p1-f2*f2/(f1*f1-f2*f2)*p2
        X0_N1[PRN_index]=update_phase_amb(p1,l1,f1,p1,p2,f1=f1,f2=f2)
        X0_N1_time[PRN_index]=rt_unix
    
    #L2载波上各星模糊度初值
    X0_N2=np.zeros(sys_sat_num,dtype=np.float64)
    X0_N2_time=np.zeros(sys_sat_num,dtype=np.float64)
    for i in range(s_num):
        si_PRN=obslist[i]['PRN']
        PRN_index=int(si_PRN[1:])-1#计算卫星在系统中的序号
    
        #观测时间&观测值
        rt_week=obs_mat[obs_index][0]['GPSweek']
        rt_sec=obs_mat[obs_index][0]['GPSsec']
        rt_unix=satpos.gpst2time(rt_week,rt_sec)
    
        p1=obslist[i]['OBS'][0]
        p2=obslist[i]['OBS'][5]
        l2=obslist[i]['OBS'][6]
    
        #伪距IF组合与相位互差求解模糊度初值
        # P_Ion_free=f1*f1/(f1*f1-f2*f2)*p1-f2*f2/(f1*f1-f2*f2)*p2
        X0_N2[PRN_index]=update_phase_amb(p2,l2,f2,p1,p2,f1=f1,f2=f2)
        X0_N2_time[PRN_index]=rt_unix
    
    #相位相关误差字典初始化
    phase_bias={}
    for i in range(s_num):
        si_PRN=obslist[i]['PRN']
        phase_bias[si_PRN]={}
        phase_bias[si_PRN]['phw']=0.0

    #状态初值
    X=np.concatenate((X0_xyztm,X0_I,X0_N1,X0_N2))
    X_time=np.concatenate((X0_xyztm_time,X0_I_time,X0_N1_time,X0_N2_time))
    X=X.reshape(X.shape[0],1)
    X_time=X_time.reshape(X_time.shape[0],1)

    #根据状态初值构建状态转移矩阵, 无动力学约束
    PHIk_1_k=np.eye(X.shape[0],dtype=np.float64)

    #系统噪声方差阵(初始值)
    Qk=np.eye(X.shape[0],dtype=np.float64)
    for i in range(X.shape[0]):
        if(i in [0,1,2,3]):                     #接收机位置、钟差(标准差100m)
            Qk[i][i]=60*60
        if(i in [4]):                           #对流层先验(标准差0.8m)
            Qk[i][i]=60*60   
        if(i in range(5,5+sys_sat_num)):            #电离层先验(标准差100m)
            Qk[i][i]=60*60
        if(i in range(5+sys_sat_num,5+2*sys_sat_num)):  #L1模糊度先验(标准差100m)
            Qk[i][i]=60*60
        if(i in range(5+2*sys_sat_num,5+3*sys_sat_num)):#L2模糊度先验(标准差100m)
            Qk[i][i]=60*60
    Pk=Qk.copy()

    #周跳检验量(每历元均更新)
    GF_sign=np.zeros((sys_sat_num),dtype=np.float64)
    Mw_sign=np.zeros((sys_sat_num),dtype=np.float64)
    #周跳标志(每历元均重置为0)
    slip_sign=np.zeros((sys_sat_num),dtype=int)
    #周跳修复累计列表(随状态初始化重置)
    dN_sign=np.zeros((sys_sat_num,2),dtype=np.float64)

    return X,Pk,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,X_time,phase_bias

def updata_PPP_state(X,Pk,spp_rr,epoch,rt_unix,X_time,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,GF_threshold,Mw_threshold,
                     sat_num,rnx_obs,out_age,f1,f2,dy_mode):
    #状态递推
    X[3]=spp_rr[3]#接收机钟差数值更新
    
    if(X[0]==100.0):
        #上历元先验重置
        print("上历元无解,重置")
        X[0]=spp_rr[0]
        X[1]=spp_rr[1]
        X[2]=spp_rr[2]
    
    
    if(dy_mode!='static'):
        X[0]=spp_rr[0]    
        X[1]=spp_rr[1]    
        X[2]=spp_rr[2]    
    
    #非首历元, 状态重置
    if(epoch):
        #计算位置/钟差/对流层状态更新时间差
        dt=rt_unix-X_time[0][0]
        #位置/钟差/对流层状态过程噪声
        if(dy_mode=='static'):
            Qk[0][0]=0.0#3600.0#坐标改正数
            Qk[1][1]=0.0#3600.0#坐标改正数
            Qk[2][2]=0.0#3600.0#坐标改正数
            Qk[3][3]=60*60#接收机钟差(白噪声)
            Qk[4][4]=1e-8*dt#对流层延迟(缓慢变化)
        else:
            Qk[0][0]=3600.0#坐标改正数
            Qk[1][1]=3600.0#坐标改正数
            Qk[2][2]=3600.0#坐标改正数
            Qk[3][3]=60*60#接收机钟差(白噪声)
            Qk[4][4]=1e-8*dt#对流层延迟(缓慢变化)

        #部分更新的状态量
        #电离层
        for j in range(5,5+sat_num):
            dt=rt_unix-X_time[j][0]
            si_PRN=j-5+1#整型PRN序号(PRN-1)
            rnx_obs_prns=[int(t['PRN'][1:]) for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age ): 
                Qk[j][j]=0.0016*dt  #电离层()
            if(si_PRN in rnx_obs_prns and dt>out_age):
                # if(dt>out_age):
                #     GF_sign[si_PRN-1]=0.0
                #     Mw_sign[si_PRN-1]=0.0
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                ion=update_ion(p1,p2,f1=f1,f2=f2)
                X[j]=ion       #重置垂直电离层估计
                Qk[j][j]=30*30  #重置过程噪声
                Pk[j,:]=0.0 
                Pk[:,j]=0.0     
        #L1模糊度
        for j in range(5+sat_num,5+2*sat_num):
            dt=rt_unix-X_time[j][0]
            si_PRN=j-(5+sat_num)+1#整型PRN序号(PRN-1)
            rnx_obs_prns=[int(t['PRN'][1:]) for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=0.0*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                if(dt>out_age):
                    GF_sign[si_PRN-1]=0.0
                    Mw_sign[si_PRN-1]=0.0
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p1,l1,f1,p1,p2,f1=f1,f2=f2)
                Qk[j][j]=30*30  #重置过程噪声
                Pk[j,:]=0.0 
                Pk[:,j]=0.0     
        #L2模糊度
        for j in range(5+2*sat_num,5+3*sat_num):
            dt=rt_unix-X_time[j][0]
            si_PRN=j-(5+2*sat_num)+1#整型PRN序号(PRN-1)
            rnx_obs_prns=[int(t['PRN'][1:]) for t in rnx_obs]#观测列表中的整型PRN序号
            if(si_PRN in rnx_obs_prns and dt<=out_age):
                Qk[j][j]=0.0*dt#模糊度(常数)
            if(si_PRN in rnx_obs_prns and dt>out_age):
                if(dt>out_age):
                    GF_sign[si_PRN-1]=0.0
                    Mw_sign[si_PRN-1]=0.0
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[j]=update_phase_amb(p2,l2,f2,p1,p2,f1=f1,f2=f2)
                Qk[j][j]=30*30  #重置过程噪声
                Pk[j,:]=0.0 
                Pk[:,j]=0.0     
        
        #周跳探测
        #GF_sign,Mw_sign,slip_sign,dN_sign=
        update_phase_slip(rnx_obs,GF_sign,Mw_sign,slip_sign,Mw_threshold,GF_threshold,f1,f2,dN_sign,dN_fix_mode=1)
        #小周跳修复/大周跳重置
        for j in range(len(slip_sign)):
            # if(slip_sign[j] and (abs(dN_sign[j][0])<500 or abs(dN_sign[j][1]<500))):
            #     #print('{} G{:02d} 周跳修复 GF: {} Mw:{} dN1:{} dN2:{}'.format(epoch,j+1,GF_sign[j],Mw_sign[j],dN_sign[j][0],dN_sign[j][1]))
            #     X[5+sat_num+j]=X[5+sat_num+j]+dN_sign[j][0]*clight/f1                
            #     X[5+2*sat_num+j]=X[5+2*sat_num+j]+dN_sign[j][1]*clight/f2
            #     Qk[5+sat_num+j][5+sat_num+j]=1e2                
            #     Qk[5+2*sat_num+j][5+2*sat_num+j]=1e2
            if(slip_sign[j] and (abs(dN_sign[j][0])>=0.0 or abs(dN_sign[j][1]>=0.0))):
                si_PRN=j+1
                rnx_obs_prns=[int(t['PRN'][1:]) for t in rnx_obs]
                p1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][0]
                p2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][5]
                l1=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][1]
                l2=rnx_obs[rnx_obs_prns.index(si_PRN)]['OBS'][6]
                X[5+sat_num+j]=update_phase_amb(p1,l1,f1,p1,p2,f1=f1,f2=f2)              
                X[5+2*sat_num+j]=update_phase_amb(p2,l2,f2,p1,p2,f1=f1,f2=f2)
                Qk[5+sat_num+j][5+sat_num+j]=60*60                
                Pk[5+sat_num+j,:]=0.0                
                Pk[:,5+sat_num+j]=0.0                
                Qk[5+2*sat_num+j][5+2*sat_num+j]=60*60
                Pk[5+2*sat_num+j,:]=0.0
                Pk[:,5+2*sat_num+j]=0.0

def KF_UCPPP(X,X_time,Pk,Qk,ion_param,peph_sat_pos,rnx_obs,ex_threshold_v,ex_threshold_sigma,rt_unix,phase_bias,f1,f2):
    #扩展卡尔曼滤波
    for KF_times in range(100):
        #相位改正值拷贝
        t_phase_bias=phase_bias.copy()
        
        #观测模型(两次构建, 验前粗差剔除)
        #print(rnx_obs)
        X,X_time,H,R,_,v,rnx_obs=createKF_HRZ_new(rnx_obs,rt_unix,X,X_time,Pk,Qk,ion_param,t_phase_bias,peph_sat_pos,f1=f1,f2=f2,exthreshold_v_sigma=ex_threshold_sigma,post=False,ex_threshold_v=ex_threshold_v)
        if(not len(rnx_obs)):
            #无先验通过数据
            #全部状态重置
            # X[0]=100.0
            # X[1]=100.0
            # X[2]=100.0
            for i in range(len(X)):
                X_time[i]=0.0
            break
        X,X_time,H,R,_,v,rnx_obs=createKF_HRZ_new(rnx_obs,rt_unix,X,X_time,Pk,Qk,ion_param,t_phase_bias,peph_sat_pos,f1=f1,f2=f2,exthreshold_v_sigma=ex_threshold_sigma,post=False,ex_threshold_v=ex_threshold_v)
        
        #系统模型(根据先验抗差得到的数据)
        t_Xk,t_Pk,t_Qk=createKF_XkPkQk(rnx_obs,X,Pk,Qk)

        #抗差滤波准备
        #R=IGGIII(v,R)
        #扩展卡尔曼滤波
        #1.状态一步预测
        PHIk_1_k=np.eye(t_Xk.shape[0],dtype=np.float64)
        X_up=PHIk_1_k.dot(t_Xk)
        #2.状态一步预测误差
        Pk_1_k=(PHIk_1_k.dot(t_Pk)).dot(PHIk_1_k.T)+t_Qk
        #3.滤波增益计算
        Kk=(Pk_1_k.dot(H.T)).dot(inv((H.dot(Pk_1_k)).dot(H.T)+R))
        #Kk=(Pk_1_k.dot(H.T)).dot(numba_inv((H.dot(Pk_1_k)).dot(H.T)+R))
        #滤波结果
        Xk_dot=X_up+Kk.dot(v)
        #滤波方差更新
        t_Pk=((np.eye(t_Xk.shape[0],dtype=np.float64)-Kk.dot(H))).dot(Pk_1_k)  
        t_Pk=t_Pk.dot((np.eye(t_Xk.shape[0],dtype=np.float64)-Kk.dot(H)).T)+Kk.dot(R).dot(Kk.T)
        #滤波暂态更新
        t_Xk_f,t_Pk_f,t_Qk_f,t_X_time=upstateKF_XkPkQk(rnx_obs,rt_unix,Xk_dot,t_Pk,t_Qk,X,Pk,Qk,X_time)
        
        #验后方差
        info='KF fixed'
        info,rnx_obs,tt_phase_bias,v=createKF_HRZ_new(rnx_obs,rt_unix,t_Xk_f,t_X_time,t_Pk_f,t_Qk_f,ion_param,t_phase_bias,peph_sat_pos,f1,f2,exthreshold_v_sigma=ex_threshold_sigma,post=True)
        #_,info=get_post_v(rnx_obs,rt_unix,Xk_dot,ion_param,phase_bias)
        if(info=='KF fixed'):    
            #验后校验通过
            X,Pk,Qk,X_time=upstateKF_XkPkQk(rnx_obs,rt_unix,Xk_dot,t_Pk,t_Qk,X,Pk,Qk,X_time)
            phase_bias=tt_phase_bias.copy()
            break

    return X,Pk,Qk,X_time,v,phase_bias,rnx_obs

def UCPPP(obs_mat,obs_start,obs_epoch,IGS,clk,
          sat_out,ion_param,sat_pcos,el_threthod,ex_threshold_v,ex_threshold_v_sigma,Mw_threshold,GF_threshold,dy_mode,
          X,Pk,Qk,phase_bias,X_time,GF_sign,Mw_sign,slip_sign,dN_sign,sat_num,out_age,f1,f2):
    
    Out_log=[]

    obs_index=obs_start
    for epoch in tqdm(range(obs_epoch)):
    
        #观测时间
        rt_week=obs_mat[obs_index+epoch][0]['GPSweek']
        rt_sec=obs_mat[obs_index+epoch][0]['GPSsec']
        rt_unix=satpos.gpst2time(rt_week,rt_sec)
    
        #单点定位
        spp_rr,rnx_obs,peph_sat_pos=SPP_from_IGS(obs_mat,obs_index+epoch,IGS,clk,sat_out,ion_param,sat_pcos,el_threthod=el_threthod,sol_mode="SF",pre_rr=[X[0][0],X[1][0],X[2][0],X[3][0]])
        #无单点定位解
        if(not len(rnx_obs)):
            print("No valid observations, Pass epoch: Week: {}, sec: {}.".format(rt_week,rt_sec))
            continue

        #PPP状态更新
        updata_PPP_state(X,Pk,spp_rr,epoch,rt_unix,X_time,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,GF_threshold,Mw_threshold,sat_num,rnx_obs,out_age,f1,f2,dy_mode)

        #PPP时间更新
        X,Pk,Qk,X_time,v,phase_bias,rnx_obs=KF_UCPPP(X,X_time,Pk,Qk,ion_param,peph_sat_pos,rnx_obs,ex_threshold_v,ex_threshold_v_sigma,rt_unix,phase_bias,f1,f2)

        #结果保存
        Out_log.append(log2out(rt_unix,v,rnx_obs,X,X_time,Pk,peph_sat_pos,f1=f1))
    
    return Out_log



if __name__=='__main__':
    
    #双频非差非组合PPP, 解算文件最小配置(观测值/观测值类型/精密星历文件/精密钟差文件/天线文件/结果输出路径)
    obs_file="D:/SMART-PNT/kagglehub/datasets/fengzhusgg/smartpnt-pos/versions/19/Dataset/Data04_20220315_FSAS_Vehicle_Complex/ROVE.22O"
    obs_type=['C1C','L1C','D1C','S1C','C2W','L2W','D2W','S2W']
    SP3_file="wum2022074.sp3"
    CLK_file="wum2022074.clk"
    ATX_file="data/ATX/igs20.atx"                                #天线文件, 支持格式转换后的npy文件和IGS天线文件

    out_path="nav_result"                                        #导航结果输出文件路径
    ion_param=[]                                                 #自定义Klobuchar电离层参数

    #可选配置(广播星历文件/DCB改正与产品选项/)
    BRDC_file="D:/SMART-PNT/kagglehub/datasets/fengzhusgg/smartpnt-pos/versions/19/Dataset/Data04_20220315_FSAS_Vehicle_Complex/brdm0740.22p"     #广播星历文件, 支持BRDC/RINEX3混合星历
    dcb_correction=0                                             #DCB修正选项
    dcb_products='NPY'                                           #DCB产品来源, 支持CODE月解文件/CAS日解文件
    dcb_file_0="D:/FDU/IGG-DCB/C1C_C2W_JFNG.npy"                                                #频间偏差文件, 支持预先转换后的.npy格式
    dcb_file_1=""
    dcb_file_2=""

    obs_start=100                                    #解算初始时刻索引
    obs_epoch=6400                                   #解算总历元数量
    out_age=2                                        #最大容忍失锁阈值时间(单位: s, 用于电离层、模糊度状态重置)
    sys_index=['G']                                  #此列表控制obsmat读取范围
    sys='GPS'                                        #配置解算系统, 支持GPS/BDS单系统
    dy_mode='dynamic'                                #PPP动态模式配置, 支持static, dynamic
    f1=1575.42*1e6                                   #配置第一频率
    f2=1227.60*1e6                                   #配置第二频率

    el_threthod=0.0                                  #设置截止高度角
    ex_threshold_v=400                               #设置先验残差阈值
    ex_threshold_v_sigma=20                          #设置后验残差阈值
    Mw_threshold=1.0                                 #设置Mw组合周跳检验阈值
    GF_threshold=0.05                                #设置GF组合周跳检验阈值

    sat_out=[]                                      #设置排除卫星


    #配置设置完毕, 以下为典型UC-PPP处理流程
    #CAS DCB产品数据读取
    if(dcb_correction==1 and dcb_products=='CAS'):
        dcb_file_0,_=CAS_DCB(dcb_file_0,obs_type[0],obs_type[4])
        dcb_file_1=''       #CAS产品同时包含码间和频间偏差
        dcb_file_2=''
    
    #读取观测文件
    sys_code=['G','C','E','R']
    sys_number=['GPS','BDS','GAL','GLO']
    obs_mat=RINEX3_to_obsmat(obs_file,obs_type,sys=sys_code[sys_number.index(sys)],dcb_correction=dcb_correction,dcb_file_0=dcb_file_0,dcb_file_1=dcb_file_1,dcb_file_2=dcb_file_2,f1=f1,f2=f2)
    #删除CAS-DCB产品辅助文件
    if(dcb_file_0!="" and dcb_products=='CAS'):
        os.unlink(dcb_file_0)

    if(not obs_epoch):
        obs_epoch=len(obs_mat)
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
    
    #读取广播星历电离层参数
    if(not len(ion_param)):
        ion_param=RINEX2ion_params(BRDC_file)
    
    #根据配置设置卫星数量
    if(sys=='GPS'):
        sat_num=32
    elif(sys=='BDS'):
        sat_num=65
    elif(sys=='GAL'):
        sat_num=37
    else:
        sat_num=0
    
    
    #初始化PPP滤波器
    X,Pk,Qk,GF_sign,Mw_sign,slip_sign,dN_sign,X_time,phase_bias=init_UCPPP(obs_mat,obs_start,IGS,clk,sat_out,ion_param,sat_pcos,sys_sat_num=sat_num,f1=f1,f2=f2)
    
    #非差非组合PPP解算
    Out_log=UCPPP(obs_mat,obs_start,obs_epoch,IGS,clk,sat_out,ion_param,sat_pcos,
                  el_threthod=el_threthod,ex_threshold_v=ex_threshold_v,ex_threshold_v_sigma=ex_threshold_v_sigma,
                  Mw_threshold=Mw_threshold,GF_threshold=GF_threshold,dy_mode=dy_mode,
                X=X,Pk=Pk,Qk=Qk,X_time=X_time,phase_bias=phase_bias,GF_sign=GF_sign,Mw_sign=Mw_sign,slip_sign=slip_sign,dN_sign=dN_sign,sat_num=sat_num,out_age=out_age,f1=f1,f2=f2)
    
    #结果以numpy数组格式保存在指定输出目录下, 若输出目录为空, 则存于nav_result
    try:
        np.save(out_path+'/{}.out'.format(os.path.basename(obs_file)),Out_log,allow_pickle=True)
    except:
        np.save('nav_result/{}.out'.format(os.path.basename(obs_file)),Out_log,allow_pickle=True)