import numpy as np
from math import *

#LD分解
def get_LD(Qaa):
    A=np.array(Qaa)
    w=A.shape[0]    #待固定模糊度
    L = np.zeros((w,w)) #
    D = np.zeros((w,w))
    try:    
        w_list=list(range(w))
        w_list.reverse()
        for i in w_list:
            if(A[i,i]<0.0):
                A[i,i]=1.0
                #return False
            D[i,i]=A[i,i]
            a=sqrt(D[i,i])
            for j in range(0,i+1):
                L[j,i]=A[j,i]/a
            for j in range(0,i): 
                for k in range(0,j+1):
                    A[k,j]=A[k,j]-L[k,i]*L[j,i]
            for j in range(0,i+1):
                L[j,i]=L[j,i]/L[i,i]
    except:
        return L,D
    return L,D


#整数高斯变换
def LAMBDA_guass(L,Z,i,j):
    #原数组拷贝
    L_=np.array(L)
    Z_=np.array(Z)
    #原数组长度
    n=L_.shape[0]
    #整数因子
    mu=int(round(L[j,i]))
    if (mu!=0):    
        for k in range(i,n):
            L_[j,k]=L_[j,k]-float(mu*L_[i,k])
        for k in range(0,n):
            Z_[j,k]=Z_[j,k]-float(mu*Z_[i,k])
    return L_,Z_

#permutations
def LAMBDA_perm(L, D, j, del_, Z):
    #待固定模糊度宽度
    n=np.array(L).shape[0]
    #原数组拷贝
    L_=np.array(L)
    D_=np.array(D)
    Z_=np.array(Z,dtype=int)
    #
    eta=D_[j,j]/del_
    lam=D_[j+1,j+1]*L_[j,j+1]/del_
    D_[j,j]=eta*D_[j+1,j+1]
    D_[j+1,j+1]=del_
    for k in range(0,j): 
        a0=L_[k,j] 
        a1=L_[k,j+1]
        L_[k,j]=-L_[j,j+1]*a0+a1
        L_[k,j+1]=eta*a0+lam*a1
    L_[j,j+1]=lam
    for k in range(j+2,n):
        temp=L_[j+1,k]
        L_[j+1,k]=L_[j,k]
        L_[j,k]=temp
    for k in range(0,n):
        temp=Z_[j+1,k]
        Z_[j+1,k]=Z_[j,k]
        Z_[j,k]=temp

    return L_,D_,Z_
#LAMBDA降相关
def LAMBDA_reduction(L,D,Z):
    #原数组长度
    n=np.array(L).shape[0]
    #拷贝原数组
    L_=np.array(L)
    D_=np.array(D)
    Z_=np.array(Z)
    #初始化参数
    del_=0.0
    j=n-2
    k=n-2
    #循环降相关
    while (j>=0): 
        if (j<=k):
            for i in range(j+1,n):
                L_,Z_=LAMBDA_guass(L_,Z_,i,j)
        del_=D_[j,j]+L_[j,j+1]*L_[j,j+1]*D_[j+1,j+1]
        if (del_+1E-6<D_[j+1,j+1]):
            L_,D_,Z_=LAMBDA_perm(L_,D_,j,del_,Z_)
            k=j
            j=n-2
        else:
            j=j-1
    return L_,D_,Z_


def SGN(x):
    if(x<=0.0):
        return -1
    else:
        return 1
def LAMBDA_search(m,L,D,zs,loopmax=10000):
    #MLAMBDA法搜索
    #注意: 在Python代码中, D阵以对角二维矩阵呈现, 较LAMBDA C++代码有不同
    #待固定模糊度宽度
    n=np.array(L).shape[0]
    newdist=0.0     #每次搜索的新距离
    maxdist=1E99    #当前搜索到的最大距离
    nn=0
    imax=0
    S=np.zeros((n,n))
    dist=np.zeros((n,1))
    zb=np.zeros((n,1))
    z=np.zeros((n,1))
    step=np.zeros((n,1))
    #生成结果数组
    zn=np.zeros((m,n))
    s=np.zeros(m)

    k=n-1
    dist[k][0]=0.0
    zb[k][0]=zs[k][0]
    z[k][0]=round(zb[k][0])
    y=zb[k][0]-z[k][0]
    step[k][0]=SGN(y)

    for c in range(0,loopmax):
        newdist=dist[k][0]+y*y/D[k][k]
        if newdist<maxdist:
            if(k!=0):
                k=k-1
                dist[k][0]=newdist
                for i in range(0,k+1):
                    S[i,k]=S[i,k+1]+(z[k+1][0]-zb[k+1][0])*L[i,k+1]
                zb[k][0]=zs[k][0]+S[k,k]
                z[k][0]=round(zb[k][0])
                y=zb[k][0]-z[k][0]
                step[k][0]=SGN(y)
            else:
                if(nn<m):
                    if(nn==0 or newdist>s[imax]):
                        imax=nn
                    for i in range(0,n):
                        zn[nn,i]=z[i][0]
                    s[nn]=newdist
                    nn=nn+1
                else:
                    if(newdist<s[imax]):
                        for i in range(0,n):
                            zn[imax,i]=z[i][0]
                        s[imax]=newdist
                        imax=0
                        for i in range(0,m):
                            if(s[imax]<s[i]):
                                imax=i
                    maxdist=s[imax]
                z[0][0]=z[0][0]+step[0][0]
                y=zb[0][0]-z[0][0]
                step[0][0]=-step[0][0]-SGN(step[0][0])
        else:
            if(k==n-1):
                break
            else:
                k=k+1
                z[k][0]=z[k][0]+step[k][0]
                y=zb[k][0]-z[k][0]
                step[k][0]=-step[k][0]-SGN(step[k][0])
            
    for i in range(0,m-1):
        for j in range(i+1,m):
            if(s[i]<s[j]):
                continue
            temp=s[i]
            s[i]=s[j]
            s[j]=temp
            for k in range(0,n):
                temp=zn[j,k]
                zn[j,k]=zn[i,k]
                zn[i,k]=temp
    if(c>loopmax):
        print("search loop count overflow")
        return-1
    return zn,s

def LAMBDA_FIX(X_float,P_float,space=3,loopmax=10000):
    L,D=get_LD(P_float)
    Z=np.eye(L.shape[0])#Z变换矩阵初始化
    L,D,Z=LAMBDA_reduction(L,D,Z)#降相关
    #整数Z变换
    z=Z.dot(X_float)
    Qzz=Z.dot(P_float).dot(Z.T)
    #MLAMBDA搜索
    E,s=LAMBDA_search(space,L,D,z,loopmax=loopmax)
    #逆Z变换恢复原始整数特
    F=np.linalg.inv(Z).dot(E.T).T
    ratio_no_Q=(F[1]-X_float.T).dot((F[1]-X_float.T).T)/(F[0]-X_float.T).dot((F[0]-X_float.T).T)
    d1=(F[1]-X_float.T).dot(P_float).dot((F[1]-X_float.T).T)
    d0=(F[0]-X_float.T).dot(P_float).dot((F[0]-X_float.T).T)
    ratio_with_Q=d1/d0

    return [ratio_with_Q,ratio_no_Q],[d0,d1],F[0]

def PAR_Search(Xfloat_N12,Pfloat_N12,id_use,ratio_threshold=2.0):

    #进行方差排序
    P_id_old=[]
    for id in id_use:
        P_id_old.append(float(Pfloat_N12[id][id]))
    P_id_rank=sorted(P_id_old,reverse=True)
    #针对排序的方差列表进行迭代
    use_id_del=[]
    del_id=0
    sub_use_id=id_use.copy()
    while len(sub_use_id)>3:
        #重置部分模糊度状态搜索情况
        sub_X_float_N12_SD=[]#部分模糊度子集
        sub_P_float_N12_SD=[]#部分模糊度子集方差#删除当前最大方差索引
        sub_use_id.remove(id_use[P_id_old.index(P_id_rank[del_id])])
        for id in sub_use_id:
            sub_X_float_N12_SD.append(Xfloat_N12[id])
            sub_P_temp=[]
            for jid in sub_use_id:
                sub_P_temp.append(Pfloat_N12[id][jid])
            sub_P_float_N12_SD.append(sub_P_temp.copy())
        try:
            ratios,ds,N12_Fix=LAMBDA_FIX(np.array(sub_X_float_N12_SD).reshape((len(sub_X_float_N12_SD),1)),sub_P_float_N12_SD)
        except:
            ratios=[0.0,0.0]
        #固定成功
        if(ratios[0]>ratio_threshold or ratios[1]>ratio_threshold):
            # global fix_counts
            # fix_counts+=1
            return sub_use_id,ratios   
        #若未固定成功则进行下一次删除
        del_id+=1
    return False