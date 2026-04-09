#多核并行计算处理
import multiprocessing
#包括ppp_yaml
from src.ptk_yaml import *

CORE_NUM=12                                                 #指定进程数量

#以yaml为函数入口, 创建单个进程
def task(PPP_YAML):
    with open(PPP_YAML,"r",encoding='utf-8') as f:
        cfg=yaml.safe_load(f)
        print("sys_index set as multi-GNSS")
        PTK_YAML_GCE(cfg)
    return 

if __name__ == '__main__':
    task_paths=os.listdir('xmls/')
    task_range=[]
    for t in task_paths:
        task_range.append('xmls/'+t)
    print("process list: ",task_paths)    
    with multiprocessing.Pool(processes=CORE_NUM) as pool:        # 创建进程池，指定进程数
        results = pool.map(task, task_range)                # 将任务分配给进程池中的进程
        print(results)