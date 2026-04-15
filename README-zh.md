# Easy4PTK: 纯Python编译的多系统PPP-RTK工具包

[[中]](./README-zh.md) &ensp; [[EN]](./README.md)

Easy4PTK是Easy4PNT工具箱的组成模块，工具箱的其他组成模块在这里给出 (单击可跳转对应工具箱目录)：[[Easy4SPP]](https://github.com/alxanderjiang/Easy4SPP), [[Easy4RTK]](https://github.com/alxanderjiang/Easy4RTK), [[Easy4PPP]](https://github.com/alxanderjiang/Easy4PPP), [[Easy4B2b]](https://github.com/alxanderjiang/Easy4B2b).

## 快速开始
1. 工具包提供一个示例jupyter notebook教程用于快速演示工具包进行实时动态精密单点定位(PPP-RTK)解算, 在使用工具包之前, 确保你已经在环境中正确安装Python发行版本和numpy(矩阵计算), ipykernel(Jupyter notebook运行插件), numba(矩阵求逆加速)和Pyyaml(配置文件读取)库;
2. 由于Github对单个文件上传大小的限制(小于25MB), 工具包将示例数据文件夹"data"和示例结果文件夹"nav_result"压缩为zip文件并上传至网盘([[谷歌网盘]](https://drive.google.com/drive/folders/1tKkHaTQvNHncI_X6PQJJ7abXlAo0ryBO?usp=drive_link), [[蓝奏云盘]](https://wwbwg.lanzouv.com/b01bjc4y2f)). 为了进行对示例数据的PPP-RTK求解, 用户需要首先下载上述网盘链接内的"data.zip"并解压到项目根目录下。示例数据的解算结果以numpy字典数组格式存储到"nav_result.zip"中，可用于直接对示例数据的结果进行可视化;
3. 解压data文件夹后，运行ptk_yaml.ipynb的所有单元格, 用户会在nav_result文件夹(确保有这样一个文件夹)内得到numpy字典数组格式的解算结果。Easy4PTK将以xmls/Easy4PTK.yaml的配置进行解算，并输出给用户以WAB2静态解为大气SSR, ZIM2的动态PPP-RTK结果, 每小时重收敛一次。更多配置细节在配置文件中详细呈现;
4. 我们提供一个对解算结果二进制文件进行可视化的示例jupyter notebook文件, 执行nav_result.ipynb的所有单元格, 用户会获取PPP-RTK收敛曲线(定位误差时序曲线), ZTD时序, STEC散点和码残差等信息的可视化。

## 下载与环境准备
1. 直接从github主页下载工具包的压缩文件“Easy4PPP-main.zip”并解压即可。如果您使用git clone工具，请运行如下命令将Easy4PTK克隆到本地:
```bash
git clone https://github.com/alxanderjiang/Easy4PTK.git
```
2. 从谷歌云盘([[https://drive.google.com/drive/folders/1tKkHaTQvNHncI_X6PQJJ7abXlAo0ryBO?usp=drive_link]](https://drive.google.com/drive/folders/1tKkHaTQvNHncI_X6PQJJ7abXlAo0ryBO?usp=drive_link))或蓝奏云盘([[https://wwbwg.lanzouv.com/b01bjc4y2f]](https://wwbwg.lanzouv.com/b01bjc4y2f))中下载"data.zip"和"nav_result.zip" 压缩包;
3. 解压示例数据文件夹"data.zip"和示例数据结果文件夹"nav_result.zip"至Easy4PTK工具包根目录, 如果是非发行版Linux窗口系统用户(例如Linux服务器), 请在命令行中运行如下命令以解压:

```bash
cd Easy4PTK
unzip data.zip
unzip nav_result
```
3. 确保 numpy、tqdm、ipykernel、numba 和 Pyyaml 在您的 Python 环境中可用。如果没有，请运行以下命令进行安装:

```bash
pip install numpy
pip install tqdm
pip install ipykernel
pip install numba
pip install Pyyaml
```

  numpy和tq​​dm用于支撑核心代码, ipykernel用于支撑Jupyter Notebook的运行。numba用于加速计算(可以通过将所有“numba_inv”函数更改为简单的“inv()”函数来忽略这一运算库)。与Easy4PPP不同，Easy4PTK不支持从具有变量定义的 __main__ 函数运行，仅支持从Yaml配置文件中读取配置进行解算。在安装或使用 numba 时可能会由于缺少scipy库而报错，请运行以下命令进行安装:

```bash
pip install scipy
```
## 基准站模式配置: 基于单站观测的状态域改正数生产
Easy4PTK的基准站模式(Base)用于生成电离层和对流层的状态域改正数(SSR), 工具包支持基准站坐标约束。
1. 将配置文件的'sta_mode'更改为'Base'以设置Easy4PTK的基准站模式：
   ```yaml
   sta_mode: Base
   ```
2. 通常情况下，Base模式不需要重新初始化, 因此reinitial_sec建议设置为0(即每历元均不重新初始化):
   ```yaml
   reinitial_sec: 0
   ```
   同样, Base模式一般静态求解, 配置如下:
   ```yaml
   dy_mode: 'static'
   ```    
4. 如果需要坐标约束，则需要设置基准站坐标'STA_P'和基准站坐标约束方差'STA_Q':
   ```yaml
   STA_P: [4331299.588262134, 567537.6703955207, 4633133.899276633]   # WGS-84, take ZIM2 for example
   STA_Q: [0.01, 0.01, 0.01]                                          # The constraint variances (X,Y,Z) of initial coordinate, expressed in meter.
   ```
5. 如果不需要基准站坐标约束, 将STA_P设置为0或直接使用直接求解的sta_mode: None模式:
   ```yaml
   STA_P: [0, 0, 0]                                                   # No coordinate constraint
   ```
Easy4PTK的基站模式生成的示例SSR可视化结果如下，测试基准站为WAB2，坐标约束关闭: 
<img src=./images/BaseMode_Products.png>

## 流动站模式配置: SSR based PPP-RTK
Easy4PTK的流动站模式用于利用Easy4PTK基准站模式或其他方法生成的同样数据格式的SSR进行PPP-RTK解算。支持电离层和对流层独立或联合约束。支持不完整的卫星SSR约束(即可用SSR数量与流动站实际观测卫星数量不同)。
1. 将配置文件的'sta_mode'更改为'Rove'以设置Easy4PTK的流动站模式：
   ```yaml
   sta_mode: Rove
   ```
2. Easy4PTK的SSR格式与Easy4PNT的解算结果文件格式相同，这意味着您可以轻松地将Easy4PPP或Easy4PTK基准站/直接求解模式生成的结果文件用作SSR数据。在配置文件中设置SSR信息文件路径如下：
   ```yaml
   rtk_info_mat : 'data/rtk_info/wab21540.25o.out.npy'    #Take WAB2-ZIM2 baseline for example
   ```
   对于区域外部大气产品的用户，也可以根据Easy4PNT的解算结果文件格式自行构建SSR数据文件。单个历元内以Python字典格式组织, 键(Key)为“GPS_week”、“GPS_sec”和“STEC”的值(value)是必需的, 多个历元间以列表格式组织, 而"std_STEC"和"azel"是可选的。同时，为了便于Easy4PTK快速从数据文件中找到对应时间的SSR，需要设置SSR的更新频率，如下:
   ```yaml
   t_interval: 30    #Take 30 seconds for example, expressed in seconds.
   ```
3. 单独配置电离层和对流层改正的衰减因子(或约束开关):
   ```yaml
   Qi_scale: 1.0                  #基线质量衰减因子(地理坐标每度衰减, 单位:m, 若为-1, 则不对电离层进行约束)
   Qt_scale: 1.0                  #基线对流层质量衰减因子(每单位基线长度衰减, 单位:m, 若为-1, 则不对对流层进行约束)
   ```
   如果上述尺度因子不设置为-1, 则表示SSR随地理纬度在基线方向的衰减程度。例如, 1.0表示每个地理纬度 SSR 约束方差放大1.0m。
4. 设置电离层SSR松弛因子，表示在电离层SSR中添加加性约束噪声削弱SSR产品质量的效果，单位为TECU: 
   ```yaml
   Qi_init:  2.0                  #初始化质量衰减因子(初始化外加质量松弛因子, 单位:TECU)
   ```
5. 设置电离层SSR使用的最小仰角，以度表示，当前站低于该仰角的卫星将不受电离层SSR的约束:
   ```yaml
   Qi_ele_threshold: 10           #电离层SSR应用最小高度角(单位:deg)
   ```
注意：无论是否模糊度固定，如果使用电离层SSR，则OSB_YES配置应设置为1。 2025年DOY 154的WAB2-ZIM2基线的PPP-RTK结果如下图所示。

<img src=./images/WAB2-ZIM2.png>

## 直接运行源码文件主函数获取单个配置文件的PPP-RTK解
1. 通过在工具包根目录下运行src/ptk_yaml.py的main函数或者在Jupyter Notebook中运行ptk_yaml.ipynb来获取示例数据的求解结果;
2. 等待处理完成后，用户将在ppprtk_result文件夹中得到一个名为"zim21540.25o.out.npy"的numpy数组格式的求解结果文件;
3. 运行nav_result.ipynb，用户将得到如上图所示的PPP-RTK定位结果和其他附属产品的可视化结果。

## 运行多线程并行脚本以获取配置文件夹下所有文件的PPP-RTK解
Easy4PTK原生支持多线程处理，可以显著加快整网求解的速度，削弱Python计算性能带来的相对于C/C++代码的耗时增加。
1. 通过运行multiprocess.py脚本获取示例多线程处理结果。源代码中的宏定义"CORE_NUM"应根据您的设备进行设置。
2. 应根据待多线程处理的所有yaml文件的文件夹目录设置"PATH"。
3. 多线程并行脚本multiprocess.py将自动​​处理PATH目录下的所有配置文件。

## 联系我们
Easy4PTK工具箱的一切内容均由武汉理工大学智能交通系统研究中心/航运学院的蒋卓君，杨泽恩，黄文静和钱闯完成。Easy4PTK目前处于测试阶段，不完善之处敬请谅解。Easy4PTK项目组欢迎一切有关工具箱的建议、意见和漏洞记录反馈，联系方式1162110359@qq.com. 
