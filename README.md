# Easy4PTK

[[中]](./README-zh.md) &ensp; [[EN]](./README.md)

An easily ported Multi-GNSS PPP-RTK Toolbox coded in Python.

This is a part of open-source toolbox Easy4PNT. Other toolboxs of Easy4PNT is listed here (clicked to jump to the target): [[Easy4SPP]](https://github.com/alxanderjiang/Easy4SPP), [[Easy4RTK]](https://github.com/alxanderjiang/Easy4RTK), [[Easy4PPP]](https://github.com/alxanderjiang/Easy4PPP), [[Easy4B2b]](https://github.com/alxanderjiang/Easy4B2b).

## Quick Start
1. We provide a set of example data and a quick-start jupyter notebook tutorial "ptk_yaml.ipynb". Make sure that you have sucessufully installed the Python as well as the numpy (for matrix computation), ipykernel (for running the Jupyter Notebook), numba (for accelerating Python computation) and Pyyaml (for configuration files reading) in your environment.
2. Due to the file size limitation of github (no more than 25MB for a file), we compress the "data" and "nav_result" folders into zip files and uploaded them to the Cloud Drive ([[Google Driver]](https://drive.google.com/drive/folders/1tKkHaTQvNHncI_X6PQJJ7abXlAo0ryBO?usp=drive_link) or [[Lanzou Driver]](https://wwbwg.lanzouv.com/b01bjc4y2f)). In order to try the provided example of GCE PPP-RTK solutions, you need first download and unzip the "data.zip" into the "data" folder in the project path. The example results for visualization is stored in 'nav_result.zip'. The above zip files are necessary to run Easy4PTK. 
3. After unzipping the "data" folder, run all the blocks of "ptk_yaml.ipynb", you all get an Easy4PNT solution log file in form of ".npy". The running script is following the configuration of xmls/Easy4PTK.yaml. Use WAB2 static PPP products as constraints, get ZIM2 PPP-RTK kinematic solutions, reinitialize per 1 hour. The details of configuration is shown in xmls/Easy4PTK.yaml.
4. We provided an example of visualizing the solution log file. Run all the blocks of nav_result.ipynb, you can get figures about the PPP-RTK convergence curve, the STEC scatter and the residuls scatter plot.

## Downloading and preperations
1. Download the **zip pakeage directly** or using git clone by running the following commend:
```bash
git clone https://github.com/alxanderjiang/Easy4PTK.git
```
2. Download the "data.zip" and "nav_result.zip" files from Google Drive ([[https://drive.google.com/drive/folders/1tKkHaTQvNHncI_X6PQJJ7abXlAo0ryBO?usp=drive_link]](https://drive.google.com/drive/folders/1tKkHaTQvNHncI_X6PQJJ7abXlAo0ryBO?usp=drive_link)) or LanZou Drive ([[https://wwbwg.lanzouv.com/b01bjc4y2f]](https://wwbwg.lanzouv.com/b01bjc4y2f)) . 
3. Unzip the sample data folders: data.zip and nav_result.zip to the same path of Easy4PTK. If linux but no GUI, please run the following commends:

```bash
cd Easy4PTK
unzip data.zip
unzip nav_result
```
3. Ensure that the numpy, tqdm, ipykernel, numba and Pyyaml is available in your Python environment. If not, please run the following commends to install:

```bash
pip install numpy
pip install tqdm
pip install ipykernel
pip install numba
pip install Pyyaml
```

  numpy and tqdm is used in the core codes while ipykernel is necessary to run Jupyter Notebook tutorials. numba is used to accelerate the computation (this can be ignored by change all the "numba_inv" function to simple "inv()" function). Unlike the Easy4PPP, Easy4PTK does not support running from a __main__ function with variables definition, only Yaml Configuration file is supported.
Some problems may happen when install or use numba because of laking the library scipy, please install it by running the following commends:

```bash
pip install scipy
```
## Base Mode Configuration: SSR Generation
The base mode of Easy4PTK is used to generate ionospheric and tropspheric state-space representation (SSR). Coordinate constraint are supported.
1. Change the 'sta_mode' of configuration file into 'Base' to set base mode of Easy4PTK as follows:
   ```yaml
   sta_mode: Base
   ```
2. Usually, the base mode does not need reinitialization so that the 'reinitial_sec' is recommended to set as:
   ```yaml
   reinitial_sec: 0
   ```
   Also, the base mode is usually set as static:
   ```yaml
   dy_mode: 'static'
   ```    
4. If coordinate constraint is needed, the 'STA_P' and 'STA_Q' is needed to set as:
   ```yaml
   STA_P: [4331299.588262134, 567537.6703955207, 4633133.899276633]   # WGS-84, take ZIM2 for example
   STA_Q: [0.01, 0.01, 0.01]                                          # The constraint variances (X,Y,Z) of initial coordinate, expressed in meter.
   ```
5. If coordinate constraint is not needed, the STA_P shold be set as:
   ```yaml
   STA_P: [0, 0, 0]                                                   # No coordinate constraint
   ```
The generated SSR by base mode of Easy4PTK is expressed as follows, the test base station is WAB2, coordinate constraint is shut off:
<img src=./images/BaseMode_Products.png>

## Rove Mode Configuration: SSR based PPP-RTK
The rove mode of Easy4PTK is used to ultilize PPP-RTK based on the generated SSR from base mode or other methods. Ionospheric and tropospheric independent or joint constraints are supported. Incomplete satellites' SSRs constraints are supported.
1. Change the 'sta_mode' of configuration file into 'Rove' to set base mode of Easy4PTK as follows:
   ```yaml
   sta_mode: Rove
   ```
2. The SSR format of Easy4PTK is as same as the solution log format of Easy4PNT, which means you can easily uses the solution logs generated by Easy4PPP or base mode as SSR data. set the file path of SSR information mat as follows:
   ```yaml
   rtk_info_mat : 'data/rtk_info/wab21540.25o.out.npy'    #Take WAB2-ZIM2 baseline for example
   ```
   For users of regional atmospheric products, SSR information can also be constructed by themselves according to the solution logs format of Easy4PNT. the values of keys: "GPS_week", "GPS_sec" and "STEC" are necessary while  "std_STEC" and "azel" are optional. At the same time, the update frequency of SSR needs to be set in order to quickly find the target time SSR as follows:
   ```yaml
   t_interval: 30    #Take 30 seconds for example, expressed in seconds.
   ```
3. Separately set the switch used for ionospheric and tropospheric corrections as follows:
   ```yaml
   Qi_scale: 1.0                  #基线质量衰减因子(地理坐标每度衰减, 单位:m, 若为-1, 则不对电离层进行约束)
   Qt_scale: 1.0                  #基线对流层质量衰减因子(每单位基线长度衰减, 单位:m, 若为-1, 则不对对流层进行约束)
   ```
   If the scale factors are not set to −1, they represent the degree of attenuation of SSR with geographical latitude in the baseline direction. For example, 1.0 indicates that the SSR constraint variance is amplified by 1.0 m for each geographical latitude.
4. The ionospheric SSR relaxation factor is set to represent the effect of adding an additive constraint noise to the ionospheric SSR to weaken the quality of the SSR product, and the unit is TECU:
   ```yaml
   Qi_init:  2.0                  #初始化质量衰减因子(初始化外加质量松弛因子, 单位:TECU)
   ```
5. The minimum elevation angle used by the ionospheric SSR is set, expressed in degrees, and the satellite with the current station below the elevation angle will not be constrained by the ionospheric state:
   ```yaml
   Qi_ele_threshold: 10           #电离层SSR应用最小高度角(单位:deg)
   ```
Attention: No matter AR or not, the OSB_YES configuration shold be set as 1 if ionospheric SSR is used. The PPP-RTK results of WAB2-ZIM2 baseline in DOY 154, 2025 is shown in Figure as follows.

<img src=./images/WAB2-ZIM2.png>

## Running from the python source code to get a singgle solution
1. Get the sample results by running the main function of src/ptk_yaml.py or running the ptk_yaml.ipynb in the Jupyter Notebook.
2. After processing, you will get a solution log named as "zim21540.25o.out.npy" in format of numpy array in the folder: ppprtk_result.
3. Running the nav_result.ipynb, you will get pictures of PPP-RTK results and products as shown in the above figures.

## Running the multiprocessing script to get a folder's solutions
Easy4PTK natively supports multi-threaded processing, which can significantly speed up the speed of the entire network solution to weaken the time-consuming enhancement caused by Python computing performance.
1. Get the sample multi-processing by running the multiprocess.py. The "CORE_NUM" shold be set according to your devices.
2. The "PATH" shold be set according to your yaml files' folder.
3. The multiprocessing script "multiprocess.py" will automatically processes all configuration files under PATH.

## Contact Authors
All the libaries and softwares in this toolbox are coded by Zhuojun Jiang, Zeen Yang, Wenjing Huang, Chuang Qian from Wuhan University of Technology. Any commends or bug reports are welcomed by sending email to 1162110359@qq.com. 
