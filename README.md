
# Easy4PTK

[[中]](./README-zh.md) &ensp; [[EN]](./README.md)

An easily ported Multi-GNSS PPP-RTK Toolbox coded in Python.

This is a part of open-source toolbox Easy4PNT. Other toolboxs of Easy4PNT is listed here (clicked to jump to the target): [[Easy4SPP]](https://github.com/alxanderjiang/Easy4SPP), [[Easy4RTK]](https://github.com/alxanderjiang/Easy4RTK), [[Easy4PPP]](https://github.com/alxanderjiang/Easy4PPP), [[Easy4B2b]](https://github.com/alxanderjiang/Easy4B2b).

## Quick Start
1. We provide a set of example data and a quick-start jupyter notebook tutorial "ptk_yaml.ipynb". Make sure that you have sucessufully installed the Python as well as the numpy (for matrix computation), ipykernel (for running the Jupyter Notebook), numba (for accelerating Python computation) and Pyyaml (for configuration files reading) in your environment.
2. Due to the file size limitation of github (no more than 25MB for a file), we compress the "data" and "nav_result" folders into zip files and uploaded them to the Cloud Drive ([[Google Driver]](https://drive.google.com/drive/folders/1tKkHaTQvNHncI_X6PQJJ7abXlAo0ryBO?usp=drive_link) or [[Lanzou Driver]](https://wwbwg.lanzouv.com/b01bjc4y2f)). In order to try the provided example of GCE PPP-RTK solutions, you need first download and unzip the "data.zip" into the "data" folder in the project path. The example results for visualization is stored in 'nav_result.zip'. The above zip files are necessary to run Easy4PTK. 
4. After unzipping the "data" folder, run all the blocks of "ptk_yaml.ipynb", you all get an Easy4PNT solution log file in form of ".npy". The running script is following the configuration of xmls/Easy4PTK.yaml. Use WAB2 static PPP products as constraints, get ZIM2 PPP-RTK kinematic solutions, reinitialize per 1 hour. The details of configuration is shown in xmls/Easy4PTK.yaml.
5. We provided an example of visualizing the solution log file. Run all the blocks of nav_result.ipynb, you can get figures about the PPP-RTK convergence curve, the STEC scatter and the residuls scatter plot.

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
## Base Mode: SSR Generation
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
Figure <img src=./images/BaseMode_Products.png>
## Rove Mode: SSR based PPP-RTK
The rove mode of Easy4PTK is used to ultilize PPP-RTK based on the generated SSR from base mode or other methods. 
