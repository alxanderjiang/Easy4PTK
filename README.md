
# Easy4PTK

[[中]](./README-zh.md) &ensp; [[EN]](./README.md)

Easy4PTK: An easily ported Multi-GNSS PPP-RTK Toolbox coded in Python.

This is a part of open-source toolbox Easy4PNT. Other toolboxs of Easy4PNT is listed here (clicked to jump to the target): [[Easy4SPP]](https://github.com/alxanderjiang/Easy4SPP), [[Easy4RTK]](https://github.com/alxanderjiang/Easy4RTK), [[Easy4PPP]](https://github.com/alxanderjiang/Easy4PPP), [[Easy4B2b]](https://github.com/alxanderjiang/Easy4B2b).

## Quick Start
1. We provide a set of example data and a quick-start jupyter notebook tutorial "ptk_yaml.ipynb". Make sure that you have sucessufully installed the Python as well as the numpy (for matrix computation), ipykernel (for running the Jupyter Notebook), numba (for accelerating Python computation) and Pyyaml (for configuration files reading) in your environment.
2. Due to the file size limitation of github (no more than 25MB for a file), we compress the "data" and "nav_result" folders into zip files and uploaded them to the Cloud Drive ([[Google Driver]](https://drive.google.com/drive/folders/1tKkHaTQvNHncI_X6PQJJ7abXlAo0ryBO?usp=drive_link) or [[Lanzou Driver]](https://wwbwg.lanzouv.com/b01bjc4y2f)). In order to try the provided example of GCE PPP-RTK precise point position (PPP), you need first download and unzip the "data.zip" into the "data" folder in the project path. The example results for visualization is stored in 'nav_result.zip'. The above zip files are necessary to run Easy4PTK. 
3. After unzipping the "data" folder, run all the blocks of "ptk_yaml.ipynb", you all get an Easy4PNT solution log file in form of ".npy". The running script is following the configuration of xmls/Easy4PTK.yaml. Use WAB2 static PPP products as constraints, get ZIM2 PPP-RTK kinematic solutions, reinitialize per 1 hour. The details of configuration is shown in xmls/Easy4PTK.yaml.
4. We provided an example of visualizing the solution log file. Run all the blocks of nav_result.ipynb, you can get figures about the PPP-RTK convergence curve, the STEC scatter and the residuls scatter plot.

## Downloading and preperations
1. Download the **zip pakeage directly** or using git clone by running the following commend:
```bash
git clone https://github.com/alxanderjiang/Easy4PTK.git
```
2. Download the "data.zip" and "nav_result.zip" files from Google Drive ([[https://drive.google.com/drive/folders/1tKkHaTQvNHncI_X6PQJJ7abXlAo0ryBO?usp=drive_link]](https://drive.google.com/drive/folders/1tKkHaTQvNHncI_X6PQJJ7abXlAo0ryBO?usp=drive_link)) or LanZou Drive ([[https://wwbwg.lanzouv.com/b01bjc4y2f]](https://wwbwg.lanzouv.com/b01bjc4y2f)) . 
3. Unzip the sample data folders: data.zip and nav_result.zip to the same path of Easy4PPP. If linux but no GUI, please run the following commends:
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
numpy and tqdm is used in the core codes while ipykernel is necessary to run Jupyter Notebook tutorials. Unlike the Easy4PPP, Easy4PTK does not support running from a __main__ function with variables definition, only Yaml Configuration file is supported.
Some problems may happen when install or use numba because of laking the library scipy, please install it by running the following commends:
```bash
pip install scipy
```


