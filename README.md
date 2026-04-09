
# Easy4PTK

[[中]](./README-zh.md) &ensp; [[EN]](./README.md)

Easy4PTK: An easily ported Multi-GNSS PPP-RTK Toolbox coded in Python.

This is a part of open-source toolbox Easy4PNT. Other toolboxs of Easy4PNT is listed here (clicked to jump to the target): [[Easy4SPP]](https://github.com/alxanderjiang/Easy4SPP), Easy4RTK, [[Easy4PPP]](https://github.com/alxanderjiang/Easy4PPP), Easy4B2b.

## Quick Start
1. We provide a set of example data and a quick-start jupyter notebook tutorial "ptk_yaml.ipynb". Make sure that you have sucessufully installed the Python as well as the numpy (for matrix computation), ipykernel (for running the Jupyter Notebook) and numba (for accelerating Python computation) in your environment.
2. Due to the file size limitation of github (no more than 25MB for a file), we compress the "data" and "nav_result" folders into zip files and uploaded them to the Cloud Drive ([[Google Driver]](https://drive.google.com/drive/folders/1tKkHaTQvNHncI_X6PQJJ7abXlAo0ryBO?usp=drive_link) or [[Lanzou Driver]](https://wwbwg.lanzouv.com/b01bjc4y2f)). In order to try the provided example of GCE PPP-RTK precise point position (PPP), you need first download and unzip the "data.zip" into the "data" folder in the project path. The example results for visualization is stored in 'nav_result.zip'. The above zip files are necessary to run Easy4PTK. 
3. After unzipping the "data" folder, run all the blocks of "ptk_yaml.ipynb", you all get an Easy4PNT solution log file in form of ".npy". The running script is following the configuration of xmls/Easy4PTK.yaml. Use WAB2 static PPP products as constraints, get ZIM2 PPP-RTK kinematic solutions, reinitialize per 1 hour. The details of configuration is shown in xmls/Easy4PTK.yaml.
4. We provided an example of visualizing the solution log file. Run all the blocks of nav_result.ipynb, you can get figures about the PPP-RTK convergence curve, the STEC scatter and the residuls scatter plot.


