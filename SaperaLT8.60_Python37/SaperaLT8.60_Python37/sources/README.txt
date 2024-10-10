********************************************************

      Windows 环境 Dalsa python 安装说明
    
                  2020-08-13

********************************************************


Python3.7 DalsaApi 安装步骤  
=============================

1.安装 python3.7

  (1)从python官网下载用于Windows(x86_64)系统的python3.7安装包并执行安装。
    
     下载网址: https://www.python.org/downloads/windows/

  (2) 在系统环境变量中添加python.exe的路径。

  (3) 在安装目录下，拷贝python.exe，命名python37.exe
 
2.安装 pip 工具

  (1) 打开网址 https://pip.pypa.io/en/stable/installing/ ，下载 get-pip.py.

  (2) 通过CMD打开DOS命令窗口，切换路径到get-pip.py所在目录。

  (3) 在DOS命令窗口中输入以下命令完成pip安装。 
    
      python get-pip.py

  (4) 在系统环境变量中添加pip.exe的路径。

3.安装 numpy 库
 
  在DOS命令窗口中输入以下命令。

      pip install numpy


注意：
=============================

  （1）示例程序可能依赖第三方库（例如 PIL），请自行安装(pip install pillow)。

  （2）Python示例程序须同dswrapper.py放置于同一目录，DalsaApi.dll放在System32目录下。

  （3）Sapera LT SDK:8.60.00.2120

  （4）系统版本：win10  64bit
  


