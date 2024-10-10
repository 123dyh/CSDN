import dswrapper as ds
from PIL import Image
import time
from ctypes import *

# Step1 : Display device list
num = ds.ViUpdateDeviceList(True)  # True: Display ; False: No Display
if 0 > num:
    ds.sys.exit(open_index)

# Step2 : Select
cam_type=0
open_index=0
while True:
    cam_type = int(input("\nPlease select the type of camera you want to open([0] Gige [1] CameraLink): "))
    if cam_type != 0 and cam_type != 1:
        print("\nPlease enter correct selector!")
    else:
        open_index = int(input("\nPlease select the index of device: "))
        break

# Step3 : Open the specify device
if cam_type == 0:
    # open gige camera
    num = ds.ViOpenDalsaDeviceGX(open_index)
else:
    # open cameralink camera
    str1 = r'C:\Users\Administrator\Desktop\ccf\P4CC_MONO.ccf'
    ccffilepath = create_string_buffer(bytes(str1,encoding='utf8'))
    num = ds.ViOpenDalsaDeviceFG(open_index,0,ccffilepath)



# Step4 : Get parameters of device

payload_size = ds.ViGetWidth(open_index,cam_type)*ds.ViGetHeight(open_index,cam_type)

width = ds.ViGetWidth(open_index,cam_type)
print("width=%d"%(width))

height = ds.ViGetHeight(open_index,cam_type)
print("height=%d"%(height))

#Callback= ds.ViCallback_number(open_index,cam_type)
#print("height=%d"%(Callback_numberr))

gain=ds.ViGetGain(open_index,cam_type)
print("gain=%f"%(gain))

epx = ds.ViGetExposureTime(open_index,cam_type)
print("exp=%f"%(epx))

# Step5 : Set parameters of device
status=ds.ViSetGain(open_index,3.0,cam_type)
if status is False:
    print("Setting gain failed.")

status=ds.ViSetExposureTime(open_index,50,cam_type)
if status is False:
    print("Setting exposureTime failed.")
        
gain=ds.ViGetGain(open_index,cam_type)
print("gain=%f"%(gain))

epx = ds.ViGetExposureTime(open_index,cam_type)
print("exp=%f"%(epx))

# Step6 : Grab
FrameCount=10
for i in range(FrameCount):
    num = ds.ViStartAcqSingleFrame(open_index,cam_type)
    time.sleep(0.001)
    # Step7 : Get Image Buffer
    raw_image = ds.get_imaging(open_index,cam_type)
    if raw_image is None:
        print("Getting image failed.")

    # Step8 : Show and Save the Image
    numpy_image = raw_image.get_numpy_array()
    if numpy_image is None:
        print(" image failed.")
    img = Image.fromarray(numpy_image, 'L')
    #img.show()
    #Callback= ds.ViCallback_number(open_index,cam_type)
    #print("height=%d"%(Callback))
    img.save("{}.jpg".format(time.time()))
    del raw_image
    del img
    

# Step 10 : Close the specify device
if cam_type == 0:
    # open gige camera
    num = ds.ViCloseDalsaDeviceGX(open_index)
else:
    # open cameralink camera
    num = ds.ViCloseDalsaDeviceFG(open_index)

