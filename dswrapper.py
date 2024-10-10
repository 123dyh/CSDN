import numpy
from ctypes import *
import sys

try:     
   dll = WinDLL('D:/Python/1.Python/Pycharm2021/yolov5-master/video_check/ruanjian/ruanjian/DalsaApi.dll')
   #dll = WinDLL('TEST.dll')
except OSError:
   print('Cannot find DalsaApi.dll.')

class MyFrame(Structure):
    _fields_ = [
        ('status', c_int),                      # The return state of the image
        ('image_buf', c_void_p),                # Image buff address
        ('width', c_int),                       # Image width
        ('height', c_int),                      # Image height
        ('Callback', c_int),                    # Image Callback_number
        ('pixel_format', c_int),                # Image PixFormat
        ('image_size', c_int),                  # Image data size, Including frame information
        ('frame_id', c_ulonglong),              # The frame id of the image
        ('timestamp', c_ulonglong),             # Time stamp of image
        ('buf_id', c_ulonglong),                # Image buff ID
        ('reserved',  c_int),                   # Reserved
    ]

    def __str__(self):
        return "MyFrame\n%s" % "\n".join("%s:\t%s" % (n, getattr(self, n[0])) for n in self._fields_)
class RawImage:
    def __init__(self, frame_data):
        self.frame_data = frame_data
        #print(self.frame_data)

        if self.frame_data.image_buf is not None:
            self.__image_array = string_at(self.frame_data.image_buf, self.frame_data.image_size)
            #print(self.frame_data.image_buf)
            #print(self.frame_data.image_size)
        else:
            self.__image_array = (c_ubyte * self.frame_data.image_size)()
            self.frame_data.image_buf = addressof(self.__image_array)
            print(self.__image_array)

    def get_numpy_array(self):
        """
        :brief      Return data as a numpy.Array type with dimension Image.height * Image.width
        :return:    numpy.Array objects
        """
        image_size = self.frame_data.width * self.frame_data.height
        #print("image_size")
        #print(image_size)
        #print("width")
        #print(self.frame_data.width)
        #print("height")
        #print(self.frame_data.height)
        #print("callback number")
        #print(self.frame_data.Callback)
        
        image_np = numpy.frombuffer(self.__image_array, dtype=numpy.ubyte, count=image_size).reshape(self.frame_data.height, self.frame_data.width)
  
        return image_np

if hasattr(dll, "ViOpenDalsaDeviceFG"):
    def ViOpenDalsaDeviceFG(nIndex,ResourceIndex,ccfName):
        """
        :biref   open camera link camera
        :param  nIndex:       
        :param  SeverName:      
        :param  ResourceIndex:       
        :param  ccfName:         
        :return: status         
      
        """
        status = dll.ViOpenDalsaDeviceFG(nIndex,ResourceIndex,ccfName)
        return status

if hasattr(dll, "ViOpenDalsaDeviceGX"):
    def ViOpenDalsaDeviceGX(nIndex):
        """
        :biref   open  GX camera
        :param  nIndex:               
        :return: status         
      
        """
        status = dll.ViOpenDalsaDeviceGX(nIndex)
        return status

if hasattr(dll, "ViGetImageData"):
    def get_imaging(nIndex,nCameraType):
        """
        :biref  Get current image buf data
        :param  nIndex: the of camera index
        :param  nCameraType: 0,Gige;1,CameraLink             
        :return: status         
      
        """
        payload_size = dll.ViGetWidth(nIndex,nCameraType)*dll.ViGetHeight(nIndex,nCameraType)
        frame_data = MyFrame()
        frame_data.image_size = payload_size
        state = dll.ViGetImageData(nIndex,byref(frame_data),nCameraType)
        raw_image = RawImage(frame_data)
        state = dll.ViDeleteImageData(nIndex,byref(frame_data),nCameraType)
        return raw_image

if hasattr(dll, "ViCloseDalsaDeviceGX"):
    def ViCloseDalsaDeviceGX(nIndex):
        """
        :biref   close  gige camera
        :param  nIndex:             
        :return: status         
      
        """
        state = dll.ViCloseDalsaDeviceGX(nIndex)
        return state

if hasattr(dll, "ViStartAcqSingleFrame"):
    def ViStartAcqSingleFrame(nIndex,nCameraType):
        """
        :biref   snap
        :param  nIndex:             
        :return: status         
      
        """
        state = dll.ViStartAcqSingleFrame(nIndex,nCameraType);
        return state

if hasattr(dll, "ViCloseDalsaDeviceFG"):
    def ViCloseDalsaDeviceFG(nIndex):
        """
        :biref   close  cameralink(cameralink-HS) camera
        :param  nIndex:             
        :return: status         
      
        """
        state = dll.ViCloseDalsaDeviceFG(nIndex)
        return state

if hasattr(dll, "ViGetWidth"):
    def ViGetWidth(nIndex,nCameraType):
        """
        :biref  get image width
        :param  nIndex:       
        :param  nCameraType:         
        :return: status         
      
        """
        state = dll.ViGetWidth(nIndex,nCameraType)
        return state


if hasattr(dll, "ViGetHeight"):
    def ViGetHeight(nIndex,nCameraType):
        """
        :biref  get image height
        :param  nIndex:       
        :param  nCameraType:         
        :return: status         
      
        """
        status = dll.ViGetHeight(nIndex,nCameraType)
        return status
      
if hasattr(dll, "ViCallback_number"):
    def ViCallback_number(nIndex,nCameraType):
        """
        :biref  get image callback_number
        :param  nIndex:       
        :param  nCameraType:         
        :return: status         
      
        """
        status = dll.ViCallback_number(nIndex,nCameraType)
        return status

if hasattr(dll, "ViUpdateDeviceList"):
    def ViUpdateDeviceList(IsShowInfo):
        """
        :biref  updateDeviceList
        :param  IsShowInfo:                
        :return: count         
      
        """
        count = dll.ViUpdateDeviceList(IsShowInfo)
        return count


if hasattr(dll, "ViGetExposureTime"):
    def ViGetExposureTime(nIndex,nCameraType):
        """
        :biref  get Exposure
        :param  nIndex:                
        :return: status         
      
        """
        dexp_c=c_double()
        dexp_c.value=0.0
        dll.ViGetExposureTime(nIndex,byref(dexp_c),nCameraType)
        return dexp_c.value      
      

if hasattr(dll, "ViGetGain"):
    def ViGetGain(nIndex,nCameraType):
        """
        :biref  get Gain
        :param  nIndex:               
        :return: status         
      
        """
        dgain_c=c_double()
        dgain_c.value=0.0
        dll.ViGetGain(nIndex,byref(dgain_c),nCameraType)
        return dgain_c.value

if hasattr(dll, "ViSetGain"):
    def ViSetGain(nIndex,arg,nCameraType):
        """
        :biref  set Gain
        :param  nIndex:       
        :param  arg:               
        :return: status         
      
        """
        gain_c = c_double(arg)
        state = dll.ViSetGain(nIndex,gain_c,nCameraType)
        return state

if hasattr(dll, "ViSetExposureTime"):
    def ViSetExposureTime(nIndex,arg,nCameraType):
        """
        :biref  set Exposure
        :param  nIndex:       
        :param  arg:               
        :return: status         
      
        """
        exp_c = c_double(arg)
        status = dll.ViSetExposureTime(nIndex,exp_c,nCameraType)
        return status

if hasattr(dll, "ViSetTriggerMode"):
    def ViSetTriggerMode(nIndex,arg):
        """
        :biref   set trigger mode
        :param   nIndex:
        :param   value            
        :return: status         
      
        """
        mode_c = c_double(arg)
        status = dll.ViSetTriggerMode(nIndex,mode_c)
        return status
      
