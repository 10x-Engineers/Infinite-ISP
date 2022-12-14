import numpy as np
from modules.crop import Crop
import util.utils as utils

class Scale: 
 
    def __init__(self, img, sensor_info, parm_sca):
        self.img = img
        self.enable = parm_sca['isEnable']
        self.sensor_info = sensor_info
        self.parm_sca = parm_sca
    
    def execute(self):
        
        print('Scale = ' + str(self.enable))
            
        if self.enable:
            scaled_img = self.apply_scaling()
            return scaled_img
        else:
            return self.img

    def apply_scaling(self):
        
        #get params
        self.get_scaling_params()
        
        #check if no change in size
        if self.old_size==self.new_size:
            print('   - Output size is the same as input size.') if self.is_debug else None
            return self.img
        
        if self.img.dtype == "float32":
            scaled_img = np.empty((self.new_size[0], self.new_size[1], 3), dtype="float32")
        else:
            scaled_img = np.empty((self.new_size[0], self.new_size[1], 3), dtype="uint8")

        #Loop over each channel to resize the image
        for i in range(3):
            
            ch_arr = self.img[:,:,i]
            scale_2d = Scale_2D(ch_arr, self.sensor_info, self.parm_sca)
            scaled_ch = scale_2d.execute()

            # If input size is invalid, the Scale_2D class returns the original image.          
            if scaled_ch.shape==self.old_size:
                return self.img
            else:    
                scaled_img[:,:,i] = scaled_ch                
            
            # Because each channel is scaled in the same way, the isDeug flag is turned 
            # off after the first channel has been scaled.    
            self.parm_sca["isDebug"]= False

        # convert uint16 img to uint8   
        # if self.img.dtype == "float32":
        #     NotImplemented
        #     #scaled_img = np.float32(np.clip(scaled_img, 0, 1))           
        # else:
        #     scaled_img = np.uint8(np.clip(scaled_img, 0, (2**8)-1))
        return scaled_img
    
    def get_scaling_params(self):
        self.is_debug = self.parm_sca["isDebug"]
        self.old_size = (self.sensor_info["height"], self.sensor_info["width"])
        self.new_size = (self.parm_sca["new_height"], self.parm_sca["new_width"])   

################################################################################
class Scale_2D:        
    def __init__(self, single_channel, sensor_info, parm_sca):
        self.single_channel = np.float32(single_channel)
        self.sensor_info = sensor_info
        self.parm_sca = parm_sca
    
    def downscale_nearest_neighbor(self, new_size):
        
        """Down scale by an integer factor using NN method using convolution."""    
        # print("Downscaling with Nearest Neighbor.")

        old_height, old_width = self.single_channel.shape[0], self.single_channel.shape[1]
        new_height, new_width = new_size[0], new_size[1]
        
        # As new_size is less than old_size, scale factor is defined s.t it is >1 for downscaling
        scale_height , scale_width = old_height/new_height, old_width/new_width

        kernel = np.zeros((int(scale_height), int(scale_width)))
        kernel[0,0] = 1

        scaled_img  = utils.stride_convolve2d(self.single_channel, kernel)
        return scaled_img.astype("float32")

    def scale_nearest_neighbor(self, new_size):
            
            """
            Upscale/Downscale 2D array by any scale factor using Nearest Neighbor (NN) algorithm.
            """

            old_height, old_width = self.single_channel.shape[0], self.single_channel.shape[1]
            new_height, new_width = new_size[0], new_size[1]
            scale_height , scale_width = new_height/old_height, new_width/old_width

            scaled_img = np.zeros((new_height, new_width), dtype = "float32")

            for y in range(new_height):
                for x in range(new_width):
                    y_nearest = int(np.floor(y/scale_height))
                    x_nearest = int(np.floor(x/scale_width))
                    scaled_img[y,x] = self.single_channel[y_nearest, x_nearest]
            return scaled_img
        
    def bilinear_interpolation(self, new_size):
            
            """
            Upscale/Downscale 2D array by any scale factor using Bilinear method.
            """

            old_height, old_width      = self.single_channel.shape[0], self.single_channel.shape[1]
            new_height, new_width      = new_size[0], new_size[1]
            scale_height , scale_width = new_height/old_height, new_width/old_width
            
            scaled_img  = np.zeros((new_height, new_width), dtype = "float32")
            old_coor    = lambda a, scale_fact: (a+0.5)/scale_fact - 0.5
            
            for y in range(new_height):
                for x in range(new_width):

                    # Coordinates in old image
                    old_y, old_x = old_coor(y, scale_height), old_coor(x, scale_width)
                    
                    x1 = 0 if np.floor(old_x)<0 else min(int(np.floor(old_x)), old_width-1)
                    y1 = 0 if np.floor(old_y)<0 else min(int(np.floor(old_y)), old_height-1)
                    x2 = 0 if np.ceil(old_x)<0 else min(int(np.ceil(old_x)), old_width-1)
                    y2 = 0 if np.ceil(old_y)<0 else min(int(np.ceil(old_y)), old_height-1)
                    
                    # Get four neghboring pixels
                    Q11 = self.single_channel[y1, x1]
                    Q12 = self.single_channel[y1, x2]
                    Q21 = self.single_channel[y2, x1]
                    Q22 = self.single_channel[y2, x2]

                    # Interpolating P1 and P2
                    weight_right = old_x- np.floor(old_x)
                    weight_left  = 1-weight_right 
                    P1 = weight_left*Q11 + weight_right*Q12
                    P2 = weight_left*Q21 + weight_right*Q22

                    # The case where the new pixel lies between two pixels
                    if x1 == x2:
                        P1 = Q11
                        P2 = Q22

                    # Interpolating P
                    weight_bottom = old_y - np.floor(old_y)
                    weight_top = 1-weight_bottom 
                    P = weight_top*P1 + weight_bottom*P2    

                    scaled_img[y,x] = P
            return scaled_img.astype("float32")

    def resize_by_non_int_fact(self, red_fact, method):
            
            """"
            Resize 2D array by non-integer factor n/d.
            Firstly, the array is upsacled n times then downscaled d times.
            Parameter:
            ---------
            red_fact: list of tuples with scale factors for height (at index 0) and width (at index 1).
            method: list with algorithms used for upscaling(at index 0) and downscaling(at index 1).
            Output: scaled img
            """
            
            for i in range(2):
                if bool(red_fact[i]):  
                    
                    # reduction factor = n/d    --->  Upscale the cropped image n times then downscale d times
                    upscale_fact   = red_fact[i][0] 
                    downscale_fact = red_fact[i][1]
                    
                    upscale_to_size = (upscale_fact*self.single_channel.shape[0], self.single_channel.shape[1]) if i==0 else \
                                      (self.single_channel.shape[0], upscale_fact*self.single_channel.shape[1])
                    if method[0]=="Nearest_Neighbor":
                        self.single_channel = self.scale_nearest_neighbor(upscale_to_size)
                    elif method[0]=="Bilinear":
                        self.single_channel = self.bilinear_interpolation(upscale_to_size)
                    else:
                        print("   - Invalid scale method. \n   - UpScaling with default Nearest Neaghibor method...") if i==0 else None
                        self.single_channel = self.scale_nearest_neighbor(upscale_to_size)
                    downscale_to_size = (int(np.round(self.single_channel.shape[0]/downscale_fact)), self.single_channel.shape[1]) if i==0 else \
                                        (self.single_channel.shape[0], int(np.round(self.single_channel.shape[1]//downscale_fact)))    
                    
                    if method[1]=="Nearest_Neighbor":
                        self.single_channel = self.downscale_nearest_neighbor(downscale_to_size)
                    elif method[1]=="Bilinear":
                        self.single_channel = self.downscale_by_int_factor(downscale_to_size)
                    else:
                        print("   - Invalid scale method. \n   - DownScaling with default Nearest Neaghibor method...") if i==0 else None
                        self.single_channel = self.downscale_nearest_neighbor(downscale_to_size)    
            return self.single_channel

    def downscale_by_int_factor(self, new_size):
            
            """
            Downscale a 2D array by an integer scale factor using Bilinear method with 2D convolution.
            
            Parameters
            ----------
            new_size: Required output size. 
            Output: 16 bit scaled image in which each pixel is an average of box nxm 
            determined by the scale factors.  
            """

            scale_height = new_size[0]/self.single_channel.shape[0]             
            scale_width  = new_size[1]/self.single_channel.shape[1]

            box_height = int(np.ceil(1/scale_height)) 
            box_width  = int(np.ceil(1/scale_width))
            
            scaled_img = np.zeros((new_size[0], new_size[1]), dtype = "float32")
            kernel     = np.ones((box_height, box_width))/(box_height*box_width)
            
            scaled_img = utils.stride_convolve2d(self.single_channel, kernel)
            return scaled_img.astype("float32")
    
    def hardware_dep_scaling(self):
        # check and set the flow of the algorithm
        scale_info = self.validate_input_output()
        
        # apply scaling according to the flow
        return self.apply_algo(scale_info, ([self.upscale_method, self.downscale_method]))
    
    def apply_algo(self, scale_info, method):
       
        """
        Scale 2D array using hardware friendly approach comprising of 3 steps:
           1. Downscale with int factor 
           2. Crop  
           3. Scale with non-integer-factor """

        # check if input size is valid
        if scale_info == [[None,None,None], [None,None,None]]:
            print("   - Invalid input size. It must be one of the following:\n"\
                  "   - 1920x1080\n"
                  "   - 2592x1536\n"
                  "   - 2592x1944")
            return self.single_channel
        
        # check if output size is valid
        if scale_info == [[1,0,None], [1,0,None]]:
            print("   - Invalid output size.")
            return self.single_channel    
        else:
            # step 1: Downscale by int fcator
            if scale_info[0][0]>1 or scale_info[1][0]>1:
                self.single_channel = self.downscale_by_int_factor((self.old_size[0]//scale_info[0][0], 
                                                        self.old_size[1]//scale_info[1][0]))
            
                print("   - Shape after downscaling by integer factor {}:  {}".format((scale_info[0][0],\
                        scale_info[1][0]),self.single_channel.shape)) if self.is_debug else None

            # step 2: crop
            if scale_info[0][1]>0 or scale_info[1][1]>0:
                self.single_channel = utils.crop(self.single_channel, scale_info[0][1], scale_info[1][1])

                print("   - Shape after cropping {}:  {}".format((scale_info[0][1],\
                        scale_info[1][1]), self.single_channel.shape)) if self.is_debug else None                                    
            
            # step 3: Scale with non-int factor
            if bool(scale_info[0][2])==True or bool(scale_info[1][2])==True:
                self.single_channel = self.resize_by_non_int_fact((scale_info[0][2], 
                                                    scale_info[1][2]), method)
                print("   - Shape after scaling by non-integer factor {}:  {}".format((scale_info[0][2],\
                        scale_info[1][2]),self.single_channel.shape)) if self.is_debug else None
            return self.single_channel
        
    def validate_input_output(self):
        
        valid_size = [(1080,1920), (1536,2592), (1944,2592)]
        sizes   = [OUT_1080x1920, OUT_1536x2592, OUT_1944x2592]
        
        # Check if input size is valid
        if self.old_size not in valid_size:
            scale_info = [[None,None,None],[None,None,None]]
            return scale_info

        idx     = valid_size.index(self.old_size) 
        size_obj   = sizes[idx](self.new_size)       
        scale_info = size_obj.scale_list
        return scale_info
        
    def execute(self):

        self.get_scaling_params()

        if self.is_hardware:
            self.single_channel = self.hardware_dep_scaling()
        else:

            self.single_channel = self.hardware_indp_scaling()    

        print("   - Shape of scaled image for a single channel = ", self.single_channel.shape) if self.is_debug else None
        return self.single_channel

    def hardware_indp_scaling(self):

        if self.algo=="Nearest_Neighbor":
            print("   - Scaling with Nearest Neighbor method...") if self.is_debug else None
            return self.scale_nearest_neighbor(self.new_size)
        elif self.algo=="Bilinear":
            print("   - Scaling with Bilinear method...") if self.is_debug else None
            return self.bilinear_interpolation(self.new_size)
        else:
            print("   - Invalid scale method.\n   - Scaling with default Nearest Neighbor method...") if self.is_debug else None
            return self.scale_nearest_neighbor(self.new_size)    

    def get_scaling_params(self):
        self.old_size = (self.sensor_info["height"], self.sensor_info["width"])
        self.new_size = (self.parm_sca["new_height"], self.parm_sca["new_width"])
        self.is_debug = self.parm_sca["isDebug"]
        self.is_hardware      = self.parm_sca["isHardware"] 
        self.algo             = self.parm_sca["Algo"] 
        self.upscale_method   = self.parm_sca["upscale_method"] 
        self.downscale_method = self.parm_sca["downscale_method"] 

#############################################################################
# The structure and working of the following three classes is exactlly the same
class OUT_1080x1920:
    """
    The hardware friendly approach can only be used for specific input and output sizes.
    This class checks if the given output size can be achieved by this approach and 
    creates a nested list (SCALE_LIST below) with corresponding constants used to execute this scaling approach 
    comprising of the following three steps:
    
    1. Downscale with int factor 
    2. Crop  
    3. Scale with non-integer-factor 
    
    The elements in each list correspond to the following constants:
    
    1. Scale factor [int]: (default 1) scale factor for downscaling.   
    2. Crop value [int] : (defaut 0) number of rows or columns to be croped.
    3. Non-int scale factor [tuple with 2 enteries]: (default None) a rational scale factor 
       of form n/d where n is the first entry (index 0) and d is the second entry(index 1)
       of this tuple.   
    
    Instance Attributes:
    -------------------  
    SCALE_LIST:  a nested list with two sublists containing constants used in order to
    scale height (index 0) and width (index 1) to the given NEW_SIZE using the three 
    steps above.
    
    """

    def __init__(self, new_size):
        self.new_size = new_size

        if self.new_size == (720, 1280):
            self.scale_list= [[1,0,(2,3)], [1,0,(2,3)]]

        elif self.new_size == (480, 640):
            self.scale_list= [[2,60,None], [3,0,None]]

        elif self.new_size == (360, 640):
            self.scale_list= [[3,0,None], [3,0,None]]        

        else:
            self.scale_list= [[1,0,None],[1,0,None]]

#############################################################################
class OUT_1536x2592:
    
    def __init__(self, new_size):
        self.new_size = new_size

        if self.new_size == (1080, 1920):
            self.scale_list= [[1,24,(5,7)], [1,32,(3,4)]]
        
        elif self.new_size == (720,1280):
            self.scale_list= [[2,48,None], [2,16,None]]
        
        elif self.new_size == (480, 640):
            self.scale_list= [[3,32,None], [4,8,None]]        
        
        elif self.new_size == (360,640):
            self.scale_list= [[4,24,None], [4,8,None]]        
        
        else:
            self.scale_list= [[1,0,None],[1,0,None]]
    
#############################################################################
class OUT_1944x2592:
    
    def __init__(self, new_size):
        self.new_size = new_size

        if self.new_size == (1440, 2560):
            self.scale_list= [[1,24,(3,4)], [1,32,None]]
        
        elif self.new_size == (1080, 1920):
            self.scale_list= [[1,54,(4,7)], [1,32,(3,4)]]
        
        elif self.new_size == (960,1280):
            self.scale_list= [[2,16,None], [2,12,None]]
        
        elif self.new_size == (720,1280):
            self.scale_list= [[2,16,(3,4)], [2,12,None]]    
        
        elif self.new_size == (480, 640):
            self.scale_list= [[4,6,None], [4,8,None]]        
        
        elif self.new_size == (360,640):
            self.scale_list= [[4,6,(3,4)], [4,8,None]]        
        
        else:
            self.scale_list= [[1,0,None],[1,0,None]]
