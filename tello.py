import socket
import threading
import time
import datetime
import numpy as np
import libh264decoder

import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

from PIL import Image
import glob

from BASNet import BASNet

import matplotlib.pyplot as plt
# zhou add, 2019.8.2
import cv2

# Tan add, 2019.9.19
import math

# load BASNet
net = BASNet(3,1)
net.load_state_dict(torch.load('./saved_models/basnet_bsi/basnet.pth'))
if torch.cuda.is_available():
    net.cuda()
net.eval()

class Tello:
    """Wrapper class to interact with the Tello drone."""

    def __init__(self, local_ip, local_port, imperial=False, command_timeout=.3, tello_ip='192.168.10.1',
                 tello_port=8889):
        """
        Binds to the local IP/port and puts the Tello into command mode.

        :param local_ip (str): Local IP address to bind.
        :param local_port (int): Local port to bind.
        :param imperial (bool): If True, speed is MPH and distance is feet.
                             If False, speed is KPH and distance is meters.
        :param command_timeout (int|float): Number of seconds to wait for a response to a command.
        :param tello_ip (str): Tello IP.
        :param tello_port (int): Tello port.
        """

        self.abort_flag = False
        self.decoder = libh264decoder.H264Decoder()
        self.command_timeout = command_timeout
        self.imperial = imperial
        self.response = None  
        self.frame = None  # numpy array BGR -- current camera output frame
        self.is_freeze = False  # freeze current camera output
        self.last_frame = None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for sending cmd
        self.socket_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for receiving video stream
        self.tello_address = (tello_ip, tello_port)
        self.local_video_port = 11111  # port for receiving video stream
        self.last_height = 0
        self.socket.bind((local_ip, local_port))

        # thread for receiving cmd ack
        self.receive_thread = threading.Thread(target=self._receive_thread)
        self.receive_thread.daemon = True

        self.receive_thread.start()

        # to receive video -- send cmd: command, streamon
        self.socket.sendto(b'command', self.tello_address)
        print ('sent: command')
        self.socket.sendto(b'streamon', self.tello_address)
        print ('sent: streamon')

        self.socket_video.bind(("0.0.0.0", self.local_video_port))

        # thread for receiving video
        self.receive_video_thread = threading.Thread(target=self._receive_video_thread)
        self.receive_video_thread.daemon = True

        self.receive_video_thread.start()

	    # zhou add, 2019.8.1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.width = 256
        self.height = 256

        # Tan modify 2019/9/2
        self.flag_adjust = False

        self.count = 0

        # add 9/21
        self.flag_command = False
        # Tan add
        self.Tset = [
            "Animals", 
            "Birds", 
            "Children", 
            "Cityscape", 
            "Family", 
            "Floral", 
            "Food and Drink", 
            "Self Portrait", 
            "Science and Technology"
            ]
        self.Tsel = ""
        
        self.Dset = [ # [size, cx, cy, bx, by]
            [0.147, 0.0290, -0.0926, -0.0169, 0.0897], # "Animals"
            [0.140, 0.0297, -0.0750, 0.0430, 0.147], # "Birds"
            [0.351, 0.00139, 0.0959, -0.0143, 0.0718], # "Children"
            [0.0503, 0.0967, 0.102, -0.0995, 0.0506], # "Cityscape"
            [0.241, 0.0278, -0.0406, -0.0505, -0.0680], # "Family"
            [0.150, -0.0982, -0.0878, -0.0990, 0.0506], # "Floral"
            [0.245, 0.0884, 0.0338, -0.0939, -0.0122], # "Food and Drink"
            [0.217, 0.0347, -0.0673, -0.0514, 0.0331], # "Self Portrait"            
            [0.0480, 0.0567, 0.0430, -0.0390, -0.0787] # "Science and Technology"
            ]
        self.Dsel = [0, 0, 0, 0, 0]

    def __del__(self):
        """Closes the local socket."""

        self.socket.close()
        self.socket_video.close()

    def PIL_to_tensor(self,image):
        image = transforms.Compose([transforms.ToTensor()])(image).unsqueeze(0)
        # print('device:', self.device)
        return image.to(self.device, torch.float)

    # find contourstkinter
    def find_contours(self, salient_img):
        # find contours
        contours, hierarchy = cv2.findContours(salient_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("find", len(contours), "contours")
        return contours

    # # draw contours and center of mass
    # def preserve_max_contours(self, img, contours):
    #     # find the area with the largest area
    #     max_area = 0
    #     for i in range(len(contours)):
    #         area = cv2.contourArea(contours[i])
    #         # print('area:', area)
    #         if(area > max_area):
    #             max_area = area
    #             max_idx = i
                
    #     # draw the center point of image
    #     pt = (round(self.width/2), round(self.height/2))
    #     # print('pt:', pt)
    #     cv2.circle(img, pt, 2, (0, 0, 255), 2)  # 画红点
    #     text = "(" + str(pt[0]) + ", " + str(pt[1]) + ")"
    #     cv2.putText(img, text, (pt[0]+10, pt[1]+10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 1, 8, 0)

    #     # draw the center of mass of max area contour
    #     if(area > 800):
    #         # draw contours
    #         cv2.drawContours(img,contours,max_idx,(0,0,255))

    #         mom = cv2.moments(contours[max_idx])
    #         pt = (int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00']))  # 使用前三个矩m00, m01和m10计算重心
    #         cv2.circle(img, pt, 2, (255, 0, 0), 2)  # 画红点
    #         text = "(" + str(pt[0]) + ", " + str(pt[1]) + ")"
    #         cv2.putText(img, text, (pt[0]+10, pt[1]+10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1, 8, 0)
    #         perimeter = cv2.arcLength(contours[max_idx],True)
                
    #     # cv2.imwrite(rgb_path, img)
    #     return contours[max_idx]

    def preserve_distinct_contours(self, contours, area_thre=1000):
        idx_list = []
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if (area > area_thre):
                idx_list.append(i)

        # print('idx_list:', idx_list)
        return idx_list

    # IQA from max contour and salient map
    # def iqa_center(self, salient_img, max_contour):
        
    #     c_x = salient_img.shape[1] / 2
    #     c_y = salient_img.shape[0] / 2
        
    #     # print('image center (x, y):', c_x, c_y)
        
    #     mom = cv2.moments(max_contour)
    #     pt = (int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00']))  # 使用前三个矩m00, m01和m10计算重心
    #     cx = (c_x - pt[0]) / c_x - self.Dsel[1]
    #     cy = (c_y - pt[1]) / c_y - self.Dsel[2]
        
    #     # print('cx, cy, area, perimeter:', cx, cy, area, perimeter)
    #     return cx, cy

    # the distance from the boundary

    def iqa_center(self, salient_img, contours, cnt_idx_list):

        half_width = salient_img.shape[1] / 2
        half_height = salient_img.shape[0] / 2

        # empty list
        x = []
        y = []
        area = []

        for i in range(len(cnt_idx_list)):
            idx = cnt_idx_list[i]

            mom = cv2.moments(contours[idx])
            pt = (int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00']))  # 使用前三个矩m00, m01和m10计算重心
            # print('Point:', pt[0], pt[1])

            x.append(pt[0])
            y.append(pt[1])
            area.append(cv2.contourArea(contours[idx]))

        alpha = np.array(area) / sum(area)
        Cx = int(sum(alpha*x))
        Cy = int(sum(alpha*y))
        Center = (Cx, Cy)
        # print('Cx, Cy:', Cx, Cy)
        cx = (half_width - Cx) / half_width - self.Dsel[1]
        cy = (half_height - Cy) / half_height - self.Dsel[2]
        return cx, cy, Center


    # def iqa_boundary(self, salient_img, max_contour):
    #     # print(max_contour[np.lexsort(max_contour.T)]) # 按照最后一列排序
    #     # print(type(max_contour), max_contour[:,:,0])
    #     x = salient_img.shape[1]
    #     y = salient_img.shape[0]
    #     # print('image size: {} * {}'.format(x, y))

    #     x_min = np.min(max_contour[:,:,0])
    #     x_max = np.max(max_contour[:,:,0])
    #     y_min = np.min(max_contour[:,:,1])
    #     y_max = np.max(max_contour[:,:,1])

    #     # Tan modified, 2019/9/19
    #     bx = ((x_min - 0) - (x - x_max)) / x - self.Dsel[3]
    #     by = ((y_min - 0) - (y - y_max)) / y - self.Dsel[4]

    #     return bx, by

    def iqa_boundary(self, salient_img, contours, cnt_idx_list):

        xmin = 9999
        ymin = 9999
        xmax = 0
        ymax = 0

        # if distinct objects detected
        if (len(cnt_idx_list)):
            for i in range(len(cnt_idx_list)):
                idx = cnt_idx_list[i]
                contour = contours[idx]
                x_min = contour[:, :, 0].min()
                x_max = contour[:, :, 0].max()
                y_min = contour[:, :, 1].min()
                y_max = contour[:, :, 1].max()
                # print('x min max,y min max:', x_min, x_max, y_min, y_max)
                if(x_min < xmin):
                    xmin = x_min
                if(y_min < ymin):
                    ymin = y_min
                if(x_max > xmax):
                    xmax = x_max
                if(y_max > ymax):
                    ymax = y_max
            # print('x min max,y min max:', xmin, xmax, ymin, ymax)
            height = salient_img.shape[0]
            width = salient_img.shape[1]
            # print('width, height:', width, height)
            bx = ((xmin - 0) - (width - xmax)) / width - self.Dsel[3]
            by = ((ymin - 0) - (height - ymax)) / height - self.Dsel[4]
            # print('dbx, dby:', dbx, dby)

        # if no distinct salient map detected
        # else:
        #     dbx = 0
        #     dby = 0
        #     return dbx, dby

        return bx, by

    # def iqa_size(self, salient_img, contours):
    #     img_area = salient_img.shape[0] * salient_img.shape[1]

    #     area_sum = 0
    #     max_area = 0

    #     for i in range(len(contours)):
    #         area = cv2.contourArea(contours[i])
    #         area_sum += area
    #         # print('area:', area)
    #         if area > max_area:
    #             max_area = area
    #             max_idx = i
    #     if area_sum > img_area:
    #         area_sum = img_area

    #     # Tan modified, 2019/9/19
    #     size = area_sum / img_area - self.Dsel[0]
    #     return size

    def iqa_size(self, salient_img, contours):
        # calculate the image size
        img_area = salient_img.shape[0]*salient_img.shape[1]
        # threshold the image
        salient_binary = cv2.threshold(salient_img, 50, 1, cv2.THRESH_BINARY)
        saliency_map_area = np.sum(salient_binary[1])

        size = saliency_map_area / img_area - self.Dsel[0]

        return size

    # infer the salient map
    def process(self):
        # print('type of frame:', type(frame))
        frame = Image.fromarray(self.frame)
        frame = frame.resize((self.width, self.height))
        inputs_test = self.PIL_to_tensor(frame)
        # print('shape of inputs:', inputs_test.shape)

        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)

        # normalization
        # print('shape of d1:', d1.shape)
        pred = d1[:,0,:,:]
        pred = (pred-torch.min(pred))/(torch.max(pred)-torch.min(pred))
        pred = pred.squeeze()
        pred_np = pred.cpu().data.numpy()

	    # save img to dir
        img = Image.fromarray(pred_np*255).convert('RGB')

        del d1,d2,d3,d4,d5,d6,d7,d8
        return img
        
    def read(self):
        """Return the last frame from camera."""
        if self.is_freeze:
            return self.last_frame
        else:
            return self.frame

    def video_freeze(self, is_freeze=True):
        """Pause video output -- set is_freeze to True"""
        self.is_freeze = is_freeze
        if is_freeze:
            self.last_frame = self.frame

    def _receive_thread(self):
        """Listen to responses from the Tello.

        Runs as a thread, sets self.response to whatever the Tello last returned.

        """
        while True:
            try:
                self.response, ip = self.socket.recvfrom(3000)
                if not self.flag_command:
                    print(self.response)
                else:
                    self.flag_command = False
            except socket.error as exc:
                print(("Caught exception socket.error : %s" % exc))

    def _receive_video_thread(self):
        """
        Listens for video streaming (raw h264) from the Tello.

        Runs as a thread, sets self.frame to the most recent frame Tello captured.

        """
        packet_data =b''
        while True:
            try:
                res_string, ip = self.socket_video.recvfrom(2048)
                
                packet_data += res_string
                # end of frame
                if len(res_string) != 1460:
                    for frame in self._h264_decod(packet_data):
                        self.frame = frame
                    packet_data = b''
                        
            except socket.error as exc:
                print(("Caught exception socket.error : %s" % exc))
    
    
    def _h264_decod(self, packet_data):
        """
        decode raw h264 format data from Tello
        
        :param packet_data: raw h264 data array
       
        :return: a list of decoded frame
        """
        res_frame_list = []
        frames = self.decoder.decode(packet_data)
                 
        for framedata in frames:
            (frame, w, h, ls) = framedata
            if frame is not None:
                # print ('frame size %i bytes, w %i, h %i, linesize %i' % (len(frame), w, h, ls))
                frame = np.frombuffer(frame, dtype = np.ubyte, count = len(frame))
                frame = (frame.reshape((h, ls//3, 3)))
                frame = frame[:, :w, :]
                res_frame_list.append(frame)

        return res_frame_list

    def command(self):
        self.flag_command = True
        self.socket.sendto(b'command', self.tello_address)


    def send_command(self, command):
        """
        Send a command to the Tello and wait for a response.

        :param command: Command to send.
        :return (str): Response from Tello.

        """

        print((">> send cmd: {}".format(command)))
        self.abort_flag = False
        timer = threading.Timer(self.command_timeout, self.set_abort_flag)

        self.socket.sendto(command.encode('utf-8'), self.tello_address)

        timer.start()
        while self.response is None:
            if self.abort_flag is True:
                break
        timer.cancel()
        
        if self.response is None:
            response = 'none_response'
        else:
            response = self.response.decode('latin-1')

        self.response = None

        return response
    
    def set_abort_flag(self):
        """
        Sets self.abort_flag to True.

        Used by the timer in Tello.send_command() to indicate to that a response
        
        timeout has occurred.

        """

        self.abort_flag = True

    def takeoff(self):
        """
        Initiates take-off.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        # modified by Tan for expriments 2019/9/6
        
        # return self.send_command('takeoff')

        return self.send_command('takeoff')     

    def set_speed(self, speed):
        """
        Sets speed.

        This method expects KPH or MPH. The Tello API expects speeds from
        1 to 100 centimeters/second.

        Metric: .1 to 3.6 KPH
        Imperial: .1 to 2.2 MPH

        Args:
            speed (int|float): Speed.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        speed = float(speed)

        if self.imperial is True:
            speed = int(round(speed * 44.704))
        else:
            speed = int(round(speed * 27.7778))

        return self.send_command('speed %s' % speed)

    def rotate_cw(self, degrees):
        """
        Rotates clockwise.

        Args:
            degrees (int): Degrees to rotate, 1 to 360.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.send_command('cw %s' % degrees)

    def rotate_ccw(self, degrees):
        """
        Rotates counter-clockwise.

        Args:
            degrees (int): Degrees to rotate, 1 to 360.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        return self.send_command('ccw %s' % degrees)

    def get_response(self):
        """
        Returns response of tello.

        Returns:
            int: response of tello.

        """
        response = self.response
        return response

    def get_height(self):
        """Returns height(dm) of tello.

        Returns:
            int: Height(dm) of tello.

        """
        height = self.send_command('height?')
        height = str(height)
        height = list(filter(str.isdigit, height))
        try:
            height = int(height)
            self.last_height = height
        except:
            height = self.last_height
            pass
        return height

    def get_battery(self):
        """Returns percent battery life remaining.

        Returns:
            int: Percent battery life remaining.

        """
        
        battery = self.send_command('battery?')

        try:
            battery = int(battery)
        except:
            pass

        return battery

    def get_flight_time(self):
        """Returns the number of seconds elapsed during flight.

        Returns:
            int: Seconds elapsed during flight.

        """

        flight_time = self.send_command('time?')

        try:
            flight_time = int(flight_time)
        except:
            pass

        return flight_time

    def get_speed(self):
        """Returns the current speed.

        Returns:
            int: Current speed in KPH or MPH.

        """

        speed = self.send_command('speed?')

        try:
            speed = float(speed)

            if self.imperial is True:
                speed = round((speed / 44.704), 1)
            else:
                speed = round((speed / 27.7778), 1)
        except:
            pass

        return speed

    def land(self):
        """Initiates landing.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.send_command('land')

    def move(self, direction, distance=.2):
        """Moves in a direction for a distance.

        This method expects meters or feet. The Tello API expects distances
        from 20 to 500 centimeters.

        Metric: .02 to 5 meters
        Imperial: .7 to 16.4 feet

        Args:
            direction (str): Direction to move, 'forward', 'back', 'right' or 'left'.
            distance (int|float): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        distance = float(distance)

        if self.imperial is True:
            distance = int(round(distance * 30.48))
        else:
            distance = int(round(distance * 100))

        return self.send_command('%s %s' % (direction, distance))

    def move_backward(self, distance):
        """Moves backward for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.move('back', distance)

    def move_down(self, distance):
        """Moves down for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.move('down', distance)

    def move_forward(self, distance):
        """Moves forward for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        return self.move('forward', distance)

    def move_left(self, distance):
        """Moves left for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        return self.move('left', distance)

    def move_right(self, distance):
        """Moves right for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """
        return self.move('right', distance)

    def move_up(self, distance):
        """Moves up for a distance.

        See comments for Tello.move().

        Args:
            distance (int): Distance to move.

        Returns:
            str: Response from Tello, 'OK' or 'FALSE'.

        """

        return self.move('up', distance)

    def take_snapshot(self, adj):

        ts = datetime.datetime.now()
        filename = "{}".format(ts.strftime("%Y-%m-%d_%H:%M:%S")) + adj + '.jpg'

        path = "./imgs/" + self.Tsel
        p = os.path.sep.join((path, filename))
        # print(path)
        cv2.imwrite(p, cv2.cvtColor(self.read(), cv2.COLOR_RGB2BGR))


    def select_type(self, n):
        self.Tsel = self.Tset[n-1]
        self.Dsel = self.Dset[n-1]

    def adjust_flag(self, flag):
        if flag:
            self.take_snapshot('_start')
            self.count = 0
        self.flag_adjust = flag
        # self.flag_adjust_center = flag
        # self.flag_adjust_boundary = flag
        # self.flag_adjust_size = flag

    def btn_adjust_relief(self):
        if not self.flag_adjust:
            return True

    def adjust(self):
        if self.flag_adjust and self.response != None:
            response =  self.response.decode(encoding='utf-8') 
            if response == 'ok':
                self.take_snapshot('')
                
                salient_map = self.process()

                salient_img = cv2.cvtColor(np.asarray(salient_map), cv2.COLOR_RGB2GRAY)

                contours = self.find_contours(salient_img)

                # preserve contours with area large than area_thre
                cnt_idx_list = self.preserve_distinct_contours(contours, area_thre=1000)

                # dsize calculation
                size = self.iqa_size(salient_img, contours)
                # dcx, dcy, dbx, dby calculation
                cx, cy, center = self.iqa_center(salient_img, contours, cnt_idx_list)
                bx, by = self.iqa_boundary(salient_img, contours, cnt_idx_list)

                # contours = self.find_contours(salient_img)
                # max_contour = self.preserve_max_contours(salient_img, contours)
                    
                # cx, cy = self.iqa_center(salient_img, max_contour)
                # bx, by = self.iqa_boundary(salient_img, max_contour)
                # size = self.iqa_size(salient_img, contours)

                M = math.sqrt(cx**2+cy**2) + math.sqrt(bx**2+by**2) + abs(size)
                print("M:", M)
                if M < 0.35:
                    # self.
                    print('----------Adjustment Finish----------')
                    self.take_snapshot('_ok')
                    self.flag_adjust = False
                    return self.land()
                if self.count % 5 == 0:
                    if bx < 0:
                        self.count += 1
                        return self.move('left')
                    if bx > 0:
                        self.count += 1
                        return self.move('right')
                elif self.count % 5 == 1:
                    if by > 0:
                        self.count += 1
                        return self.move('down')
                    if by < 0:
                        self.count += 1
                        return self.move('up')
                elif self.count % 5 == 2:
                    if size < 0:
                        self.count += 1
                        return self.move('forward')
                    if size > 0 :
                        self.count += 1
                        return self.move('back')
                elif self.count % 5 == 3:
                    if cx > 0:
                        self.count += 1
                        return self.move('left')
                    if cx < 0:
                        self.count += 1
                        return self.move('right')
                else:
                    if cy < 0:
                        self.count += 1
                        return self.move('down')
                    if cy > 0:
                        self.count += 1
                        return self.move('up')
