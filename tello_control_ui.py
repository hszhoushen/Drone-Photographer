from PIL import Image
from PIL import ImageTk
import tkinter as tki
from tkinter import Toplevel, Scale
import threading
import datetime
import cv2
import os
import time
import numpy as np

class TelloUI:
    """Wrapper class to enable the GUI."""

    def __init__(self,tello):
        """
        Initial all the element of the GUI,support by Tkinter

        :param tello: class interacts with the Tello drone.

        Raises:
            RuntimeError: If the Tello rejects the attempt to enter command mode.
        """        

        self.tello = tello # videostream device
        self.outputPath = './imgs' # the path that save pictures
        self.thread = None # thread of the Tkinter mainloop
        self.stopEvent = None  
        
        # control variables
        self.distance = 0.2  # default distance for 'move' cmd
        self.degree = 5  # default degree for 'cw' or 'ccw' cmd

        # if the flag is TRUE,the auto-takeoff thread will stop waiting for the response from tello
        self.quit_waiting_flag = False
        
        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel_img = None
        self.panel_sal = None

        # radiobuttons
        self.btns = []

        # adjust button
        self.btn_adjust = None

        # create buttons
        self.btn_battery = tki.Button(self.root, text="Get Battery",
                                        command=self.tello.get_battery)
        self.btn_battery.pack(side="bottom", fill="both",
                               expand="yes", padx=10, pady=5)

        self.btn_snapshot = tki.Button(self.root, text="Take Snapshot",
                                       command=self.takeSnapshot)
        self.btn_snapshot.pack(side="bottom", fill="both",
                               expand="yes", padx=10, pady=5)

        self.btn_pause = tki.Button(self.root, text="Pause Video", relief="raised", command=self.pauseVideo)
        self.btn_pause.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_landing = tki.Button(
            self.root, text="Land", relief="raised", command=self.telloLanding)
        self.btn_landing.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

        self.btn_open = tki.Button(
            self.root, text="Open Adjust Panel", relief="raised", command=self.openAdjWindow)
        self.btn_open.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5) 

        self.btn_takeoff = tki.Button(
            self.root, text="Takeoff", relief="raised", command=self.telloTakeOff)
        self.btn_takeoff.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

        self.tmp = tki.Frame(self.root, width=100, height=2)
        self.tmp.bind('<KeyPress-w>', self.on_keypress_w)
        self.tmp.bind('<KeyPress-s>', self.on_keypress_s)
        self.tmp.bind('<KeyPress-a>', self.on_keypress_a)
        self.tmp.bind('<KeyPress-d>', self.on_keypress_d)
        self.tmp.bind('<KeyPress-Up>', self.on_keypress_up)
        self.tmp.bind('<KeyPress-Down>', self.on_keypress_down)
        self.tmp.bind('<KeyPress-Left>', self.on_keypress_left)
        self.tmp.bind('<KeyPress-Right>', self.on_keypress_right)
        self.tmp.pack(side="left")
        self.tmp.focus_set()

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("Tello Video")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

        # the sending_command will send command to tello every 5 seconds
        self.sending_command_thread = threading.Thread(target = self._sendingCommand)
    def videoLoop(self):
        """
        The mainloop thread of Tkinter 
        Raises:
            RuntimeError: To get around a RunTime error that Tkinter throws due to threading.
        """
        try:
            # start the thread that get GUI image
            time.sleep(0.5)
            self.sending_command_thread.start()
            while not self.stopEvent.is_set():                

                # read the frame for GUI show
                frame = self.tello.read()
                # print('type of frame:', type(frame))
                if frame is None or frame.size == 0:
                    continue 
                else:
                    self.tello.adjust()
                    if self.btn_adjust != None and self.tello.btn_adjust_relief():
                        self.btn_adjust.config(relief="raised")
                        for i in range(9):
                            self.btns[i].config(state="normal")

                # transfer the format from frame to image         
                image = Image.fromarray(frame) 	# array to pil image
                salient_map = self.tello.process()
                self._updateGUIImage(image, salient_map)
                                                       
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

           
    def _updateGUIImage(self,image,prcsd_img):
        """
        Main operation to initial the object of image,and update the GUI panel 
        """  
        image = ImageTk.PhotoImage(image)
        prcsd_img = ImageTk.PhotoImage(prcsd_img)
        # if the panel none ,we need to initial it
        if self.panel_img is None:
            self.panel_img = tki.Label(self.root,image=image)
            self.panel_img.pack(side="left", padx=10, pady=10)
            self.panel_sal = tki.Label(self.root,image=prcsd_img)
            self.panel_sal.pack(side="right", padx=10, pady=10)

        # otherwise, simply update the panel
        else:
            self.panel_img.configure(image=image)
            self.panel_img.image = image
            self.panel_sal.configure(image=prcsd_img)
            self.panel_sal.image = prcsd_img
            
    def _sendingCommand(self):
        """
        start a while loop that sends 'command' to tello every 5 second
        """    

        while True:
            self.tello.command()
            time.sleep(5)

    def _setQuitWaitingFlag(self):  
        """
        set the variable as TRUE,it will stop computer waiting for response from tello  
        """       
        self.quit_waiting_flag = True        
   
    # Tan modified 2019/9/4
    def openAdjWindow(self):
        """
        open the adj window and initial all the button and text
        """        
        panel = Toplevel(self.root)
        panel.wm_title("Adjust Panel")
        self.v = tki.IntVar()
        self.v.set(0)
        font = (None, 12)

        Types = [
            ("Animals", 1), ("Birds", 2),
            ("Children", 3), ("Cityscape", 4),
            ("Family", 5), ("Floral", 6),
            ("Food and Drink", 7), ("Self Portrait", 8),
            ("Science and Technology", 9)]
        
        for t, n in Types:
            b = tki.Radiobutton(
                panel, text=t, font=font, value=n, variable=self.v)
            b.pack(side="top", fill="both",
                              expand="yes", padx=10, pady=5)
            self.btns.append(b)

        self.tmp_f = tki.Frame(panel, width=100, height=2)
        self.tmp_f.bind('<KeyPress-w>', self.on_keypress_w)
        self.tmp_f.bind('<KeyPress-s>', self.on_keypress_s)
        self.tmp_f.bind('<KeyPress-a>', self.on_keypress_a)
        self.tmp_f.bind('<KeyPress-d>', self.on_keypress_d)
        self.tmp_f.bind('<KeyPress-Up>', self.on_keypress_up)
        self.tmp_f.bind('<KeyPress-Down>', self.on_keypress_down)
        self.tmp_f.bind('<KeyPress-Left>', self.on_keypress_left)
        self.tmp_f.bind('<KeyPress-Right>', self.on_keypress_right)
        self.tmp_f.pack(side="top")
        self.tmp_f.focus_set()

        self.btn_adjust = tki.Button(
            panel, text="Adjust", font=(None,15), relief="raised", command=self.telloAdjust)
        self.btn_adjust.pack(side="top", fill="both",
                              expand="yes", padx=10, pady=5)

    def takeSnapshot(self):
        """
        save the current frame of the video as a jpg file and put it into outputpath
        """

        # grab the current timestamp and use it to construct the filename
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H:%M:%S_Snapshot!"))

        p = os.path.sep.join((self.outputPath, filename))

        # save the file
        cv2.imwrite(p, cv2.cvtColor(self.tello.read(), cv2.COLOR_RGB2BGR))
        print(("[INFO] saved {}".format(filename)))

    def pauseVideo(self):
        """
        Toggle the freeze/unfreze of video
        """
        if self.btn_pause.config('relief')[-1] == 'sunken':
            self.btn_pause.config(relief="raised")
            self.tello.video_freeze(False)
        else:
            self.btn_pause.config(relief="sunken")
            self.tello.video_freeze(True)
    
    # Tan add 2019/9/2
    def telloAdjust(self):
        # abort adjustment
        # click btn from sunken to raised
        if self.btn_adjust.config('relief')[-1] == 'sunken':
            self.btn_adjust.config(relief="raised")
            self.tello.adjust_flag(False)
            for i in range(9):
                self.btns[i].config(state="normal")

            print("----------Adjustment Abort----------")
        
        # start adjustment
        # click btn from raised to sunken
        else:
            self.btn_adjust.config(relief="sunken")
            n = self.v.get()
            self.tello.select_type(n)
            self.tello.adjust_flag(True)
            for i in range(9):
                self.btns[i].config(state="disabled")

            print("----------Adjustment Start----------")
    
    def telloSelectType(self, n):
        self.tello.select_type(n)

    def telloTakeOff(self):
        return self.tello.takeoff()

    def telloLanding(self):
        return self.tello.land()

    def telloCW(self, degree):
        return self.tello.rotate_cw(degree)

    def telloCCW(self, degree):
        return self.tello.rotate_ccw(degree)

    def telloMoveForward(self, distance):
        return self.tello.move_forward(distance)

    def telloMoveBackward(self, distance):
        return self.tello.move_backward(distance)

    def telloMoveLeft(self, distance):
        return self.tello.move_left(distance)

    def telloMoveRight(self, distance):
        return self.tello.move_right(distance)

    def telloUp(self, dist):
        return self.tello.move_up(dist)

    def telloDown(self, dist):
        return self.tello.move_down(dist)

    def on_keypress_w(self, event):
        print("manually move up %.2f m" % self.distance)
        return self.telloUp(self.distance)

    def on_keypress_s(self, event):
        print("manually move down %.2f m" % self.distance)
        return self.telloDown(self.distance)

    def on_keypress_a(self, event):
        print("manually move ccw %.2f degree" % self.degree)
        return self.tello.rotate_ccw(self.degree)

    def on_keypress_d(self, event):
        print("manually move cw %.2f m" % self.degree)
        return self.tello.rotate_cw(self.degree)

    def on_keypress_up(self, event):
        print("manually move forward %.2f m" % self.distance)
        return self.telloMoveForward(self.distance)

    def on_keypress_down(self, event):
        print("manually move backward %.2f m" % self.distance)
        return self.telloMoveBackward(self.distance)

    def on_keypress_left(self, event):
        print("manually move left %.2f m" % self.distance)
        return self.telloMoveLeft(self.distance)

    def on_keypress_right(self, event):
        print("manually move right %.2f m" % self.distance)
        return self.telloMoveRight(self.distance)

    def onClose(self):
        """
        set the stop event, cleanup the camera, and allow the rest of
        
        the quit process to continue
        """
        print("[INFO] closing...")
        self.stopEvent.set()
        del self.tello
        self.root.quit()

