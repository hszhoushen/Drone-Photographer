import tello as tello
from tello_control_ui import TelloUI


def main():

    drone = tello.Tello('', 8889)
    vplayer = TelloUI(drone)
    
    # start the Tkinter mainloop
    vplayer.root.mainloop() 

if __name__ == "__main__":
    main()
