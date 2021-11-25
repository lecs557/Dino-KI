import pyautogui
import PIL.ImageOps
import numpy
#interpreter
import pytesseract
#plot
import matplotlib.pyplot as plt
import time

class FrameGrabber:
    def __init__(self):
        pass

    def frame_grabber(self):
        screen = pyautogui.screenshot()
        #check for darkmode
        bw = screen.crop((685, 230, 1260, 310))
        bw = bw.convert('1')
        pixels=bw.getdata()
        color=numpy.mean(pixels)
        # if dark mode -> invert image
        if color<50:
            screen = PIL.ImageOps.invert(screen)
            frame = screen.crop((685, 230, 1260, 320))
            self.pointframe = screen.crop((1190, 185, 1250, 205))
            self.lostframe = screen.crop((850, 210, 1060, 240))

        #light mode frames have to be cropped different than dark mode screens
        else:
            frame = screen.crop((685, 230, 1260, 310))  # works if dinogame is fullscreen and centered
            self.pointframe = screen.crop((1190, 175, 1250, 195))
            self.lostframe = screen.crop((860, 200, 1060, 230))


        #gameover and points are interpreted
        self.interpreter()

        #the cropped frame and the points are returned. If the game is over, the string GAME OVER is returned
        if self.gameover=="GAME OVER":
            return "GAME OVER", self.points
        else:
            return frame, self.points

    #the frame where the points are and the gameover message is, are interpreted as string and points are converted to ints
    def interpreter(self):
        self.gameover = pytesseract.image_to_string(self.lostframe)
        self.points = pytesseract.image_to_string(self.pointframe)
        try:
            self.points=int(self.points)
        except:
            pass


#test with matplotlib
def testPlt():
    while True:
        frame, points = FrameGrabber.frame_grabber()
        if frame=="GAME OVER":
            break
        plt.imshow(frame)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    print(points)

#the testframes are saved
def testSave():
    i=0
    while True:
        frame, points = FrameGrabber.frame_grabber()
        print(points)
        if frame=="GAME OVER":
            break
        frame.save("frame"+str(i)+".png")
        time.sleep(2)
        i+=1
    print(points)
