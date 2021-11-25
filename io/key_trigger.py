"""
This class implements a key trigger.

It can be used to simulate key events useful for our interaction with the game.

Example:
    kt = KeyTrigger()
    ...
    if should_jump:
        kt.trigger_up()
"""
import pyautogui


class KeyTrigger(object):
    
    def trigger_up(self):
        # here the 'up-arrow' will be triggered (only once)
        pyautogui.press("up")
        


