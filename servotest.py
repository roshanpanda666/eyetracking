import time
from pyfirmata import Arduino, util

board = Arduino('COM8')

servo_pin = board.get_pin('d:13:s') 


time.sleep(1)
while True:
    val=int(input(print("enter a value 0 to 180")))
    servo_pin.write(val)



