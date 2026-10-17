from pymodbus.client import ModbusTcpClient
import socket
import time 
import subprocess 
import sys
import os
def switch_to_relay_wifi():
    subprocess.run(["networksetup", "-setairportnetwork", "en0", "JY_172ab1"])
    print("Switched to relay Wi-Fi.")
    

    result = subprocess.run(
        ["networksetup", "-getairportnetwork", "en0"],
        capture_output=True,
        text=True
    )
    current_ssid = result.stdout.strip().split(": ")[-1]
    print(f"Current WiFi: {current_ssid}")  
    time.sleep(5)  


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) 

def send_close_signal():
    switch_to_relay_wifi()  
    modbus_command_open = "01050000FF008C3A"  # open output1   
    modbus_command_close = "010500000000CDCA"  # close output1
# these are the modbus commands for the network relay. (very specific to a device, developer obtained it from the graphical software of the device)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.connect(("192.168.10.1", 10000))
        print("Connection succeed")

        s.sendall(bytes.fromhex(modbus_command_open))
        print("Command sent to open output1.")

        time.sleep(2) 

        s.sendall(bytes.fromhex(modbus_command_close))
        print("Command sent to close output1.")
        
    except socket.error as e:
        print(f"Socket error: {e}")

    finally:
     
        print("Closing the socket connection.")
        s.close() 
        os.execv(sys.executable, ['python'] + ['/Users/apple/Desktop/research/TNRproject/代码2.0/backend/video_processing/process_video.py'])
        #Above is the address of the process_video.py file on developer's computer. The code is also attached in the appendix.

