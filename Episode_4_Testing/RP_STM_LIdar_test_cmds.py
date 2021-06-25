# Modules
import serial
from rplidar import RPLidar
import pandas as pd
import time
'''
#$pip install rplidar-roboticia
----------------------------------------------------------------------------------------------------------------
'''
# Defines
angulo_ref = 180
threshold_angle = 20
dist_alpha = 222 #milimetros
dist_beta = 140 #milimetros
dist_gamma = 150 #milimetros
impurity = False
RUN = False
contador = 0
'''
----------------------------------------------------------------------------------------------------------------
'''
# Serial config
stm =serial.Serial('/dev/ttyACM0',115200, bytesize = 8, stopbits = 1,timeout = 0, parity='N')
lidar = RPLidar('/dev/ttyUSB0')

info = lidar.get_info()
print(info)

health = lidar.get_health()
print(health)

# Model Config

'''
----------------------------------------------------------------------------------------------------------------
'''
point = 0
try:
    while(1):

        # Recollect data
        data = pd.DataFrame (columns=['quality', 'angle', 'distance'])
        
        for i, scan in enumerate(lidar.iter_scans()):
            # print(len(scan))
            df_new= pd.DataFrame (scan,columns=['quality', 'angle', 'distance'])

            # if(i > 10):
            data= pd.concat([data, df_new], ignore_index=True)
            if i == 5:
                break
        # Recollect data Lidar
        data.sort_values(by=['angle'], inplace = True)
        maximum =data[['distance']].idxmax()
        calibrate =data[['distance']].idxmin()
        min_values = data['distance'].nsmallest(40)
        angulo_test = float(data['angle'].iloc[data[['distance']].idxmin()]) # Min distance


        #----------------------------------------------------------------------------------------------------------------
        # Algorithm
        # Cmd 2 send STM
        # Scan
        point = data.iloc[data[['distance']].idxmin()] # Min distance
        
        scape = data.loc[data[['angle']].idxmax()] # Max distance

        
    # ----------------------------------------------------------------------------------------------------------------------
        # SEND CMDs

        stm.write(cmd.encode())
        while True:

            buffer = stm.read()           # Wait forever for anything
            time.sleep(0.01)              # Sleep (or inWaiting() doesn't give the correct value)
            buffer_left = stm.inWaiting()  # Get the number of characters ready to be read
            
            buffer += stm.read(buffer_left) # Do the read and combine it with the first character
            # INFO: https://stackoverflow.com/questions/13017840/using-pyserial-is-it-possible-to-wait-for-data
            break

        del buffer

        lidar.stop()
        # lidar.stop_motor()
        # lidar.clean_input()



except serial.SerialException:
    print('\n*** Serial ports ABRUPTLY closed ***')
    stm.close()
    lidar.disconnect()
    
    # End functions
except KeyboardInterrupt:
    print('\n*** Serial ports closed ***')
    stm.close()
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()

