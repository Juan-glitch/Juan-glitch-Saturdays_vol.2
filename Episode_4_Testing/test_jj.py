
# import pandas as pd
# from rplidar import RPLidar
# # 1. Descargar modulo 
# # $pip install rplidar-roboticia 

# # 2. Problemas comunicacion
# # sudo usermod -a -G dialout $USER

from rplidar import RPLidar
import pandas as pd

lidar = RPLidar('/dev/ttyUSB0')

info = lidar.get_info()
print(info)

health = lidar.get_health()
print(health)

data = pd.DataFrame (columns=['quality', 'angle', 'distance'])
for i, scan in enumerate(lidar.iter_scans()):
    print(len(scan))
    df_new= pd.DataFrame (scan,columns=['quality', 'angle', 'distance'])
    data= pd.concat([data, df_new], ignore_index=True)
    if i == 20:
        break

data.sort_values(by=['angle'], inplace = True)

print(data.head(), data.shape) 
data.to_csv('./pruebas.csv', index=False)
# End functions
lidar.stop()
lidar.stop_motor()
lidar.disconnect()

