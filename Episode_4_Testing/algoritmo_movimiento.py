#!/usr/bin/env python
# coding: utf-8

# In[136]:


import pandas as pd
# Modules
import serial
import time

# ser=serial.Serial('/dev/ttyACM0',115200, bytesize = 8, stopbits = 1,
#                   timeout = 0, parity='N')

data = pd.read_csv('./pruebas.csv')
data.head()

maximum =data[['distance']].idxmax()
calibrate =data[['distance']].idxmin()
data.loc[calibrate]


# In[137]:


data.loc[data[['distance']].idxmax()]


# In[138]:


# Constantes
angulo_ref = 180
threshold_angle = 20
dist_alpha = 222 #milimetros
dist_beta = 140 #milimetros
dist_gamma = 150 #milimetros
impurity = False

RUN = False
# scan_dist_b = 150 #milimetros


# In[139]:


data.shape


# In[141]:


# Todo dentro de un bucle while
#  El angulo y lo otro tambien

# Scan
point = data.iloc[data[['distance']].idxmin()] # Min distance
scape = data.loc[data[['distance']].idxmax()] # Max distance

# Exception mode
if(RUN):
    # GAMMA LAYER RUN
    if(  (float(rounout['distance']) > (angulo_ref - threshold_angle)) and 
         (float(rounout['distance']) < (angulo_ref + threshold_angle))):
        
        print('continua pa lante gamma') 
        rounout -=100
        
    else:
        # Turn LEFT
        if(float(rounout['angle']) - angulo_ref < 0):
            print('LEFT gamma', abs(float(rounout['angle']) - angulo_ref))
        # Turn RIGHT
        else:
            print('RIGHT gamma', abs(float(rounout['angle']) - angulo_ref)) 
            
# Normal Mode
else:
    # VOID LAYER
    if(float(point['distance']) >= dist_alpha):
        print('seguir palante void layer')
        
    # ALPHA LAYER
    elif(float(point['distance']) >= dist_beta):
        # Keep Fwd alpha band
        if(  (float(point['distance']) > (angulo_ref - threshold_angle)) and 
             (float(point['distance']) < (angulo_ref + threshold_angle))):

            print('continua pa lante ALPHA')  
        else:
            # Turn LEFT
            if(float(point['angle']) - angulo_ref < 0):
                print('LEFT alpha', abs(float(point['angle']) - angulo_ref))
            # Turn RIGHT
            else:
                print('RIGHT alpha', abs(float(point['angle']) - angulo_ref))  
                
    # BETA LAYER    
    elif(float(point['distance']) >= dist_gamma):
        if(impurity):
            print('ALARM send')
        else:
            printf('Keep Forward')
    # GAMMA LAYER        
    else:
        # Save Longest distance point
        rounout = scape
        RUN = True
        print('Ejecutar GAMMA')
    
#     # Rright
#     if( (data['angle'][calibrate]) - angulo_ref) < 0 and (data['angle'][calibrate]) - angulo_ref):
#         print('He entrado')
    


# In[ ]:


if(data.loc[calibrate] <)


# In[ ]:




