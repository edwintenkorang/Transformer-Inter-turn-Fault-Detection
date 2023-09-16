import numpy as np
import imutils
from imutils import paths
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
prefix = "C:\\Users\\Edwin\\Desktop\\Career\\Project\\Fault Generation\\Data\\"
phases = ['Phase A', 'Phase B', 'Phase C']
sides = ['Primary','Secondary']
turns = (np.arange(0.02, 0.99,0.02))
turns.tolist()
turns  = np.round(turns, 2).tolist()

sections = [1,2,3]
triggertimes = [0.06,0.1,0.14]
folders = ['input_current','output_current']

data = []
health_Labels = []
phase_Labels = []
side_Labels = []
percent_Labels = []
image_count = 0
#for Faulty Images:
for phase in phases:
    for side in sides:
        for turn in turns:
            print(turn)
            for sec in sections:
                for time in triggertimes:
                    vconcate = []  
                    print("Processing {}, {}, {} turns, Section {}, {} secs images".format(phase,side,turn, sec, time))
                    for folder in folders:
                      
                        path_name = prefix + "Faulty\\" + phase + "\\" + side + "\\" + "p_" +str(turn) + "\\sec_" + str(sec) +"\\b_" + str(time) + "\\"+ folder 
                        imagePaths = sorted(list(paths.list_images(path_name)))
                        newlist = []
                        for imagePath in imagePaths:
                            
                            image = cv2.imread(imagePath)
                            image = cv2.resize(image, [256,256])
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
                            #image = keras.preprocessing.image.img_to_array(image)                           
                            newlist.append(image)                                                  
                        hconcat_image = np.concatenate(newlist,axis = 1)
                        vconcate.append(hconcat_image)                        
                    #vconcat_image = cv2.vconcat(vconcate)
                    vconcat_image = np.concatenate(vconcate,axis=0)
                   #image = keras.preprocessing.image.img_to_array(vconcat_image)
                    image = cv2.resize(vconcat_image, [256,256])
                    print(image.shape)
                    data.append(image)
                    health_Labels.append('Faulty')
                    phase_Labels.append(phase) 
                    side_Labels.append(side)
                    percent_Labels.append(turn)
                    image_count +=1
                    
                
       
#healthy
for i in range(400):
    vconcate = []
    for folder in folders:
        path_name = prefix + "\\Healthy\\" + folder               
        
        imagePaths = sorted(list(paths.list_images(path_name)))
        newlist = []
        for imagePath in imagePaths:
            
            image = cv2.imread(imagePath)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                            
            newlist.append(image)        
        #hconcat_image = cv2.hconcat(newlist)
        hconcat_image = np.concatenate(newlist,axis=1)
        vconcate.append(hconcat_image)                        
    #vconcat_image = cv2.vconcat(vconcate)
    vconcat_image = np.concatenate(vconcate,axis=0)
    #image = keras.preprocessing.image.img_to_array(vconcat_image)
    image = cv2.resize(vconcat_image, [256,256])
    
    data.append(image)
    health_Labels.append('Healthy')
    phase_Labels.append('None') 
    side_Labels.append('None')
    percent_Labels.append('0')  
    image_count+=1      

    
print(f"Processed {image_count} images")
np.savez("C:\\Users\\Edwin\\Desktop\\Career\\Project\\fault_tester",data= data, healthLabels= health_Labels,sideLabels=side_Labels,phaseLabels=phase_Labels, percentLabels=percent_Labels)           