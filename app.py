#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## CODE TO EXECUTE IT 
## streamlit run app.py

"""

EXECUTE THE CODE LOCALLY: 

streamlit run app.py 


Pushing code to github 

in the folder you want to commit 

git add . 

git commit -m "some updated message here"

git push -u origin master 

"""

#################################################################
#### IMPORTS 
#################################################################

import math
import streamlit as st 
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from typing import List, Dict, Union

#################################################################
#### CONFIG 
#################################################################

st.set_option('deprecation.showPyplotGlobalUse', False)


## set the page config to wide mode and expanded sidebar, the page icon should be a cell
st.set_page_config(layout="wide", page_title="My App", page_icon="🧬")


## put a title to the side bar 
st.sidebar.title("My App ")

## put a text to the side bar
st.sidebar.text("This is my app")

## add a header 
st.header("Mitochondria-Quant")

## add a subheader
st.subheader("Instructions")

## add a text
st.text("""Use this app to automatically analyse size and circularity of mitochondria. Simply upload your microscopic image, select a region of interest (ROI), and tune parameters for optimal image contours.
""")

## ACCEPTED FILE TYPES: .png, .jpg, .jpeg,
ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg"]

## add a subheader
st.subheader("Parameters")

## add a text
st.markdown("## ROI" "Roi is really cool")

## add a button to upload an image either png or jpg
uploaded_file = st.file_uploader("Upload your image", type=ALLOWED_EXTENSIONS)

#################################################################
#### PARAMETERS 
#################################################################


## separator 
st.markdown("---")

## section for the parameters 
st.sidebar.subheader("Parameters")

## the following parameters are needed 

## REGION OF INTEREST
roi_x = st.sidebar.number_input("ROI X", min_value=0, max_value=1000, value=200, step=1)
roi_y = st.sidebar.number_input("ROI Y", min_value=0, max_value=1000, value=400, step=1)

## ROI 
REGION_OF_INTEREST = (roi_x, roi_y)

## BRITHNESS MULTIPLIERS - a slider 
BRIGHTNESS_MULTIPLIER = st.sidebar.slider("Brightness Multiplier", min_value=0.0, max_value=100.0, value=2.0, step=0.1)

## BLUR KERNEL - a tuple of (n,n) - a slider 
BLUR_KERNEL = st.sidebar.slider("Blur Kernel", min_value=0, max_value=100, value=3, step=1)
BLUR_KERNEL = (BLUR_KERNEL, BLUR_KERNEL)

## BLUR SIGMA 
BLUR_SIGMA = st.sidebar.slider("Blur Sigma", min_value=0.0, max_value=100.0, value=1.0, step=0.1)

## THRESHOLD - a slider
THRESHOLD_STAGE_1 = st.sidebar.slider("Canny Edge Detection - Threshold Stage 1", min_value=0, max_value=255, value=100, step=1)

## THRESHOLD - a slider
THRESHOLD_STAGE_2 = st.sidebar.slider("Canny Edge Detection - Threshold Stage 2", min_value=0, max_value=255, value=100, step=1)

## DILATION KERNEL - a slider, 2x2 
DILATION_KERNEL = st.sidebar.slider("Dilation Kernel", min_value=0, max_value=100, value=2, step=1)

## EROSION KERNEL - a slider, 1x1
EROSION_KERNEL = st.sidebar.slider("Erosion Kernel", min_value=0, max_value=100, value=1, step=1)

## DILATION ITERATIONS - a slider, 1
DILATION_ITERATIONS = st.sidebar.slider("Dilation Iterations", min_value=0, max_value=100, value=1, step=1)

## EROSION ITERATIONS - a slider, 2
EROSION_ITERATIONS = st.sidebar.slider("Erosion Iterations", min_value=0, max_value=100, value=2, step=1)


#################################################################
#### FUNCTIONS 
#################################################################

## function to read an image with cv2, takes in bytes and returns a numpy array
## type hinting and docstring
def read_image(image_bytes: bytes) -> np.ndarray:
    """
    Reads an image with cv2
    :param image_bytes: bytes of the image
    :return: numpy array of the image
    """
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    return image

## function to turn a list of images into a zipfile 
## type hinting and docstring


## the function to get the contours


#################################################################
#### MAIN APP
#################################################################

in_memory_image = read_image(uploaded_file.read()) if uploaded_file else None
## create a button 
if isinstance(in_memory_image, np.ndarray):
        #extract ROI
    ROI = in_memory_image[REGION_OF_INTEREST[0]:REGION_OF_INTEREST[1], REGION_OF_INTEREST[0]:REGION_OF_INTEREST[1]]

    #make a brighter image
    matrix = np.ones(ROI.shape) * BRIGHTNESS_MULTIPLIER
    IMG_BRIGHTER = np.uint8(cv2.multiply(np.float64(ROI),matrix))

    ## grey the image 
    IMG_GRAY = cv2.cvtColor(IMG_BRIGHTER,cv2.COLOR_BGR2GRAY)

    ## blur the image
    IMG_BLUR = cv2.GaussianBlur(IMG_GRAY,(BLUR_KERNEL),BLUR_SIGMA)
    ## canny edge 

    IMG_CANNY = cv2.Canny(IMG_BLUR,THRESHOLD_STAGE_1, THRESHOLD_STAGE_2)

    ## dilate 
    kerneldilate = np.ones(DILATION_KERNEL)

    ## erosion
    kernelerode = np.ones(EROSION_KERNEL)
    IMG_DIAL = cv2.dilate(IMG_CANNY, kerneldilate, iterations = DILATION_ITERATIONS)
    IMG_EROD = cv2.erode(IMG_DIAL, kernelerode, iterations=EROSION_ITERATIONS)

    #if showCanny: cv2.imshow("Canny",imgCanny)
    contours, _ = cv2.findContours(IMG_EROD, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    ## list of areas 
    arealst = [cv2.contourArea(cnt) for cnt in contours]

    ## perimeter list
    perimeter = [cv2.arcLength(cnt,True) for cnt in contours]

    ## DRAWING CONTOURS
    cv2.drawContours(IMG_BRIGHTER, contours, -1, (255,0,0),1)


    ## dictionary with images: IMG_BRIGHTER, ROI, img, IMG_BLUR, IMG_DIAL, IMG_EROD, IMG_CANNY
    ## and the area list and perimeter list 

    payload = {
        ## original image 
        "original_img": in_memory_image,
        ## ROI
        "roi": ROI,
        ## brighter image
        "brighter_img": IMG_BRIGHTER,
        ## blur image
        "blur_img": IMG_BLUR,
        ## dilate image
        "dilate_img": IMG_DIAL,
        ## erode image
        "erode_img": IMG_EROD,
        ## canny image
        "canny_img": IMG_CANNY,
        ## area list
        "area_lst": arealst,
        ## perimeter list
        "perimeter_lst": perimeter,
        ## contours
        "contours": contours
    }
    # if st.button("Show Image"):
    #     ## show a spinner
    #     with st.spinner("Processing Image..."):
    #         ## now get the contours of the image
    results_dict = payload
    

    ## first show the original image 
    # a, _ = st.columns(2)
    # a.image(results_dict["original_img"], caption="Original Image", use_column_width=True)

    ## now in a 3x2 grid, show the ROI, brighter image, blur image, dilate image, erode image, canny image
    col1, col2, col3 = st.columns(3)
    # with colextra: 
    #     ## show the original image
    #     st.image(results_dict["original_img"], caption="original_img", use_column_width=True)
    with col1:
        st.image(results_dict["roi"], caption="ROI", use_column_width=True)
    with col2:
        st.image(results_dict["brighter_img"], caption="Contours Image", use_column_width=True)
    with col3:
        st.image(results_dict["blur_img"], caption="Blur Image", use_column_width=True)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.image(results_dict["dilate_img"], caption="Dilate Image", use_column_width=True)
    with col5:
        st.image(results_dict["erode_img"], caption="Erode Image", use_column_width=True)
    with col6:
        st.image(results_dict["canny_img"], caption="Canny Image", use_column_width=True)
    
    ## now here show the results as a dataframe 
    ## dataframe of Area and Perimeter 
    df = pd.DataFrame({"Area": results_dict["area_lst"], "Perimeter": results_dict["perimeter_lst"]})
    df["Circularity"] = (df.Perimeter ** 2)/(4*(math.pi)*df.Area)
    # Remove infitiy values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #drop all NAN values
    df = df.dropna(axis=0)
    # Only display values with area greater than 10
    ## button to show the dataframe 
    ## 3 tabs: Dataframe, dataframe describe and plots
    tab1, tab2, tab3 = st.tabs(["Dataframe", "Dataframe Describe", "Plots"])
    with tab1:
        st.dataframe(df)
    with tab2:
        st.dataframe(df.describe())
    with tab3:
        ## show two plots per line 
        cola, colb = st.columns(2)
        ## show the plots with plt 
        with cola:
            ## Area vs Perimeter
            plt.scatter(df.Area, df.Perimeter)
            plt.xlabel("Area")
            plt.ylabel("Perimeter")
            plt.title("Area vs Perimeter")
            st.pyplot()
        with colb:
            ## Area vs Circularity
            plt.scatter(df.Area, df.Circularity)
            plt.xlabel("Area")
            plt.ylabel("Circularity")
            plt.title("Area vs Circularity")
            st.pyplot()
        
        colc, cold = st.columns(2)
        with colc:
            ## Perimeter vs Circularity
            plt.scatter(df.Perimeter, df.Circularity)
            plt.xlabel("Perimeter")
            plt.ylabel("Circularity")
            plt.title("Perimeter vs Circularity")
            st.pyplot()
        
        with cold:
            ## Histogram of Area
            plt.hist(df.Area)
            plt.xlabel("Area")
            plt.ylabel("Frequency")
            plt.title("Histogram of Area")
            st.pyplot()
        
        cole, colf = st.columns(2)
        with cole:
            ## Histogram of Perimeter
            plt.hist(df.Perimeter)
            plt.xlabel("Perimeter")
            plt.ylabel("Frequency")
            plt.title("Histogram of Perimeter")
            st.pyplot()
        
        with colf:
            ## Histogram of Circularity
            plt.hist(df.Circularity)
            plt.xlabel("Circularity")
            plt.ylabel("Frequency")
            plt.title("Histogram of Circularity")
            st.pyplot()

    ## put all the images in a list 
    imglst = [results_dict["original_img"], results_dict["roi"], results_dict["brighter_img"], results_dict["blur_img"], results_dict["dilate_img"], results_dict["erode_img"], results_dict["canny_img"]]

    ## convert all the images to numpy.ascontiguousarray
    imglst = [np.ascontiguousarray(img) for img in imglst]
    ## download images as a zip file
    ## create a zip file
    zip_file = zipfile.ZipFile("images.zip", "w")
    ## add all the images to the zip file
    for i in range(len(imglst)):
        ## save the image as a png file
        tmp_name = "image_{}.png".format(i)
        ## save the image to the zip file
        zip_file.writestr(f"image{i}.png", imglst[i])
    ## close the zip file
    zip_file.close()
    ## download the zip file
    st.download_button(label="Download Images", data=open("images.zip", "rb").read(), file_name="images.zip", mime="application/zip")