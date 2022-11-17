#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#################################################################
####
#### IMPORTS
####
#################################################################

import io 
import tempfile
import os
import math
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from typing import List


#################################################################
####
#### CONFIG
####
#################################################################

st.set_option("deprecation.showPyplotGlobalUse", False)


## set the page config to wide mode and expanded sidebar, the page icon should be a cell
st.set_page_config(layout="wide", page_title="Mitochondria Quant", page_icon="🧬")


## put a title to the side bar
st.sidebar.title("Parameters")

## put a text to the side bar
st.sidebar.text("Adjust the parameters as needed!")

## add a header
st.header("Mitochondria Quant")

## add a subheader
st.subheader("Instructions")

## add set of instructions
st.markdown(
    """
- Upload the image that you want to analyze
- Adjust the parameters as needed
- Download the results
"""
)

## ACCEPTED FILE TYPES: .png, .jpg, .jpeg,
ALLOWED_EXTENSIONS: List[str] = ["png", "jpg", "jpeg"]

## add a button to upload an image either png or jpg
uploaded_file: bytes = st.file_uploader("Upload your image", type=ALLOWED_EXTENSIONS)


#################################################################
####
#### PARAMETERS
####
#################################################################


## separator
st.markdown("---")

## section for the parameters
st.sidebar.subheader("Parameters")

## the following parameters are needed
## button to reset the parameters
# if st.sidebar.button("Reset Parameters"):
#     ## clear cache 
#     st.cache(clear_cache=True)



## checkbox to save the plots or not 
# SAVE_PLOTS_FLAG = st.sidebar.checkbox("Save Plots to ZIP File?", value=True)
SAVE_PLOTS_FLAG = False

## outlier removal method
outlier_removal_method: str = st.sidebar.selectbox(
    "Outlier Removal Method", ["IQR", "Z-Score", "MAD"]
)
## three columns
multipliers1, multipliers2, multipliers3 = st.sidebar.columns(3)
## iqr multiplication factor - number input

IQR_MULTIPLIER: float = multipliers1.number_input(
    "IQR Multiplier", value=1.5, min_value=-5.0, max_value=10.0, step=0.1
)
## z-score threshold slider
Z_SCORE_THRESHOLD: float = multipliers2.number_input(
    "Z-Score Threshold", min_value=-5.0, max_value=10.0, value=3.0, step=0.1
)
## mad multiplication factor slider
MAD_MULTIPLIER: float = multipliers3.number_input(
    "MAD Multiplier", min_value=-5.0, max_value=10.0, value=3.0, step=0.1
)

## REGION OF INTERESTS 

## make 2 columns
st.sidebar.write("Select the region of interest")
roi_col_x_start, roi_col_x_end = st.sidebar.columns(2)
## roi_x start
ROI_X_START: int = roi_col_x_start.number_input("Start - ROI X", min_value=0, max_value=1000, value=200, step=1)
## roi_x end
ROI_X_END: int = roi_col_x_end.number_input("End - ROI X", min_value=0, max_value=1000, value=400, step=1)

roi_col_y_start, roi_col_y_end = st.sidebar.columns(2)
## roi_y start
ROI_Y_START: int = roi_col_y_start.number_input("Start - ROI Y", min_value=0, max_value=1000, value=200, step=1)
## roi_y end
ROI_Y_END: int = roi_col_y_end.number_input("End - ROI Y", min_value=0, max_value=1000, value=400, step=1)


## ROI
# REGION_OF_INTEREST: Tuple[float, float, float, float] = (roi_x_start, roi_x_end, roi_y_start, roi_y_end)

## BRITHNESS MULTIPLIERS - a slider
BRIGHTNESS_MULTIPLIER: float = st.sidebar.slider(
    "Brightness Multiplier", min_value=0.0, max_value=100.0, value=2.0, step=0.1
)

## BLUR KERNEL - a tuple of (n,n) - a slider
BLUR_KERNEL: float = st.sidebar.slider(
    "Blur Kernel", min_value=0, max_value=100, value=3, step=1
)
BLUR_KERNEL:float = (BLUR_KERNEL, BLUR_KERNEL)

## BLUR SIGMA
BLUR_SIGMA:float = st.sidebar.slider(
    "Blur Sigma", min_value=0, max_value=100, value=1, step=1
)

## THRESHOLD - a slider
THRESHOLD_STAGE_1: int = st.sidebar.slider(
    "Canny Edge Detection - Threshold Stage 1",
    min_value=0,
    max_value=255,
    value=100,
    step=1,
)

## THRESHOLD - a slider
THRESHOLD_STAGE_2: int = st.sidebar.slider(
    "Canny Edge Detection - Threshold Stage 2",
    min_value=0,
    max_value=255,
    value=100,
    step=1,
)

## DILATION KERNEL - a slider, 2x2
DILATION_KERNEL: int = st.sidebar.slider(
    "Dilation Kernel", min_value=0, max_value=100, value=2, step=1
)

## EROSION KERNEL - a slider, 1x1
EROSION_KERNEL: int = st.sidebar.slider(
    "Erosion Kernel", min_value=0, max_value=100, value=1, step=1
)

## DILATION ITERATIONS - a slider, 1
DILATION_ITERATIONS: int = st.sidebar.slider(
    "Dilation Iterations", min_value=0, max_value=100, value=1, step=1
)

## EROSION ITERATIONS - a slider, 2
EROSION_ITERATIONS: int = st.sidebar.slider(
    "Erosion Iterations", min_value=0, max_value=100, value=2, step=1
)


#################################################################
#### 
#### FUNCTIONS
####
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


## 3 methods of Outlier Removal

## 1. IQR
def remove_outliers_iqr(
    data: List[float], iqr_multiplier: float = IQR_MULTIPLIER
) -> List[float]:
    """
    Removes outliers using the IQR method
    :param data: list of data
    :param iqr_multiplier: multiplier for the IQR
    :return: list of data without outliers
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (iqr_multiplier * iqr)
    upper_bound = q3 + (iqr_multiplier * iqr)
    return [x for x in data if lower_bound <= x <= upper_bound]


## 2. Z-Score
def remove_outliers_z_score(
    data: List[float], z_score_threshold: float = Z_SCORE_THRESHOLD
) -> List[float]:
    """
    Removes outliers using the Z-Score method
    :param data: list of data
    :param z_score_threshold: threshold for the Z-Score
    :return: list of data without outliers
    """
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(x - mean) / std for x in data]
    return [
        x
        for x, z in zip(data, z_scores)
        if -z_score_threshold <= z <= z_score_threshold
    ]


## 3. MAD
def remove_outliers_mad(
    data: List[float], mad_multiplier: float = MAD_MULTIPLIER
) -> List[float]:
    """
    Removes outliers using the MAD method
    :param data: list of data
    :param mad_multiplier: multiplier for the MAD
    :return: list of data without outliers
    """
    median = np.median(data)
    median_absolute_deviation = np.median([np.abs(x - median) for x in data])
    modified_z_scores = [
        0.6745 * (x - median) / median_absolute_deviation for x in data
    ]
    return [
        x
        for x, z in zip(data, modified_z_scores)
        if -mad_multiplier <= z <= mad_multiplier
    ]
    

## function to save the plot as io.BytesIO() object
def save_plot(fig: plt.Figure) -> io.BytesIO:
    """
    Saves the plot as io.BytesIO() object
    :param fig: the figure to save
    :return: io.BytesIO() object
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf
# fn = 'scatter.png'
# img = io.BytesIO()
# plt.savefig(img, format='png')

# btn = st.download_button(
#    label="Download image",
#    data=img,
#    file_name=fn,
#    mime="image/png"

## function that given a dataframe, and a outlier removal method, returns a dataframe without outliers
def remove_outliers(
    df: pd.DataFrame, outlier_removal_method: str, column: str
) -> pd.DataFrame:
    """
    Removes outliers from a dataframe
    :param df: dataframe
    :param outlier_removal_method: outlier removal method
    :param column: column to remove outliers from
    :return: dataframe without outliers
    """
    values_to_clean = df[column].values.tolist()
    if outlier_removal_method == "IQR":
        cleaned_values = remove_outliers_iqr(values_to_clean)
    elif outlier_removal_method == "Z-Score":
        cleaned_values = remove_outliers_z_score(values_to_clean)
    elif outlier_removal_method == "MAD":
        cleaned_values = remove_outliers_mad(values_to_clean)
    return cleaned_values


#################################################################
####
#### MAIN APP
####
#################################################################

in_memory_image = read_image(uploaded_file.read()) if uploaded_file else None
# TEST_IMAGE = "/Users/eric/Documents/Locaria/Projects/repos/mitochondria_detection/images/original_img.png"
# in_memory_image = cv2.imread(TEST_IMAGE)

## create a button
if isinstance(in_memory_image, np.ndarray):
    # extract ROI
    ## get the dimensions of the image 
    height, width = in_memory_image.shape[:2]
    ROI: np.array = in_memory_image[
        ROI_X_START : ROI_X_END,
        ROI_Y_START : ROI_Y_END,
    ]
    
    ## grey the image
    IMG_GRAY = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

    ## multiply the matrix & keep one in color 
    matrix = np.uint8(np.ones_like(IMG_GRAY) * BRIGHTNESS_MULTIPLIER)
    matrix_color = np.uint8(np.ones_like(ROI) * BRIGHTNESS_MULTIPLIER)
    ## remove last channel from matrix 

    ## make the image brighter
    IMG_BRIGHTER = np.uint8(cv2.multiply(ROI, matrix_color))

    ## blur the image
    IMG_BLUR = cv2.GaussianBlur(IMG_GRAY, (BLUR_KERNEL), BLUR_SIGMA)
    ## canny edge

    IMG_CANNY = cv2.Canny(IMG_BLUR, THRESHOLD_STAGE_1, THRESHOLD_STAGE_2)

    ## dilate
    kerneldilate = np.ones(DILATION_KERNEL)

    ## erosion
    kernelerode = np.ones(EROSION_KERNEL)
    IMG_DIAL = cv2.dilate(IMG_CANNY, kerneldilate, iterations=DILATION_ITERATIONS)
    IMG_EROD = cv2.erode(IMG_DIAL, kernelerode, iterations=EROSION_ITERATIONS)

    # if showCanny: cv2.imshow("Canny",imgCanny)
    contours, _ = cv2.findContours(IMG_EROD, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ## list of areas
    arealst = [cv2.contourArea(cnt) for cnt in contours]

    ## perimeter list
    perimeter = [cv2.arcLength(cnt, True) for cnt in contours]

    ## DRAWING CONTOURS
    cv2.drawContours(IMG_BRIGHTER, contours, -1, (255, 0, 0), 1)

    ## dictionary with images: IMG_BRIGHTER, ROI, img, IMG_BLUR, IMG_DIAL, IMG_EROD, IMG_CANNY
    ## and the area list and perimeter list

    results_dict = {
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
        "contours": contours,
    }

    ## now in a 3x2 grid, show the ROI, brighter image, blur image, dilate image, erode image, canny image
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(results_dict["roi"], caption="ROI", use_column_width=True)
    with col2:
        st.image(
            results_dict["brighter_img"],
            caption="Brighter Image",
            use_column_width=True,
        )
    with col3:
        st.image(results_dict["blur_img"], caption="Blur Image", use_column_width=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.image(
            results_dict["dilate_img"], caption="Dilate Image", use_column_width=True
        )
    with col5:
        st.image(
            results_dict["erode_img"], caption="Erode Image", use_column_width=True
        )
    with col6:
        st.image(
            results_dict["canny_img"], caption="Canny Image", use_column_width=True
        )

    ## now here show the results as a dataframe
    ## dataframe of Area and Perimeter
    df = pd.DataFrame(
        {"Area": results_dict["area_lst"], "Perimeter": results_dict["perimeter_lst"]}
    )
    df["Circularity"] = (df.Perimeter**2) / (4 * (math.pi) * df.Area)
    # Remove infitiy values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # drop all NAN values
    df = df.dropna(axis=0)

    ## 3 tabs: Dataframe, dataframe describe and plots
    tab1, tab2, tab3 = st.tabs(["Dataframe", "Dataframe Describe", "Plots"])
    with tab1:
        st.dataframe(df)
    with tab2:
        st.dataframe(df.describe())
    if SAVE_PLOTS_FLAG: 
        PLOT_LISTS = []
    with tab3:
        ## three columns: Area vs Perimeter, Area vs Circularity, Perimeter vs Circularity
        cola, colb, colc = st.columns(3)
        ## these should be scatter plots
        with cola:
            ## area vs perimeter
            ## create the figure 
            fig = plt.scatter(df.Area, df.Perimeter)
            plt.xlabel("Area")
            plt.ylabel("Perimeter")
            plt.title("Area vs Perimeter")
            ## save the figure as a png
            if SAVE_PLOTS_FLAG:
                fn = "Area_vs_Perimeter.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()
        with colb:
            ## area vs circularity
            ## create the figure
            fig = plt.scatter(df.Area, df.Circularity)
            plt.xlabel("Area")
            plt.ylabel("Circularity")
            plt.title("Area vs Circularity")
            if SAVE_PLOTS_FLAG:
                fn = "Area_vs_Circularity.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()
        with colc:
            ## perimeter vs circularity
            ## create the figure
            fig = plt.scatter(df.Perimeter, df.Circularity)
            # plt.scatter(df.Perimeter, df.Circularity)
            plt.xlabel("Perimeter")
            plt.ylabel("Circularity")
            plt.title("Perimeter vs Circularity")
            if SAVE_PLOTS_FLAG:
                fn = "Perimeter_vs_Circularity.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()

        ## now 3 more columns for the histograms
        cold, cole, colf = st.columns(3)
        with cold:
            ## area histogram
            fig = plt.hist(df.Area)
            # plt.hist(df.Area)
            plt.title("Area Histogram")
            plt.xlabel("Area")
            plt.ylabel("Frequency")
            if SAVE_PLOTS_FLAG:
                fn = "Area_Histogram.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()

        with cole:
            ## perimeter histogram
            # plt.hist(df.Perimeter)
            fig = plt.hist(df.Perimeter)
            plt.title("Perimeter Histogram")
            plt.xlabel("Perimeter")
            plt.ylabel("Frequency")
            if SAVE_PLOTS_FLAG:
                fn = "Perimeter_Histogram.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()

        with colf:
            ## circularity histogram
            # plt.hist(df.Circularity)
            fig = plt.hist(df.Circularity)
            plt.title("Circularity Histogram")
            plt.xlabel("Circularity")
            plt.ylabel("Frequency")
            if SAVE_PLOTS_FLAG:
                fn = "Circularity_Histogram.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()

        ## now 3 more columns for the boxplots
        colg, colh, coli = st.columns(3)
        with colg:
            ## area boxplot
            # plt.boxplot(df.Area, vert=False)
            fig = plt.boxplot(df.Area, vert=False)
            plt.title("Area Boxplot")
            plt.xlabel("Area")
            if SAVE_PLOTS_FLAG:
                fn = "Area_Boxplot.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()

        with colh:
            ## perimeter boxplot
            # plt.boxplot(df.Perimeter, vert=False)
            fig = plt.boxplot(df.Perimeter, vert=False)
            plt.title("Perimeter Boxplot")
            plt.xlabel("Perimeter")
            if SAVE_PLOTS_FLAG:
                fn = "Perimeter_Boxplot.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()

        with coli:
            ## circularity boxplot
            plt.boxplot(df.Circularity, vert=False)
            fig = plt.boxplot(df.Circularity, vert=False)
            plt.title("Circularity Boxplot")
            plt.xlabel("Circularity")
            if SAVE_PLOTS_FLAG:
                fn = "Circularity_Boxplot.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()

        ## now 3 more columns for the violin plots
        colj, colk, coll = st.columns(3)
        with colj:
            ## area violin plot
            # plt.violinplot(df.Area)
            fig = plt.violinplot(df.Area)
            plt.title("Area Violin Plot")
            plt.xlabel("Area")
            if SAVE_PLOTS_FLAG:
                fn = "Area_Violin_Plot.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()

        with colk:
            ## perimeter violin plot
            # plt.violinplot(df.Perimeter)
            fig = plt.violinplot(df.Perimeter)
            plt.title("Perimeter Violin Plot")
            plt.xlabel("Perimeter")
            if SAVE_PLOTS_FLAG:
                fn = "Perimeter_Violin_Plot.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()

        with coll:
            ## circularity violin plot
            # plt.violinplot(df.Circularity)
            fig = plt.violinplot(df.Circularity)
            plt.title("Circularity Violin Plot")
            plt.xlabel("Circularity")
            if SAVE_PLOTS_FLAG:
                fn = "Circularity_Violin_Plot.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()

        ## show the same boxplots, but remove the outliers for the speciified column on the
        st.write(f"Removed Outliers with Method: {outlier_removal_method}")
        removed_outliers_area = remove_outliers(
            df, outlier_removal_method=outlier_removal_method, column="Area"
        )
        removed_outliers_perimeter = remove_outliers(
            df, outlier_removal_method=outlier_removal_method, column="Perimeter"
        )
        removed_outliers_circularity = remove_outliers(
            df, outlier_removal_method=outlier_removal_method, column="Circularity"
        )
        col1_no_outliers, col2_no_outliers, col3_no_outliers = st.columns(3)
        with col1_no_outliers:
            ## area boxplot - no outliers

            # plt.boxplot(removed_outliers_area, vert=False)
            fig = plt.boxplot(removed_outliers_area, vert=False)
            plt.title(f"Area Boxplot - No Outliers {outlier_removal_method}")
            plt.xlabel("Area")
            if SAVE_PLOTS_FLAG:
                fn = f"Area_Boxplot_No_Outliers_{outlier_removal_method}.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()
        ## perimeter boxplot - no outliers
        with col2_no_outliers:

            # plt.boxplot(removed_outliers_perimeter, vert=False)
            fig = plt.boxplot(removed_outliers_perimeter, vert=False)
            plt.title(f"Perimeter Boxplot - No Outliers {outlier_removal_method}")
            plt.xlabel("Perimeter")
            if SAVE_PLOTS_FLAG:
                fn = f"Perimeter_Boxplot_No_Outliers_{outlier_removal_method}.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()
        ## circularity boxplot - no outliers
        with col3_no_outliers:
            # plt.boxplot(removed_outliers_circularity, vert=False)
            fig = plt.boxplot(removed_outliers_circularity, vert=False)
            plt.title(f"Circularity Boxplot - No Outliers {outlier_removal_method}")
            plt.xlabel("Circularity")
            if SAVE_PLOTS_FLAG:
                fn = f"Circularity_Boxplot_No_Outliers_{outlier_removal_method}.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()

        ## the same with the violin plots
        col4_no_outliers, col5_no_outliers, col6_no_outliers = st.columns(3)
        with col4_no_outliers:
            ## area violin plot - no outliers
            # plt.violinplot(removed_outliers_area)
            fig = plt.violinplot(removed_outliers_area)
            plt.title(f"Area Violin Plot - No Outliers {outlier_removal_method}")
            plt.xlabel("Area")
            if SAVE_PLOTS_FLAG:
                fn = f"Area_Violin_Plot_No_Outliers_{outlier_removal_method}.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()
        ## perimeter violin plot - no outliers
        with col5_no_outliers:
            # plt.violinplot(removed_outliers_perimeter)
            fig = plt.violinplot(removed_outliers_perimeter)
            plt.title(f"Perimeter Violin Plot - No Outliers {outlier_removal_method}")
            plt.xlabel("Perimeter")
            if SAVE_PLOTS_FLAG:
                fn = f"Perimeter_Violin_Plot_No_Outliers_{outlier_removal_method}.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()
        ## circularity violin plot - no outliers
        with col6_no_outliers:
            # plt.violinplot(removed_outliers_circularity)
            fig = plt.violinplot(removed_outliers_circularity)
            plt.title(f"Circularity Violin Plot - No Outliers {outlier_removal_method}")
            plt.xlabel("Circularity")
            if SAVE_PLOTS_FLAG:
                fn = f"Circularity_Violin_Plot_No_Outliers_{outlier_removal_method}.png"
                PLOT_LISTS.append((fn, save_plot(fig)))
            st.pyplot()

    ## put all the images in a list
    imglst = [
        results_dict["original_img"],
        results_dict["roi"],
        results_dict["brighter_img"],
        results_dict["blur_img"],
        results_dict["dilate_img"],
        results_dict["erode_img"],
        results_dict["canny_img"],
    ]

    ## SAVING ALL THE IMAGES INTO A ZIP FILE 
    ## convert all the images to numpy.ascontiguousarray
    imglst = [np.ascontiguousarray(img) for img in imglst]

    ## create a zipfile at a temporary location
    with tempfile.TemporaryDirectory() as tmpdir:
        ## create a zip file
        zip_file = zipfile.ZipFile(os.path.join(tmpdir, "images.zip"), "w")
        ## add all the images to the zip file
        for i, img in enumerate(imglst):
            ## the image name is the variable name of the image
            imgname = list(results_dict.keys())[i]
            ## save the image to the zip file
            ## convert the image to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            zip_file.writestr(f"{imgname}.png", cv2.imencode(".png", img)[1].tobytes())
        ## add the dataframe to the zip file
        zip_file.writestr("dataframe.csv", df.to_csv())
        ## iterate over the plot list and add the plots to the zip file
        if SAVE_PLOTS_FLAG:
            for fn, plot in PLOT_LISTS:
                zip_file.writestr(fn, plot)
            
        ## close the zip file
        zip_file.close()
        ## now download the zip file
        st.download_button(
            label="Download Images as zip",
            data=open(zip_file.filename, "rb").read(),
            file_name="images.zip",
            mime="application/zip",
        )
