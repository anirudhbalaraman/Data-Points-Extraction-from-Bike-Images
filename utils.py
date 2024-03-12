from PIL import Image
from PIL import Image, ImageFilter
from PIL import ImageEnhance

import cv2
import numpy as np
from sklearn.cluster import KMeans
from difflib import SequenceMatcher

from color_recognition import color_histogram_feature_extraction
from color_recognition import knn_classifier

import keras_ocr

def crop_image(img_path):
    #Crop images of chassis
    
    image = Image.open('dataset/'+img_path+'.jpg')  

    if abs(image.size[0] - 3000) < abs(image.size[0] - 6000) :
        # Define the cropping coordinates (left, upper, right, lower)
        left = 1000
        upper = 300
        right = 2000
        lower = 1300

    else:

        # Define the cropping coordinates (left, upper, right, lower)
        left = 2000
        upper = 1000
        right = 4000
        lower = 2500

    # Crop the image
    cropped_image = image.crop((left, upper, right, lower))
    cropped_image.save('dataset/'+img_path+'_cropped.jpg')
    
def find_closest_string(target, string_list):
    closest_similarity = 0
    closest_string = None
    for string in string_list:
        similarity = SequenceMatcher(None, target, string).ratio()
        if similarity > closest_similarity:
            closest_similarity = similarity
            closest_string = string
    return closest_string, closest_similarity

def find_highest_matching_strings(list1, list2):
    max_matching_string_list1 = None
    max_matching_string_list2 = None
    max_matching_similarity = 0
    for target_string_list1 in list1:
        for target_string_list2 in list2:
            _, similarity = find_closest_string(target_string_list1, [target_string_list2])
            if similarity > max_matching_similarity:
                max_matching_similarity = similarity
                max_matching_string_list1 = target_string_list1
                max_matching_string_list2 = target_string_list2
    return max_matching_string_list1, max_matching_string_list2


def contrast_image(img_path):
    image = cv2.imread('dataset/'+img_path+'.jpg')


    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # K-means clustering to group similar pixels together
    num_clusters = 3  # 3 clusters
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)

    # Cluster labels and reshape them to the original image shape
    cluster_labels = kmeans.labels_.reshape(image.shape[:2])

    # Generate contrasting colors for highlighting clusters
    contrast_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Example contrasting colors

    # Apply the contrasting colors to the image based on cluster labels
    highlighted_image = np.zeros_like(image)
    for label in range(num_clusters):
        highlighted_image[cluster_labels == label] = contrast_colors[label]

    cv2.imwrite('dataset/'+img_path+'_highlighted.jpg', cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))
    
 
def create_roi_lamp(img_path):
    
    image_path = 'dataset/'+img_path+'.jpg'
    img = Image.open(image_path).convert("L")
    binary_img = img.point(lambda p: 255 if p > 230 else 0)
    inverted_img = Image.eval(binary_img, lambda x: 255 - x)
    
    black_img = Image.new("RGB", inverted_img.size, "black")
    if inverted_img.size[1]>3000:
        roi_box=(3500,500,4500,2000)
    else:
        roi_box=(1750,250,2250,770)
    
    roi = inverted_img.crop(roi_box)
    black_img.paste(roi, roi_box)
    black_img.save('dataset/roi_lamp/roi_lamp_'+img_path+'.jpg')
    
    
def predict_color(source_image):
    
    ########## Generate Mask ########## 

    # Convert the image to grayscale
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to segment the objects
    _, thresholded = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    ###################################

    if thresholded.shape[1] > 1500:
        segmented_image = thresholded[:,700:1750]

        inverted_segmented_image = 255 - segmented_image


        # get the prediction
        color_histogram_feature_extraction.color_histogram_of_test_image(source_image[:,700:1750], inverted_segmented_image)
        prediction = knn_classifier.main('training.data', 'test.data')


    else:
        segmented_image = thresholded[:,250:750]

        inverted_segmented_image = 255 - segmented_image


        # get the prediction
        color_histogram_feature_extraction.color_histogram_of_test_image(source_image[:,250:750], inverted_segmented_image)
        prediction = knn_classifier.main('training.data', 'test.data')
        
    return prediction

def predict_brand(brands, images):
    
    pipeline = keras_ocr.pipeline.Pipeline()
    
    prediction_groups = pipeline.recognize(images)
    prediction_list = []
    for i in list(range(len(prediction_groups[0]))):
        prediction_list.append(prediction_groups[0][i][0])

    brands = [i.lower() for i in brands]
    prediction_list = [i.lower() for i in prediction_list]

    list1 = prediction_list
    list2 = brands


    highest_matching_string_list1, highest_matching_string_list2 = find_highest_matching_strings(list1, list2)
    print(f"The string from list1 with the highest matching is '{highest_matching_string_list1}'")
    print(f"The corresponding string from list2 is '{highest_matching_string_list2}'")
    print(highest_matching_string_list2)
    
    return highest_matching_string_list2

    
    