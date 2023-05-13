#Q1 part a
import cv2
import numpy as np
import os
def write_img(path):  # this function will update the final image
    grayscaling=cv2.imread("static/images/input.jpg") # convert to grayscale
    cv2.imwrite("static/images/final.jpg",grayscaling)  # writting the updated image
    print ("written")
    return grayscaling #returing the output

def convert_img(grayscaling): #this function will convert image to grayscale
    image = cv2.imdecode(np.frombuffer(grayscaling, np.uint8), cv2.COLOR_BGR2GRAY) #reading the img
    if len(image.shape) == 3: # checking grayscale status
        grayscaling =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # if not then convert to grayscale
        cv2.imwrite("static/images/input.jpg",grayscaling) #writing the img
    else:
         cv2.imwrite("static/images/input.jpg",image) #writting the img
    print ("hi")
    return grayscaling #returing the output

def blur(path,size_k): #this function will make the image blur
    grayscaling = cv2.imread(path) #loading the iamge
    if len(grayscaling.shape) == 3:
        grayscaling =cv2.cvtColor(grayscaling, cv2.COLOR_BGR2GRAY) #converting to grayscale
    blur_image = cv2.GaussianBlur(grayscaling, (size_k,size_k), 0)  #appling gaussain blur to reduce noise
    cv2.imwrite("static/images/final.jpg",blur_image) #writting the img
    print ("hi")
    return blur_image #returing the output
def sharpen_image(path):
    
    image1 = cv2.imread(path) #loading the iamge
    grayscaling =cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) #converting to grayscale
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened_image = cv2.filter2D(grayscaling, -1, kernel)# Apply the kernel to the input image
    cv2.imwrite("static/images/final.jpg",sharpened_image)  #writting the img
    return sharpened_image #returing the output

def detect_edges_canny(path,size_k):
    image1 = cv2.imread(path) #loading the iamge
    grayscaling =cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) #converting to grayscale
    blurred = cv2.GaussianBlur(grayscaling, (size_k, size_k), 0) # Apply Gaussian blur to reduce noise
    edges = cv2.Canny(blurred, 100, 200) # Apply Canny edge detection with the specified thresholds
    cv2.imwrite("static/images/final.jpg",edges)  #writting the img
    return edges #returing the output
def threshold_image(path, threshold_value):
    image1 = cv2.imread(path) #loading the iamge
    grayscaling =cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) #converting to grayscale
    _, thresholded_image = cv2.threshold(grayscaling, threshold_value, 255, cv2.THRESH_BINARY) #appliy threshlod
    cv2.imwrite("static/images/final.jpg",thresholded_image)  #writting the img
    return thresholded_image #returing the output

def edge_based_thresholding(path, low_threshold, high_threshold):
    image1 = cv2.imread(path) #loading the iamge
    grayscaling =cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) #converting to grayscale
    edges = cv2.Canny(grayscaling, low_threshold, high_threshold) #apply canny
    _, thresholded_image = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY) #apply threshold
    cv2.imwrite("static/images/final.jpg",thresholded_image)  #writting the img
    return thresholded_image #returing the output

def region_based_thresholding(path, block_size, constant):
    image1 = cv2.imread(path) #loading the iamge
    grayscaling =cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) #converting to grayscale
    thresholded_image = cv2.adaptiveThreshold(grayscaling, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)  #apply threshold
    cv2.imwrite("static/images/final.jpg",thresholded_image)  #writting the img
    return thresholded_image #returing the output

def lossless_compress_image(path,level):
    image1 = cv2.imread(path) #loading the iamge
    grayscaling =cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) #converting to grayscale
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), level]  # Use maximum compression level
    _, compressed = cv2.imencode('.jpg', grayscaling, encode_param) #compress the image
    compressed_image = cv2.imdecode(np.frombuffer(compressed, np.uint8), cv2.IMREAD_UNCHANGED) #decode the image
    cv2.imwrite("static/images/final.jpg",compressed_image)  #writting the img
    return compressed_image #returing the output

def lossy_compress_image(path,quality):
    image1 = cv2.imread(path) #loading the iamge
    grayscaling =cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) #converting to grayscale
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality] #set parameter for encode
    _, compressed = cv2.imencode('.jpg', grayscaling, encode_param) #apply encoding
    compressed_image = cv2.imdecode(np.frombuffer(compressed, np.uint8), cv2.IMREAD_UNCHANGED) # apply decoding
    cv2.imwrite("static/images/final.jpg",compressed_image)  #writting the img
    return compressed_image #returing the output

def crop_image(path,x=0,y=0, width=0, height=0):
    image = cv2.imread(path)
    x_end = x + width # Define the crop region
    y_end = y + height
    cropped_image = image[y:y_end, x:x_end] # Crop the image
    cv2.imwrite("static/images/final.jpg",cropped_image)  #writting the img
    return cropped_image #returing the output

def restore(path):
    image = cv2.imread(path) # Load the image
    cv2.imwrite("static/images/final.jpg",image)  #writting the img
    return image #returing the output

def rotate_left(path):
    img = cv2.imread(path) # Load the image
    img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # Rotate the image to the left
    cv2.imwrite("static/images/final.jpg",img_rotated)  #writting the img
    return img_rotated #returing the output

def rotate_right(path):
    img = cv2.imread(path)  # Load the image
    rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # Rotate the image to the left
    cv2.imwrite("static/images/final.jpg",rotated_img)  #writting the img
    return rotated_img #returing the output

def rotate_top(path):
    img = cv2.imread(path) # Load the image
    rotated_image = cv2.rotate(img, cv2.ROTATE_180) # Rotate the image 180 degrees
    cv2.imwrite("static/images/final.jpg",rotated_image)  #writting the img
    return rotated_image #returing the output

def rotate_bottom(path):
    img = cv2.imread(path)  # Load the image
    img_rotated = cv2.rotate(img, cv2.ROTATE_180) # Rotate the image by 180 degrees clockwise (to the bottom)
    cv2.imwrite("static/images/final.jpg",img_rotated)  #writting the img
    return img_rotated #returing the output

def colarization(path):
    # Paths to load the model
    PROTOTXT = "models/colorization_deploy_v2.prototxt"
    POINTS = "models/pts_in_hull.npy"
    MODEL = "models/colorization_release_v2.caffemodel"
    print("Load model")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL) # Load the model
    pts = np.load(POINTS)  #load the points

    class8 = net.getLayerId("class8_ab") # Load centers for ab(green-red, blue-yellow) channel quantization .
    conv8 = net.getLayerId("conv8_313_rh") 
    pts = pts.transpose().reshape(2, 313, 1, 1) #reshapping the img
    net.getLayer(class8).blobs = [pts.astype("float32")] #convert to float32
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")] #getting layer

    image = cv2.imread(path) # Load the input image
    scaled = image.astype("float32") / 255.0 #scale to 255
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB) # convert color from bgr to lab

    resized = cv2.resize(lab, (224, 224)) #reszing
    L = cv2.split(resized)[0] # getting L layer
    L -= 50 #subtracting 50 from L

    print("Colorizing the image")
    net.setInput(cv2.dnn.blobFromImage(L)) #applig colorization
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    cv2.imwrite("static/images/final.jpg",colorized)  #writting the img
    return colorized #returing the output