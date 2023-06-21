#importing necessary libraries for image processing and numerical operations
import cv2
import numpy as np

#defining function to show image in window
def imshow(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#loading image and saving it ito variable and 0 indicates images should be loaded as gray scale.
mask = cv2.imread("OpenCV/images/img5.png", 0)

#convering gray scale image to BGR format
output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

#Inverse threshold to convert image in binary format 
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

#Finding contours in binary images
contours, [hist] = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#creating empty list to store information about the rectangle found in image.
rectangles = []

#This starts a loop to iterate over each contour and its corresponding hierarchy information.
for rect, (nxt, prv, first_child, parent) in zip(contours, hist):
    
    if parent == 0:
        
        _, _, line_index, _ = hist[first_child]
        line = contours[line_index]
        _, (width, height), _ = cv2.minAreaRect(line)
        line_length = max(width, height)      
        rectangle = cv2.minAreaRect(rect)
        box = cv2.boxPoints(rectangle)
        box = np.intp(box) 
        box_center = rectangle[0]
        
        rectangles.append((line_length, box, box_center))

#Sorting by line length
rectangles.sort()

#Assigning numbers to rectangles based on length
num_rectangles = len(rectangles)
for index, (_, box, (x, y)) in enumerate(rectangles):
    if index < num_rectangles * 0.25:
        number = 1
    elif index < num_rectangles * 0.5:
        number = 2
    elif index < num_rectangles * 0.75:
        number = 3
    else:
        number = 4
    
    #These lines iterate over the sorted rectangles list and assign a number (1, 2, 3, or 4) based on the index of each rectangle.
    cv2.drawContours(output, [box], 0, (0, 0, 255), 3)
    cv2.putText(output, str(number), (round(x), round(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 9)
    cv2.putText(output, str(number), (round(x), round(y)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

#displaying the final output of the image 
imshow(output)
