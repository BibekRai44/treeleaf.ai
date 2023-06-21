#importing necessary libaries for image processing and numerical operations
import cv2
import numpy as np

#defining a function that takes an image as input and returns a list of aligned rectangle images.
def align_rectangles(image):
    #converting input imagae into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #performing threshold on grayscale image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #finding contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #making empty list to store information about the rectangle found in image.
    aligned_images = []
    
    #iterating over each contour
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        #determing the angle of rotation for aligning the rectangle.
        width, height = rect[1]
        if width > height:
            angle = rect[2]
        else:
            angle = rect[2] + 90
        
        #creating rotation matrix and applying rotation transformation
        rotation_matrix = cv2.getRotationMatrix2D(rect[0], angle, 1)
        aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        
        #Croping the aligned image to remove any borders or areas
        rect_points = cv2.transform(np.array([box]), rotation_matrix).squeeze().astype(int)
        x, y, w, h = cv2.boundingRect(rect_points)
        aligned_image = aligned_image[y:y+h, x:x+w]
        
        aligned_images.append(aligned_image)
    
    return aligned_images

#reading images 
image = cv2.imread('OpenCV/images/img5.png')
aligned_images = align_rectangles(image)

#displaying the aligned images
for i, aligned_image in enumerate(aligned_images):
    cv2.imshow(f'Aligned Image {i+1}', aligned_image)

#closing displayed windows
cv2.waitKey(0)
cv2.destroyAllWindows()
