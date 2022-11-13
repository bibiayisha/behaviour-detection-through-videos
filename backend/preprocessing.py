import cv2

def preprocessing(img): 
    # print('preprocessing')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    sift = cv2.SIFT_create() 
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors
