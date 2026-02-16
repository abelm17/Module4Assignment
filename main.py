import cv2, numpy as np, matplotlib.pyplot as plt

img= cv2.imread("giraffe.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Image not loaded properly")

blur= cv2.GaussianBlur(img, (11,11), 0)

ret, _ = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
adjusted_thresh = ret # Can add or subtract here to adjust the threshold as needed

_, thresh = cv2.threshold(
    blur,
    adjusted_thresh,
    255,
    cv2.THRESH_BINARY
)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    cv2.drawContours(thresh, contours, i, 255, -1)

contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output= cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for c in contours:
    cv2.drawContours(output, [c], -1, (0, 255, 0), 2)

cv2.imshow("Boundary", output)
cv2.waitKey(0)
cv2.destroyAllWindows()