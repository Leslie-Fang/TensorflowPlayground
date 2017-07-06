import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./pic/figure_9.png')
img = cv2.GaussianBlur(img,(5,5),0)

print("origin")
plt.imshow(img,cmap = 'gray', interpolation = 'bicubic')
plt.show()

print("grey")
GrayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(GrayImage,cmap = 'gray', interpolation = 'bicubic')
plt.show()

print("thresh")
ret,threshImage = cv2.threshold(GrayImage,60,255,cv2.THRESH_BINARY_INV)
plt.imshow(threshImage,cmap = 'gray', interpolation = 'bicubic')
plt.show()

#threshImage = cv2.GaussianBlur(threshImage,(5,5),0)
#OpenCV structure element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
eroded = cv2.erode(threshImage,kernel)
dilated = cv2.dilate(eroded,kernel)
print("dilated's size: ")
print(dilated.shape)
print("erode and dilated: ")
plt.imshow(dilated,cmap = 'gray', interpolation = 'bicubic')
plt.show()
#print("canny")
#canny = cv2.Canny(threshImage, 50, 150)
#plt.imshow(canny,cmap = 'gray', interpolation = 'bicubic')
#plt.show()

print("contours")
im2,contours,hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
print("area")
max=cv2.contourArea(contours[0])
number=0
for i in range(len(contours)):
    print(cv2.contourArea(contours[i]))
    if cv2.contourArea(contours[i]) >= max:
        number = i

cnt = contours[number]
#M = cv2.moments(cnt)
#print( M )

#cv2.drawContours(dilated, cnt, 0, (0,255,0), 3)

x,y,w,h = cv2.boundingRect(cnt)
print("x,y,w,h")
print x
print y
print w
print h
#cv2.rectangle(dilated,(x,y),(x+w,y+h),(0,255,0),2)

#print("rectangle")
#plt.imshow(dilated,cmap = 'gray', interpolation = 'bicubic')
#plt.show()

# put the digit in the middle of the image
# enlarge the image to have more paddleing space
n = 100
xbegin = x-n
xend = x + w + n
ybegin = y - n
yend = y + h + n
if xbegin < 0:
    xbegin = 0
if xend > dilated.shape[1]:
    xend = dilated.shape[1]
if ybegin < 0:
    ybegin = 0
if yend > dilated.shape[0]:
    yend = dilated.shape[0]

digit = dilated[ybegin:yend,xbegin:xend]
print("digit's size: ")
print(digit.shape)

print("subimage")
plt.imshow(digit,cmap = 'gray', interpolation = 'bicubic')
plt.show()

# resize the shape to 28x28
res = cv2.resize(digit,(28,28),interpolation=cv2.INTER_CUBIC)
res = cv2.dilate(res,kernel)
res = cv2.erode(res,kernel)
#res = cv2.resize(dilated,(28,28),interpolation=cv2.INTER_CUBIC)
print("result")
plt.imshow(res,cmap = 'gray', interpolation = 'bicubic')
plt.show()
cv2.imwrite("res.png",res)


