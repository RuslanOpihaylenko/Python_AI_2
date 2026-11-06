import cv2
import numpy as np

img = cv2.imread('../assets/image.jpg')

if img is None:
    print("Помилка! Зображення image.jpg не знайдено.")
    exit()

height, width = img.shape[:2]

small_img = cv2.resize(
    img,
    (width // 2, height // 2)
)

cv2.imshow('Original Image', img)
cv2.imshow('Resized Image', small_img)

swapped_img = img.copy()

swapped_img[:, :, [0, 1]] = swapped_img[:, :, [1, 0]]

cv2.imshow('Green and Blue Swapped', swapped_img)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Black and White Image', gray_img)


blur1 = cv2.GaussianBlur(gray_img, (3, 3), 0)
blur2 = cv2.GaussianBlur(gray_img, (5, 5), 0)
blur3 = cv2.GaussianBlur(gray_img, (7, 7), 0)

edges1 = cv2.Canny(blur1, 50, 150)
edges2 = cv2.Canny(blur2, 100, 200)
edges3 = cv2.Canny(blur3, 150, 250)

cv2.imshow('Canny 3x3', edges1)
cv2.imshow('Canny 5x5', edges2)
cv2.imshow('Canny 7x7', edges3)

cv2.imwrite('contours.jpg', edges2)

print("Зображення успішно оброблено!")
print("Результат збережено у файлі contours.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()
