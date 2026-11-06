from ultralytics import YOLO
import cv2

# Завантажуємо навчену модель
model = YOLO("best_HW_03_11.pt")

img = input("Enter img name: ")
# Передаємо зображення моделі
results = model.predict(
    source=f"./assets/{img}",
    conf=0.25
)

# Отримуємо результат
result = results[0]

# Малюємо рамки навколо знайдених об'єктів
image = result.plot()

# Показуємо зображення
cv2.imshow("YOLO Ship", image)

cv2.waitKey(0)
cv2.destroyAllWindows()