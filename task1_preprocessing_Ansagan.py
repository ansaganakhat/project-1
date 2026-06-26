"""
ТАПСЫРМА 1: Суретті алдын ала өңдеу конвейері
================================================

Мақсат: бір суретке бірнеше өңдеу қадамын ретімен қолдану.

Нұсқаулық:
  1) monkey.webp суретін оқыңыз.
  2) Оны сұр түске айналдырыңыз.
  3) Гаусс сүзгісімен бұлдыратыңыз (шуды азайту үшін).
  4) Canny әдісімен жиектерді табыңыз.
  5) Әр нәтижені бөлек файлға сақтаңыз.

Қажет: opencv-python (pip install opencv-python)
Әр TODO жерін өзің толтыр.
"""

import cv2


def load_image(path):
    
    img = cv2.imread(path)

    
    if img is None:
        raise FileNotFoundError(f"Сурет табылмады: {path}")

    return img


def to_gray(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def blur_image(gray):
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def find_edges(blurred):
    
    edges = cv2.Canny(blurred, 100, 200)
    return edges


def main():
    img = load_image("C:\\Users\\Ansagan\\Downloads\\monkey.webp")

    gray = to_gray(img)
    cv2.imwrite("out_gray.jpg", gray)

    blurred = blur_image(gray)
    cv2.imwrite("out_blur.jpg", blurred)

    edges = find_edges(blurred)
    cv2.imwrite("out_edges.jpg", edges)

    print("Дайын!")


if __name__ == "__main__":
    main()