import pyautogui
import time

def mover_y_hacer_click():
    while True:
        # Mueve el mouse un poco para simular actividad
        pyautogui.move(0, 1)
        pyautogui.move(0, -1)
        
        # Hacer clic en la posición actual del mouse
        pyautogui.click()
        print("click")
        
        # Esperar 20 segundos antes de hacer otro clic
        time.sleep(20)

        pyautogui.press('shift')  # Simula una pulsación de la tecla Shift

        print('shift')

        time.sleep(20)

if __name__ == "__main__":
    try:
        mover_y_hacer_click()
    except KeyboardInterrupt:
        print("Programa detenido por el usuario.")
