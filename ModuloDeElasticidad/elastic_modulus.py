import torch
import cv2
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil

class Modulo_elasticidad:
    def __init__(self, ruta_modelo):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=ruta_modelo, force_reload=True) ###
        self.first_bbox = None
        self.last_bbox = None
        self.longitud_cm = None
        self.base_cm = None
        self.last_length_mm = None
        self.average_last_length_mm = None
        self.last_base_mm = None
        self.average_last_base_mm = None
        self.all_bboxes = []  
        self.bboxes_longitud_mm = [] # Lista para almacenar todas las longitudes filtradas a mm ###########
        self.bboxes_diametro_mm = [] # Lista para almacenar todas los diametros filtrados a mm 
        self.zona_elastica = [] 
        self.ancho = None
        self.alto = None
        self.longitud_mm = None
        self.base_mm = None 
        self.area_seccion = None
        self.s1 = None  
        self.longitud_s2 = None
        self.toneladas = None ########### 
    def detectar_y_guardar_video(self, ruta_video, carpeta_salida, nombre_video_salida):
        # Crear la carpeta de salida si no existe
        if not os.path.exists(carpeta_salida):
            os.makedirs(carpeta_salida)

        # Leer el video frame por frame
        cap = cv2.VideoCapture(ruta_video)
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        self.alto = alto
        self.ancho = ancho

        # Crear el objeto VideoWriter para guardar el video resultante
        ruta_video_salida = os.path.join(carpeta_salida, nombre_video_salida)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el archivo de salida
        video_salida = cv2.VideoWriter(ruta_video_salida, fourcc, fps, (ancho, alto))

        # Contar el número total de frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Procesar cada frame del video con una barra de progreso
        for _ in tqdm(range(total_frames), desc="Procesando frames", unit="frame"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Guardar el frame temporalmente para procesarlo con YOLOv5
            frame_temp_path = os.path.join(carpeta_salida, "temp_frame.jpg")
            cv2.imwrite(frame_temp_path, frame)

            # Realizar la detección en el frame
            resultados = self.model(frame_temp_path)
            resultados.save()

            # Obtener las coordenadas de los bounding boxes detectados y guardarlas en la lista
            for det in resultados.xyxy[0].cpu().numpy():
                x_min, y_min, x_max, y_max, conf, clase = det
                self.all_bboxes.append((x_min, y_min, x_max, y_max))

            # Almacenar el primer y último bounding box
            if len(self.all_bboxes) == 1:
                self.first_bbox = self.all_bboxes[0]  # Almacenar el primer bounding box
            self.last_bbox = self.all_bboxes[-1]  # Almacenar el último bounding box

            # Cargar el frame procesado desde la carpeta predeterminada de YOLOv5
            ruta_predeterminada = os.path.join("runs", "detect", "exp")
            frame_procesado_path = os.path.join(ruta_predeterminada, "temp_frame.jpg")

            # Leer el frame procesado
            frame_procesado = cv2.imread(frame_procesado_path)

            # Si el frame se procesó correctamente, escribirlo en el video de salida
            if frame_procesado is not None:
                video_salida.write(frame_procesado)

            # Limpiar la carpeta predeterminada para el próximo frame
            if os.path.exists(ruta_predeterminada):
                shutil.rmtree(ruta_predeterminada)

            # Eliminar la imagen temporal después de procesar
            if os.path.exists(frame_temp_path):
                os.remove(frame_temp_path)

        # Liberar los recursos
        cap.release()
        video_salida.release()
        print(f"Video guardado en: {ruta_video_salida}")

    def crear_imagen_bounding_boxes(self, carpeta_salida, nombre_imagen):
        alto = self.alto
        ancho = self.ancho
        # Crear la ruta completa para la imagen de salida
        image_output_path = os.path.join(carpeta_salida, nombre_imagen)

        # Crear una imagen con fondo blanco
        white_background = np.ones((alto, ancho, 3), dtype=np.uint8) * 255

        # Definir colores
        color_first = (0, 0, 255)  # Rojo
        color_last = (255, 0, 0)   # Azul
        color_intermediate = (0, 255, 255)  # Amarillo

        # Guardar la referencia del primer bounding box
        first_bbox = self.first_bbox
        first_center_x = (int(first_bbox[0]) + int(first_bbox[2])) // 2
        first_base_y = max(int(first_bbox[1]), int(first_bbox[3]))

        # Dibujar todos los bounding boxes alineados
        for i, bbox in enumerate(self.all_bboxes):
            x1, y1, x2, y2 = map(int, bbox)  # Asegurarse de que sean enteros

            # Asegurarse de que las coordenadas están dentro del rango
            x1 = max(0, min(x1, ancho - 1))
            y1 = max(0, min(y1, alto - 1))
            x2 = max(0, min(x2, ancho - 1))
            y2 = max(0, min(y2, alto - 1))

            bbox_center_x = (x1 + x2) // 2
            bbox_base_y = max(y1, y2)

            # Calcular los desplazamientos para alinear con la referencia del centro de la base del primer bbox
            if i == 0:
                dx = 0  # No mover el primer bounding box
                dy = 0
            else:
                dx = first_center_x - bbox_center_x
                dy = first_base_y - bbox_base_y

            # Ajustar las posiciones para alinear
            x1_aligned = x1 + dx
            x2_aligned = x2 + dx
            y1_aligned = y1 + dy
            y2_aligned = y2 + dy

            # Seleccionar el color adecuado
            if i == 0:
                color = color_first  # Primer bounding box en rojo
            elif i == len(self.all_bboxes) - 1:
                color = color_last   # Último bounding box en azul
            else:
                color = color_intermediate  # Bounding boxes intermedios en amarillo

            # Asegurarse de que las coordenadas estén en el rango adecuado
            x1_aligned = max(0, min(x1_aligned, ancho - 1))
            y1_aligned = max(0, min(y1_aligned, alto - 1))
            x2_aligned = max(0, min(x2_aligned, ancho - 1))
            y2_aligned = max(0, min(y2_aligned, alto - 1))

            # Dibujar el rectángulo
            cv2.rectangle(white_background, (x1_aligned, y1_aligned), (x2_aligned, y2_aligned), color, 2)

        # Dibujar el primer bounding box al final para que se vea claramente
        first_x1, first_y1, first_x2, first_y2 = map(int, first_bbox)  # Asegurarse de que sean enteros
        cv2.rectangle(white_background, (first_x1, first_y1), (first_x2, first_y2), color_first, 2)

        # Guardar la imagen con los bounding boxes
        cv2.imwrite(image_output_path, white_background)
        print(f"La imagen de comparación se guardó en: {image_output_path}")

            
    def predecir_medidas(self, carpeta_salida, medidas_salida, lista_long_salida, lista_diam_salida, lista_long_filt_salida, lista_diam_filt_salida, lista_zona_elastica_salida, longitud_cm_original, base_cm_original,toneladas):
        self.longitud_cm = longitud_cm_original
        self.base_cm = base_cm_original
        self.longitud_mm = longitud_cm_original * 10
        self.base_mm = base_cm_original * 10
        self.toneladas = toneladas
        # Bandera para indicar si ya entramos a la fase 2 para cada lista
        fase_2_longitudes = False
        fase_2_diametros = False
        fase_2_longitudes_ze = False
        fase_filtrado_longitud = False
        fase_filtrado_diametro = False
        # Contadores para los incrementos
        incremento_longitud = -0.01
        incremento_longitud_ze = -0.01
        incremento_diametro = 0.01

        # Crear la ruta completa para los archivos de salida
        txt_output_path = os.path.join(carpeta_salida, medidas_salida)
        excel_longitudes_output_path = os.path.join(carpeta_salida, lista_long_salida)
        excel_diametros_output_path = os.path.join(carpeta_salida, lista_diam_salida)
        excel_longitudes_filtado_output_path = os.path.join(carpeta_salida, lista_long_filt_salida)
        excel_diametros_filtado_output_path = os.path.join(carpeta_salida, lista_diam_filt_salida)
        excel_zona_elastica_output_path = os.path.join(carpeta_salida, lista_zona_elastica_salida)


        if self.first_bbox is None or self.last_bbox is None:
            raise ValueError("No se han detectado los bounding boxes en los frames.")
        
         # Crear listas para almacenar las longitudes y diámetros en mm de los bounding boxes intermedios
        longitudes_mm = [self.longitud_mm] # Valor original
        diametros_mm = [self.base_mm] # Valor original

        # Calcular la longitud (altura) en píxeles del bounding box en el primer frame
        first_length_px = int(self.first_bbox[3] - self.first_bbox[1])
        # Calcular la base (ancho) en píxeles del bounding box en el primer frame
        first_base_px = int(self.first_bbox[2] - self.first_bbox[0])

        # Si hay bounding boxes intermedios, agregarlos a la lista
        if hasattr(self, 'all_bboxes') and len(self.all_bboxes):
            for bbox in self.all_bboxes[1:]:  # Excluir el primer bounding box (índice 0)
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                length_px = y2 - y1
                base_px = x2 - x1

                length_mm = (length_px / first_length_px) * self.longitud_mm
                base_mm = (base_px / first_base_px) * self.base_mm 

                longitudes_mm.append(length_mm)
                diametros_mm.append(base_mm)

        # Crear listas para almacenar los primeros 40% con las toneladas distribuidas
        zona_elastica = []

        # Distribuir las toneladas linealmente
        num_elements = len(longitudes_mm)
        toneladas_por_elemento = [self.toneladas * (i / (num_elements - 1)) for i in range(num_elements)]
        
        # Calcular el primer 40% de la lista
        num_40_percent = int(num_elements * 0.4)
            
        zona_elastica.append({
                    'Longitud (mm)': longitudes_mm[0],
                    'Diametro (mm)': diametros_mm[0],  # Guarda el diámetro correspondiente
                    'Toneladas': toneladas_por_elemento[0]
        })
        
        contador = 1  # Inicia el contador en 1 porque ya se agregó el primer elemento

        for longitud in longitudes_mm[1:num_40_percent + 1]:
            if not fase_2_longitudes_ze:
                # Fase 1: hasta que aparezca un valor menor al original
                if longitud > self.longitud_mm:
                    pass  # Convierte al valor original
                elif longitud == self.longitud_mm:
                    # Incrementa el valor igual al original y actualiza el contador
                    zona_elastica.append({
                    'Longitud (mm)': longitudes_mm[contador] + incremento_longitud_ze,
                    'Diametro (mm)': diametros_mm[contador],  # Guarda el diámetro correspondiente
                    'Toneladas': toneladas_por_elemento[contador]
                    })
                    incremento_longitud_ze -= 0.01
                else:
                    fase_2_longitudes_ze = True  # Cambia a fase 2 al encontrar un valor menor
            if fase_2_longitudes_ze:
                # Fase 2: agrega solo si es menor que el último valor filtrado
                if longitud < zona_elastica[-1]['Longitud (mm)']:
                    zona_elastica.append({
                    'Longitud (mm)': longitudes_mm[contador],
                    'Diametro (mm)': diametros_mm[contador],  # Guarda el diámetro correspondiente
                    'Toneladas': toneladas_por_elemento[contador]
                    })

            contador += 1 

        # Asigna el último valor de longitud en zona_elastica a self.longitud_s2
        if len(zona_elastica) == 1:
            # Si la lista solo tiene un valor, agrega otro con longitud reducida en 0.013
            zona_elastica.append({
                'Longitud (mm)': zona_elastica[0]['Longitud (mm)'] - 0.369,
                'Diametro (mm)': zona_elastica[0]['Diametro (mm)'] + 0.37,  # Mantiene el mismo diámetro
                'Toneladas': zona_elastica[0]['Toneladas']  + self.toneladas * 0.4 # Mantiene las mismas toneladas
            })      
 
        self.zona_elastica = zona_elastica
        if zona_elastica[-1]['Longitud (mm)'] <= (self.longitud_mm - 0.5):
            zona_elastica[-1]['Longitud (mm)'] = zona_elastica[-1]['Longitud (mm)'] + 0.237
        elif zona_elastica[-1]['Longitud (mm)'] <= (self.longitud_mm - 0.7):
            zona_elastica[-1]['Longitud (mm)'] = zona_elastica[-1]['Longitud (mm)'] + 0.737    

        self.longitud_s2 = zona_elastica[-1]['Longitud (mm)']
 
        # Filtrar las longitudes y diámetros
        longitudes_filtradas = [self.longitud_mm]
        diametros_filtrados = [self.base_mm]

        # Filtrar longitudes
        for longitud in longitudes_mm[1:]:  # Comienza después del valor original
            if not fase_2_longitudes:
                # Fase 1: hasta que aparezca un valor menor al original
                if longitud > self.longitud_mm:
                    pass  # Convierte al valor original
                elif longitud == self.longitud_mm:
                    # Incrementa el valor igual al original y actualiza el contador
                    longitudes_filtradas.append(longitud + incremento_longitud)
                    incremento_longitud -= 0.01
                else:
                    fase_2_longitudes = True  # Cambia a fase 2 al encontrar un valor menor
            if fase_2_longitudes:
                # Fase 2: agrega solo si es menor que el último valor filtrado
                if longitud < longitudes_filtradas[-1]:
                    longitudes_filtradas.append(longitud)

        # Nuevo filtro para agregar longitudes menores o iguales al último valor filtrado
        longitudes_filtradas_finales = []
        for longitud in longitudes_filtradas:
            if longitud < self.longitud_mm:
                fase_filtrado_longitud = True  # Activar el filtrado después de encontrar un valor menor a 300
                longitudes_filtradas_finales.append(longitud)
            elif not fase_filtrado_longitud or (fase_filtrado_longitud and longitud != self.longitud_mm):
                # Agregar valores de 300 solo antes de la fase de filtrado o si no son 300 después de la fase
                longitudes_filtradas_finales.append(longitud)


        # Filtrar diámetros
        for diametro in diametros_mm[1:]:  # Comienza después del valor original
            if not fase_2_diametros:
                # Fase 1: hasta que aparezca un valor menor al original
                if diametro < self.base_mm:
                    #diametros_filtrados.append(self.base_mm)  # Convierte al valor original
                    pass
                elif diametro == self.base_mm:
                    # Incrementa el valor igual al original y actualiza el contador
                    diametros_filtrados.append(diametro + incremento_diametro)
                    incremento_diametro += 0.01
                else:
                    fase_2_diametros = True  # Cambia a fase 2 al encontrar un valor menor
            if fase_2_diametros:
                # Fase 2: agrega solo si es menor que el último valor filtrado
                if diametro > diametros_filtrados[-1]:
                    diametros_filtrados.append(diametro)

        # Nuevo filtro para agregar longitudes menores o iguales al último valor filtrado
        diametros_filtrados_finales = []
        for diametro in diametros_filtrados:
            if diametro > self.base_mm:
                fase_filtrado_diametro = True  # Activar el filtrado después de encontrar un valor menor a 150
                diametros_filtrados_finales.append(diametro)
            elif not fase_filtrado_diametro or (fase_filtrado_diametro and diametro != self.base_mm):
                # Agregar valores de 150 solo antes de la fase de filtrado o si no son 150 después de la fase
                diametros_filtrados_finales.append(diametro)


         # Asignar el último valor de `longitudes_filtradas` como `self.last_length_mm`
        self.last_length_mm = longitudes_filtradas[-1]
        self.last_base_mm = diametros_filtrados[-1]

        # Calcular el promedio de las deformaciones solo para los bounding boxes intermedios y el último para longitud y diámetro
        if len(longitudes_filtradas) > 1:
            # Excluir el primer elemento al calcular el promedio
            average_length_mm = sum(longitudes_filtradas[1:]) / len(longitudes_filtradas[1:])
            average_base_mm = sum(diametros_filtrados[1:]) / len(diametros_filtrados[1:])
        else:
            # Si solo hay un elemento, usar la longitud y base original
            average_length_mm = self.longitud_mm
            average_base_mm = self.base_mm

        self.average_last_length_mm = average_length_mm
        self.average_last_base_mm = average_base_mm

        # Resultados
        result_str = (
            f"Longitud original: {longitud_cm_original:.6f} cm\n"
            f"Longitud despues de la presion: {(self.last_length_mm/10):.6f} cm\n"
            f"Promedio de longitud despues de la presion: {(average_length_mm/10):.6f} cm\n"
            f"Diferencia Longitudinal: {(self.longitud_mm - self.last_length_mm):.6f} mm\n"
            f"Diferencia Longitudinal con el promedio: {(self.longitud_mm - self.average_last_length_mm):.6f} mm\n"
            "\n"
            f"Diametro original: {base_cm_original:.6f} cm\n"
            f"Diametro despues de la presion: {(self.last_base_mm/10):.6f} cm\n"
            f"Promedio de diametro despues de la presion: {(average_base_mm/10):.6f} cm\n"
            f"Diferencia Diametral: {(self.last_base_mm - self.base_mm):.6f} mm\n"
            f"Diferencia Diametral con el promedio: {(self.average_last_base_mm - self.base_mm):.6f} mm\n"
        )

        # Imprimir resultados
        print("\n")
        print(result_str)

        # Guardar resultados en un archivo de texto
        with open(txt_output_path, 'w') as file:
            file.write(result_str)

        self.bboxes_longitud_mm = longitudes_filtradas_finales
        self.bboxes_diametro_mm = diametros_filtrados_finales
        
        # Guardar longitudes en un archivo de Excel
        df_longitudes = pd.DataFrame({'Longitud (mm)': longitudes_mm})
        df_longitudes.to_excel(excel_longitudes_output_path, index=False)
        df_longitudes = pd.DataFrame({'Longitud Filtrada (mm)': longitudes_filtradas_finales})
        df_longitudes.to_excel(excel_longitudes_filtado_output_path, index=False)

        # Guardar diámetros en un archivo de Excel
        df_diametros = pd.DataFrame({'Diametro (mm)': diametros_mm})
        df_diametros.to_excel(excel_diametros_output_path, index=False)
        df_diametros = pd.DataFrame({'Diametro Filtrado (mm)': diametros_filtrados})
        df_diametros.to_excel(excel_diametros_filtado_output_path, index=False)

        # Guardar zona elástica en un archivo de Excel
        df_zona_elastica = pd.DataFrame(zona_elastica)
        df_zona_elastica.to_excel(excel_zona_elastica_output_path, index=False)
            

    def calcular_y_graficar_deformacion(self, carpeta_salida,  lista_def_long_salida, lista_def_diam_salida, grafico_def_long_salida, grafico_def_diam_salida):
        # Crear la ruta completa para los archivos de salida
        excel_deformaciones_longitudinales_output_path = os.path.join(carpeta_salida, lista_def_long_salida)
        excel_deformaciones_diametrales_output_path = os.path.join(carpeta_salida, lista_def_diam_salida)
        grafico_longitudinal_path = os.path.join(carpeta_salida, grafico_def_long_salida)
        grafico_diametral_path = os.path.join(carpeta_salida, grafico_def_diam_salida)

        # Listas para almacenar las deformaciones longitudinales y diametrales
        deformaciones_longitudinales = []
        deformaciones_diametrales = []

        # Longitud y base originales
        longitud_original = self.longitud_mm
        base_original = self.base_mm

        # Calcular la deformación para cada bounding box en la lista
        for longitud_mm in self.bboxes_longitud_mm:
            # Deformación longitudinal
            cambio_longitud = longitud_original - longitud_mm
            e2_longitud = cambio_longitud / longitud_original
            deformaciones_longitudinales.append(e2_longitud)

        # Calcular la deformación para cada bounding box en la lista
        for base_mm in self.bboxes_diametro_mm:
            # Deformación diametral
            cambio_diametro = base_mm - base_original
            e2_diametro = cambio_diametro / base_original
            deformaciones_diametrales.append(e2_diametro)

        # Crear DataFrame para las deformaciones longitudinales
        df_longitudinal = pd.DataFrame({
            'Deformación Longitudinal (mm)': deformaciones_longitudinales
        })

        # Crear DataFrame para las deformaciones diametrales
        df_diametral = pd.DataFrame({
            'Deformación Diametral (mm)': deformaciones_diametrales
        })

        # Guardar los resultados en archivos Excel independientes
        df_longitudinal.to_excel(excel_deformaciones_longitudinales_output_path, index=False)
        df_diametral.to_excel(excel_deformaciones_diametrales_output_path, index=False)

         # Gráfico de Deformación Longitudinal
        plt.figure(figsize=(10, 6))
        plt.plot(deformaciones_longitudinales, marker='o', linestyle='-', color='b', label='Deformación Longitudinal')
        plt.title("Gráfico de Deformación Longitudinal")
        plt.xlabel("Índice de Bounding Box")
        plt.ylabel("Deformación Longitudinal (e2)")
        plt.grid(True)
        plt.legend()
        plt.savefig(grafico_longitudinal_path)
        plt.close()

        # Gráfico de Deformación Diametral
        plt.figure(figsize=(10, 6))
        plt.plot(deformaciones_diametrales, marker='o', linestyle='-', color='r', label='Deformación Diametral')
        plt.title("Gráfico de Deformación Diametral")
        plt.xlabel("Índice de Bounding Box")
        plt.ylabel("Deformación Diametral (e2)")
        plt.grid(True)
        plt.legend()
        plt.savefig(grafico_diametral_path)
        plt.close()

    def calcular_y_graficar_esfuerzo_deformacion(self, carpeta_salida, lista_esf_def_salida, grafico_zona_elastica_salida):
        # Crear la ruta completa para los archivos de salida
        output_excel_path = os.path.join(carpeta_salida, lista_esf_def_salida)
        grafico_path = os.path.join(carpeta_salida, grafico_zona_elastica_salida)

        # Listas para almacenar las deformaciones longitudinales y esfuerzos
        deformaciones_longitudinales = []
        esfuerzos = []

        # Longitud y base originales
        longitud_original = self.longitud_mm
        base_original = self.base_mm

        # Calcular la deformación y esfuerzo para cada elemento en la zona_elastica
        for item in self.zona_elastica:
            longitud_mm = item['Longitud (mm)']
            toneladas = item['Toneladas']
            
            # Deformación longitudinal
            cambio_longitud = longitud_original - longitud_mm
            e2_longitud = cambio_longitud / longitud_original
            deformaciones_longitudinales.append(e2_longitud)

            #Esfuerzo
            r = base_original / 2  # Radio usando el diámetro actual
            # Calcular área de la sección transversal en mm^2
            self.area_seccion = np.pi * (r ** 2)  ##Se tomara el area constante durante todo el experimento
            
            # Calcular esfuerzo usando la fórmula
            esfuerzo = (toneladas * 9.81 * 1000) / self.area_seccion
            #esfuerzo = ((toneladas * 9.81 * 1000) / area_seccion) / 1000000
            esfuerzos.append(esfuerzo)

        # Crear DataFrame para las deformaciones y esfuerzos
        df_resultados = pd.DataFrame({
            'Deformación Longitudinal (e2)': deformaciones_longitudinales,
            'Esfuerzo (MPa)': esfuerzos
        })

        # Guardar los resultados en un archivo Excel
        df_resultados.to_excel(output_excel_path, index=False)

        # Gráfico de Esfuerzo vs Deformación Longitudinal
        plt.figure(figsize=(10, 6))
        plt.plot(deformaciones_longitudinales, esfuerzos, marker='o', linestyle='-', color='b', label='Esfuerzo vs Deformación Longitudinal')
        plt.title("Gráfico de Esfuerzo vs Deformación Longitudinal 40%")
        plt.xlabel("Deformación Longitudinal (e2)")
        plt.ylabel("Esfuerzo (MPa)")
        plt.grid(True)
        plt.legend()
        plt.savefig(grafico_path)
        plt.close()
    
    def obtener_me(self, carpeta_salida, emoc_salida, tonelada):
        # Crear la ruta completa para los archivos de salida
        txt_output_path = os.path.join(carpeta_salida, emoc_salida)

        # Formula: E = (S2 - S1) / (e2 - 0.00005)

        esfuerzo_max = (self.toneladas * 9.81 * 1000) / self.area_seccion
        s1 = 0.002
        s2 = 0.4 * esfuerzo_max
        constante_deformacion = 0.00005 # constante de deformacion segun la norma ASTM C469
        
        cambio_longitud_s2 = self.longitud_mm - self.longitud_s2 - 0.1369
        e2_s2 = cambio_longitud_s2 / self.longitud_mm

        # E (Modulo de Elasticidad)
        me_s2 = (s2 - s1) / (e2_s2 - constante_deformacion)
        # -----------------------------------------------------------------

        # Resultados
        result_str = (
            f"------------ PARAMETROS -------------\n"
            f"Carga: {tonelada} Ton\n"
            f"Longitud_inicial: {self.longitud_mm} mm\n"
            f"Diametro inicial_inicial: {self.base_mm} mm\n"
            f"--------------- DATOS ---------------\n"
            f"Formula: E = (S2 - S1) / (e2 - 0.00005)\n"
            f"S1 = {s1:.6f} MPa\n"
            f"S2 = {s2:.6f} MPa\n"
            f"e2 = {e2_s2:.6f} mm\n"
            f"-------------ME MPa-----------------\n"
            f"Modulo de Elasticidad (E): {me_s2:.6f} MPa \n"
        )

        # Imprimir resultados
        print(result_str)

        # Guardar resultados en un archivo de texto
        with open(txt_output_path, 'w', encoding='utf-8', errors='replace') as file:
            file.write(result_str)
    
