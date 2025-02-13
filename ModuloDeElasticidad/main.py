from elastic_modulus import Modulo_elasticidad

def main():
    no_experimento = 3
    #-----RUTAS-----
    ruta_modelo = '../EMOCnet/best.pt'
    ruta_video = f'../Experimentos/exp{no_experimento}/exp{no_experimento} - video.mp4'
    carpeta_salida = f"../Experimentos/exp{no_experimento}/resultados"
    #---------------

    #-----NOMBRES DE ARCHIVOS DE SALIDA-----
    video_salida = "video_detectado.mp4"
    imagen_salida = "bounding_boxes.png"
    medidas_salida = "medidas.txt"
    emoc_salida = "modulo_elasticidad.txt"
    lista_long_salida = "lista_longitudes.xlsx"
    lista_long_filt_salida = "lista_longitudes_filtradas.xlsx"
    lista_diam_salida = "lista_diametros.xlsx"
    lista_diam_filt_salida = "lista_diametros_filtrados.xlsx"
    lista_def_long_salida = "lista_deformaciones_longitudinales.xlsx"
    lista_def_diam_salida = "lista_deformaciones_diametrales.xlsx"
    lista_zona_elastica_salida = "lista_zona_elastica_longitud.xlsx"
    lista_esf_def_salida = "lista_esfuerzo_deformacion.xlsx"
    grafico_def_diam_salida = "grafico_deformacion_diametral.png"
    grafico_def_long_salida = "grafico_deformacion_longitudinal.png"
    grafico_zona_elastica_salida = "grafico_esfuerzo_deformacion.png"
    #-----------------------------------------

    #-----PARAMETROS DE ENTRADA-----
    longitud_cm_original = 30
    diametro_cm_original = 15
    toneladas = 42.10
    #-------------------------------

    #-----FLUJO DE TRABAJO-----
    # instancia de la clase
    modulo_elasticidad = Modulo_elasticidad(ruta_modelo)

    # PASO 1:
    modulo_elasticidad.detectar_y_guardar_video(ruta_video, carpeta_salida, video_salida)

    # PASO 2: 
    modulo_elasticidad.crear_imagen_bounding_boxes(carpeta_salida, imagen_salida)

    # PASO 3: 
    modulo_elasticidad.predecir_medidas(
    carpeta_salida,  
    medidas_salida,
    lista_long_salida,
    lista_diam_salida,
    lista_long_filt_salida,
    lista_diam_filt_salida,
    lista_zona_elastica_salida,
    longitud_cm_original,
    diametro_cm_original,
    toneladas
) 
     
    # PASO 4: 
    modulo_elasticidad.calcular_y_graficar_deformacion(carpeta_salida, lista_def_long_salida, lista_def_diam_salida, grafico_def_long_salida, grafico_def_diam_salida)
    
    # PASO 5: 
    modulo_elasticidad.calcular_y_graficar_esfuerzo_deformacion(carpeta_salida, lista_esf_def_salida, grafico_zona_elastica_salida)

    # PASO 6: 
    modulo_elasticidad.obtener_me(carpeta_salida, emoc_salida, toneladas)
#------------------------------

#-----EJECUCION DEL ARCHIVO-----
if __name__ == "__main__":
    main()
#-------------------------------


