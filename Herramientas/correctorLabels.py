import os

label_dir = 'C:\\Users\\carlo\\Desktop\\TESIS\\DATASET\\Dataset_EM\\train\\labels'  # Cambia a tu ruta de etiquetas

for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        file_path = os.path.join(label_dir, label_file)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        modified = False
        for line in lines:
            parts = line.strip().split()
            if parts[0] != '0':  # Si el class_id es distinto de 0
                print(f"Error encontrado en el archivo: {label_file}")
                print(f"Contenido del archivo: {line.strip()}")
                modified = True
        
        # Si hubo algún error, corrige el archivo
        if modified:
            with open(file_path, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    parts[0] = '0'  # Cambiar class_id a 0
                    f.write(' '.join(parts) + '\n')

print("Proceso completado.")
