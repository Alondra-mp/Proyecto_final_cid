import streamlit as st
import numpy as np
from PIL import Image
import cv2
import sys
import os

# Añadir el directorio raíz al path para importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.vision import detect_reference_object, four_point_transform
from utils.calculations import calculate_ac_tonnage

st.set_page_config(page_title="Cuarto & AC", page_icon="❄️", layout="wide")

st.title("❄️ Medición de Cuarto para Aire Acondicionado")

st.sidebar.header("Parámetros de la Habitación")
height_m = st.sidebar.slider("Altura del techo (m)", min_value=2.0, max_value=4.0, value=2.4, step=0.1)
occupants = st.sidebar.number_input("Número de ocupantes regulares", min_value=1, value=2)
sun_exposure = st.sidebar.radio("Exposición al sol", ["baja", "media", "alta"])

source_option = st.radio("Fuente de imagen", ["Subir archivo", "Usar cámara"])
img_file_buffer = None

if source_option == "Subir archivo":
    img_file_buffer = st.file_uploader("Sube una imagen del piso de tu cuarto", type=["jpg", "jpeg", "png"])
else:
    img_file_buffer = st.camera_input("Toma una foto del piso de tu cuarto")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        cv_img = img_array
        
    col1, col2 = st.columns(2)
    
    with st.spinner("Procesando imagen e identificando dimensiones..."):
        ref_contour, px_per_cm = detect_reference_object(cv_img)
        
        if ref_contour is None:
            st.warning("No se pudo detectar la hoja de referencia (tamaño carta). Por favor, asegúrate de que sea visible.")
        else:
            st.success(f"¡Objeto de referencia detectado! Escala: {px_per_cm:.2f} px/cm")
            
            st.markdown("### Selecciona las 4 esquinas del piso del cuarto")
            st.info("Ajusta los porcentajes para marcar las 4 esquinas del área a medir.")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                tl_x = st.slider("Sup-Izq X % ", 0, 100, 10)
                tl_y = st.slider("Sup-Izq Y % ", 0, 100, 10)
            with c2:
                tr_x = st.slider("Sup-Der X % ", 0, 100, 90)
                tr_y = st.slider("Sup-Der Y % ", 0, 100, 10)
            with c3:
                br_x = st.slider("Inf-Der X % ", 0, 100, 90)
                br_y = st.slider("Inf-Der Y % ", 0, 100, 90)
            with c4:
                bl_x = st.slider("Inf-Izq X % ", 0, 100, 10)
                bl_y = st.slider("Inf-Izq Y % ", 0, 100, 90)
                
            h, w = cv_img.shape[:2]
            pts = np.array([
                [w * tl_x / 100, h * tl_y / 100],
                [w * tr_x / 100, h * tr_y / 100],
                [w * br_x / 100, h * br_y / 100],
                [w * bl_x / 100, h * bl_y / 100]
            ], dtype="float32")
            
            img_with_pts = cv_img.copy()
            cv2.drawContours(img_with_pts, [ref_contour], -1, (0, 0, 255), 3)
            for pt in pts:
                cv2.circle(img_with_pts, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), -1)
                
            with col1:
                st.subheader("Imagen Original con Puntos")
                st.image(cv2.cvtColor(img_with_pts, cv2.COLOR_BGR2RGB), use_container_width=True)
                
            try:
                room_img = four_point_transform(cv_img, pts)
                
                if room_img.size == 0 or room_img.shape[0] == 0 or room_img.shape[1] == 0:
                    st.error("El área seleccionada es inválida.")
                else:
                    room_h_px, room_w_px = room_img.shape[:2]
                    room_w_m = (room_w_px / px_per_cm) / 100
                    room_h_m = (room_h_px / px_per_cm) / 100
                    
                    btus, tonnage = calculate_ac_tonnage(
                        room_w_m, room_h_m, height_m, occupants, sun_exposure
                    )
                    
                    with col2:
                        st.subheader("Perspectiva Corregida del Piso")
                        st.image(cv2.cvtColor(room_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                    st.markdown("### Resultados del Cálculo de A/C")
                    
                    suggested_tonnage = max(1.0, round(np.ceil(tonnage * 2) / 2, 1))
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Área del Cuarto", f"{room_w_m * room_h_m:.2f} m²")
                    m2.metric("BTUs Requeridos", f"{int(btus):,}")
                    m3.metric("Toneladas Calculadas", f"{tonnage:.2f}")
                    
                    st.info(f"💡 **Sugerencia de Equipo:** Te recomendamos instalar un equipo de **{suggested_tonnage} Toneladas**.")
                    
                    st.write(f"**Dimensiones corregidas:** {room_w_m:.2f}m x {room_h_m:.2f}m")
            except Exception as e:
                st.error(f"Error en los cálculos: {e}")
