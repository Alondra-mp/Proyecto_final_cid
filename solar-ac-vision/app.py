import streamlit as st

st.set_page_config(
    page_title="Visión Artificial - Solar & AC",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Visión Artificial para Proyectos de Instalación")

st.markdown("""
Bienvenido a la aplicación de **Visión Artificial para Instalaciones**. 

Esta herramienta te ayuda a medir áreas y calcular requerimientos de equipos basándose en imágenes que tomes de tu azotea o cuarto.

### 🛠️ Herramientas Disponibles

1. **🏠 Azotea y Paneles**: 
   Mide tu azotea usando una imagen y calcula automáticamente cuántos paneles solares caben en el área utilizable.
2. **❄️ Cuarto y A/C**: 
   Calcula la capacidad en toneladas de Aire Acondicionado que necesitas midiendo el espacio con una simple fotografía.

### 📄 Instrucciones Generales
Para que el sistema funcione correctamente, **debes colocar una hoja tamaño carta (21.6 x 27.9 cm)** en el piso o superficie que deseas medir. 
El algoritmo detectará la hoja y la utilizará como objeto de referencia para determinar la escala de la imagen (píxeles por centímetro).
¡Asegúrate de que la hoja sea claramente visible y no esté doblada o cubierta!
""")

st.info("Selecciona una herramienta en el menú de la izquierda para comenzar.")
