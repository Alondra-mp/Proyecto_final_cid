# Visión Artificial - Solar & AC

Proyecto de visión artificial con Streamlit para la medición de espacios usando un objeto de referencia (hoja tamaño carta).

## Funcionalidades
1. **Azotea y Paneles Solares**: Mide el área de una azotea y calcula cuántos paneles solares caben.
2. **Cuarto y Aire Acondicionado**: Mide el área de un cuarto y calcula las toneladas de A/C necesarias.

## Cómo correr localmente
1. Clona el repositorio.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta la aplicación:
   ```bash
   streamlit run app.py
   ```

## Despliegue en Streamlit Community Cloud
1. Sube tu código a un repositorio público en GitHub.
2. Entra a [Streamlit Community Cloud](https://share.streamlit.io/).
3. Crea una nueva app, selecciona el repositorio, la rama principal y `app.py` como el archivo principal.
4. Haz clic en "Deploy".

## Instrucciones de uso
Es crucial utilizar un objeto de referencia de tamaño conocido para que la aplicación pueda calcular la escala píxeles-por-centímetro. Por defecto, el sistema está calibrado para usar una hoja tamaño carta (21.6 x 27.9 cm). Coloca la hoja en el piso de tu cuarto o en la azotea y asegúrate de que sea claramente visible en la imagen.

## Tecnologías
- Python
- Streamlit
- OpenCV
- NumPy
- Pillow
