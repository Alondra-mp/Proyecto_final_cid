import streamlit as st
import numpy as np
from PIL import Image
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.vision import detect_target_and_reference, four_point_transform, draw_panel_grid, order_points
from utils.calculations import calculate_panels

st.set_page_config(page_title="Azotea & Paneles", layout="wide")

st.title("Medición de Azotea para Paneles Solares")

# ── Parámetros del panel ──────────────────────────────────────────────────────
st.sidebar.header("Parámetros del Panel")
panel_w_cm = st.sidebar.number_input("Ancho del panel (cm)", min_value=1.0, value=100.0)
panel_h_cm = st.sidebar.number_input("Alto del panel (cm)", min_value=1.0, value=165.0)
spacing_cm = st.sidebar.number_input("Separación entre paneles (cm)", min_value=0.0, value=10.0)

# ── Método de calibración ────────────────────────────────────────────────────
st.sidebar.header("Calibración de Escala")
cal_method = st.sidebar.radio(
    "Método de calibración",
    ["Hoja de referencia (carta)", "Sin hoja — medida conocida"],
    help=(
        "Con hoja carta: coloca una hoja tamaño carta (21.6×27.9 cm) en la foto.\n"
        "Sin hoja: indica cuánto mide un lado conocido de la azotea."
    ),
)
known_width_cm = None
if cal_method == "Sin hoja — medida conocida":
    known_width_cm = st.sidebar.number_input(
        "Ancho conocido de la azotea (cm)",
        min_value=10.0,
        max_value=5000.0,
        value=500.0,
        step=10.0,
        help="El ancho del área que vas a seleccionar manualmente (de izquierda a derecha).",
    )

# ── Fuente de imagen ─────────────────────────────────────────────────────────
source_option = st.radio("Fuente de imagen", ["Subir archivo", "Usar cámara"])
img_file_buffer = None

if source_option == "Subir archivo":
    img_file_buffer = st.file_uploader("Sube una imagen de tu azotea", type=["jpg", "jpeg", "png"])
else:
    st.info("Asegúrate de permitir el acceso a la cámara en tu navegador cuando se te solicite.")
    img_file_buffer = st.camera_input("Toma una foto de tu azotea")

# ── Procesamiento ─────────────────────────────────────────────────────────────
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        cv_img = img_array

    h, w = cv_img.shape[:2]
    col1, col2 = st.columns(2)

    # ── Calibración por hoja de referencia ───────────────────────────────────
    if cal_method == "Hoja de referencia (carta)":
        with st.spinner("Detectando hoja de referencia y área..."):
            target_contour, ref_contour, px_per_cm = detect_target_and_reference(cv_img)

        if ref_contour is None:
            st.error(
                "No se detectó la hoja de referencia tamaño carta. "
                "Verifica que esté visible, bien iluminada y sin obstrucciones, "
                "o cambia a 'Sin hoja — medida conocida' en el panel lateral."
            )
            st.stop()

        st.success(f"Hoja de referencia detectada — Escala: {px_per_cm:.2f} px/cm")

        auto_detect = False
        if target_contour is not None:
            st.success("Área de azotea detectada automáticamente.")
            auto_detect = st.checkbox("Usar área detectada automáticamente", value=True)
        else:
            st.info("No se detectó el área automáticamente. Usa los controles manuales.")

        if auto_detect and target_contour is not None:
            pts = target_contour.reshape(4, 2).astype("float32")
        else:
            st.markdown("### Selecciona las 4 esquinas de la azotea")
            st.info("Ajusta los porcentajes para marcar las 4 esquinas del área útil.")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                tl_x = st.slider("Sup-Izq X %", 0, 100, 10)
                tl_y = st.slider("Sup-Izq Y %", 0, 100, 10)
            with c2:
                tr_x = st.slider("Sup-Der X %", 0, 100, 90)
                tr_y = st.slider("Sup-Der Y %", 0, 100, 10)
            with c3:
                br_x = st.slider("Inf-Der X %", 0, 100, 90)
                br_y = st.slider("Inf-Der Y %", 0, 100, 90)
            with c4:
                bl_x = st.slider("Inf-Izq X %", 0, 100, 10)
                bl_y = st.slider("Inf-Izq Y %", 0, 100, 90)

            pts = np.array([
                [w * tl_x / 100, h * tl_y / 100],
                [w * tr_x / 100, h * tr_y / 100],
                [w * br_x / 100, h * br_y / 100],
                [w * bl_x / 100, h * bl_y / 100],
            ], dtype="float32")

        # Dibujar contornos en imagen original
        img_with_pts = cv_img.copy()
        cv2.drawContours(img_with_pts, [ref_contour], -1, (0, 0, 255), 3)
        if auto_detect and target_contour is not None:
            cv2.drawContours(img_with_pts, [target_contour], -1, (0, 255, 0), 3)
        else:
            for pt in pts:
                cv2.circle(img_with_pts, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), -1)

    # ── Calibración por medida conocida (sin hoja) ────────────────────────────
    else:
        st.info(
            "Modo sin hoja: selecciona las 4 esquinas de la azotea con los controles "
            "y asegúrate de que el ancho conocido que ingresaste corresponda al ancho "
            "horizontal del área que estás seleccionando."
        )

        st.markdown("### Selecciona las 4 esquinas de la azotea")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tl_x = st.slider("Sup-Izq X %", 0, 100, 10, key="nf_tl_x")
            tl_y = st.slider("Sup-Izq Y %", 0, 100, 10, key="nf_tl_y")
        with c2:
            tr_x = st.slider("Sup-Der X %", 0, 100, 90, key="nf_tr_x")
            tr_y = st.slider("Sup-Der Y %", 0, 100, 10, key="nf_tr_y")
        with c3:
            br_x = st.slider("Inf-Der X %", 0, 100, 90, key="nf_br_x")
            br_y = st.slider("Inf-Der Y %", 0, 100, 90, key="nf_br_y")
        with c4:
            bl_x = st.slider("Inf-Izq X %", 0, 100, 10, key="nf_bl_x")
            bl_y = st.slider("Inf-Izq Y %", 0, 100, 90, key="nf_bl_y")

        pts = np.array([
            [w * tl_x / 100, h * tl_y / 100],
            [w * tr_x / 100, h * tr_y / 100],
            [w * br_x / 100, h * br_y / 100],
            [w * bl_x / 100, h * bl_y / 100],
        ], dtype="float32")

        # Calcular px_per_cm desde la medida conocida del ancho seleccionado
        pts_ordered = order_points(pts)
        tl_o, tr_o, _, _ = pts_ordered
        width_px_ref = np.sqrt(((tr_o[0] - tl_o[0]) ** 2) + ((tr_o[1] - tl_o[1]) ** 2))
        if width_px_ref < 10:
            st.error("El área seleccionada es demasiado pequeña. Ajusta los controles.")
            st.stop()
        px_per_cm = width_px_ref / known_width_cm
        ref_contour = None

        img_with_pts = cv_img.copy()
        for pt in pts:
            cv2.circle(img_with_pts, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), -1)

    # ── Mostrar imagen con puntos ─────────────────────────────────────────────
    with col1:
        st.subheader("Imagen original con puntos")
        st.image(cv2.cvtColor(img_with_pts, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ── Corrección de perspectiva y cálculos ──────────────────────────────────
    try:
        roof_img = four_point_transform(cv_img, pts)

        if roof_img.size == 0 or roof_img.shape[0] == 0 or roof_img.shape[1] == 0:
            st.error("El área seleccionada es inválida.")
        else:
            roof_h_px, roof_w_px = roof_img.shape[:2]
            roof_w_cm = roof_w_px / px_per_cm
            roof_h_cm = roof_h_px / px_per_cm

            results = calculate_panels(
                roof_w_cm / 100,
                roof_h_cm / 100,
                panel_w_cm / 100,
                panel_h_cm / 100,
                spacing_cm / 100,
            )

            annotated_roof, panel_count = draw_panel_grid(
                roof_img,
                (roof_w_cm, roof_h_cm),
                (panel_w_cm + spacing_cm, panel_h_cm + spacing_cm),
                px_per_cm,
            )

            cv2.putText(
                img_with_pts,
                f"{roof_w_cm/100:.2f}m x {roof_h_cm/100:.2f}m",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
            )
            cv2.putText(
                annotated_roof,
                f"W: {roof_w_cm/100:.2f}m",
                (int(roof_w_px / 2) - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
            )
            cv2.putText(
                annotated_roof,
                f"H: {roof_h_cm/100:.2f}m",
                (10, int(roof_h_px / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
            )

            with col2:
                st.subheader("Perspectiva corregida y paneles")
                st.image(cv2.cvtColor(annotated_roof, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("### Resultados del Cálculo")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total de Paneles", f"{results['total_panels']}")
            m2.metric("Área Útil Cubierta", f"{results['usable_area']:.2f} m²")
            m3.metric("Eficiencia de Espacio", f"{results['efficiency_percentage']:.1f} %")

            st.write(f"**Distribución:** {results['panels_per_row']} columnas x {results['rows']} filas")
            st.write(f"**Dimensiones corregidas:** {roof_w_cm/100:.2f} m × {roof_h_cm/100:.2f} m")

    except Exception as e:
        st.error(f"Error procesando la perspectiva o los cálculos: {e}")
