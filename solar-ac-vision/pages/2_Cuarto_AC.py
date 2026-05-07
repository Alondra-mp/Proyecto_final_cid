import streamlit as st
import numpy as np
from PIL import Image
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.vision import detect_target_and_reference, four_point_transform, order_points
from utils.calculations import calculate_ac_tonnage

st.set_page_config(page_title="Cuarto & AC", layout="wide")

st.title("Medición de Cuarto para Aire Acondicionado")

# ── Parámetros de la habitación ───────────────────────────────────────────────
st.sidebar.header("Parámetros de la Habitación")
height_m = st.sidebar.slider("Altura del techo (m)", min_value=2.0, max_value=4.0, value=2.4, step=0.1)
occupants = st.sidebar.number_input("Número de ocupantes regulares", min_value=1, value=2)
sun_exposure = st.sidebar.radio("Exposición al sol", ["baja", "media", "alta"])

# ── Método de medición ────────────────────────────────────────────────────────
st.markdown("### Método de medición")
measurement_method = st.radio(
    "¿Cómo deseas obtener las dimensiones del cuarto?",
    [
        "Ingresar dimensiones manualmente",
        "Desde imagen con hoja de referencia (carta)",
        "Desde imagen con medida conocida (sin hoja)",
    ],
    help=(
        "Manual: introduce ancho y largo directamente.\n"
        "Con hoja carta: coloca una hoja tamaño carta (21.6×27.9 cm) en la foto.\n"
        "Sin hoja: indica un lado conocido y selecciona las esquinas manualmente."
    ),
)

# ═══════════════════════════════════════════════════════════════════════════════
# OPCIÓN 1 — Dimensiones manuales
# ═══════════════════════════════════════════════════════════════════════════════
if measurement_method == "Ingresar dimensiones manualmente":
    st.info("Introduce las dimensiones del cuarto y calcula el equipo de A/C necesario.")
    col_a, col_b = st.columns(2)
    with col_a:
        room_w_m = st.number_input("Ancho del cuarto (m)", min_value=0.5, max_value=30.0, value=3.0, step=0.1)
    with col_b:
        room_h_m = st.number_input("Largo del cuarto (m)", min_value=0.5, max_value=30.0, value=4.0, step=0.1)

    if st.button("Calcular A/C"):
        btus, tonnage = calculate_ac_tonnage(room_w_m, room_h_m, height_m, occupants, sun_exposure)
        suggested_tonnage = max(1.0, round(np.ceil(tonnage * 2) / 2, 1))

        st.markdown("### Resultados del Cálculo de A/C")
        m1, m2, m3 = st.columns(3)
        m1.metric("Área del Cuarto", f"{room_w_m * room_h_m:.2f} m²")
        m2.metric("BTUs Requeridos", f"{int(btus):,}")
        m3.metric("Toneladas Calculadas", f"{tonnage:.2f}")

        st.info(f"Sugerencia de Equipo: Te recomendamos instalar un equipo de **{suggested_tonnage} Toneladas**.")
        st.write(f"**Dimensiones ingresadas:** {room_w_m:.2f} m × {room_h_m:.2f} m")

# ═══════════════════════════════════════════════════════════════════════════════
# OPCIONES 2 y 3 — Desde imagen
# ═══════════════════════════════════════════════════════════════════════════════
else:
    uses_ref_sheet = measurement_method == "Desde imagen con hoja de referencia (carta)"

    if uses_ref_sheet:
        st.info(
            "Toma la foto del piso del cuarto desde arriba (sube a una silla o escalera). "
            "Coloca una hoja tamaño carta (21.6×27.9 cm) sobre el piso para que la cámara "
            "la vea completa. Marca las 4 esquinas del piso con los controles si la "
            "detección automática no es precisa."
        )
    else:
        known_width_cm = st.number_input(
            "Ancho conocido del cuarto (cm)",
            min_value=10.0,
            max_value=3000.0,
            value=300.0,
            step=10.0,
            help="El ancho horizontal del área que vas a delimitar con los 4 puntos.",
        )
        st.info(
            "Toma la foto del piso desde arriba. Selecciona las 4 esquinas del cuarto "
            "con los controles. El ancho que ingresaste arriba debe corresponder al lado "
            "superior del área seleccionada."
        )

    source_option = st.radio("Fuente de imagen", ["Subir archivo", "Usar cámara"])
    img_file_buffer = None

    if source_option == "Subir archivo":
        img_file_buffer = st.file_uploader(
            "Sube una imagen del piso de tu cuarto", type=["jpg", "jpeg", "png"]
        )
    else:
        st.info("Asegúrate de permitir el acceso a la cámara en tu navegador.")
        img_file_buffer = st.camera_input("Toma una foto del piso de tu cuarto")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)

        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            cv_img = img_array

        h, w = cv_img.shape[:2]
        col1, col2 = st.columns(2)

        # ── Con hoja de referencia ────────────────────────────────────────────
        if uses_ref_sheet:
            with st.spinner("Detectando hoja de referencia..."):
                target_contour, ref_contour, px_per_cm = detect_target_and_reference(cv_img)

            if ref_contour is None:
                st.error(
                    "No se detectó la hoja tamaño carta. Asegúrate de que esté "
                    "completamente visible y bien iluminada, o usa la opción "
                    "'Sin hoja — medida conocida'."
                )
                st.stop()

            st.success(f"Hoja de referencia detectada — Escala: {px_per_cm:.2f} px/cm")

            auto_detect = False
            if target_contour is not None:
                st.success("Área del piso detectada automáticamente.")
                auto_detect = st.checkbox("Usar área detectada automáticamente", value=False)
            else:
                st.info(
                    "No se detectó el área del piso automáticamente. "
                    "Usa los controles manuales para delimitar el cuarto."
                )

            if auto_detect and target_contour is not None:
                pts = target_contour.reshape(4, 2).astype("float32")
            else:
                st.markdown("### Selecciona las 4 esquinas del piso del cuarto")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    tl_x = st.slider("Sup-Izq X %", 0, 100, 10, key="r_tl_x")
                    tl_y = st.slider("Sup-Izq Y %", 0, 100, 10, key="r_tl_y")
                with c2:
                    tr_x = st.slider("Sup-Der X %", 0, 100, 90, key="r_tr_x")
                    tr_y = st.slider("Sup-Der Y %", 0, 100, 10, key="r_tr_y")
                with c3:
                    br_x = st.slider("Inf-Der X %", 0, 100, 90, key="r_br_x")
                    br_y = st.slider("Inf-Der Y %", 0, 100, 90, key="r_br_y")
                with c4:
                    bl_x = st.slider("Inf-Izq X %", 0, 100, 10, key="r_bl_x")
                    bl_y = st.slider("Inf-Izq Y %", 0, 100, 90, key="r_bl_y")

                pts = np.array([
                    [w * tl_x / 100, h * tl_y / 100],
                    [w * tr_x / 100, h * tr_y / 100],
                    [w * br_x / 100, h * br_y / 100],
                    [w * bl_x / 100, h * bl_y / 100],
                ], dtype="float32")

            img_with_pts = cv_img.copy()
            cv2.drawContours(img_with_pts, [ref_contour], -1, (0, 0, 255), 3)
            if auto_detect and target_contour is not None:
                cv2.drawContours(img_with_pts, [target_contour], -1, (0, 255, 0), 3)
            else:
                for pt in pts:
                    cv2.circle(img_with_pts, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), -1)

        # ── Sin hoja (medida conocida) ─────────────────────────────────────────
        else:
            st.markdown("### Selecciona las 4 esquinas del piso del cuarto")
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

            pts_ordered = order_points(pts)
            tl_o, tr_o, _, _ = pts_ordered
            width_px_ref = np.sqrt(((tr_o[0] - tl_o[0]) ** 2) + ((tr_o[1] - tl_o[1]) ** 2))
            if width_px_ref < 10:
                st.error("El área seleccionada es demasiado pequeña.")
                st.stop()
            px_per_cm = width_px_ref / known_width_cm

            img_with_pts = cv_img.copy()
            for pt in pts:
                cv2.circle(img_with_pts, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), -1)

        # ── Imagen con puntos ─────────────────────────────────────────────────
        with col1:
            st.subheader("Imagen original con puntos")
            st.image(cv2.cvtColor(img_with_pts, cv2.COLOR_BGR2RGB), use_container_width=True)

        # ── Corrección de perspectiva y cálculos ──────────────────────────────
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

                cv2.putText(
                    img_with_pts,
                    f"{room_w_m:.2f}m x {room_h_m:.2f}m",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                )
                cv2.putText(
                    room_img,
                    f"W: {room_w_m:.2f}m",
                    (int(room_w_px / 2) - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                )
                cv2.putText(
                    room_img,
                    f"H: {room_h_m:.2f}m",
                    (10, int(room_h_px / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                )

                with col2:
                    st.subheader("Perspectiva corregida del piso")
                    st.image(cv2.cvtColor(room_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.markdown("### Resultados del Cálculo de A/C")
                suggested_tonnage = max(1.0, round(np.ceil(tonnage * 2) / 2, 1))

                m1, m2, m3 = st.columns(3)
                m1.metric("Área del Cuarto", f"{room_w_m * room_h_m:.2f} m²")
                m2.metric("BTUs Requeridos", f"{int(btus):,}")
                m3.metric("Toneladas Calculadas", f"{tonnage:.2f}")

                st.info(
                    f"Sugerencia de Equipo: Te recomendamos instalar un equipo de "
                    f"**{suggested_tonnage} Toneladas**."
                )
                st.write(f"**Dimensiones corregidas:** {room_w_m:.2f} m × {room_h_m:.2f} m")

        except Exception as e:
            st.error(f"Error en los cálculos: {e}")
