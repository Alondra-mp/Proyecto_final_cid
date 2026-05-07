import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def _find_quads(image, min_area=1000):
    """Encuentra cuadriláteros convexos en la imagen. Retorna lista de (approx, area)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Dilatar bordes para cerrar huecos pequeños y mejorar cierre de contornos
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    quads = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append((approx, area))
    return quads


def _calc_quad_ratio(quad):
    """Retorna min/max de (ancho, alto) del cuadrilátero ordenado."""
    pts = quad.reshape(4, 2).astype("float32")
    rect = order_points(pts)
    tl, tr, br, bl = rect
    w = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    h = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    if w < 1 or h < 1:
        return None, w, h
    return min(w, h) / max(w, h), w, h


def four_point_transform(image, points):
    """
    Corrige la perspectiva de la imagen usando 4 puntos con cv2.getPerspectiveTransform y cv2.warpPerspective.
    """
    rect = order_points(points)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    if maxWidth == 0 or maxHeight == 0:
        return np.array([])
        
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def detect_reference_object(image):
    """
    Detecta la hoja de referencia tamaño carta (21.6 x 27.9 cm).
    Primero busca por proporción (~0.774), luego usa la más pequeña como fallback.
    """
    quads = _find_quads(image, min_area=500)
    if not quads:
        return None, None

    LETTER_RATIO = 21.6 / 27.9
    RATIO_TOL = 0.10

    ref_contour = None
    for q, area in quads:
        ratio, _, _ = _calc_quad_ratio(q)
        if ratio is not None and abs(ratio - LETTER_RATIO) < RATIO_TOL:
            ref_contour = q
            break

    if ref_contour is None:
        ref_contour = quads[0][0]

    pts = ref_contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_px = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    height_px = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_px = max(width_px, height_px)
    px_per_cm = max_px / 27.9

    return ref_contour, px_per_cm

def detect_target_and_reference(image):
    """
    Detecta el área objetivo y la hoja de referencia carta (21.6 x 27.9 cm).

    Estrategia:
    1. Busca la hoja carta por su proporción característica (~0.774 ± 0.10).
    2. El área objetivo es el cuadrilátero más grande que NO sea la hoja.
    3. Si no hay coincidencia por proporción, usa fallback por tamaño
       (segundo cuadrilátero más grande = referencia).

    Retorna: target_contour, ref_contour, px_per_cm
    """
    quads = _find_quads(image, min_area=1000)
    if not quads:
        return None, None, None

    LETTER_RATIO = 21.6 / 27.9  # ~0.774
    RATIO_TOL = 0.10

    ref_contour = None
    target_contour = None
    ref_candidates = []
    other_quads = []

    for q, area in quads:
        ratio, _, _ = _calc_quad_ratio(q)
        if ratio is not None and abs(ratio - LETTER_RATIO) < RATIO_TOL:
            ref_candidates.append((q, area))
        else:
            other_quads.append((q, area))

    if ref_candidates:
        # De los candidatos a hoja, tomar el de menor área (la hoja es más pequeña que el target)
        ref_contour = min(ref_candidates, key=lambda x: x[1])[0]
        if other_quads:
            target_contour = other_quads[0][0]  # mayor área entre los no-referencia
    else:
        # Fallback: el más grande = target, el segundo = referencia
        if len(quads) >= 2:
            target_contour = quads[0][0]
            ref_contour = quads[1][0]
        elif quads:
            ref_contour = quads[0][0]

    if ref_contour is None:
        return None, None, None

    pts = ref_contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_px = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    height_px = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_px = max(width_px, height_px)
    px_per_cm = max_px / 27.9

    return target_contour, ref_contour, px_per_cm

def draw_panel_grid(image, area_dims, panel_dims, px_per_cm):
    """
    Dibuja rectángulos con cv2.rectangle representando los paneles, calcula cuántos caben.
    Retorna la imagen anotada y el conteo de paneles.
    area_dims: (ancho_cm, alto_cm)
    panel_dims: (ancho_panel_cm, alto_panel_cm)
    """
    annotated_img = image.copy()
    
    area_w_cm, area_h_cm = area_dims
    panel_w_cm, panel_h_cm = panel_dims
    
    cols = int(area_w_cm // panel_w_cm)
    rows = int(area_h_cm // panel_h_cm)
    
    total_panels = cols * rows
    
    panel_w_px = int(panel_w_cm * px_per_cm)
    panel_h_px = int(panel_h_cm * px_per_cm)
    
    for row in range(rows):
        for col in range(cols):
            x_start = col * panel_w_px
            y_start = row * panel_h_px
            x_end = x_start + panel_w_px
            y_end = y_start + panel_h_px
            
            cv2.rectangle(annotated_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            
    return annotated_img, total_panels
