import cv2
import numpy as np

def order_points(pts):
    """
    Ordena 4 puntos como top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

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
    Detecta el contorno rectangular más grande asumiendo que es la hoja de referencia.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
        
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    ref_contour = None
    for c in contours:
        if cv2.contourArea(c) < 500: continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            ref_contour = approx
            break
            
    if ref_contour is None:
        return None, None
        
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
    Detecta los dos contornos rectangulares más grandes (área objetivo y hoja de referencia).
    Usa cv2.RETR_LIST para encontrar hojas dentro de otras áreas.
    Retorna: target_contour, ref_contour, px_per_cm
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None
        
    rects = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for c in contours:
        if cv2.contourArea(c) < 1000:
            continue
            
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            if not rects or cv2.contourArea(rects[-1]) / cv2.contourArea(approx) > 1.2:
                rects.append(approx)
            if len(rects) == 2:
                break
                
    if len(rects) == 0:
        return None, None, None
    elif len(rects) == 1:
        target_contour = None
        ref_contour = rects[0]
    else:
        target_contour = rects[0]
        ref_contour = rects[1]
        
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
