def calculate_panels(roof_width_m, roof_height_m, panel_width_m, panel_height_m, spacing_m=0.1):
    """
    Calcula la cantidad de paneles que caben en una azotea.
    Retorna un diccionario con las métricas.
    """
    # Sumar el espacio al tamaño del panel
    effective_panel_w = panel_width_m + spacing_m
    effective_panel_h = panel_height_m + spacing_m
    
    cols = int(roof_width_m // effective_panel_w)
    rows = int(roof_height_m // effective_panel_h)
    
    total_panels = cols * rows
    usable_area = total_panels * (panel_width_m * panel_height_m)
    total_area = roof_width_m * roof_height_m
    
    efficiency_percentage = (usable_area / total_area) * 100 if total_area > 0 else 0
    
    return {
        "total_panels": total_panels,
        "panels_per_row": cols,
        "rows": rows,
        "usable_area": usable_area,
        "efficiency_percentage": efficiency_percentage
    }

def calculate_ac_tonnage(width_m, length_m, height_m=2.4, occupants=2, sun_exposure='medium'):
    """
    Calcula los BTUs y las toneladas requeridas para un cuarto.
    """
    area_m2 = width_m * length_m
    # Usa fórmula BTU = area_m² * 600 como base
    base_btu = area_m2 * 600
    
    # Ajuste: +10% por cada ocupante sobre 2
    extra_occupants = max(0, occupants - 2)
    occupant_adjustment = extra_occupants * 0.10 * base_btu
    
    # Ajuste: +10% si sun_exposure='alta', etc
    sun_adjustment = 0
    if sun_exposure == 'alta':
        sun_adjustment = 0.10 * base_btu
    elif sun_exposure == 'baja':
        sun_adjustment = -0.10 * base_btu
        
    total_btu = base_btu + occupant_adjustment + sun_adjustment
    tonnage = total_btu / 12000
    
    return total_btu, tonnage
