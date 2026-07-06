def limpiar_accion(accion_texto):
    texto = accion_texto.replace('(', '').replace(')', '')
    return texto.strip().lower()

def calcular_score_plan(plan_generado, plan_optimo):
    P = [limpiar_accion(p) for p in plan_generado if p.strip()]
    G = [limpiar_accion(p) for p in plan_optimo if p.strip()]
    
    L_P = len(P)
    L_G = len(G)
    
    if L_P == 0:
        return 0.0
        
    score_horizonte = 2.0 if L_P == L_G else 0.0
    
    l_match = 0
    for p_accion, g_accion in zip(P, G):
        if p_accion == g_accion:
            l_match += 1
        else:
            break 
            
    score_progreso = 3.0 * (l_match / L_G)
    score_exacto = 5.0 if (l_match == L_G and L_P == L_G) else 0.0
    
    return round(score_horizonte + score_progreso + score_exacto, 2)
