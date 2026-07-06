import json
from llm_engine import qwen
from student_agent import AssemblyAgent

ARCHIVO_EVALUACION = "Task.json"
ARCHIVO_SALIDA = "submission.json"

def main():
    print(f"Iniciando ejecucion sobre: {ARCHIVO_EVALUACION}")
    with open(ARCHIVO_EVALUACION, 'r') as f:
        casos = json.load(f)

    agente = AssemblyAgent()
    resultados_entrega = []

    for i, caso in enumerate(casos):
        task_id = caso['assembly_task_id']
        print(f"Procesando caso {i+1}/{len(casos)} (ID: {task_id})...")

        try:
            plan_generado = agente.solve(caso['scenario_context'], qwen)

            resultados_entrega.append({
                "assembly_task_id": task_id,
                "complexity_level": len(plan_generado),
                "target_action_sequence": plan_generado
            })

        except Exception as e:
            print(f"ERROR critico en el caso {task_id}: {e}")
            print("Corrige tu codigo. https://www.youtube.com/watch?v=Y-U1calv6X8")
            return

    with open(ARCHIVO_SALIDA, 'w') as f:
        json.dump(resultados_entrega, f, indent=4)

    print("-" * 50)
    print(f"Exito. Archivo '{ARCHIVO_SALIDA}' generado correctamente.")

if __name__ == "__main__":
    main()
