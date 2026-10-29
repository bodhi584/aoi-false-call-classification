import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Cargar los datos
    data_path = os.path.join("data", "dataset.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "raw", "dataset.csv")
        
    print(f"Cargando datos desde {data_path}...")
    df = pd.read_csv(data_path)

    # Crear carpeta de salida para los reportes si no existe
    output_dir = os.path.join("reports", "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Detectar la columna objetivo (etiqueta)
    if 'label' in df.columns:
        target_col = 'label'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        target_col = df.columns[-1]

    # 2. Figura 1: Distribución general de clases (True Defects vs False Calls)
    plt.figure(figsize=(7, 5))
    counts = df[target_col].value_counts()
    plt.bar(['False Call (0)', 'True Defect (1)'], [counts.get(0, 0), counts.get(1, 0)], color=['#2b5c8f', '#d9534f'])
    plt.title("Distribución Global de Alarmas en AOI")
    plt.ylabel("Número de Casos")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig1_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figura 1 guardada en: {fig1_path}")

    # 3. Figura 2: Evolución por día (si existe columna de día o fecha)
    day_col = [col for col in df.columns if 'day' in col.lower() or 'date' in col.lower()]
    if day_col:
        col_name = day_col[0]
        daily = df.groupby([col_name, target_col]).size().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 6))
        if 0 in daily.columns:
            plt.plot(daily.index, daily[0], label='False Calls', color='#2b5c8f', linewidth=2)
        if 1 in daily.columns:
            plt.plot(daily.index, daily[1], label='True Defects', color='#d9534f', linewidth=2)
            
        plt.title("Evolución Diaria de False Calls vs True Defects")
        plt.xlabel("Día de Producción")
        plt.ylabel("Cantidad de Alarmas")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        fig2_path = os.path.join(output_dir, "daily_false_calls_trend.png")
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Figura 2 guardada en: {fig2_path}")

    print("¡Análisis y figuras generadas con éxito!")

if __name__ == "__main__":
    main()