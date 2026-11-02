import os
import matplotlib
matplotlib.use('Agg')  # Vital para que guarde las imágenes correctamente desde la terminal
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Load dataset
    data_path = os.path.join("data", "dataset.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "raw", "dataset.csv")
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully with {len(df)} rows.")

    # Output directory required by Issue #35
    output_dir = os.path.join("docs", "img")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Chart 1: Share of False Calls per Inspection Type
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    if 'inspection_type' in df.columns and 'class' in df.columns:
        min_class = df['class'].min()
        false_calls = df[df['class'] == min_class].groupby('inspection_type').size()
        total_counts = df.groupby('inspection_type').size()
        share_fc = (false_calls / total_counts).fillna(0) * 100
        
        share_fc.plot(kind='bar', color='#2b5c8f')
        plt.title("Share of False Calls per Inspection Type")
        plt.xlabel("Inspection Type")
        plt.ylabel("False Call Share (%)")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    path1 = os.path.join(output_dir, "data_false_calls_per_inspection_type.png")
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # Chart 2: Flagged Boards and True Defects per Week
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['week'] = df['datetime'].dt.isocalendar().week
        
        weekly_flagged = df.groupby('week').size()
        max_class = df['class'].max() if 'class' in df.columns else None
        weekly_defects = df[df['class'] == max_class].groupby('week').size() if max_class is not None else pd.Series(dtype=int)
        
        plt.plot(weekly_flagged.index, weekly_flagged.values, label='Flagged Boards', marker='o', color='#2b5c8f', linewidth=2)
        if not weekly_defects.empty:
            plt.plot(weekly_defects.index, weekly_defects.values, label='True Defects', marker='s', color='#d9534f', linewidth=2)
            
        plt.title("Flagged Boards and True Defects per Week")
        plt.xlabel("Calendar Week")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    
    path2 = os.path.join(output_dir, "data_flagged_boards_and_true_defects_per_week.png")
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # Chart 3: Pure Matplotlib Horizontal Bar Chart (Guaranteed Non-Blank)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    if 'inspection_type' in df.columns:
        counts = df['inspection_type'].value_counts().head(10)
        print("-> Chart 3 values to plot:", counts.to_dict())
        
        y_labels = counts.index.astype(str).tolist()
        x_values = counts.values.tolist()
        
        plt.barh(y_labels, x_values, color='#5cb85c')
        plt.title("Total Inspections per Inspection Type (Top 10)")
        plt.xlabel("Inspection Count")
        plt.ylabel("Inspection Type")
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
    path3 = os.path.join(output_dir, "data_false_calls_per_component_type.png")
    plt.savefig(path3, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n¡Listo! Los 3 gráficos se han generado con éxito y se guardaron en docs/img/")

if __name__ == "__main__":
    main()