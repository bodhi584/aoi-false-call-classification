import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Load dataset
    data_path = os.path.join("data", "dataset.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "raw", "dataset.csv")
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Output directory required by Issue #35
    output_dir = os.path.join("docs", "img")
    os.makedirs(output_dir, exist_ok=True)

    # Normalize column names for safe matching
    df.columns = [c.strip().lower() for c in df.columns]

    # Identify key columns dynamically
    label_col = next((c for c in df.columns if 'label' in c or 'target' in c or 'class' in c), df.columns[-1])
    insp_col = next((c for c in df.columns if 'inspection' in c or 'type' in c), df.columns[1] if len(df.columns) > 1 else df.columns[0])
    date_col = next((c for c in df.columns if 'date' in c or 'time' in c or 'day' in c), None)
    comp_col = next((c for c in df.columns if 'comp' in c or 'component' in c), None)

    # -------------------------------------------------------------
    # Chart 1: Share of false calls per inspection type
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    if insp_col in df.columns and label_col in df.columns:
        # Assuming 0 or 'False' indicates a false call
        false_calls = df[df[label_col] == 0].groupby(insp_col).size()
        total_counts = df.groupby(insp_col).size()
        share_fc = (false_calls / total_counts).fillna(0) * 100
        
        share_fc.plot(kind='bar', color='#2b5c8f')
        plt.title("Share of False Calls per Inspection Type")
        plt.xlabel("Inspection Type")
        plt.ylabel("False Call Share (%)")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    path1 = os.path.join(output_dir, "data_false_calls_by_inspection_type.png")
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart 1 saved to: {path1}")

    # -------------------------------------------------------------
    # Chart 2: Flagged boards and true defects per week
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    if date_col:
        df['week'] = pd.to_datetime(df[date_col], errors='coerce').dt.isocalendar().week
        weekly_flagged = df.groupby('week').size()
        weekly_defects = df[df[label_col] == 1].groupby('week').size() if label_col in df.columns else pd.Series(dtype=int)
        
        plt.plot(weekly_flagged.index, weekly_flagged.values, label='Flagged Boards', marker='o', color='#2b5c8f', linewidth=2)
        if not weekly_defects.empty:
            plt.plot(weekly_defects.index, weekly_defects.values, label='True Defects', marker='s', color='#d9534f', linewidth=2)
            
        plt.title("Flagged Boards and True Defects per Week")
        plt.xlabel("Calendar Week")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    
    path2 = os.path.join(output_dir, "data_weekly_flagged_and_defects.png")
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart 2 saved to: {path2}")

    # -------------------------------------------------------------
    # Chart 3: False calls per component type (Top 10)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    if comp_col and label_col in df.columns:
        top_comps = df[df[label_col] == 0][comp_col].value_counts().head(10)
        top_comps.plot(kind='barh', color='#5cb85c')
        plt.title("Top 10 Component Types with Most False Calls")
        plt.xlabel("Number of False Calls")
        plt.ylabel("Component Type")
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
    path3 = os.path.join(output_dir, "data_false_calls_by_component.png")
    plt.savefig(path3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart 3 saved to: {path3}")

    print("All required charts successfully generated in English inside docs/img/!")

if __name__ == "__main__":
    main()