import sys
import os
import io
import contextlib

sys.path.append(r"d:\LIDAR\ml")
import train_xgb

folders = [
    "D:/synth_data/processed_slices_debug",
    "D:/synth_data/processed_slices_icp",
    "D:/synth_data/processed_slices_tunnel"
]

output_file = r"d:\LIDAR\ml\training_stats.txt"

with open(output_file, "w", encoding="utf-8") as out:
    for folder in folders:
        print(f"\n=============================================")
        print(f"TRAINING ON FOLDER: {folder}")
        print(f"=============================================")
        
        out.write(f"\n=============================================\n")
        out.write(f"TRAINING ON FOLDER: {folder}\n")
        out.write(f"=============================================\n")
        
        # Monkey patch default arguments
        train_xgb.load_and_prepare_data.__defaults__ = (folder, 1.0)
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                train_xgb.search()
                train_xgb.train()
            except Exception as e:
                print(f"Error: {e}")
                out.write(f"Error: {e}\n")
                
        stdout_str = f.getvalue()
        print(stdout_str)
        out.write(stdout_str)
        
        model_name = f"xgb_obstacle_{os.path.basename(folder)}.json"
        if os.path.exists("xgb_obstacle.json"):
            if os.path.exists(model_name):
                os.remove(model_name)
            os.rename("xgb_obstacle.json", model_name)
            msg = f"\nModel saved to {model_name}\n"
            print(msg)
            out.write(msg)
        else:
            msg = f"\nWarning: xgb_obstacle.json not found!\n"
            print(msg)
            out.write(msg)

print(f"\nAll done! Stats saved to {output_file}")
