import os
import sys
from pathlib import Path

sys.path.append(r"d:\LIDAR")

import export_background_gif
import export_clearance_gif

def main():
    base_dir = Path(r"D:\Датасет")
    if not base_dir.exists():
        print(f"Ошибка: Директория {base_dir} не найдена.")
        return

    trials = [d.name for d in base_dir.iterdir() if d.is_dir()]
    
    print(f"Найдено {len(trials)} триалов: {trials}")

    for trial in trials:
        print(f"\n" + "="*50)
        print(f"Обрабатываем триал: {trial}")
        print("="*50)
        
        print(f"\n[1/2] Генерация 2D Background GIF для {trial}...")
        try:
            export_background_gif.export_gif(trial)
        except Exception as e:
            print(f"Ошибка background gif: {e}")
            
        print(f"\n[2/2] Генерация 3D Clearance GIF для {trial}...")
        try:
            export_clearance_gif.export_gif(trial)
        except Exception as e:
            print(f"Ошибка clearance gif: {e}")

    print("\n" + "="*50)
    print("ГЕНЕРАЦИЯ ВСЕХ GIF ЗАВЕРШЕНА!")
    print("="*50)

if __name__ == "__main__":
    main()

