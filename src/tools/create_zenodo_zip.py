import os
import zipfile
from pathlib import Path

def create_zenodo_zip():
    root_dir = Path(__file__).resolve().parent.parent.parent
    zip_path = root_dir / "opti_fab_zenodo_upload.zip"
    print(f"Creating Zenodo upload archive at: {zip_path}")
    
    # Files and folders to include
    files_to_include = [
        ".zenodo.json",
        "README.md",
        "requirements.txt"
    ]
    
    folders_to_include = [
        "src",
        "paper",
        "results"
    ]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add top-level files
        for f in files_to_include:
            file_path = root_dir / f
            if file_path.exists():
                zipf.write(file_path, f)
                print(f"  Added: {f}")
                
        # 2. Add folders (excluding __pycache__ and logs)
        for folder in folders_to_include:
            folder_path = root_dir / folder
            for root, dirs, files in os.walk(folder_path):
                # Modify dirs in-place to exclude pycache and logs
                dirs[:] = [d for d in dirs if d not in ("__pycache__", "logs", "trt_engine_cache")]
                for file in files:
                    # Skip logs inside src
                    if file.endswith(".log"):
                        continue
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(root_dir)
                    zipf.write(full_path, rel_path)
            print(f"  Added folder: {folder}")
            
        # 3. Add specific models (exclude large Keras model to stay under 50MB)
        models_to_add = ["opti_fab_model.onnx", "opti_fab_model_mc.onnx"]
        for m in models_to_add:
            model_path = root_dir / "models" / m
            if model_path.exists():
                zipf.write(model_path, Path("models") / m)
                print(f"  Added: models/{m}")
                
        # 4. Add test dataset subset (5 images per class)
        test_dir = root_dir / "dataset" / "test"
        if test_dir.exists():
            for class_name in os.listdir(test_dir):
                class_path = test_dir / class_name
                if class_path.is_dir():
                    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    images.sort()
                    # Keep only first 5 images
                    for img in images[:5]:
                        full_img_path = class_path / img
                        rel_img_path = Path("dataset") / "test" / class_name / img
                        zipf.write(full_img_path, rel_img_path)
            print("  Added test dataset subset (5 images per class)")
            
    print("Zenodo zip creation done!")

def create_paper_source_zip():
    root_dir = Path(__file__).resolve().parent.parent.parent
    zip_path = root_dir / "opti_fab_paper_source.zip"
    print(f"Creating paper source archive at: {zip_path}")
    
    paper_files = ["main.tex", "references.bib", "IEEEtran.cls", "IEEEtran.bst"]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in paper_files:
            file_path = root_dir / "paper" / f
            if file_path.exists():
                zipf.write(file_path, f)
                print(f"  Added: {f}")
                
    print("Paper source zip creation done!")

if __name__ == "__main__":
    create_zenodo_zip()
    print("-" * 50)
    create_paper_source_zip()
