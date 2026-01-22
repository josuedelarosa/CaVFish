import argparse
import os
import json
import subprocess

# ======================
# ARGUMENTOS (CLI)
# ======================

parser = argparse.ArgumentParser(description="Run keypoint inference over several CavFish datasets.")

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Ruta al archivo de configuración (.py) del modelo."
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Ruta al checkpoint (.pth) del modelo."
)

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Nombre del modelo para crear la carpeta principal dentro de CavFish."
)

args = parser.parse_args()

config_path = args.config
checkpoint_path = args.checkpoint
model_name = args.model


# ======================
# CONFIG GENERAL
# ======================

dataset_root = '/data/Datasets/Fish/CavFish'

DATASETS = [
    '2020 Bojonawi',
    '2020 Bajo Cauca Magdalena',
    '2021 Guaviare',
    '2022 Ayapel',
    '2023 Peces San Cipriano Buenaventura',
    '2024 Tarapoto',
]

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


# ======================
# FUNCIONES AUXILIARES
# ======================

def is_image_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTS


def build_out_paths(full_image_path: str, output_root: str) -> tuple[str, str]:
    """ Construye rutas de salida reflejando la estructura del dataset. """
    rel_path = os.path.relpath(full_image_path, dataset_root)
    rel_dir = os.path.dirname(rel_path)
    img_name = os.path.basename(rel_path)

    out_dir = os.path.join(output_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_base = os.path.join(out_dir, img_name)

    root, _ = os.path.splitext(out_base)
    out_json = f"{root}_keypoints.json"

    return out_base, out_json


# ======================
# PROCESAR CADA DATASET
# ======================

def run_inference_for_folder(folder_name: str):
    input_folder = os.path.join(dataset_root, folder_name)

    if not os.path.isdir(input_folder):
        print(f"❌ La carpeta de entrada no existe: {input_folder}")
        return

    # Tag para conservar orden en nombres
    tag = folder_name.lower().replace(' ', '-')

    # 🔥 Nuevo output: CavFish/<MODEL_NAME>/inference_<dataset>/
    output_root = os.path.join(dataset_root, model_name, f"inference_{tag}")

    merged_json_name = f'all_keypoints_predicted_{tag}.json'
    merged_json_path = os.path.join(output_root, merged_json_name)

    print("\n" + "=" * 80)
    print(f"📂 Procesando dataset: {folder_name}")
    print(f"   Carpeta entrada : {input_folder}")
    print(f"   Carpeta salida  : {output_root}")
    print(f"   JSON final      : {merged_json_path}")
    print("=" * 80 + "\n")

    # ---------- Listar imágenes ----------
    image_paths = []
    for root, dirs, files in os.walk(input_folder):
        for fname in files:
            if is_image_file(fname):
                image_paths.append(os.path.join(root, fname))

    image_paths.sort()

    print(f"Total de imágenes encontradas: {len(image_paths)}\n")

    if not image_paths:
        print("⚠️ No se encontraron imágenes en este dataset.")
        return

    # ---------- Inferencia o uso previo ----------
    all_predictions = []

    for idx, full_path in enumerate(image_paths, start=1):
        relative_path = os.path.relpath(full_path, dataset_root)
        out_base, out_json = build_out_paths(full_path, output_root)

        # 1) Intentar reutilizar JSON previo si existe
        reused_ok = False
        if os.path.exists(out_json):
            print(f"[{idx}/{len(image_paths)}] ✅ Ya procesada (intentando reutilizar JSON): {relative_path}")
            try:
                with open(out_json, "r") as f:
                    content = f.read().strip()
                if not content:
                    raise ValueError("JSON vacío")

                pred = json.loads(content)
                pred["image"] = relative_path
                all_predictions.append(pred)
                reused_ok = True

            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️ Problema leyendo {out_json}: {e}")
                print(f"   → Se volverá a ejecutar la inferencia para esta imagen.")
                # Opcional: eliminar el archivo corrupto para no volver a usarlo
                # os.remove(out_json)

        if reused_ok:
            # Todo bien con el JSON previo, pasamos a la siguiente imagen
            continue

        # 2) Si no hay JSON o estaba corrupto, ejecutar inferencia
        print(f"[{idx}/{len(image_paths)}] 🔁 Procesando {relative_path} ...")

        cmd = [
            "python", "demo/image_demo.py",
            full_path,
            config_path,
            checkpoint_path,
            "--out-file", out_base,
            "--draw-heatmap",
            "--show-kpt-idx"
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"⚠️ Error ejecutando inferencia para {relative_path} (code {result.returncode})")
            continue

        # 3) Intentar leer el JSON recién generado
        if os.path.exists(out_json):
            try:
                with open(out_json, "r") as f:
                    content = f.read().strip()
                if not content:
                    raise ValueError("JSON vacío tras inferencia")

                pred = json.loads(content)
                pred["image"] = relative_path
                all_predictions.append(pred)

            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️ JSON inválido incluso después de inferencia en {out_json}: {e}")
                # Aquí ya solo saltamos esta imagen
                continue
        else:
            print(f"⚠️ JSON esperado no encontrado: {out_json}")


# ======================
# MAIN
# ======================

if __name__ == "__main__":
    for ds in DATASETS:
        run_inference_for_folder(ds)
