import argparse
import os
import json
import time
import logging

import torch
from mmengine.logging import print_log
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples


# ======================
# CONFIG GENERAL
# ======================

DEFAULT_DATASET_ROOT = '/data/Datasets/Fish/CavFish'

DEFAULT_DATASETS = [
    '2020 Bojonawi',
    '2020 Bajo Cauca Magdalena',
    '2021 Guaviare',
    '2022 Ayapel',
    '2023 Peces San Cipriano Buenaventura',
    '2024 Tarapoto',
]

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

KEYPOINT_NAMES = [
    '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9', '10', '11',
    '12', '13', '14', '15', '16',
    '17', '18', '19', '20'
]


# ======================
# ARGUMENTOS (CLI)
# ======================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run keypoint inference over CavFish datasets (JSON only, no visualizer)."
    )

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
        help="Nombre del modelo para crear la carpeta principal dentro de CavFish (CavFish/<MODEL_NAME>/...)."
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help=f"Ruta raíz de los datasets CavFish (default: {DEFAULT_DATASET_ROOT})"
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Lista de datasets a procesar (por nombre exacto de carpeta dentro de dataset-root). "
            "Si se omite, se usan todos los datasets por defecto."
        )
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Dispositivo para inferencia (por ejemplo 'cuda:0' o 'cpu')."
    )

    parser.add_argument(
        "--with-heatmap",
        action="store_true",
        help="Si se activa, se piden heatmaps al modelo (solo en los outputs internos; no se dibuja nada)."
    )

    return parser.parse_args()


# ======================
# FUNCIONES AUXILIARES
# ======================

def is_image_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTS


def build_out_paths(full_image_path: str, dataset_root: str, output_root: str):
    """Construye rutas de salida reflejando la estructura del dataset."""
    rel_path = os.path.relpath(full_image_path, dataset_root)
    rel_dir = os.path.dirname(rel_path)
    img_name = os.path.basename(rel_path)

    out_dir = os.path.join(output_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_base = os.path.join(out_dir, img_name)
    root, _ = os.path.splitext(out_base)
    out_json = f"{root}_keypoints.json"

    return out_base, out_json, rel_path


# ======================
# MODELO (SIN VISUALIZADOR)
# ======================

def build_model(
    config_path,
    checkpoint_path,
    device='cuda:0',
    with_heatmap=False,
):
    """Inicializa el modelo UNA vez y lo devuelve (sin visualizador)."""
    cfg_options = None
    if with_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

    model = init_model(
        config=config_path,
        checkpoint=checkpoint_path,
        device=device,
        cfg_options=cfg_options
    )

    # --- Debug: check actual device ---
    device_obj = next(model.parameters()).device
    print(f"\n🧠 Model is on device: {device_obj}")
    if device_obj.type == "cuda":
        print("   CUDA available:", torch.cuda.is_available())
        try:
            print("   GPU name:", torch.cuda.get_device_name(device_obj.index or 0))
        except Exception:
            pass

    # Enable cuDNN autotune for conv nets with fixed sizes
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    # Parchar dataset_meta para tus 20 keypoints
    model.dataset_meta = {
        'dataset_name': 'FishPose20Kpt',
        'num_keypoints': 20,
        'keypoint_names': KEYPOINT_NAMES,
        'keypoint_id2name': {i: name for i, name in enumerate(KEYPOINT_NAMES)},
        'keypoint_name2id': {name: i for i, name in enumerate(KEYPOINT_NAMES)},
        'skeleton': [],
        'keypoint_colors': [[255, 0, 0]] * 20,
        'skeleton_links_color': [[0, 255, 0]] * 7,
        'flip_pairs': [],
        'flip_indices': list(range(20)),
        'upper_body_ids': [0, 2, 3, 4, 5],
        'lower_body_ids': [1, 6, 7],
        'dataset_keypoint_weights': [1.0] * 20,
        'sigmas': [0.05] * 20,
        'CLASSES': ['fish']
    }

    print('\n✅ DEBUG: Patched model.dataset_meta keys =', model.dataset_meta.keys(), '\n')

    # Pipeline de test (igual que en tu image_demo original)
    # Ajusta si tu config ya lo trae bien:
    model.cfg.test_dataloader.dataset.pipeline = [
        dict(type='LoadImage'),
        dict(type='GetBBoxCenterScale', padding=1.25),
        dict(type='TopdownAffine', input_size=(192, 256), use_udp=True),
        dict(type='PackPoseInputs')
    ]

    return model


def infer_single_image(
    model,
    img_path,
    save_json_path,
):
    """Corre inferencia sobre UNA imagen usando modelo ya cargado. Sólo JSON, sin visualización."""
    t0 = time.time()

    dev = next(model.parameters()).device
    use_cuda = dev.type == "cuda"

    # Usamos la API de MMPose exactamente como en tu image_demo original:
    # inference_topdown(model, args.img)
    # pero con no_grad() y autocast() para ser más rápido.
    with torch.no_grad():
        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                batch_results = inference_topdown(model, img_path)
        else:
            batch_results = inference_topdown(model, img_path)

    results = merge_data_samples(batch_results)

    keypoints = results.pred_instances.keypoints
    scores = results.pred_instances.keypoint_scores

    # Construir estructura JSON (una sola vez por imagen)
    predicted_data = {
        'image': os.path.basename(img_path),
        'keypoints': [
            {
                'name': model.dataset_meta['keypoint_names'][idx],
                'x': float(xy[0]),
                'y': float(xy[1]),
                'score': float(score_val)
            }
            for idx, (xy, score_val) in enumerate(zip(keypoints[0], scores[0]))
        ]
    }

    # Guardar JSON
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    with open(save_json_path, 'w') as f:
        json.dump(predicted_data, f, indent=4)

    dt = time.time() - t0
    print(f"    ⏱ tiempo inferencia (incl. JSON): {dt:.2f}s")

    return predicted_data


# ======================
# PROCESAR CADA DATASET
# ======================

def run_inference_for_folder(
    folder_name: str,
    model,
    args
):
    dataset_root = args.dataset_root
    model_name = args.model

    input_folder = os.path.join(dataset_root, folder_name)

    if not os.path.isdir(input_folder):
        print(f"❌ La carpeta de entrada no existe: {input_folder}")
        return

    tag = folder_name.lower().replace(' ', '-')

    output_root = os.path.join(dataset_root, model_name, f"inference_{tag}")
    os.makedirs(output_root, exist_ok=True)

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

    all_predictions = []

    # ---------- Procesar cada imagen ----------
    for idx, full_path in enumerate(image_paths, start=1):
        out_base, out_json, relative_path = build_out_paths(
            full_image_path=full_path,
            dataset_root=dataset_root,
            output_root=output_root
        )

        # 1) Intentar reutilizar JSON previo si existe
        reused_ok = False
        if os.path.exists(out_json):
            print(f"[{idx}/{len(image_paths)}] ✅ Ya procesada (reutilizando JSON): {relative_path}")
            try:
                with open(out_json, "r") as f:
                    content = f.read().strip()
                if not content:
                    raise ValueError("JSON vacío")

                pred = json.loads(content)
                # Forzar ruta relativa global
                pred["image"] = relative_path
                all_predictions.append(pred)
                reused_ok = True

            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️ Problema leyendo {out_json}: {e}")
                print(f"   → Se volverá a ejecutar la inferencia para esta imagen.")

        if reused_ok:
            continue

        # 2) Inferencia normal con modelo ya cargado
        print(f"[{idx}/{len(image_paths)}] 🔁 Procesando {relative_path} ...")

        try:
            pred = infer_single_image(
                model=model,
                img_path=full_path,
                save_json_path=out_json,
            )
            # Usar ruta relativa respecto a CavFish (coherente con tu script original)
            pred["image"] = relative_path
            all_predictions.append(pred)

        except Exception as e:
            print(f"⚠️ Error en inferencia para {relative_path}: {e}")
            continue

    # ---------- Guardar JSON combinado del dataset ----------
    try:
        with open(merged_json_path, "w") as f:
            json.dump(all_predictions, f, indent=2)
        print(f"\n✅ JSON combinado guardado en: {merged_json_path}")
    except Exception as e:
        print(f"⚠️ Error guardando JSON combinado {merged_json_path}: {e}")


# ======================
# MAIN
# ======================

def main():
    args = parse_args()

    datasets = args.datasets if args.datasets is not None else DEFAULT_DATASETS

    print("\n" + "=" * 80)
    print("🚀 Iniciando inferencia batch CavFish (JSON only)")
    print(f"Config        : {args.config}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Modelo        : {args.model}")
    print(f"Device        : {args.device}")
    print(f"Root          : {args.dataset_root}")
    print(f"Datasets      : {datasets}")
    print(f"Heatmaps flag : {'ON' if args.with_heatmap else 'OFF'}")
    print("=" * 80 + "\n")

    # 1) Cargar modelo UNA vez (sin visualizador)
    model = build_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        with_heatmap=args.with_heatmap,
    )

    # 2) Procesar cada dataset
    for ds in datasets:
        run_inference_for_folder(
            folder_name=ds,
            model=model,
            args=args
        )

    print("\n✅ Todo terminado.\n")


if __name__ == "__main__":
    main()
