# Real-time stream inference visualization using Pygame

import os
os.add_dll_directory(r"C:\Users\harih\AppData\Local\Programs\Python\Python310\Lib\site-packages\tensorrt_libs")
os.add_dll_directory(r"C:\Users\harih\AppData\Local\Programs\Python\Python310\Lib\site-packages\nvidia\cudnn\bin")
os.add_dll_directory(r"C:\Users\harih\AppData\Local\Programs\Python\Python310\Lib\site-packages\nvidia\cublas\bin")
os.add_dll_directory(r"C:\Users\harih\AppData\Local\Programs\Python\Python310\Lib\site-packages\nvidia\cuda_nvrtc\bin")

import sys
import time
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MODEL_ONNX, TEST_DIR, IMG_SIZE, CLASS_NAMES,
    SCAN_SPEED, CONFIDENCE_THRESHOLD,
    DEFECT_CLASSES, get_logger,
)

import pygame

log = get_logger(__name__)

# Max entropy-based uncertainty threshold
UNCERTAINTY_MAX = 0.35

# UI Layout configs
WIN_W       = 1100
WIN_H       = 680
FPS         = 60

WAFER_X     = 40
WAFER_Y     = 80
WAFER_SIZE  = 380

PANEL_X     = 480
PANEL_Y     = 80
PANEL_W     = 580
PANEL_H     = 560

# Palette
BG          = (10,  10,  18)
SURFACE     = (18,  18,  30)
BORDER      = (50,  50,  70)
GREEN       = (0,   255, 136)
AMBER       = (255, 170, 0)
RED         = (255, 60,  60)
BLUE        = (80,  160, 255)
WHITE       = (220, 220, 220)
GRAY        = (100, 100, 120)
DARK_GRAY   = (40,  40,  55)

SPEED_MAP   = {"slow": 80, "normal": 40, "fast": 15}


def get_session():
    # Setup ORT session with TensorRT provider
    trt_cache = str(Path(__file__).resolve().parent.parent / "benchmarks" / "trt_engine_cache")
    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_max_workspace_size" : 1 << 30,
                "trt_fp16_enable"        : True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path"  : trt_cache,
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    return ort.InferenceSession(str(MODEL_ONNX), providers=providers)


def run_inference(session, buffer):
    input_name = session.get_inputs()[0].name
    inp        = buffer[np.newaxis, ...]

    t0         = time.perf_counter()
    preds      = session.run(None, {input_name: inp})[0][0]
    infer_ms   = (time.perf_counter() - t0) * 1000

    pred_class = int(np.argmax(preds))
    confidence = float(preds[pred_class])

    # Entropy calculation
    eps         = 1e-9
    entropy     = float(-np.sum(preds * np.log(preds + eps)))
    max_entropy = float(np.log(len(preds)))
    uncertainty = entropy / max_entropy

    return pred_class, confidence, uncertainty, infer_ms


# Pygame layout helpers
def draw_text(surf, text, x, y, font, color=WHITE, anchor="topleft"):
    rendered = font.render(text, True, color)
    rect     = rendered.get_rect(**{anchor: (x, y)})
    surf.blit(rendered, rect)
    return rect


def draw_bar(surf, x, y, w, h, value, max_val, color, bg=DARK_GRAY):
    pygame.draw.rect(surf, bg, (x, y, w, h), border_radius=3)
    fill_w = int(w * min(value / max_val, 1.0))
    if fill_w > 0:
        pygame.draw.rect(surf, color, (x, y, fill_w, h), border_radius=3)


def draw_panel(surf, x, y, w, h, title, font_sm):
    pygame.draw.rect(surf, SURFACE, (x, y, w, h), border_radius=8)
    pygame.draw.rect(surf, BORDER,  (x, y, w, h), 1, border_radius=8)
    draw_text(surf, title, x + 16, y + 12, font_sm, GRAY)


def reset_state():
    return {
        "circular_buffer": np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32),
        "row"            : 0,
        "decided"        : False,
        "decision"       : "STREAMING",
        "decision_color" : GREEN,
        "pred_class"     : 0,
        "confidence"     : 0.0,
        "uncertainty"    : 0.0,
        "infer_ms"       : 0.0,
        "tick_log"       : [],
        "start_time"     : time.perf_counter(),
    }


def run_demo(img_path=None, speed="normal"):
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("OPTI-FAB — Stream Inference Demo")
    clock  = pygame.time.Clock()

    font_lg = pygame.font.SysFont("consolas", 22, bold=True)
    font_md = pygame.font.SysFont("consolas", 16)
    font_sm = pygame.font.SysFont("consolas", 12)
    font_xl = pygame.font.SysFont("consolas", 32, bold=True)

    tick_delay_ms = SPEED_MAP.get(speed, 40)

    log.info("Loading ONNX session...")
    session = get_session()
    log.info("Session ready")

    # Gather test images
    test_images = []
    if img_path:
        test_images = [Path(img_path)]
    else:
        for cls_dir in TEST_DIR.iterdir():
            if cls_dir.is_dir():
                for f in list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")):
                    test_images.append(f)
    test_images.sort()
    img_index = 0

    def load_image(idx):
        path = test_images[idx % len(test_images)]
        pil  = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        arr  = np.array(pil, dtype=np.float32) / 255.0
        return arr[:, :, np.newaxis], path, pil

    def pil_to_surface(pil, size):
        rgb  = pil.convert("RGB").resize((size, size))
        return pygame.image.fromstring(rgb.tobytes(), (size, size), "RGB")

    img_arr, current_path, pil_img = load_image(img_index)
    wafer_surf = pil_to_surface(pil_img, WAFER_SIZE)
    state      = reset_state()
    paused     = False
    last_tick  = pygame.time.get_ticks()
    FILE_LATENCY = 8.3

    running = True
    while running:
        clock.tick(FPS)
        now = pygame.time.get_ticks()

        # ---- EVENTS ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    img_index += 1
                    img_arr, current_path, pil_img = load_image(img_index)
                    wafer_surf = pil_to_surface(pil_img, WAFER_SIZE)
                    state      = reset_state()
                    last_tick  = now

        # ---- SIMULATION TICK ----
        s = state
        if not paused and not s["decided"] and (now - last_tick) >= tick_delay_ms:
            last_tick = now

            if s["row"] < IMG_SIZE:
                chunk = img_arr[s["row"]: s["row"] + SCAN_SPEED, :, :]
                s["circular_buffer"]                    = np.roll(s["circular_buffer"], -SCAN_SPEED, axis=0)
                s["circular_buffer"][-SCAN_SPEED:, :, :] = chunk
                s["row"] += SCAN_SPEED

                if s["row"] >= IMG_SIZE // 2:
                    pc, conf, unc, ims = run_inference(session, s["circular_buffer"])
                    s["pred_class"]  = pc
                    s["confidence"]  = conf
                    s["uncertainty"] = unc
                    s["infer_ms"]    = ims

                    s["tick_log"].append({
                        "row"        : s["row"],
                        "class_name" : CLASS_NAMES[pc],
                        "confidence" : conf,
                        "uncertainty": unc,
                    })
                    if len(s["tick_log"]) > 6:
                        s["tick_log"].pop(0)

                    # Decision gate
                    if conf >= CONFIDENCE_THRESHOLD and unc <= UNCERTAINTY_MAX:
                        s["decided"] = True
                        if pc in DEFECT_CLASSES:
                            s["decision"]       = f"EARLY REJECT  {CLASS_NAMES[pc].upper()}"
                            s["decision_color"] = RED
                        else:
                            s["decision"]       = f"EARLY ACCEPT  {CLASS_NAMES[pc].upper()}"
                            s["decision_color"] = GREEN
            else:
                s["decided"]        = True
                s["decision"]       = f"FULL FRAME  {CLASS_NAMES[s['pred_class']].upper()}"
                s["decision_color"] = AMBER

        # ---- DRAW ----
        screen.fill(BG)

        # Header
        draw_text(screen, "OPTI-FAB", 40, 20, font_lg, GREEN)
        draw_text(screen, "stream inference demo", 165, 25, font_sm, GRAY)
        draw_text(screen, current_path.name, WIN_W - 40, 25, font_sm, GRAY, anchor="topright")

        # Wafer panel
        draw_panel(screen, WAFER_X - 10, WAFER_Y - 30,
                   WAFER_SIZE + 20, WAFER_SIZE + 110, "WAFER IMAGE", font_sm)
        screen.blit(wafer_surf, (WAFER_X, WAFER_Y))

        # Scanned overlay
        scan_y_px = int((s["row"] / IMG_SIZE) * WAFER_SIZE)
        ov = pygame.Surface((WAFER_SIZE, min(scan_y_px, WAFER_SIZE)), pygame.SRCALPHA)
        ov.fill((0, 255, 136, 22))
        screen.blit(ov, (WAFER_X, WAFER_Y))

        # Scan line
        if not s["decided"]:
            ly = WAFER_Y + min(scan_y_px, WAFER_SIZE - 2)
            pygame.draw.rect(screen, GREEN, (WAFER_X, ly, WAFER_SIZE, 2))

        # Row progress
        draw_text(screen,
                  f"{s['row']} / {IMG_SIZE} rows  ({int(s['row']/IMG_SIZE*100)}%)",
                  WAFER_X, WAFER_Y + WAFER_SIZE + 8, font_sm, GRAY)

        # Latency comparison
        cy = WAFER_Y + WAFER_SIZE + 34
        draw_text(screen, "file-based",  WAFER_X + 10,  cy,      font_sm, GRAY)
        draw_text(screen, f"{FILE_LATENCY:.1f} ms", WAFER_X + 10,  cy + 18, font_md, RED)
        draw_text(screen, "opti-fab",    WAFER_X + 180, cy,      font_sm, GRAY)
        draw_text(screen, f"{s['infer_ms']:.2f} ms", WAFER_X + 180, cy + 18, font_md, GREEN)
        draw_text(screen, "trt / frame", WAFER_X + 310, cy,      font_sm, GRAY)
        draw_text(screen, "0.79 ms",     WAFER_X + 310, cy + 18, font_md, BLUE)

        # Right panel
        draw_panel(screen, PANEL_X, PANEL_Y - 30, PANEL_W, PANEL_H, "LIVE INFERENCE", font_sm)

        py = PANEL_Y + 10

        # Confidence
        draw_text(screen, "CONFIDENCE", PANEL_X + 16, py, font_sm, GRAY)
        draw_text(screen, f"{s['confidence']:.4f}", PANEL_X + PANEL_W - 20, py,
                  font_sm, WHITE, anchor="topright")
        py += 18
        draw_bar(screen, PANEL_X + 16, py, PANEL_W - 32, 10, s["confidence"], 1.0, GREEN)
        tx = PANEL_X + 16 + int((PANEL_W - 32) * CONFIDENCE_THRESHOLD)
        pygame.draw.rect(screen, AMBER, (tx, py - 3, 2, 16))
        py += 26

        # Uncertainty
        draw_text(screen, "UNCERTAINTY (entropy)", PANEL_X + 16, py, font_sm, GRAY)
        draw_text(screen, f"{s['uncertainty']:.4f}", PANEL_X + PANEL_W - 20, py,
                  font_sm, WHITE, anchor="topright")
        py += 18
        draw_bar(screen, PANEL_X + 16, py, PANEL_W - 32, 10, s["uncertainty"], 1.0, AMBER)
        ux = PANEL_X + 16 + int((PANEL_W - 32) * UNCERTAINTY_MAX)
        pygame.draw.rect(screen, RED, (ux, py - 3, 2, 16))
        py += 26

        # Class + frame
        draw_text(screen, "PREDICTED CLASS", PANEL_X + 16, py, font_sm, GRAY)
        cls_col = RED if s["pred_class"] in DEFECT_CLASSES else GREEN
        draw_text(screen, CLASS_NAMES[s["pred_class"]].upper(),
                  PANEL_X + PANEL_W - 20, py, font_md, cls_col, anchor="topright")
        py += 26

        draw_text(screen, "FRAME PROGRESS", PANEL_X + 16, py, font_sm, GRAY)
        draw_text(screen, f"{int(s['row']/IMG_SIZE*100)}%",
                  PANEL_X + PANEL_W - 20, py, font_md, WHITE, anchor="topright")
        py += 30

        # Divider
        pygame.draw.rect(screen, BORDER, (PANEL_X + 16, py, PANEL_W - 32, 1))
        py += 14

        # Tick log
        draw_text(screen, "TICK LOG", PANEL_X + 16, py, font_sm, GRAY)
        py += 18
        for i, entry in enumerate(s["tick_log"]):
            is_last = (i == len(s["tick_log"]) - 1)
            col     = WHITE if is_last else GRAY
            line    = (f"[{entry['row']:03d}] {entry['class_name']:<12} "
                       f"conf:{entry['confidence']:.3f}  unc:{entry['uncertainty']:.3f}")
            draw_text(screen, line, PANEL_X + 16, py, font_sm, col)
            py += 16
        py += 8

        # Divider
        pygame.draw.rect(screen, BORDER, (PANEL_X + 16, py, PANEL_W - 32, 1))
        py += 16

        # Decision box
        box_h   = 58
        bg_cols = {
            id(RED)  : (55, 10, 10),
            id(GREEN): (10, 50, 30),
            id(AMBER): (50, 38, 5),
        }
        bg_col = bg_cols.get(id(s["decision_color"]), (28, 28, 40))
        pygame.draw.rect(screen, bg_col,
                         (PANEL_X + 16, py, PANEL_W - 32, box_h), border_radius=6)
        pygame.draw.rect(screen, s["decision_color"],
                         (PANEL_X + 16, py, PANEL_W - 32, box_h), 1, border_radius=6)
        draw_text(screen, s["decision"],
                  PANEL_X + 16 + (PANEL_W - 32) // 2, py + box_h // 2,
                  font_md, s["decision_color"], anchor="center")

        # Controls
        draw_text(screen, "SPACE=pause    R=next image    Q=quit",
                  WIN_W // 2, WIN_H - 16, font_sm, GRAY, anchor="midbottom")

        # Paused overlay
        if paused:
            ov2 = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            ov2.fill((0, 0, 0, 130))
            screen.blit(ov2, (0, 0))
            draw_text(screen, "PAUSED", WIN_W // 2, WIN_H // 2,
                      font_xl, AMBER, anchor="center")

        pygame.display.flip()

    pygame.quit()


# CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OPTI-FAB Stream Demo")
    parser.add_argument("--img_path", type=str, default=None)
    parser.add_argument("--speed",    type=str, default="normal",
                        choices=["slow", "normal", "fast"])
    args = parser.parse_args()

    log.info("Starting stream demo")
    run_demo(args.img_path, args.speed)