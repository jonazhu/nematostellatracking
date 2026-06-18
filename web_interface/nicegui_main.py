from nicegui import events, ui
import os
import pandas as pd

# ── Class definitions ──────────────────────────────────────────────────────────
CLASSES = {
    'Class A': 'SkyBlue',
    'Class B': 'Tomato',
}
YOLO_COLOR = 'MediumSeaGreen'

# ── State ──────────────────────────────────────────────────────────────────────
start_x        = None
start_y        = None
boxes          = []   # {'class', 'x1','y1','x2','y2'}  — all boxes treated equally
current_class  = list(CLASSES.keys())[0]
current_image  = None
selected_index = None
yolo_model     = None

all_annotations = pd.DataFrame(columns=['filename', 'class', 'x1', 'y1', 'x2', 'y2'])

# ── Drawing & selection ────────────────────────────────────────────────────────
def hit_test(x, y):
    for i in range(len(boxes) - 1, -1, -1):
        b = boxes[i]
        if b['x1'] <= x <= b['x2'] and b['y1'] <= y <= b['y2']:
            return i
    return None

def mouse_handler(e: events.MouseEventArguments):
    global start_x, start_y, selected_index
    if e.type == 'mousedown':
        start_x, start_y = e.image_x, e.image_y
    elif e.type == 'mouseup' and start_x is not None:
        x1 = min(start_x, e.image_x);  y1 = min(start_y, e.image_y)
        x2 = max(start_x, e.image_x);  y2 = max(start_y, e.image_y)
        if (x2 - x1) <= 5 and (y2 - y1) <= 5:
            hit = hit_test(e.image_x, e.image_y)
            selected_index = hit
            if hit is not None:
                ui.notify(f'Selected box {hit + 1} ({boxes[hit]["class"]})')
        else:
            selected_index = None
            boxes.append({'class': current_class, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
            ui.notify(f'[{current_class}] ({x1:.0f},{y1:.0f}) → ({x2:.0f},{y2:.0f})')
        redraw()
        update_delete_button()
        start_x = start_y = None

def box_color(cls):
    if cls in CLASSES:
        return CLASSES[cls]
    # Any YOLO-predicted class not in CLASSES dict gets the YOLO color
    return YOLO_COLOR

def redraw():
    svg = ''
    for i, box in enumerate(boxes):
        color  = box_color(box['class'])
        is_sel = (i == selected_index)
        x, y   = box['x1'], box['y1']
        w, h   = box['x2'] - box['x1'], box['y2'] - box['y1']
        is_yolo = box['class'] not in CLASSES
        label   = f"{i + 1}: {box['class']}"
        stroke  = '5' if is_sel else '3'
        fill    = 'rgba(255,255,255,0.15)' if is_sel else 'none'
        # Draw the bounding rectangle for every box
        svg += (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="{fill}" stroke="{color}" stroke-width="{stroke}"/>'
        )
        # Only draw the filled label tag for manual boxes
        if not is_yolo:
            svg += (
                f'<rect x="{x}" y="{y}" width="{len(label)*8+8}" height="20" fill="{color}"/>'
                f'<text x="{x+4}" y="{y+14}" fill="white" font-size="12" font-weight="bold">{label}</text>'
            )
    ii.content = svg

# ── Class selector & box ops ───────────────────────────────────────────────────
def set_class(cls):
    global current_class
    current_class = cls
    for name, btn in class_buttons.items():
        btn.props('color=primary unelevated' if name == cls else 'color=grey outlined')

def update_delete_button():
    if selected_index is not None:
        delete_btn.props('color=negative')
        delete_btn.set_text(f'Delete Box {selected_index + 1}')
        delete_btn.enable()
    else:
        delete_btn.props('color=negative outlined')
        delete_btn.set_text('Delete Selected')
        delete_btn.disable()

def delete_selected():
    global selected_index
    if selected_index is None:
        return
    boxes.pop(selected_index)
    selected_index = None
    redraw()
    update_delete_button()
    update_status()

def undo_box():
    global selected_index
    if boxes:
        boxes.pop()
        selected_index = None
        redraw()
        update_delete_button()
        ui.notify('Last box removed')

def clear_boxes():
    global selected_index
    boxes.clear()
    selected_index = None
    ii.content = ''
    update_delete_button()
    ui.notify('All boxes cleared')

# ── YOLO inference ─────────────────────────────────────────────────────────────
def load_yolo_model():
    global yolo_model
    pt_path = model_input.value.strip()
    if not pt_path:
        ui.notify('Enter a path to a .pt file first', type='warning')
        return
    if not os.path.isfile(pt_path):
        ui.notify('File not found', type='negative')
        return
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(pt_path)
        ui.notify(f'Model loaded: {os.path.basename(pt_path)}', type='positive')
        run_yolo_btn.enable()
    except Exception as ex:
        ui.notify(f'Failed to load model: {ex}', type='negative')

def iou(a, b):
    """Intersection-over-Union for two boxes given as dicts with x1/y1/x2/y2."""
    ix1 = max(a['x1'], b['x1']);  iy1 = max(a['y1'], b['y1'])
    ix2 = min(a['x2'], b['x2']);  iy2 = min(a['y2'], b['y2'])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
    area_b = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
    return inter / (area_a + area_b - inter)

def deduplicate(candidates, iou_threshold=0.5):
    """
    Greedy duplicate removal: sort by confidence descending, then suppress
    any later box whose IoU with an already-kept box exceeds the threshold.
    """
    ranked = sorted(candidates, key=lambda b: b['conf'], reverse=True)
    kept = []
    for candidate in ranked:
        if all(iou(candidate, k) <= iou_threshold for k in kept):
            kept.append(candidate)
    return kept

def run_yolo():
    """Run inference and add predictions as ordinary saveable boxes."""
    if yolo_model is None:
        ui.notify('Load a model first', type='warning')
        return
    if current_image is None:
        ui.notify('No image loaded', type='warning')
        return
    try:
        results = yolo_model(current_image, verbose=False)[0]
        # Collect raw predictions
        candidates = []
        for box in results.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            cls_name = results.names[int(box.cls)]
            conf     = float(box.conf)
            label    = f'{cls_name} {conf:.0%}'
            candidates.append({'class': label, 'x1': x1, 'y1': y1,
                                'x2': x2, 'y2': y2, 'conf': conf})
        # Remove overlapping duplicates (IoU > 50%)
        kept = deduplicate(candidates, iou_threshold=0.5)
        removed = len(candidates) - len(kept)
        for b in kept:
            boxes.append({k: v for k, v in b.items() if k != 'conf'})
        redraw()
        update_status()
        msg = f'YOLO added {len(kept)} box(es)'
        if removed:
            msg += f' — {removed} duplicate(s) removed'
        ui.notify(msg, type='positive')
    except Exception as ex:
        ui.notify(f'Inference error: {ex}', type='negative')

# ── Persistence ────────────────────────────────────────────────────────────────
def save_current_boxes():
    global all_annotations
    if current_image is None:
        return
    fname = os.path.basename(current_image)
    all_annotations = all_annotations[all_annotations['filename'] != fname]
    if boxes:
        new_rows = pd.DataFrame([
            {'filename': fname, 'class': b['class'],
             'x1': round(b['x1']), 'y1': round(b['y1']),
             'x2': round(b['x2']), 'y2': round(b['y2'])}
            for b in boxes
        ])
        all_annotations = pd.concat([all_annotations, new_rows], ignore_index=True)

def restore_boxes_for(filename):
    global selected_index
    boxes.clear()
    selected_index = None
    fname = os.path.basename(filename)
    rows = all_annotations[all_annotations['filename'] == fname]
    for _, row in rows.iterrows():
        boxes.append({
            'class': row['class'],
            'x1': float(row['x1']), 'y1': float(row['y1']),
            'x2': float(row['x2']), 'y2': float(row['y2']),
        })

# ── File / directory ───────────────────────────────────────────────────────────
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif'}

def load_directory():
    path = dir_input.value.strip()
    if not os.path.isdir(path):
        ui.notify('Directory not found', type='negative')
        return
    files = sorted(f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    if not files:
        ui.notify('No image files found', type='warning')
        return
    file_select.options = files
    file_select.value   = files[0]
    file_select.update()
    switch_to_image(os.path.join(path, files[0]))
    ui.notify(f'{len(files)} image(s) found')
    update_nav_buttons()

def on_file_select(e):
    switch_to_image(os.path.join(dir_input.value.strip(), e.value))

def step_image(delta):
    """Move forward (+1) or backward (-1) through the file list."""
    files = file_select.options
    if not files or file_select.value not in files:
        return
    idx = files.index(file_select.value) + delta
    idx = max(0, min(idx, len(files) - 1))
    file_select.value = files[idx]
    file_select.update()
    switch_to_image(os.path.join(dir_input.value.strip(), files[idx]))

def update_nav_buttons():
    files = file_select.options
    if not files or file_select.value not in files:
        prev_btn.disable(); next_btn.disable()
        return
    idx = files.index(file_select.value)
    prev_btn.set_enabled(idx > 0)
    next_btn.set_enabled(idx < len(files) - 1)

def switch_to_image(full_path):
    global current_image
    save_current_boxes()
    current_image = full_path
    restore_boxes_for(full_path)
    ii.set_source(full_path)
    redraw()
    update_delete_button()
    annotation_row.set_visibility(True)
    update_status()
    update_nav_buttons()
    # Auto-run YOLO if a model is loaded and this image has no existing annotations
    if yolo_model is not None and len(boxes) == 0:
        run_yolo()

def export_csv():
    save_current_boxes()
    if all_annotations.empty:
        ui.notify('No annotations to export', type='warning')
        return
    out = os.path.join(dir_input.value.strip(), 'annotations.csv')
    all_annotations.to_csv(out, index=False)
    ui.notify(f'Saved → {out}')

def update_status():
    if current_image is None:
        return
    fname = os.path.basename(current_image)
    status_label.set_text(
        f'{fname}  ·  {len(boxes)} box(es) on this image  ·  {len(all_annotations)} total saved'
    )

# ── UI ─────────────────────────────────────────────────────────────────────────
ui.label('Image Annotation Tool').classes('text-xl font-bold mb-2')

with ui.row().classes('items-center gap-2 w-full mb-1'):
    dir_input = ui.input(placeholder='Image directory, e.g. /Users/you/images').classes('flex-grow')
    ui.button('Load', icon='folder_open', on_click=load_directory).props('color=primary')

with ui.row().classes('items-center gap-2 w-full mb-2'):
    model_input = ui.input(placeholder='Path to .pt model file, e.g. /Users/you/best.pt').classes('flex-grow')
    ui.button('Load Model', icon='smart_toy', on_click=load_yolo_model).props('color=secondary')
    run_yolo_btn = ui.button('Run YOLO', icon='auto_fix_high', on_click=run_yolo).props('color=secondary outlined')
    run_yolo_btn.disable()

with ui.row().classes('items-center gap-2 w-full mb-2'):
    prev_btn = ui.button(icon='chevron_left', on_click=lambda: step_image(-1)).props('outline')
    file_select = ui.select(options=[], label='Select image', on_change=on_file_select).classes('flex-grow')
    next_btn = ui.button(icon='chevron_right', on_click=lambda: step_image(1)).props('outline')
prev_btn.disable()
next_btn.disable()

with ui.row().classes('items-center gap-4 mb-2') as annotation_row:
    ui.label('Active Class:').classes('font-semibold')
    class_buttons = {}
    for cls in CLASSES:
        btn = ui.button(cls, on_click=lambda c=cls: set_class(c))
        class_buttons[cls] = btn
    ui.separator().props('vertical')
    ui.button('Undo', on_click=undo_box).props('color=warning outlined')
    delete_btn = ui.button('Delete Selected', on_click=delete_selected).props('color=negative outlined')
    ui.button('Clear All', on_click=clear_boxes).props('color=negative outlined')
    ui.separator().props('vertical')
    ui.button('Export CSV', icon='download', on_click=export_csv).props('color=secondary')

annotation_row.set_visibility(False)

status_label = ui.label('').classes('text-sm text-gray-500 mb-1')
ui.label('Drag to draw · Click a box to select it · YOLO boxes shown in green') \
    .classes('text-xs text-gray-400 mb-1')

ii = ui.interactive_image(
    '', on_mouse=mouse_handler,
    events=['mousedown', 'mouseup'],
    cross='white', sanitize=False,
)

set_class(current_class)
update_delete_button()

ui.run()