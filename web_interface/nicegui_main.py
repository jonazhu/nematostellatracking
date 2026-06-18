from nicegui import events, ui
import os
import pandas as pd

# ── Class definitions ──────────────────────────────────────────────────────────
CLASSES = {
    'Class A': 'SkyBlue',
    'Class B': 'Tomato',
}
YOLO_COLOR   = 'MediumSeaGreen'   # SVG color for YOLO prediction boxes
YOLO_CLASS   = '__yolo__'          # Internal sentinel stored in the boxes list

# ── State ──────────────────────────────────────────────────────────────────────
start_x        = None
start_y        = None
boxes          = []                # {'class', 'x1','y1','x2','y2', 'conf'(opt)}
current_class  = list(CLASSES.keys())[0]
current_image  = None
selected_index = None

yolo_model     = None              # Loaded ultralytics YOLO instance (or None)

all_annotations = pd.DataFrame(columns=['filename','class','x1','y1','x2','y2'])

# ── Hit-test & drawing ─────────────────────────────────────────────────────────
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
                cls = boxes[hit]['class']
                label = 'YOLO prediction' if cls == YOLO_CLASS else cls
                ui.notify(f'Selected box {hit + 1} ({label})')
        else:
            selected_index = None
            boxes.append({'class': current_class, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
            ui.notify(f'[{current_class}] ({x1:.0f},{y1:.0f}) → ({x2:.0f},{y2:.0f})')
        redraw()
        update_delete_button()
        start_x = start_y = None

def redraw():
    svg = ''
    for i, box in enumerate(boxes):
        is_yolo    = box['class'] == YOLO_CLASS
        color      = YOLO_COLOR if is_yolo else CLASSES.get(box['class'], 'White')
        is_sel     = (i == selected_index)
        x, y       = box['x1'], box['y1']
        w, h       = box['x2'] - box['x1'], box['y2'] - box['y1']

        if is_yolo:
            conf  = box.get('conf', 0)
            label = f"{i+1}: {box.get('yolo_class','?')} {conf:.0%}"
            dash  = 'stroke-dasharray="6 3"'
        else:
            label = f"{i+1}: {box['class']}"
            dash  = ''

        stroke_w = '5' if is_sel else '3'
        fill     = 'rgba(255,255,255,0.15)' if is_sel else 'none'
        svg += (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="{fill}" stroke="{color}" stroke-width="{stroke_w}" {dash}/>'
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

def run_yolo():
    """Run inference on the current image and add predictions as YOLO boxes."""
    if yolo_model is None:
        ui.notify('Load a model first', type='warning')
        return
    if current_image is None:
        ui.notify('No image loaded', type='warning')
        return

    # Remove any existing YOLO boxes so re-running replaces rather than appends
    boxes[:] = [b for b in boxes if b['class'] != YOLO_CLASS]

    try:
        results = yolo_model(current_image, verbose=False)[0]
        added = 0
        for box in results.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            cls_idx  = int(box.cls)
            cls_name = results.names[cls_idx]
            conf     = float(box.conf)
            boxes.append({
                'class':      YOLO_CLASS,
                'yolo_class': cls_name,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'conf': conf,
            })
            added += 1
        redraw()
        update_status()
        ui.notify(f'YOLO found {added} object(s)', type='positive')
    except Exception as ex:
        ui.notify(f'Inference error: {ex}', type='negative')

# ── Persistence ────────────────────────────────────────────────────────────────
def save_current_boxes():
    global all_annotations
    if current_image is None:
        return
    fname = os.path.basename(current_image)
    all_annotations = all_annotations[all_annotations['filename'] != fname]
    # Only persist manually drawn boxes (skip raw YOLO predictions)
    manual = [b for b in boxes if b['class'] != YOLO_CLASS]
    if manual:
        new_rows = pd.DataFrame([
            {'filename': fname, 'class': b['class'],
             'x1': round(b['x1']), 'y1': round(b['y1']),
             'x2': round(b['x2']), 'y2': round(b['y2'])}
            for b in manual
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

def on_file_select(e):
    switch_to_image(os.path.join(dir_input.value.strip(), e.value))

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
    fname   = os.path.basename(current_image)
    manual  = sum(1 for b in boxes if b['class'] != YOLO_CLASS)
    pred    = sum(1 for b in boxes if b['class'] == YOLO_CLASS)
    total   = len(all_annotations)
    parts   = [f'{fname}', f'{manual} manual box(es)']
    if pred:
        parts.append(f'{pred} YOLO prediction(s)')
    parts.append(f'{total} total saved')
    status_label.set_text('  ·  '.join(parts))

# ── UI ─────────────────────────────────────────────────────────────────────────
ui.label('Image Annotation Tool').classes('text-xl font-bold mb-2')

# Image directory row
with ui.row().classes('items-center gap-2 w-full mb-1'):
    dir_input = ui.input(placeholder='Image directory, e.g. /Users/you/images').classes('flex-grow')
    ui.button('Load', icon='folder_open', on_click=load_directory).props('color=primary')

# YOLO model row
with ui.row().classes('items-center gap-2 w-full mb-2'):
    model_input = ui.input(placeholder='Path to .pt model file, e.g. /Users/you/best.pt').classes('flex-grow')
    ui.button('Load Model', icon='smart_toy', on_click=load_yolo_model).props('color=secondary')
    run_yolo_btn = ui.button('Run YOLO', icon='auto_fix_high', on_click=run_yolo).props('color=secondary outlined')
    run_yolo_btn.disable()

# File selector
file_select = ui.select(options=[], label='Select image', on_change=on_file_select).classes('w-full mb-2')

# Annotation toolbar
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
ui.label('Drag to draw a box · Click a box to select it · YOLO boxes shown with dashed outlines') \
    .classes('text-xs text-gray-400 mb-1')

ii = ui.interactive_image(
    '', on_mouse=mouse_handler,
    events=['mousedown', 'mouseup'],
    cross='white', sanitize=False,
)

set_class(current_class)
update_delete_button()

ui.run()