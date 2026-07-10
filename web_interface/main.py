from nicegui import events, ui
import tkinter as tk
from tkinter import filedialog

# ── Notification wrapper ───────────────────────────────────────────────────────
notifications_enabled = False   # off by default

def notify(message, type='info'):
    """Show a notification only if enabled, or if it's a warning/error."""
    if notifications_enabled or type in ('warning', 'negative'):
        ui.notify(message, type=type)
import os
import pandas as pd

# ── Class definitions ──────────────────────────────────────────────────────────
CLASSES = {
    'Planula': 'SkyBlue',
    'Polyp': 'Tomato',
}
# Palette cycled through for YOLO-predicted classes (one color assigned per class name)
YOLO_PALETTE = [
    'MediumSeaGreen', 'Orchid', 'DodgerBlue', 'Orange',
    'Gold', 'Coral', 'MediumSlateBlue', 'Teal',
    'HotPink', 'SaddleBrown', 'LimeGreen', 'DarkOrange',
]
yolo_class_colors = {}   # populated at runtime: yolo_class_name -> color

# ── Path picker helpers ───────────────────────────────────────────────────────
def pick_directory():
    """Open a native OS folder picker and populate the directory input."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(title='Select image directory')
    root.destroy()
    if path:
        dir_input.value = path
        dir_input.update()
        load_directory()

def pick_model_file():
    """Open a native OS file picker for .pt files and populate the model input."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        title='Select YOLO model file',
        filetypes=[('PyTorch model', '*.pt'), ('All files', '*.*')],
    )
    root.destroy()
    if path:
        model_input.value = path
        model_input.update()
        load_yolo_model()

# ── State ──────────────────────────────────────────────────────────────────────
start_x        = None
start_y        = None
boxes          = []   # {'class', 'x1','y1','x2','y2'}  — all boxes treated equally
current_class  = list(CLASSES.keys())[0]
current_image  = None
selected_index = None
yolo_model     = None
conf_threshold = 0.5   # 0.0–1.0; boxes below this confidence are discarded
deleted_yolo_boxes = []  # YOLO boxes the user explicitly deleted; used to suppress similar future predictions
suppression_enabled = True   # toggle for the suppression filter
remembered_boxes = []  # manual boxes to auto-add on future images if no overlapping YOLO prediction exists
zoom_level     = 1.0   # display scale; stored box coords are always in natural pixels

all_annotations = pd.DataFrame(columns=['filename', 'class', 'yolo_class', 'x1', 'y1', 'x2', 'y2', 'from_yolo'])

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
                notify(f'Selected box {hit + 1} ({boxes[hit]["class"]})')
        else:
            selected_index = None
            boxes.append({'class': current_class, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
            notify(f'[{current_class}] ({x1:.0f},{y1:.0f}) → ({x2:.0f},{y2:.0f})')
        redraw()
        update_selection_controls()
        start_x = start_y = None

def yolo_color_for(raw_class_name):
    """Return (and lazily assign) a palette color for a YOLO class name."""
    if raw_class_name not in yolo_class_colors:
        yolo_class_colors[raw_class_name] = YOLO_PALETTE[len(yolo_class_colors) % len(YOLO_PALETTE)]
    return yolo_class_colors[raw_class_name]

def box_color(box):
    """Return the display color for a box dict."""
    if box['class'] in CLASSES:
        return CLASSES[box['class']]
    # YOLO boxes carry the raw class name separately for color lookup
    return yolo_color_for(box.get('yolo_class', box['class']))

def redraw():
    svg = ''
    for i, box in enumerate(boxes):
        color  = box_color(box)
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

def reassign_class(new_cls):
    """Change the class of the currently selected box."""
    if selected_index is None or not new_cls:
        return
    box = boxes[selected_index]
    old_cls = box['class']
    box['class'] = new_cls
    # If reassigning to a manual class, drop any YOLO-specific metadata
    if new_cls in CLASSES:
        box.pop('yolo_class', None)
    redraw()
    notify(f'Box {selected_index + 1}: "{old_cls}" → "{new_cls}"')

def remember_selected():
    """Store the selected manual box as a template to auto-add on future images."""
    if selected_index is None:
        return
    box = boxes[selected_index]
    if box.get('from_yolo'):
        notify('Only manual boxes can be remembered', type='warning')
        return
    entry = {k: box[k] for k in ('class', 'x1', 'y1', 'x2', 'y2')}
    # Avoid storing duplicates
    if not any(iou(entry, r) > 0.5 and entry['class'] == r['class'] for r in remembered_boxes):
        remembered_boxes.append(entry)
        notify(f'Box remembered ({entry["class"]}) — will auto-add on future images if missing')
    else:
        notify('A similar box is already remembered', type='warning')

def update_selection_controls():
    """Show/hide and populate the reassign select based on current selection."""
    if selected_index is not None:
        box = boxes[selected_index]
        is_manual = not box.get('from_yolo')
        all_classes = list(CLASSES.keys()) + [
            cls for cls in yolo_class_colors if cls not in CLASSES
        ]
        reassign_select.options = all_classes
        reassign_select.value = box['class']
        reassign_select.update()
        selection_row.set_visibility(True)
        delete_btn.props('color=negative')
        delete_btn.set_text(f'Delete Box {selected_index + 1}')
        delete_btn.enable()
        # Show remember button only for manual boxes
        remember_btn.set_visibility(is_manual)
    else:
        selection_row.set_visibility(False)
        delete_btn.props('color=negative outlined')
        delete_btn.set_text('Delete Selected')
        delete_btn.disable()
        remember_btn.set_visibility(False)

def delete_selected():
    global selected_index
    if selected_index is None:
        return
    removed = boxes.pop(selected_index)
    # Remember deleted YOLO boxes so future predictions in the same area are suppressed
    if removed.get('from_yolo'):
        deleted_yolo_boxes.append({
            'x1': removed['x1'], 'y1': removed['y1'],
            'x2': removed['x2'], 'y2': removed['y2'],
        })
    selected_index = None
    redraw()
    update_selection_controls()
    update_status()

def undo_box():
    global selected_index
    if boxes:
        boxes.pop()
        selected_index = None
        redraw()
        update_selection_controls()
        notify('Last box removed')

def clear_boxes():
    global selected_index
    boxes.clear()
    selected_index = None
    ii.content = ''
    update_selection_controls()
    notify('All boxes cleared')

# ── YOLO inference ─────────────────────────────────────────────────────────────
def load_yolo_model():
    global yolo_model
    pt_path = model_input.value.strip()
    if not pt_path:
        notify('Enter a path to a .pt file first', type='warning')
        return
    if not os.path.isfile(pt_path):
        notify('File not found', type='negative')
        return
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(pt_path)
        notify(f'Model loaded: {os.path.basename(pt_path)}', type='positive')
        run_yolo_btn.enable()
    except Exception as ex:
        notify(f'Failed to load model: {ex}', type='negative')

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
        notify('Load a model first', type='warning')
        return
    if current_image is None:
        notify('No image loaded', type='warning')
        return
    try:
        # Remove any boxes that were previously added by YOLO so re-running
        # at a different confidence threshold doesn't stack duplicates
        boxes[:] = [b for b in boxes if not b.get('from_yolo')]
        results = yolo_model(current_image, verbose=False)[0]
        # Collect raw predictions
        candidates = []
        for box in results.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            cls_name = results.names[int(box.cls)]
            conf     = float(box.conf)
            label    = f'{cls_name} {conf:.0%}'
            candidates.append({'class': label, 'yolo_class': cls_name,
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'conf': conf})
        # Filter by confidence threshold
        threshold = conf_threshold
        candidates = [c for c in candidates if c['conf'] >= threshold]
        # Suppress predictions that overlap with a previously deleted YOLO box
        before_suppress = len(candidates)
        if suppression_enabled and deleted_yolo_boxes:
            candidates = [
                c for c in candidates
                if all(iou(c, d) <= 0.5 for d in deleted_yolo_boxes)
            ]
        removed_suppressed = before_suppress - len(candidates)
        # Remove overlapping duplicates (IoU > 50%)
        total_before_dedup = len(candidates)
        kept = deduplicate(candidates, iou_threshold=0.5)
        removed_dedup = total_before_dedup - len(kept)
        for b in kept:
            boxes.append({k: v for k, v in b.items() if k != 'conf'} | {'from_yolo': True})
        # Auto-add remembered manual boxes that have no overlapping YOLO prediction
        auto_added = 0
        all_yolo_boxes = [b for b in boxes if b.get('from_yolo')]
        for template in remembered_boxes:
            has_overlap = any(iou(template, y) > 0.5 for y in all_yolo_boxes)
            already_present = any(iou(template, b) > 0.5 for b in boxes if not b.get('from_yolo'))
            if not has_overlap and not already_present:
                boxes.append({**template, 'from_yolo': False})
                auto_added += 1
        redraw()
        update_status()
        msg = f'YOLO added {len(kept)} box(es)'
        if removed_dedup:
            msg += f' — {removed_dedup} duplicate(s) removed'
        if removed_suppressed:
            msg += f' — {removed_suppressed} suppressed from deletion history'
        if auto_added:
            msg += f' — {auto_added} remembered box(es) auto-added'
        notify(msg, type='positive')
        update_legend()
    except Exception as ex:
        notify(f'Inference error: {ex}', type='negative')

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
             'yolo_class': b.get('yolo_class', ''),
             'x1': round(b['x1']), 'y1': round(b['y1']),
             'x2': round(b['x2']), 'y2': round(b['y2']),
             'from_yolo': b.get('from_yolo', False)}
            for b in boxes
        ])
        all_annotations = pd.concat([all_annotations, new_rows], ignore_index=True)
    autosave_csv()

def restore_boxes_for(filename):
    global selected_index
    boxes.clear()
    selected_index = None
    fname = os.path.basename(filename)
    rows = all_annotations[all_annotations['filename'] == fname]
    for _, row in rows.iterrows():
        box = {
            'class': row['class'],
            'x1': float(row['x1']), 'y1': float(row['y1']),
            'x2': float(row['x2']), 'y2': float(row['y2']),
            'from_yolo': bool(row.get('from_yolo', False)),
        }
        yc = row.get('yolo_class', '')
        if yc:
            box['yolo_class'] = yc
            yolo_color_for(yc)   # ensure color is registered in the palette map
        boxes.append(box)
    update_legend()

# ── CSV autosave & resume ─────────────────────────────────────────────────────
def csv_path():
    """Canonical autosave path inside the current image directory."""
    return os.path.join(dir_input.value.strip(), 'annotations.csv')

def autosave_csv():
    """Write all_annotations to disk silently (called on every image switch)."""
    if all_annotations.empty:
        return
    try:
        all_annotations.to_csv(csv_path(), index=False)
    except Exception:
        pass   # Never block navigation due to a save error

def try_resume_from_csv(directory):
    """If annotations.csv exists in directory, load it into all_annotations."""
    global all_annotations
    path = os.path.join(directory, 'annotations.csv')
    if not os.path.isfile(path):
        return False
    try:
        loaded = pd.read_csv(path)
        # Ensure yolo_class column exists even in CSVs saved before that column was added
        if 'yolo_class' not in loaded.columns:
            loaded['yolo_class'] = ''
        if 'from_yolo' not in loaded.columns:
            loaded['from_yolo'] = False
        all_annotations = loaded
        # Re-register any YOLO class colors so the legend rebuilds correctly
        for yc in all_annotations['yolo_class'].dropna().unique():
            if yc:
                yolo_color_for(yc)
        return True
    except Exception as ex:
        notify(f'Could not load existing annotations: {ex}', type='warning')
        return False

# ── File / directory ───────────────────────────────────────────────────────────
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif'}

def load_directory():
    global all_annotations
    path = dir_input.value.strip()
    if not os.path.isdir(path):
        notify('Directory not found', type='negative')
        return
    files = sorted(f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    if not files:
        notify('No image files found', type='warning')
        return
    # Reset annotations then try to resume from an existing CSV
    all_annotations = pd.DataFrame(columns=['filename', 'class', 'yolo_class', 'x1', 'y1', 'x2', 'y2', 'from_yolo'])
    resumed = try_resume_from_csv(path)
    file_select.options = files
    file_select.value   = files[0]
    file_select.update()
    switch_to_image(os.path.join(path, files[0]))
    if resumed:
        n = len(all_annotations)
        notify(f'{len(files)} image(s) found · Resumed {n} saved annotation(s)', type='positive')
    else:
        notify(f'{len(files)} image(s) found · No existing annotations')
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
    update_selection_controls()
    annotation_row.set_visibility(True)
    update_status()
    update_nav_buttons()
    # Auto-run YOLO if a model is loaded and this image has no existing annotations
    if yolo_model is not None and len(boxes) == 0:
        run_yolo()

def export_csv():
    save_current_boxes()
    if all_annotations.empty:
        notify('No annotations to export', type='warning')
        return
    out = os.path.join(dir_input.value.strip(), 'annotations.csv')
    all_annotations.to_csv(out, index=False)
    notify(f'Saved → {out}')

def update_legend():
    """Rebuild the YOLO class legend in the left panel."""
    legend_row.clear()
    if not yolo_class_colors:
        legend_row.set_visibility(False)
        return
    with legend_row:
        for cls_name, color in yolo_class_colors.items():
            with ui.row().classes('items-center gap-2'):
                ui.html(
                    f'<svg width="14" height="14">'
                    f'<circle cx="7" cy="7" r="6" fill="{color}"/>'
                    f'</svg>'
                )
                ui.label(cls_name).classes('text-sm')
    legend_row.set_visibility(True)

ZOOM_STEP = 0.25
ZOOM_MIN  = 0.25
ZOOM_MAX  = 4.0

def apply_zoom():
    """Resize the interactive image element and update the zoom label."""
    pct = int(zoom_level * 100)
    ii.style(f'width: {pct}%; max-width: none;')
    zoom_label.set_text(f'{pct}%')

def zoom_in():
    global zoom_level
    zoom_level = min(ZOOM_MAX, round(zoom_level + ZOOM_STEP, 2))
    apply_zoom()

def zoom_out():
    global zoom_level
    zoom_level = max(ZOOM_MIN, round(zoom_level - ZOOM_STEP, 2))
    apply_zoom()

def update_status():
    if current_image is None:
        return
    fname = os.path.basename(current_image)
    status_label.set_text(
        f'{fname}  ·  {len(boxes)} box(es) on this image  ·  {len(all_annotations)} total saved'
    )

# ── UI ─────────────────────────────────────────────────────────────────────────
ui.label('Nematostella Image Annotation Tool').classes('text-xl font-bold mb-2')

with ui.row().classes('w-full gap-0 items-start').style('height: calc(100vh - 80px);'):

    # ── Left panel (30%) ──────────────────────────────────────────────────────
    with ui.column().classes('gap-3 p-3 border-r').style('width: 30%; min-width: 220px; height: 100%; overflow-y: auto;'):

        ui.label('Image Directory').classes('text-sm font-semibold text-gray-500')
        with ui.row().classes('items-center gap-2 w-full'):
            dir_input = ui.input(placeholder='/Users/you/images').classes('flex-grow')
            ui.button(icon='folder_open', on_click=load_directory).props('color=primary dense').tooltip('Load typed path')
        ui.button('Browse…', icon='drive_folder_upload', on_click=pick_directory)             .props('color=primary outline w-full').tooltip('Open folder picker')

        ui.separator()

        ui.label('YOLO Model').classes('text-sm font-semibold text-gray-500')
        with ui.row().classes('items-center gap-2 w-full'):
            model_input = ui.input(placeholder='/Users/you/best.pt').classes('flex-grow')
            ui.button(icon='smart_toy', on_click=load_yolo_model).props('color=secondary dense').tooltip('Load typed path')
        ui.button('Browse…', icon='file_open', on_click=pick_model_file)             .props('color=secondary outline w-full').tooltip('Open file picker')
        run_yolo_btn = ui.button('Run YOLO', icon='auto_fix_high', on_click=run_yolo).props('color=secondary outlined w-full')
        run_yolo_btn.disable()

        with ui.row().classes('items-center justify-between w-full'):
            ui.label('Min. confidence').classes('text-sm text-gray-500')
            conf_label = ui.label('50%').classes('text-sm font-semibold')

        def on_conf_change(e):
            global conf_threshold
            conf_threshold = e.value / 100
            conf_label.set_text(f'{int(e.value)}%')

        ui.slider(min=0, max=100, step=1, value=50, on_change=on_conf_change)             .props('label color=secondary').classes('w-full')

        ui.separator()

        ui.label('Class Legend').classes('text-sm font-semibold text-gray-500')
        legend_row_main = ui.column().classes('gap-1 w-full')
        with legend_row_main:
            for cls_name, color in CLASSES.items():
                with ui.row().classes('items-center gap-2'):
                    ui.html(
                        f'<svg width="14" height="14">'
                        f'<circle cx="7" cy="7" r="6" fill="{color}"/>'
                        f'</svg>'
                    )
                    ui.label(cls_name).classes('text-sm')

        ui.label('YOLO Class Legend').classes('text-sm font-semibold text-gray-500')
        legend_row = ui.column().classes('gap-1 w-full')
        legend_row.set_visibility(False)

        ui.separator()

        def toggle_notifications():
            global notifications_enabled
            notifications_enabled = not notifications_enabled
            notif_btn.props(
                'color=positive' if notifications_enabled else 'color=grey outlined'
            )
            notif_btn.set_text(
                'Notifications: On' if notifications_enabled else 'Notifications: Off'
            )

        notif_btn = ui.button('Notifications: Off', icon='notifications_off',
                              on_click=toggle_notifications)             .props('color=grey outlined w-full')

    # ── Right panel (70%) ─────────────────────────────────────────────────────
    with ui.column().classes('gap-2 p-3 flex-grow').style('width: 70%; height: 100%; overflow-y: auto;'):

        # File navigation
        with ui.row().classes('items-center gap-2 w-full'):
            prev_btn = ui.button(icon='chevron_left', on_click=lambda: step_image(-1)).props('outline')
            file_select = ui.select(options=[], label='Select image', on_change=on_file_select).classes('flex-grow')
            next_btn = ui.button(icon='chevron_right', on_click=lambda: step_image(1)).props('outline')
        prev_btn.disable()
        next_btn.disable()

        # Annotation toolbar
        with ui.row().classes('items-center gap-3 flex-wrap') as annotation_row:
            ui.label('Active Class:').classes('font-semibold')
            class_buttons = {}
            for cls in CLASSES:
                btn = ui.button(cls, on_click=lambda c=cls: set_class(c))
                class_buttons[cls] = btn
            ui.separator().props('vertical')
            ui.button('Undo', on_click=undo_box).props('color=warning outlined')
            ui.button('Clear All', on_click=clear_boxes).props('color=negative outlined')
            ui.separator().props('vertical')
            ui.button('Export CSV', icon='download', on_click=export_csv).props('color=secondary')
        annotation_row.set_visibility(False)

        # Reassign class row — visible only when a box is selected
        with ui.row().classes('items-center gap-3 flex-wrap') as selection_row:
            ui.label('Selected box class:').classes('text-sm font-semibold')
            reassign_select = ui.select(
                options=list(CLASSES.keys()),
                label='Reassign to…',
                on_change=lambda e: reassign_class(e.value),
            ).classes('w-48')
            remember_btn = ui.button('Remember Box', icon='bookmark',
                                     on_click=remember_selected).props('color=teal outlined')
            delete_btn = ui.button('Delete Selected', on_click=delete_selected).props('color=negative outlined')
        selection_row.set_visibility(False)
        remember_btn.set_visibility(False)

        status_label = ui.label('').classes('text-sm text-gray-500')
        ui.label('Drag to draw · Click a box to select it · YOLO box colors shown in legend')             .classes('text-xs text-gray-400')

        # Zoom controls + image
        with ui.row().classes('items-center gap-2'):
            ui.button(icon='zoom_out', on_click=zoom_out).props('outline').tooltip('Zoom out')
            zoom_label = ui.label('100%').classes('text-sm w-12 text-center')
            ui.button(icon='zoom_in', on_click=zoom_in).props('outline').tooltip('Zoom in')

        with ui.scroll_area().classes('w-full border rounded').style('height: 600px;'):
            ii = ui.interactive_image(
                '', on_mouse=mouse_handler,
                events=['mousedown', 'mouseup'],
                cross='white', sanitize=False,
            ).style('width: 100%; max-width: none;')

ui.label("Keyboard shortcuts").classes('text font-bold mb-2')
ui.label("(Delete/Backspace) to delete a selected annotation box")
ui.label("(Left/Right) to move to previous/next image in the directory")

def clear_suppression():
    deleted_yolo_boxes.clear()
    notify('Deletion history cleared', type='positive')

with ui.row().classes('items-center justify-left w-full'):
    ui.label('Suppression filter').classes('text-sm text-gray-500')
    def toggle_suppression(e):
        global suppression_enabled
        suppression_enabled = e.value

    ui.switch(value=True, on_change=toggle_suppression)                 .tooltip('When on, YOLO predictions overlapping deleted boxes are suppressed')

    ui.button('Clear Suppression History', icon='history', on_click=clear_suppression)             .props('color=grey outlined w-full')             .tooltip('Re-allow YOLO predictions that were previously deleted')

    ui.separator()

    ui.label('Remembered Boxes').classes('text-sm font-semibold text-gray-500')
    ui.label('Click a manual box then "Remember Box" to auto-add it on future images when YOLO misses it.')             .classes('text-xs text-gray-400')

    def clear_remembered():
        remembered_boxes.clear()
        notify('Remembered boxes cleared', type='positive')

    ui.button('Clear Remembered Boxes', icon='bookmark_remove', on_click=clear_remembered)             .props('color=teal outlined w-full')             .tooltip('Stop auto-adding remembered boxes on future images')

ui.separator()

ui.label("To report technical issues or bugs please contact Jonathan Zhu (jzhu@uark.edu)")

def handle_key(e):
    if e.action.repeat:
        return
    if e.action.keydown:
        if e.key in ('Delete', 'Backspace') and selected_index is not None:
            delete_selected()
        elif e.key.arrow_left:
            step_image(-1)
        elif e.key.arrow_right:
            step_image(1)

ui.keyboard(on_key=handle_key, ignore=[])

set_class(current_class)
update_selection_controls()

def build_favicon(png_path='icon.png', ico_path='icon.ico'):
    """Convert icon.png to a properly formatted .ico file for use as a favicon."""
    try:
        from PIL import Image
        img = Image.open(png_path).convert('RGBA')
        # .ico supports multiple sizes; 32x32 and 16x16 are the standard favicon sizes
        img.save(ico_path, format='ICO', sizes=[(32, 32), (16, 16)])
        return ico_path
    except FileNotFoundError:
        return None   # icon.png not found — fall back to NiceGUI default
    except Exception:
        return None   # Pillow not installed or conversion failed — fall back silently

ui.run(title='Nematostella Annotation', favicon=build_favicon())