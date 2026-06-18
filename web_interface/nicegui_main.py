from nicegui import events, ui
import os

ui.label('Image Testing')

x = ui.textarea(label='Image Directory', placeholder='type a directory',
            on_change=lambda e: result.set_text('you typed: ' + e.value),
            validation=lambda value: 'Directory Not Found' if check_directory(value) == False else None)
result = ui.label()

img_test = "/Users/jonathanzhu/nematostella_videos/imgs/rpa_bb_16C_9_dpf_celldish_04_02_2026/rpa_bb_16C_9_dpf_celldish_04_02_20260001.png"

def mouse_handler(e: events.MouseEventArguments):
    color = 'SkyBlue' if e.type == 'mousedown' else 'SteelBlue'
    ii.content += f'<circle cx="{e.image_x}" cy="{e.image_y}" r="15" fill="none" stroke="{color}" stroke-width="4" />'
    ui.notify(f'{e.type} at ({e.image_x:.1f}, {e.image_y:.1f})')

def check_directory(dir: str):
    try:
        files = os.listdir(dir)
    except:
        return False
    else:
        return None

ii = ui.interactive_image(img_test, on_mouse=mouse_handler, events=['mousedown', 'mouseup'], cross='white', sanitize=False)

ui.run()