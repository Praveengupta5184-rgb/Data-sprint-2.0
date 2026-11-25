"""Inspect the PPT template structure."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu

prs = Presentation(r'C:\Users\prave\.gemini\antigravity\brain\c40e39a6-3e4d-46da-bd41-d73e15fa842a\.user_uploaded\media_1795529018870_78b8d98d.pptx')

print('=== Slide dimensions ===')
print(f'Width: {prs.slide_width} EMU = {prs.slide_width/914400:.2f} in')
print(f'Height: {prs.slide_height} EMU = {prs.slide_height/914400:.2f} in')

print('\n=== Slide Layouts ===')
for i, layout in enumerate(prs.slide_layouts):
    print(f'\nLayout {i}: "{layout.name}"')
    for ph in layout.placeholders:
        print(f'  PH idx={ph.placeholder_format.idx}: "{ph.name}" type={ph.placeholder_format.type}')

print('\n=== Existing Slides ===')
for i, slide in enumerate(prs.slides):
    print(f'\nSlide {i+1}: layout="{slide.slide_layout.name}"')
    for shape in slide.shapes:
        t = shape.shape_type
        print(f'  Shape: type={t}, name="{shape.name}"')
        print(f'    pos=({shape.left},{shape.top}) size=({shape.width},{shape.height})')
        if hasattr(shape, 'text') and shape.text:
            txt = shape.text[:300].replace('\n', '|')
            print(f'    Text: {txt}')
        if shape.has_text_frame:
            for j, para in enumerate(shape.text_frame.paragraphs):
                for run in para.runs:
                    f = run.font
                    try:
                        c = str(f.color.rgb) if f.color and f.color.type else None
                    except:
                        c = 'theme'
                    print(f'    Para{j} Run: font={f.name}, size={f.size}, bold={f.bold}, italic={f.italic}, color={c}')
        # Check for images
        if shape.shape_type == 13:  # Picture
            print(f'    IMAGE: {shape.image.content_type}')
