import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import datetime

st.title('Interactive Object Manipulation')

global_render = st.empty() # test 'container' for outside-of-widget rendering

# helper function to initialize session state if not already present
def init_session_state():
    defaults = {
        'image': None,
        'object_points': set(),
        'contact_points': set(),
        'place_points': set(),
        'sweep_path': [],
        'holding_left_hand': False,
        'holding_right_hand': False,
        # 'action': None,
        'hand_choice': None,
        'original_image': None,
        'history': [],
        # 'done': True,
    }
    for key, value in defaults.items():
        st.session_state[key] = value
    print("init session state")

# resize image to fit to screen
def resize_img(img: Image.Image, max_size: int = 512) -> Image.Image:
    if max(img.size) > max_size:
        scale_factor = max_size / max(img.size)
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        img = img.resize(new_size)
    return img

# upload image & save to state
# TODO: fix intermittent 'MediaFileHandler: Missing file' error
def process_img_upload():
    print("changed file")
    file = st.session_state.get('img_file', None)
    if file is not None:
        raw = Image.open(file)
        img = resize_img(raw)
        print()
        if st.session_state.get('original_image', None) is None:
            print(f'{st.session_state.get('original_image', None)=}')
            st.session_state['base_dims'] = raw.size
            st.session_state['original_image'] = img
            st.session_state['image'] = img
            print(f'setting image at {datetime.datetime.now()}')
    else:
        st.session_state['image'] = st.session_state['original_image'] = None

# function to handle image upload
def upload_image():
    st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'], key='img_file', on_change=process_img_upload)

    if st.session_state.get('image', None) is not None:
        st.sidebar.image(st.session_state['image'], caption='Preview of image', use_column_width=True) # or st.session_state['image']

def set_action_active():
    st.session_state['done'] = False

# function to handle action selection
def select_action():
    actions = ['Pick object up', 'Sweep', 'Place object down']
    st.selectbox(
        'Select an action to perform',
        actions,
        index=None,
        on_change=set_action_active,
        key='action',
    )

def scale_canvas_coords(x: float, y: float) -> tuple:
    curr_dims = st.session_state['image'].size
    base_dims = st.session_state['base_dims']
    
    x_scaled = x * base_dims[0] / curr_dims[0]
    y_scaled = y * base_dims[1] / curr_dims[1]
    
    return x_scaled, y_scaled

# function to handle hand selection based on the current action
def handle_hand_selection():
    if st.session_state['action'] == 'Pick object up':
        if not (st.session_state['holding_left_hand'] and st.session_state['holding_right_hand']):
            hand_options = []
            if not st.session_state['holding_left_hand']:
                hand_options.append('Left Hand')
            if not st.session_state['holding_right_hand']:
                hand_options.append('Right Hand')
                
            st.session_state['hand_choice'] = st.radio('Choose hand to pick up object', hand_options)
        else: # tODO: don't render rest of the form if invalid state
            st.warning('Both hands are holding objects already.')
            # reset_form() # doesn't work since 'action' key cannot be manually set
            # st.session_state['done'] = True # will instantly remove all warnings/toasts/etc. too fast

    if st.session_state['action'] == 'Place object down':
        if st.session_state['holding_left_hand'] or st.session_state['holding_right_hand']:
            hand_options = []
            if st.session_state['holding_left_hand']:
                hand_options.append('Left Hand')
            if st.session_state['holding_right_hand']:
                hand_options.append('Right Hand')
                
            st.session_state['hand_choice'] = st.radio('Choose hand to place object', hand_options)
        else: # tODO: don't render rest of the form if invalid state
            st.warning('Both hands are empty. You need to pick up an object first.')

def handle_pick_action():
    if not st.session_state.get('mask_submitted', False):
        select_mask()
        # print('showing after mask annotate')
        # st.session_state['image'].show()
    # if st.session_state.get('mask_submitted', False):
    else:
        # print('showing before contact pt')
        # st.session_state['image'].show()
        select_contact_points()
    # st.session_state['action'] = None

# function to handle object annotation
def select_mask():
    st.write('Click on points to identify the object. Press submit to confirm.')
    img = st.session_state['image']

    canvas_result = st_canvas(
        fill_color='rgba(255, 165, 0, 0.3)',  # color for object mask
        stroke_width=3,
        background_image=img,
        update_streamlit=True,
        height=img.height,
        width=img.width,
        drawing_mode='point',
        key='object_canvas',
    )
    print(canvas_result.json_data)
    if canvas_result.json_data is not None:
        # st.session_state['object_points'] = set()
        for obj in canvas_result.json_data['objects']:
            if obj['type'] == 'Circle':
                cx = obj['left'] + obj['radius']
                cy = obj['top'] + obj['radius']
                
                cx, cy = scale_canvas_coords(cx, cy)
                
                if (cx, cy) not in st.session_state['object_points']:
                    st.session_state['object_points'].add((cx, cy))

        st.button('Submit Object Mask', on_click=handle_mask_submit, args=(img,), disabled=(len(st.session_state.get('object_points', [])) == 0))

    st.write('Object Points Selected: ', st.session_state['object_points'])
    
def handle_mask_submit(img: Image.Image):
    mask_coordinates = np.array(st.session_state['object_points'])
    print('Mask coordinates (for backend): ', mask_coordinates)
    print(f'{st.session_state['original_image']=}')
    
    st.session_state['history'].append('Object mask submitted.')

    mask = get_mask(img, st.session_state['object_points']) # tODO: send this to backend instead
    img_with_mask = mask_color_img(img, mask)
    st.session_state['image'] = img_with_mask
    # st.session_state['image'].show()
    # select_contact_points()
    st.session_state['mask_submitted'] = True
    # st.rerun()

# function to handle contact point annotation for object interaction
def select_contact_points():
    st.write('Now, select contact points for hand grasping.')
    img: Image.Image = st.session_state['image']
    print(f'{st.session_state['mask_submitted']=}')
    # img.show()

    contact_canvas = st_canvas(
        fill_color='rgba(0, 255, 0, 0.3)',  # color for contact points
        stroke_width=3,
        background_image=img,
        update_streamlit=True,
        height=img.height,
        width=img.width,
        drawing_mode='point',
        key='contact_canvas',
    )

    if contact_canvas.json_data is not None:
        # st.session_state['contact_points'] = set()
        for obj in contact_canvas.json_data['objects']:
            if obj['type'] == 'Circle':
                cx = obj['left'] + obj['radius']
                cy = obj['top'] + obj['radius']
                
                cx, cy = scale_canvas_coords(cx, cy)
                
                if (cx, cy) not in st.session_state['contact_points']:
                    st.session_state['contact_points'].add((cx, cy))

        st.button('Submit Contact Points', on_click=handle_contact_point_submit, disabled=(len(st.session_state.get('contact_points', [])) == 0))

    # if st.session_state['contact_points']:
    st.write('Contact Points Selected: ', st.session_state['contact_points'])
    
def handle_contact_point_submit():
    # st.write('Contact points submitted.')

    # st.session_state['image'] = 
    print('Contact points (for backend): ', st.session_state['contact_points']) # tODO: send this to backend

    st.session_state['history'].append('Contact points submitted.')

    update_hand_state()
    reset_form()
    # reset_all()
    st.session_state['done'] = True
    # st.session_state['action_set'] = None
    # st.rerun()

# function to update the hand state after object interactions
def update_hand_state():
    if st.session_state['hand_choice'] == 'Left Hand':
        st.session_state['holding_left_hand'] = not st.session_state['holding_left_hand']
    elif st.session_state['hand_choice'] == 'Right Hand':
        st.session_state['holding_right_hand'] = not st.session_state['holding_right_hand']
    
# function to reset object and contact points
def reset_form():
    st.session_state['object_points'] = set()
    st.session_state['contact_points'] = set()
    st.session_state['place_points'] = set()
    st.session_state['sweep_path'] = []
    st.session_state['mask_submitted'] = False
    # st.session_state['action'] = None
    st.session_state['action'] = None
    st.session_state['done'] = True
    # st.session_state['image'] = st.session_state['original_image'] # if we want to reset the canvas after every action
    # st.rerun()

# function to handle sweeping action
def handle_sweep_action():
    img = st.session_state['image']

    st.write('Draw the trajectory to sweep')
    sweep_canvas = st_canvas(
        fill_color='rgba(0, 0, 255, 0.3)',
        stroke_width=3,
        background_image=img,
        update_streamlit=True,
        height=img.height,
        width=img.width,
        drawing_mode='freedraw',
        key='sweep_canvas',
    )

    if sweep_canvas.json_data is not None:
        for obj in sweep_canvas.json_data['objects']:
            if 'path' in obj:
                for command in obj['path']:
                    # skip the command letter and take the coordinates as pairs
                    coords = command[1:]
                    print(coords)
                    if len(coords) % 2 == 0:
                        coord_pairs = [scale_canvas_coords(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                        st.session_state['sweep_path'].extend(coord_pairs)
    
        st.button('Submit Sweep Path', on_click=handle_sweep_submit, disabled=(len(st.session_state.get('sweep_path', [])) == 0))
            
def handle_sweep_submit():
    # st.write('Sweep path submitted.')
    print('Sweep Path: ', st.session_state['sweep_path']) # tODO: send this to backend
    
    st.session_state['history'].append('Sweep path submitted.')
    
    st.session_state['done'] = True
    reset_form()

# function to handle placing object down
def handle_place_action():
    img = st.session_state['image']
    st.write('Click on points where the object should be placed.')
    place_canvas = st_canvas(
        fill_color='rgba(255, 0, 0, 0.3)',  # red for placing the object
        stroke_width=3,
        background_image=img,
        update_streamlit=True,
        height=img.height,
        width=img.width,
        drawing_mode='point',
        key='place_canvas',
    )

    if place_canvas.json_data is not None:
        # st.session_state['place_points'] = set()
        for obj in place_canvas.json_data['objects']:
            if obj['type'] == 'Circle':
                cx = obj['left'] + obj['radius']
                cy = obj['top'] + obj['radius']
                
                cx, cy = scale_canvas_coords(cx, cy)
                
                if (cx, cy) not in st.session_state['place_points']:
                    st.session_state['place_points'].add((cx, cy))

        st.button('Submit Points', on_click=handle_place_submit, disabled=(len(st.session_state.get('place_points', [])) == 0))
            

def handle_place_submit():
    # st.write('Place location submitted.')

    # st.session_state['image'] = 
    print('Place points (for backend): ', st.session_state['place_points']) # tODO: send this to backend

    st.session_state['history'].append('Place location submitted.')

    update_hand_state()
    reset_form()
    st.session_state['done'] = True

def reset_all():
    init_session_state()
    # print(st.session_state)

# main application flow
def main():
    if len(st.session_state) == 0:
        init_session_state()
    
    # step 1: Upload Image
    upload_image()
    
    # reset button to clear session state
    st.button('Reset All', on_click=reset_all)
    
    if st.session_state['image']:
        # step 2: Select Action
        select_action()

        if st.session_state.get('action',None) and not st.session_state.get('done', True):
            print('Action selected: ', st.session_state['action'])
            # print(st.session_state)
            # step 3: Handle Hand Selection if necessary
            handle_hand_selection()

            # step 4: Perform actions based on the selected action
            if st.session_state['action'] == 'Pick object up':
                handle_pick_action()
            elif st.session_state['action'] == 'Sweep':
                handle_sweep_action()
                # st.session_state['action'] = None
            elif st.session_state['action'] == 'Place object down':
                handle_place_action()
                # st.session_state['action'] = None
        
        with st.sidebar:
            # tODO: persist these
            st.write('Left Hand Holding: ', st.session_state['holding_left_hand'])
            st.write('Right Hand Holding: ', st.session_state['holding_right_hand'])
            
            for i,entry in enumerate(st.session_state['history']):
                st.write(f'{i+1}. {entry}')

# masking function
def mask_color_img(img: Image.Image, mask, color=[0, 0, 0], alpha=0.3) -> Image.Image:
    print('masking func')
    opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    out = opencv_image.copy()
    img_layer = opencv_image.copy()
    img_layer[mask] = color
    cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

# temp function to generate random rectangular 2d mask coordinates
# tODO: fetch actual mask from backend
def get_mask(img: Image.Image, points: np.ndarray):
    # create a 2D mask for a grayscale image with False values initially
    mask = np.zeros((img.height, img.width), dtype=bool)

    region_width = int(img.width * 0.2)
    region_height = int(img.height * 0.2)

    center_x, center_y = img.width // 2, img.height // 2

    start_x_min = max(0, center_x - region_width // 2)
    start_x_max = min(img.width - region_width, center_x + region_width // 2)
    start_y_min = max(0, center_y - region_height // 2)
    start_y_max = min(img.height - region_height, center_y + region_height // 2)

    start_x = random.randint(start_x_min, start_x_max)
    start_y = random.randint(start_y_min, start_y_max)

    end_x = min(start_x + region_width, img.width)
    end_y = min(start_y + region_height, img.height)

    # set the mask region to True within the selected rectangle
    mask[start_y:end_y, start_x:end_x] = True

    # convert the 2D mask to match the 3-channel (RGB) image format
    # return np.dstack([mask] * 3)
    return mask

# run the app
if __name__ == '__main__':
    main()