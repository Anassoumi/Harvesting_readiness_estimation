import os
import os
import time

import numpy as np
from kivy import Config
from kivy.animation import Animation
from kivy.app import App
from kivy.core.image import Texture
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.screenmanager import Screen
from kivy.uix.screenmanager import ScreenManager
from kivy.utils import get_color_from_hex
from ultralytics import YOLO

# Set the window size
Config.set('graphics', 'width', '800')
Config.set('graphics', 'height', '500')


def detect_fruit(model_path, image_path):
    import cv2
    from ultralytics import YOLO

    # Load the YOLOv8 model
    model = YOLO('Models/yolov8n.pt')

    # Read the image
    image = cv2.imread(image_path)

    # Run YOLOv8 inference on the image
    results = model(image)

    # Get the class probabilities from the results
    probs = results[0].probs

    # Visualize the results on the image
    annotated_image = results[0].plot(img=image, boxes=True, masks=True, probs=True)

    # Save the annotated image
    cv2.imwrite("annotated_image.jpg", annotated_image)

    # Iterate over the boxes and print the type of object detected
    for i, box in enumerate(results[0].boxes.data):
        class_index = int(box[5])  # The class index is at index 5 in the box tensor
        print(class_index)
        # Determine the fruit type based on confidence rates
        if class_index == 46:
            return "Banana"
        elif class_index == 49:
            return "Orange"

    return "No fruit detected"


class MyButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(mouse_pos=self.on_mouseover)

    def on_mouseover(self, window, pos):
        if self.collide_point(*pos):
            self.font_size = '18sp'
            self.background_color = (1, 0.6, 0, 1)
            self.background_normal = 'Buttons/START.png'
            self.color = (0, 0, 0, 1)
        else:
            self.font_size = '14sp'
            self.background_color = (1, 1, 1, 1)
            self.background_normal = 'Buttons/START.png'
            self.color = (1,1, 1, 1)


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        layout = FloatLayout()
        self.background_image = Image(
            source='Backgrounds/2207.q803.014.S.m012.c10.banana ripe stages realictic.jpg',
            allow_stretch=True, keep_ratio=False)
        self.capture_button = MyButton(size_hint=(0.1, 0.1), text='START', background_normal='Buttons/START.png',
                                       background_down='Buttons/START.png')
        self.capture_button.bind(on_release=self.switch_to_camera)
        welcome_label = Label(text='Welcome to the Fruits Maturity Estimator', font_size='24sp', bold=True,
                              color=get_color_from_hex('#000000'), pos_hint={'center_x': 0.5, 'center_y': 0.93})
        layout.add_widget(self.background_image)
        layout.add_widget(self.capture_button)
        layout.add_widget(welcome_label)
        self.add_widget(layout)


    def on_button_enter(self):
        self.capture_button.size_hint = (0.15, 0.15)

    def on_button_leave(self):
        self.capture_button.size_hint = (0.1, 0.1)

    def on_button_release(self):
        self.capture_button.color = (1, 0.5, 0, 1)
        self.switch_to_camera(self.capture_button)

    def switch_to_camera(self, *args):
        app.screen_manager.transition.direction = 'left'
        app.screen_manager.current = 'camera'
        app.camera_screen.activate_camera()

    def on_enter(self):
        self.capture_button.pos_hint = {'x': 0.05, 'center_y': 0.07}
        animation_background = Animation(x=50, duration=2) + Animation(x=0, duration=2)
        animation_background.repeat = True
        animation_background.start(self.background_image)

        animation_button = Animation(x=50, duration=2) + Animation(x=0, duration=2)
        animation_button.repeat = True
        animation_button.start(self.capture_button)





from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import RoundedRectangle
from kivy.clock import Clock
import cv2
import datetime

class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        self.layout = FloatLayout()
        self.add_widget(self.layout)

        # Add background image
        background_image = Image(source='Backgrounds/background black .jpeg', allow_stretch=True, keep_ratio=False)
        self.layout.add_widget(background_image)

        self.camera = None
        self.capture_button = Button(text='', size_hint=(0.16, 0.16), pos_hint={'x': 0.4, 'y': 0.03},
                                     background_normal='Buttons/Screenshot 2023-06-07 at 12.32.47-cutout.png')
        self.capture_button.bind(on_release=self.capture_image)
        self.layout.add_widget(self.capture_button)

    def activate_camera(self):
        self.camera = Camera(resolution=(1920, 1080), play=True)  # Adjust the resolution here

        # Calculate the size and position of the camera widget
        window_width = self.width
        window_height = self.height - dp(50)  # Adjust for the capture button's height
        camera_ratio = self.camera.image_ratio

        camera_width = window_width
        camera_height = window_width / camera_ratio

        camera_x = 20
        camera_y = 90

        self.camera.size = (camera_width, camera_height)
        self.camera.pos = (camera_x, camera_y)

        self.layout.add_widget(self.camera)

    def capture_image(self, *args):
        if self.camera:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_path = os.path.abspath(f"captured_image_{timestamp}.png")

            # Retrieve the image texture from the camera
            texture = self.camera.texture

            # Get the actual texture size
            texture_size = self.camera.texture_size
            texture_width, texture_height = texture_size

            # Get the camera widget size
            camera_width, camera_height = self.camera.size

            # Calculate the crop region
            crop_width = int(camera_width * texture_width / camera_width)
            crop_height = int(camera_height * texture_height / camera_height)
            crop_x = int((texture_width - crop_width) / 2)
            crop_y = int((texture_height - crop_height) / 2)

            # Crop the texture
            cropped_texture = texture.get_region(crop_x, crop_y, crop_width, crop_height)

            # Convert the cropped texture to a numpy array
            np_image = np.frombuffer(cropped_texture.pixels, np.uint8).reshape((crop_height, crop_width, 4))

            # Rotate the numpy image upside down
            rotated_image = np.flipud(np_image)

            # Convert the rotated image back to a texture
            rotated_texture = Texture.create(size=(crop_width, crop_height))
            rotated_texture.blit_buffer(rotated_image.tobytes(), colorfmt='rgba', bufferfmt='ubyte')

            # Save the rotated texture image without modifications
            rotated_texture.save(image_path)

            # Load the YOLOv8 model
            model = YOLO('Models/yolov8n.pt')

            # Read the captured image
            captured_image = cv2.imread(image_path)

            # Run YOLOv8 inference on the captured image
            results = model(captured_image)

            # Get the boxes from the results
            boxes = results[0].boxes

            # Check if bananas or oranges are detected
            is_banana_detected = any(box[5] == 46 for box in boxes.data)
            is_orange_detected = any(box[5] == 49 for box in boxes.data)

            if is_banana_detected:
                popup_label = Label(text='Initializing Model...', bold=True)

                # Create a custom BoxLayout for popup content
                popup_content = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
                popup_content.add_widget(popup_label)

                # Create a rounded rectangle background for the popup
                popup_background = RoundedRectangle(radius=[dp(10)])

                popup = Popup(title='Banana detected',
                              content=popup_content,
                              size_hint=(0.6, None),
                              size=(dp(200), dp(200)),
                              auto_dismiss=False,
                              background='Backgrounds/verified.png')

                # Add the background to the popup canvas
                popup_content.canvas.before.add(popup_background)

                def on_popup_dismiss():
                    app.display_screen.show_image(image_path)

                # Bind the on_popup_dismiss function to the dismiss event of the popup
                popup.bind(on_dismiss=lambda instance: on_popup_dismiss())

                popup.open()

                def update_popup_text(dt):
                    popup_label.text = 'Processing image...'

                # Schedule the popup text update after 2 seconds
                Clock.schedule_once(update_popup_text, 2)

                def process_image(dt):
                    # Run YOLOv8 inference on the captured image
                    image = cv2.imread(image_path)

                    # Load the YOLOv8 model
                    model = YOLO(r"Models/last_best.pt")

                    # Run YOLOv8 inference on the image
                    results = model(image)

                    # Print the result parameters
                    print("Boxes:")

                    # Visualize the results on the image
                    annotated_image = results[0].plot(img=image, boxes=True, masks=True, probs=True)

                    # Save the annotated image with a unique filename
                    current_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    image_filename = f"Maturity_{current_date}.jpg"
                    cv2.imwrite(image_filename, annotated_image)

                    app.display_screen.display_image(image_filename)
                    app.screen_manager.transition.direction = 'right'
                    app.screen_manager.current = 'display'

                # Schedule the image processing after 4 seconds
                Clock.schedule_once(process_image, 4)

                # Schedule the popup to be dismissed after 3 seconds
                Clock.schedule_once(lambda dt: popup.dismiss(), 3)

            elif is_orange_detected:
                # Handle orange detection
                pass

            else:
                popup_label = Label(text='Please try again !', bold=True)

                # Create a custom BoxLayout for popup content
                popup_content = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
                popup_content.add_widget(popup_label)

                # Create a rounded rectangle background for the popup
                popup_background = RoundedRectangle(radius=[dp(10)])

                popup = Popup(title='No Banana Detected ',
                              content=popup_content,
                              size_hint=(0.6, None),
                              size=(dp(150), dp(150)),
                              auto_dismiss=True,
                              background='Backgrounds/verified2.png')

                # Add the background to the popup canvas
                popup_content.canvas.before.add(popup_background)

                popup.open()

                # Schedule the popup to be dismissed after 3 seconds
                Clock.schedule_once(lambda dt: popup.dismiss(), 5)

                # If no bananas or oranges are detected, use the original captured image


            self.deactivate_camera(image_path)

    def deactivate_camera(self,image_path):
        if self.camera:
            self.camera.play = False  # Turn off the camera
            self.layout.remove_widget(self.camera)
            self.camera = None
            app.display_screen.display_image(image_path)
            app.screen_manager.transition.direction = 'right'
            app.screen_manager.current = 'display'


class DisplayScreen(Screen):
    def __init__(self, **kwargs):
        super(DisplayScreen, self).__init__(**kwargs)

        layout = BoxLayout(orientation='vertical')
        self.image = Image(size_hint=(1, 1))
        layout.add_widget(self.image)


        self.info_layout = BoxLayout(orientation='horizontal', padding=(30, 0), spacing=10, size_hint=(1, 0.1))
        layout.add_widget(self.info_layout)

        self.back_button = Button(text='New Experiment', size_hint=(.8, 1),background_normal='Buttons/EXPERIMENT.png')
        self.back_button.bind(on_release=self.switch_to_main)
        self.info_layout.add_widget(self.back_button)

        self.exit_button = Button(text='EXIT', color='white',size_hint=(.5, 1),
                                  background_normal='Buttons/EXIT.png')
        self.exit_button.bind(on_release=self.exit_app)
        self.info_layout.add_widget(self.exit_button)

        self.add_widget(layout)

    def exit_app(self, *args):
        # Create the popup window
        popup_label = Label(text='Exiting app...')
        popup_content = BoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
        popup_content.add_widget(popup_label)
        popup = Popup(title='Exit App', content=popup_content, size_hint=(0.6, None), size=(dp(200), dp(200)))

        # Dismiss the popup and stop the app after a delay
        def dismiss_and_exit(dt):
            popup.dismiss()
            App.get_running_app().stop()
            Window.close()

        # Schedule the dismiss_and_exit function to be called after 2 seconds
        Clock.schedule_once(dismiss_and_exit, 2)

        popup.open()

    def display_image(self, image_path):
        self.image.source = image_path

    def show_image(self, image_path):
        self.image.source = image_path

    def switch_to_previous_screen(self, *args):
        self.manager.current = "main"

    def show_no_fruit_popup(self):
        content = BoxLayout(orientation='vertical', spacing=30, padding=(30, 30))
        popup_label = Label(text='No fruit detected', font_size=20)
        back_button = Button(text='Back', size_hint=(1, None), height=40)
        content.add_widget(popup_label)
        content.add_widget(back_button)

        popup = Popup(title='No Fruit Detected', content=content, size_hint=(None, None), size=(600, 400))
        back_button.bind(on_release=popup.dismiss)  # Dismiss the popup when the back button is clicked
        popup.open()

    def switch_to_main(self, *args):
        app.screen_manager.transition.direction = 'right'
        app.screen_manager.current = 'camera'
        app.camera_screen.activate_camera()



class MyApp(App):
    def build(self):
        self.screen_manager = ScreenManager()
        self.main_screen = MainScreen(name='main')
        self.camera_screen = CameraScreen(name='camera')
        self.display_screen = DisplayScreen(name='display')
        self.screen_manager.add_widget(self.main_screen)
        self.screen_manager.add_widget(self.camera_screen)
        self.screen_manager.add_widget(self.display_screen)
        return self.screen_manager


if __name__ == '__main__':
    app = MyApp()
    app.run()
