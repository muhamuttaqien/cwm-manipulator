from pynput import keyboard
import ai2thor.controller

# Initialize the AI2-THOR environment
controller = ai2thor.controller.Controller(
    scene="FloorPlan10",
    width=640,
    height=480
)

# Function to handle key presses
def on_press(key):
    try:
        if key.char == 'w':
            controller.step(dict(action='MoveAhead'))
        elif key.char == 's':
            controller.step(dict(action='MoveBack'))
        elif key.char == 'a':
            controller.step(dict(action='MoveLeft'))
        elif key.char == 'd':
            controller.step(dict(action='MoveRight'))
        elif key == keyboard.Key.space:
            controller.step(dict(action='ToggleObject', objectId='Cup'))
    except AttributeError:
        # Handle special keys like shift or control
        pass

# Listen for key events
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
