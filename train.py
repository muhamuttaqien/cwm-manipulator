import os
import cv2
import random; random.seed(0)
import numpy as np; np.random.seed(0)

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from pynput import keyboard
import torch
import clip

is_cuda = torch.cuda.is_available()

if is_cuda: device = torch.device('cuda')
else: device = torch.device('cpu')


# Import Resnet-based Clip model
clip_model, preprocess = clip.load("RN50", device=device) 

def process_inputs(frame, instruction):
    
    image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
    
    text = clip.tokenize([instruction]).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return image_features, text_features

# Function for creating meta-information box
def put_text_with_background(
    frame, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0),
    thickness=2, padding=5
):
    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    x, y = org

    cv2.rectangle(
        frame,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        bg_color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

# Function for manual control using the keyboard
def manual_control_policy(controller, action_space, instruction):
    image_count = [1]  # Initialize image counter
    def on_press(key):
        # nonlocal image_count # Access the nonlocal variable
        try:
            if key.char == 'w':
                controller.step(action="MoveAgent", ahead=0.25, returnToStart=False)
            elif key.char == 's':
                controller.step(action="MoveAgent", ahead=-0.25, returnToStart=False)
            elif key.char == 'a':
                controller.step(action="MoveAgent", right=-0.25, returnToStart=False)
            elif key.char == 'd':
                controller.step(action="MoveAgent", right=0.25, returnToStart=False)
            elif key.char == 'r':
                controller.step(action="RotateAgent", degrees=30, returnToStart=False)
            elif key.char == 'f':
                controller.step(action="RotateAgent", degrees=-30, returnToStart=False)
            elif key.char == 'u':
                controller.step(action="LookUp")
            elif key.char == 'j':
                controller.step(action="LookDown")
            elif key.char == 'p':
                controller.step(action="MoveArm", position={"x": 0.0, "y": 1.0, "z": 0.0}, coordinateSpace="armBase", restrictMovement=False, returnToStart=False)
            
            # Process and calculate similarity
            current_frame = controller.last_event.frame
            image_features, text_features = process_inputs(current_frame, instruction)
            similarity = torch.cosine_similarity(image_features, text_features)
            print(f"Action: {key.char}, Similarity={similarity.item():.4f}")

            # Display the current frame
            plt.imshow(controller.last_event.frame)
            plt.title(f"Agent's View - Similarity: {similarity.item():.4f}")
            plt.axis("off")

            # Create the directory if it doesn't exist
            save_dir = "angga/img"
            os.makedirs(save_dir, exist_ok=True)

            # Save the plot with the desired filename
            filename = f"image_{image_count[0]}_{key.char}_{similarity.item():.4f}.png"
            plt.savefig(os.path.join(save_dir, filename))   # You can add a path here if needed 
            plt.show()
            import time
            time.sleep(0.5)
            image_count[0] += 1  # Increment image counter
        except AttributeError:
            # Handle special keys (e.g., shift, ctrl)
            pass

    # Start listening for keyboard input
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def manual_control_policy(controller, action_space, instruction, 
                          BASE_STEP=0.05, ARM_STEP=0.05, ROT_STEP=1.0, CAMERA_STEP=1.0, ARM_BASE = 0.5, ARM_BASE_STEP = 0.05, 
                          img_dir="angga/img", video_dir="angga/video"):

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    image_count = 1
    video_count = 1

    print("""
            W/S/A/D = move     Q/E = rotate
            R/F     = look     I/K/J/L/U/O = arm
            G/H     = pickup/release
            [/]     = Arm base up/down
            P       = print state
            0       = reset
            SPACE   = start/stop recording
            ESC     = quit
            """)

    last_similarity = None

    recording = False 
    video_writer = None
    
    while True:
        frame = controller.last_event.cv2img.copy()
        h, w = frame.shape[:2]

        # Similarity
        similarity_text = (
            f"CLIP's Score Similarity: {last_similarity:.4f}"
            if last_similarity is not None
            else "CLIP's Score Similarity:"
        )
        
        put_text_with_background(
            frame,
            similarity_text,
            (15, h - 140)
        )

        # EE Position
        arm = controller.last_event.metadata["arm"]
        ee_pos = arm["handSphereCenter"]
        
        ee_text = (
            f"EE Position: "
            f"x={ee_pos['x']:.2f}, "
            f"y={ee_pos['y']:.2f}, "
            f"z={ee_pos['z']:.2f}"
        )
        
        put_text_with_background(
            frame,
            ee_text,
            (15, h - 100)
        )
        
        # Visible objects        
        visible_objects = sorted(set(
            obj["objectType"]
            for obj in controller.last_event.metadata["objects"]
            if obj["visible"]
        ))
        
        visible_text = (
            "Visible Objects: " + ", ".join(visible_objects)
            if visible_objects
            else "Visible Objects:"
        )
        
        put_text_with_background(
            frame,
            visible_text,
            (15, h - 60)
        )

        # Pickup Ready
        pickupable_objects = arm["pickupableObjects"]

        pickupable_objects = [
            object_id.split("|")[0]
            for object_id in arm["pickupableObjects"]
        ]
        
        pickup_text = (
            "Pickup Ready: " + ", ".join(sorted(set(pickupable_objects)))
            if pickupable_objects
            else "Pickup Ready: None"
        )
        
        put_text_with_background(
            frame,
            pickup_text,
            (15, h - 20)
        )

        # Recording indicator 
        if recording: 
            cv2.putText(
                frame, "REC", 
                (w - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 0, 255), 2, cv2.LINE_AA 
            )

        # Record annotated frame 
        if recording and video_writer is not None: 
            video_writer.write(frame)
        
        cv2.imshow("ManipulaTHOR Robot", frame)
        key = cv2.waitKey(0) & 0xFF

        # No key pressed 
        if key == 255: 
            continue

        # Exit
        if key == 27:
            break

        # Video recording
        elif key == 32: # SPACE
            if not recording:
                h, w = frame.shape[:2]
        
                filename = f"angga/video/video_{video_count:03d}.mp4"
                os.makedirs("angga", exist_ok=True)
        
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                
                video_writer = cv2.VideoWriter(
                    filename, fourcc, 5.0, (w, h)
                )
        
                recording = True
                print("Recording started:", filename)
        
            else:
                recording = False
        
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
        
                print("Recording stopped.")
                video_count += 1

        action_name = None
        
        # Reset position
        if key == ord("0"):
            controller.step(
                action="Teleport",
                position={"x": 0.15, "y": 0.90, "z": -0.75},
                rotation={"x": 0, "y": 270, "z": 0},
                horizon=30
            )
            controller.step(action="MoveArmBase", y=0.5)
            action_name = "reset"
            
        elif key == ord("w"):
            controller.step(action="MoveAgent", ahead=BASE_STEP, right=0, returnToStart=True)
            action_name = "forward"

        elif key == ord("s"):
            controller.step(action="MoveAgent", ahead=-BASE_STEP, right=0, returnToStart=True)
            action_name = "backward"

        elif key == ord("a"):
            controller.step(action="MoveAgent", ahead=0, right=-BASE_STEP, returnToStart=True)
            action_name = "left"

        elif key == ord("d"):
            controller.step(action="MoveAgent", ahead=0, right=BASE_STEP, returnToStart=True)
            action_name = "right"

        elif key == ord("q"):
            controller.step(action="RotateAgent", degrees=-ROT_STEP, returnToStart=True)
            action_name = "rotate_left"

        elif key == ord("e"):
            controller.step(action="RotateAgent", degrees=ROT_STEP, returnToStart=True)
            action_name = "rotate_right"

        elif key == ord("r"):
            controller.step(action="LookUp", degrees=CAMERA_STEP)
            action_name = "look_up"

        elif key == ord("f"):
            controller.step(action="LookDown", degrees=CAMERA_STEP)
            action_name = "look_down"

        elif key == ord("["):
            ARM_BASE = min(1.0, ARM_BASE + ARM_BASE_STEP)
            controller.step(action="MoveArmBase", y=ARM_BASE, speed=1, returnToStart=True)
            action_name = "arm_base_up"
        
        elif key == ord("]"):
            ARM_BASE = max(0.0, ARM_BASE - ARM_BASE_STEP)
            controller.step(action="MoveArmBase", y=ARM_BASE, speed=1, returnToStart=True)
            action_name = "arm_base_down"

        elif key in map(ord, "ikjluo"):
            moves = {
                ord("i"): (0, ARM_STEP, 0, "arm_forward"),
                ord("k"): (0, -ARM_STEP, 0, "arm_backward"),
                ord("j"): (-ARM_STEP, 0, 0, "arm_left"),
                ord("l"): (ARM_STEP, 0, 0, "arm_right"),
                ord("u"): (0, 0, ARM_STEP, "arm_up"),
                ord("o"): (0, 0, -ARM_STEP, "arm_down"),
            }
            x, y, z, action_name = moves[key]
            controller.step(
                action="MoveArm",
                position={"x": x, "y": y, "z": z},
                coordinateSpace="wrist",
                restrictMovement=True,
                returnToStart=True
            )

        elif key == ord("g"):
            controller.step(action="PickupObject")
            action_name = "pickup"

        elif key == ord("h"):
            controller.step(action="ReleaseObject")
            action_name = "release"

        elif key == ord("p"):
            e = controller.last_event.metadata
            print("\nPosition:", e["agent"]["position"])
            print("Rotation:", e["agent"]["rotation"])
            print("Camera horizon:", e["agent"]["cameraHorizon"])
            print("End-effector:", e["arm"]["handSphereCenter"])
            continue

        if action_name is None:
            continue

        event = controller.last_event

        if not event.metadata["lastActionSuccess"]:
            print("Action failed:", event.metadata["errorMessage"])

        image_features, text_features = process_inputs(event.frame, instruction)
        similarity = torch.cosine_similarity(image_features, text_features).item()

        last_similarity = similarity

        filename = f"image_{image_count:04d}_{action_name}_{similarity:.4f}.png"
        # cv2.imwrite(os.path.join(save_dir, filename), frame)

        image_count += 1

    cv2.destroyAllWindows()

def random_policy(controller, action_space, instruction, num_steps):

    for step in range(num_steps):
    
        action = random.choice(action_space)
        
        if action == "MoveAgent":
            controller.step(
                action="MoveAgent",
                ahead=random.uniform(-0.25, 0.25), # Move forward/backward
                right=random.uniform(-0.25, 0.25), # Move left/right
                returnToStart=False
            )
      
        elif action == "RotateAgent":
            controller.step(
                action="RotateAgent",
                degrees=random.choice([30, -30]), # Rotate left/right
                returnToStart=False
            )
        elif action == "MoveArm":
           
            random_position = {
                "x": random.uniform(-0.5, 0.5),
                "y": random.uniform(0.0, 1.0),
                "z": random.uniform(-0.5, 0.5),
            }
            controller.step(
                action="MoveArm",
                position=random_position,
                coordinateSpace="armBase",
                restrictMovement=False,
                returnToStart=False
            )
            
        elif action == "MoveArmBase":
            controller.step(
                action="MoveArmBase",
                y=random.uniform(0.0, 1.0), # Move arm base up/down
                returnToStart=False
            )
            
        elif action == "LookUp":
            controller.step(action="LookUp")
            
        elif action == "LookDown":
            controller.step(action="LookDown")
        
        current_frame = controller.last_event.frame
        image_features, text_features = process_inputs(current_frame, instruction)
        
        similarity = torch.cosine_similarity(image_features, text_features)
        print(f"Step {step + 1}: Action={action}, Instruction: \"{instruction}\", Similarity={similarity.item():.4f}")
        
        plt.imshow(controller.last_event.frame)
        plt.title("Agent's View")
        plt.axis("off")
        plt.show()
        import time
        time.sleep(1)
        
def train_ppo(controller, ppo_agent, action_space, state_dim, num_episodes=1000, max_timesteps=200):
    
    """
    Train the PPO agent in the AI2-THOR environment.

    Parameters:
    - controller: AI2-THOR controller
    - ppo_agent: Instance of PPOAgent
    - action_space: List of possible actions
    - state_dim: Dimensionality of the state representation
    - num_episodes: Number of training episodes
    - max_timesteps: Maximum timesteps per episode
    """

    for episode in range(num_episodes):
    
        controller.reset("FloorPlan20")
        state = np.zeros(state_dim) # Replace with actual state representation logic
        total_reward = 0

        for timestep in range(max_timesteps):
            action_idx = ppo_agent.select_action(state)
            action = action_space[action_idx]

            # Perform the action in AI2-THOR
            event = controller.step(action=action)
            next_state = np.zeros(state_dim) # Replace with actual state representation logic
            reward = 0 # Replace with appropriate reward function
            done = False # Replace with terminal state logic

            # Store in memory
            ppo_agent.memory.rewards.append(reward)
            ppo_agent.memory.is_terminals.append(done)

            # Update state
            state = next_state
            total_reward += reward

            # Break if done
            if done:break

        # PPO Update
        if (episode + 1) % ppo_agent.update_timestep == 0:
            ppo_agent.update()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
