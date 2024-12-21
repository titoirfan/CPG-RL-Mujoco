import time
import numpy as np
import mujoco
import mujoco.viewer
import copy
import pytz
import datetime
import os
import imageio


from A1_description.model_xml import FlatXml

script_dir = os.path.dirname(__file__)


view = True
capture = False
episode_length = 5 # Capture 5 seconds
framerate = 20 # 20 fps

if view:
    view = True
    capture = False

def main():

    xml = FlatXml(additional_trunk_mass=0, limb_mass_scaling_factor=1,skeleton=False,friction=2.5)
    

    model = mujoco.MjModel.from_xml_string(xml)

    data = mujoco.MjData(model)

    joint_names = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
    
    target_q = [0,0.8,-1.6,0,0.8,-1.6,0,0.8,-1.6,0,0.8,-1.6]
    target_dq = [0,0,0,0,0,0,0,0,0,0,0,0]
    
    Kp = [100,100,100,100,100,100,100,100,100,100,100,100]
    Kd = [2,2,2,2,2,2,2,2,2,2,2,2]
    
    Kp = np.array(Kp) * 0.2
    Kd = np.array(Kd) * 0.2

    
    
    frames = []
    sim_time = 0
    
    if view:
        viewer = mujoco.viewer.launch_passive(model, data)
    if capture:
        renderer = mujoco.Renderer(model,1080,1920)
    
    data.joint("free").qvel = [0,0,0,0,0,0]
    data.joint("free").qpos = [0,0,0.45,1,0,0,0]

    done = False

    while not done:

        
        sim_time += 0.001
        
        joint_q = []
        joint_dq = []
        for joint_name in joint_names:
            joint_q.append(data.joint(joint_name).qpos[0])
            joint_dq.append(data.joint(joint_name).qvel[0])
            
        diff_q = np.array(target_q) - np.array(joint_q)
        diff_dq = np.array(target_dq) - np.array(joint_dq)
        data.ctrl[:] = np.array(Kp)*diff_q +np.array(Kd)*diff_dq
        
        mujoco.mj_step(model, data)

        if view:
            viewer.sync()
        
        if len(frames) < sim_time * framerate and capture:
            # renderer.update_scene(data, camera="fixed")
            renderer.update_scene(data, camera="track")
            pixels = renderer.render()
            frames.append(pixels)

        if capture and sim_time > episode_length:
                done = True
                
        
    if len(frames) > 0:
                
        timezone = pytz.timezone('Asia/Tokyo')
        current_datetime = datetime.datetime.now(timezone)
        formatted_datetime = current_datetime.strftime("%y%m%d_%H%M%S")
        
        video_folder = f"{script_dir}/A1_description/videos"
        
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        

        video_path = video_folder + f"/{formatted_datetime}.mp4"
        
        
        
        imageio.mimsave(video_path, frames, fps=framerate, macro_block_size=1)
        print(f"Saved video to {video_path}")
        


if __name__ == '__main__':
    main()




