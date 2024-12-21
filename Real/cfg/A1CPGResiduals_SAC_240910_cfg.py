from dataclasses import dataclass
from typing import List, Union
import json

from dataclasses import dataclass, field
from typing import List

# dataclassの定義
@dataclass
class Config:
    gpu: int = 2
    seed: int = 123456
    
    # SAC
    policy: str = "Gaussian"
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 0.0001
    alpha: float = 0.2
    automatic_entropy_tuning: bool = True
    batch_size: int = 256
    num_steps: int = 10000000
    hidden_size: List[int] = field(default_factory=lambda: [512, 256, 128]) # 512, 256, 128
    updates_interval: int = 1
    updates_per_step: int = 1
    log_interval: int = 1000
    start_steps: int = 100000
    target_update_interval: int = 1
    replay_size: int = 5000000
    capture_interval: int = 500

    # Environment
    dt: float = 0.001
    delay: List[int] = field(default_factory=lambda: [30,15,35]) # defalut min max
    terrain: str = "flat"
    dekoboko: List[float] = field(default_factory=lambda: [0.10,0.00001,0.12]) # default min max
    
    box_size: List[float] = field(default_factory=lambda: [0.4,0.3,0.5]) # default min max
    step_height: List[float] = field(default_factory=lambda: [0.0,0.0,0.0]) # default min max
    row_length: float = 5 # 縦
    col_length: float = 10 # 横
    step_noise: List[float] = field(default_factory=lambda: [0.02,0.00,0.01]) # default min max
    step_down: bool = True
    
    
    curriculum_learn_terrain: bool = True
    curriculum_start_end_terrain: List[int] = field(default_factory=lambda: [500000,5000000])
    curriculum_terrain: List[str] = field(default_factory=lambda: ["dekoboko"])
    curriculum_dekoboko_max: List[float] = field(default_factory=lambda: [0.001,0.20]) # default min max
    curriculum_dekoboko_min: List[float] = field(default_factory=lambda: [0.00,0.10])
    
    curriculum_learn_gc: bool = False
    curriculum_start_end_gc: List[int] = field(default_factory=lambda: [2000000,5000000])
    curriculum_gc_max: List[float] = field(default_factory=lambda: [0.05,0.15]) # min max
    curriculum_gc_min: List[float] = field(default_factory=lambda: [0.05,0.15])
    
    curriculum_learn_step_noise: bool = False
    curriculum_start_end_box_noise: List[int] = field(default_factory=lambda: [2000000,5000000])
    curriculum_step_noise_max: List[float] = field(default_factory=lambda: [0.00,0.12]) # min max
    curriculum_step_noise_min: List[float] = field(default_factory=lambda: [0.00,0.04]) # min max
    
    curriculum_learn_step_height: bool = True
    curriculum_start_end_step_height: List[int] = field(default_factory=lambda: [500000,5000000])
    curriculum_step_height_max: List[float] = field(default_factory=lambda: [0.00,0.15]) # min max
    curriculum_step_height_min: List[float] = field(default_factory=lambda: [0.00,0.05]) # min max
    
    
    # CPG
    cpg_log:str = None
    cpg_net:str = None
    a: int = 150
    d: float = 0.15
    gc: List[float] = field(default_factory=lambda: [0.08,0.08,0.12]) # default min max
    gp: List[float] = field(default_factory=lambda: [0.02,0.00001,0.02]) # default min max
    h: List[float] = field(default_factory=lambda: [0.25,0.19,0.30]) # default min max  
    mu: List[float] = field(default_factory=lambda: [1.5,1.0,2.0]) # default min max
    omega: List[float] = field(default_factory=lambda: [0.0,0.0,4.5]) # default min max
    psi: List[float] = field(default_factory=lambda: [0.0,-1.5,1.5]) # default min max
    
    # Residuals e_dq 5
    e_dq: List[float] = field(default_factory=lambda: [0.0,-5.0,5.0]) # default min max
    # e_q_max: List[float] = field(default_factory=lambda: [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,])
    # e_q_min: List[float] = field(default_factory=lambda: [-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,])
    e_q_max: List[float] = field(default_factory=lambda: [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,])
    e_q_min: List[float] = field(default_factory=lambda: [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,])
    e_ddq: List[float] = field(default_factory=lambda: [0.0,-5.0,5.0]) # default min max
    
    
    # Train
    observation_space: List[Union[str, int]] = field(default_factory=lambda: ["full", 87])
    observation_space_cpg: List[Union[str, int]] = field(default_factory=lambda: ["full", 63])
    command_x: List[float] = field(default_factory=lambda: [0.4, 0.0, 0.6]) # default min max
    command_y: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0]) # default min max
    command_w: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0]) # default min max
    episodeLength_s: int = 20
    command_change_interval: int = 5
    external_push_interval: int = 15
    external_push_vel: float = 0.5
    
    # Control
    Kp: float = 100.0
    Kd: float = 2.0
    control_dt_per_mujoco_dt: int = 1
    NN_dt_per_mujoco_dt: int = 10
    hip_limit: List[float] = field(default_factory=lambda: [-0.8,0.8]) # min max
    thigh_limit: List[float] = field(default_factory=lambda: [-0.524,3.927]) # min max
    calf_limit: List[float] = field(default_factory=lambda: [-2.69,-0.9163]) # min max
    
    # Reward
    rewardScale_ang_vel_x: float = -0.05
    rewardScale_ang_vel_y: float = -0.05
    rewardScale_ang_vel_z: float = 0.5
    rewardScale_lin_vel_x: float = 3.0
    rewardScale_lin_vel_y: float = 0.75
    rewardScale_lin_vel_z: float = -2.0
    # rewardScale_work: float = -0.001
    rewardScale_work: float = -0.0
    rewardScale_e_q: float = -1.0
    rewardScale_e_dq: float = -0.00
    
    # Randomization
    additional_trunk_mass: List[float] = field(default_factory=lambda: [0.0,0.0,5.0]) # default min max
    limb_mass_scaling_factor: List[float] = field(default_factory=lambda: [1.0,0.8,1.2]) # default min max
    friction: List[float] = field(default_factory=lambda: [1.5,0.5,2.5]) # default min max
    
    # Visualization
    framerate: int = 25
    frame_height: int = 1080
    frame_width: int = 1920
    track_cam: bool = True
    track_id: int = 0
