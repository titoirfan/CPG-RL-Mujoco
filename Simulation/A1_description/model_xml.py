import os
import sys
import numpy as np


script_dir = os.path.dirname(__file__)    # .../A1_WS/Envs
parent_dir = os.path.dirname(script_dir)  # .../A1_WS
sys.path.append(parent_dir)
from .a1_xml import a1_xml
# from a1_xml import a1_xml


def FlatXml(additional_trunk_mass=0.0, limb_mass_scaling_factor=1.0,skeleton=True,friction=0.8):
    a1_model = a1_xml(additional_trunk_mass=additional_trunk_mass, limb_mass_scaling_factor=limb_mass_scaling_factor,skeleton=skeleton)
    
    xml = f"""
    <mujoco model="a1 Flat Ground">
        <option gravity='0 0 -9.806' iterations='50' solver='Newton' timestep='0.001'/> <!--追記-->

        {a1_model}

        <statistic center="0 0 0.1" extent="0.8"/>

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080"/>
        </visual>

        <asset>
            <!-- <texture type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/> -->
            <!--<texture type="2d" name="groundplane" builtin="checker" rgb1="0.59 0.6 0.66" rgb2="0.49 0.5 0.56"  width="300" height="300"/>-->
            <texture type="2d" name="groundplane" builtin="checker" rgb1="0.9804 0.9882 0.9882" rgb2="0.9804 0.9882 0.9882"  width="500" height="500" mark="edge" markrgb="0.1 0.1 0.1" />
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
        </asset>

        <worldbody>
            <geom name="floor" size="100 100 0.05" type="plane" material="groundplane" friction="{friction}"/>
        </worldbody>

    </mujoco>
    """
    return xml