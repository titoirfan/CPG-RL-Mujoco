import socket
import struct
import numpy as np
import glob
import sys
import os


from xbox360_controller import Xbox360Controller
import pytz
import datetime
import csv

script_dir = os.path.dirname(__file__)    
parent_dir = os.path.dirname(script_dir)  
sys.path.append(parent_dir)

script_name = os.path.basename(__file__)[: -len(".py")]

from SoftActorCritic.SAC import SAC_Eval
from cfg.cfg import Config as CPG_Config
from cfg.Save_Load_cfg import json_to_dataclass,dataclass_to_json


class Server:
    def __init__(self):
        
        ####################################################################################
        self.host = "169.254.187.12"
        self.port = 12345
        
        
        forward_path = os.path.join(parent_dir,"NN","241021_173827")
        forwardturn_path = os.path.join(parent_dir,"NN","240917_150210")
        backward_path = os.path.join(parent_dir,"NN","240917_151044")
        turnleft_path = os.path.join(parent_dir,"NN","240921_171819")
        turnright_path = os.path.join(parent_dir,"NN","240921_172331")

        
        
        
        self.h = 0.27
        self.gc = 0.20
        self.gp = 0.00
        self._kp = 60.0
        self._kd = 2.0
        
        
        
        self.command = np.array([0.0, 0.0, 0.0])
        
        self.observation_cpg = np.zeros(63)
        
        
        
        ####################################################################################
        
        self.controller = Xbox360Controller()

        
        

        # モデルの読み込み
        forward_cfg_data = json_to_dataclass(forward_path + "/config.json")
        forwardturn_cfg_data = json_to_dataclass(forwardturn_path + "/config.json")
        backward_cfg_data = json_to_dataclass(backward_path + "/config.json")
        turnleft_cfg_data = json_to_dataclass(turnleft_path + "/config.json")
        turnright_cfg_data = json_to_dataclass(turnright_path + "/config.json")
        
        self.forward_cfg = CPG_Config(**forward_cfg_data)
        self.forwardturn_cfg = CPG_Config(**forwardturn_cfg_data)
        self.backward_cfg = CPG_Config(**backward_cfg_data)
        self.turnleft_cfg = CPG_Config(**turnleft_cfg_data)
        self.turnright_cfg = CPG_Config(**turnright_cfg_data)
        
        self.forward_cfg.gpu = 0
        self.forwardturn_cfg.gpu = 0
        self.backward_cfg.gpu = 0
        self.turnleft_cfg.gpu = 0
        self.turnright_cfg.gpu = 0

        self.forward_agent = SAC_Eval(63,12,self.forward_cfg)
        self.forwardturn_agent = SAC_Eval(63,12,self.forwardturn_cfg)
        self.backward_agent = SAC_Eval(63,12,self.backward_cfg)
        self.turnleft_agent = SAC_Eval(63,12,self.turnleft_cfg)
        self.turnright_agent = SAC_Eval(63,12,self.turnright_cfg)
        
        
        forward_net = self.get_latest_net(forward_path)
        forwardturn_net = self.get_latest_net(forwardturn_path)
        backward_net = self.get_latest_net(backward_path)
        turnleft_net = self.get_latest_net(turnleft_path)
        turnright_net = self.get_latest_net(turnright_path)
        
        self.forward_agent.load_checkpoint(forward_net)
        self.forwardturn_agent.load_checkpoint(forwardturn_net)
        self.backward_agent.load_checkpoint(backward_net)
        self.turnleft_agent.load_checkpoint(turnleft_net)
        self.turnright_agent.load_checkpoint(turnright_net)
        
        self.mu = np.ones(4)
        self.omega = np.zeros(4)
        self.psi = np.zeros(4)
        
        self.Kp = np.zeros(4)
        self.Kd = np.zeros(4)
        self.last_Kp = self.Kp
        self.last_Kd = self.Kd
        self.last_last_Kp = self.last_Kp
        self.last_last_Kd = self.last_Kd
        self.last_joint_q = np.zeros(12)
        
        self.hat_vert_push = False
        self.hat_horz_push = False
        self.start_push = False
        self.home_push = False
        
        
        self.motiontime = 0
        self.last_motiontime = 0
        self.mode = 0
        self.mode_count = 0
            
        # ソケットの作成
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server is listening on {self.host}:{self.port}")
        
    def get_latest_net(self,path):
        network_files = glob.glob(f"{path}/Networks/episode_*.pt")
        if network_files:
            latest_network = max(network_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            return latest_network
        else:
            print("No network files found")
            sys.exit()
    
    def set_command(self):

        self.command[0] = self.controller.Ljoy_vert * 0.2
        self.command[1] = self.controller.Ljoy_horz * 0.6
        self.command[2] = self.controller.Rjoy_horz * 1.0
        
    
    def set_mode(self):
        # mode 0: Passive(0 Torque), mode 1: (Stand by PD Control), mode 2: walk (Use CPG Policy)
        if self.motiontime > 1000:
 
            if (self.mode == 0 or self.mode == 2) and (self.controller.RT > 0 or self.controller.LB > 0):
                self.mode = 1
                self.mode_count = 0
                
            elif (self.mode == 1 or self.mode == 2) and self.controller.LT > 0:
                self.mode = 0
                self.mode_count = 0

            elif self.mode == 1 and self.controller.RB > 0:
                self.mode = 2
                self.mode_count = 0
                
                
            self.mode_count += 1
        
    def set_param(self):
        
        if np.abs(self.controller.hat_vert) > 0.5 and self.hat_vert_push == False:
            # print(f"self.controller.hat_vert: {self.controller.hat_vert}")
            self.h = self.controller.hat_vert * 0.01 + self.h
            self.h = np.round(self.h,3)
            self.hat_vert_push = True
        elif np.abs(self.controller.hat_vert) > 0.5:
            self.hat_vert_push = True
        else:
            self.hat_vert_push = False
            
        if np.abs(self.controller.hat_horz) > 0.5 and self.hat_horz_push == False:
            self.gc = self.controller.hat_horz * 0.01 + self.gc
            self.gc = np.round(self.gc,3)
            self.hat_horz_push = True
        elif np.abs(self.controller.hat_horz) > 0.5:
            self.hat_horz_push = True
        else:
            self.hat_horz_push = False
        
        if self.h < self.gc + 0.07:
            self.h = self.gc + 0.07
            
        if self.controller.start > 0 and self.start_push == False:
            self.gp =  -0.01 + self.gp
            self.gp = np.round(self.gp,3)
            self.start_push = True
        elif self.controller.start > 0:
            self.start_push = True
        else:
            self.start_push = False
            
        if self.controller.home > 0 and self.home_push == False:
            self.gp = 0.01 + self.gp
            self.gp = np.round(self.gp,3)
            self.home_push = True
        elif self.controller.home > 0:
            self.home_push = True
        else:
            self.home_push = False
            
        
            
    def set_gain(self,A=0.005):
        
        def func(x,mid):
            return 0.5 * np.exp(-4*(np.abs(x-2*mid/3))**2) + 0.5

        if self.mode == 0:
            _Kp=np.full(4,0.0)
            _Kd=np.full(4,0.0)
        elif self.mode == 1 or self.mode == 2:
            _Kp=np.full(4,self.forward_cfg.Kp)
            _Kp = np.full(4,self._kp)
            
            _Kd = np.full(4,self._kd)
            
        if self.mode_count < 1500:
            
            if np.sum(np.abs(self.Kp - _Kp)) < 0.8*_Kp[0] and  (self.mode == 1 or self.mode == 2):
                self.Kp = _Kp
            else:
                self.Kp = A * func(self.Kp,self.forward_cfg.Kp/2) * (_Kp-self.Kp) + self.Kp
            if np.sum(np.abs(self.Kd - _Kd)) < 0.8*_Kd[0] and  (self.mode == 1 or self.mode == 2):
                self.Kd = _Kd
            else:
                self.Kd = A * func(self.Kd,self.forward_cfg.Kd/2) * (_Kd-self.Kd) + self.Kd
            
        else:
            self.Kp = _Kp
            self.Kd = _Kd

            
    def get_observation(self,array):

        
        self.motiontime = array[0]
        
        array[25:29] = np.where(array[25:29] > 0.0, 1.0, 0.0)
        
        self.observation_cpg = np.concatenate((array[1:61], self.command))
        
        self.last_joint_q = array[1:13]
        
    def StandCPGPolicy(self,A=0.5):
        self.mu = np.full(4,1.0)
        self.omega = np.full(4,0.0)
        self.psi = np.full(4,0.0)
        
        if self.observation_cpg[40] < np.pi:
            self.omega[0] = A * (0 - self.observation_cpg[40]) 
        else:
            self.omega[0] = A * (2*np.pi - self.observation_cpg[40]) 
        
        
        self.omega[1] = A * (np.pi - self.observation_cpg[41]) 
        
        self.omega[2] = A * (np.pi - self.observation_cpg[42]) 
        
        if self.observation_cpg[43] < np.pi:
            self.omega[3] = A * (0 - self.observation_cpg[43]) 
        else:
            self.omega[3] = A * (2*np.pi - self.observation_cpg[43]) 
            
        
        self.psi = A * (0 - self.observation_cpg[44:48])
        

    
        
        
    def get_action(self):
        if self.mode == 0 or self.mode == 1:
            self.StandCPGPolicy()
            
        elif self.mode == 2:
            if self.command[0] >= 0.0 and np.abs(self.command[2]) == 0.0:
                self.observation_cpg[61] = 0.0
                self.observation_cpg[62] = 0.0
                action_cpg =self.forward_agent.select_action(self.observation_cpg)
                self.mu = ((self.forward_cfg.mu[2]-self.forward_cfg.mu[1])/2)*action_cpg[0:4] + ((self.forward_cfg.mu[1]+self.forward_cfg.mu[2])/2)
                
                self.omega = ((self.forward_cfg.omega[2]-self.forward_cfg.omega[1])/2)*action_cpg[4:8] + ((self.forward_cfg.omega[1]+self.forward_cfg.omega[2])/2)
                
                self.psi = ((self.forward_cfg.psi[2]-self.forward_cfg.psi[1])/2)*action_cpg[8:12] + ((self.forward_cfg.psi[1]+self.forward_cfg.psi[2])/2)
                
                
                
                    
            elif self.command[0] > 0 and np.abs(self.command[2]) > 0:
                self.observation_cpg[61] = 0.0
                action_cpg = self.forwardturn_agent.select_action(self.observation_cpg)
                self.mu = ((self.forwardturn_cfg.mu[2]-self.forwardturn_cfg.mu[1])/2)*action_cpg[0:4] + ((self.forwardturn_cfg.mu[1]+self.forwardturn_cfg.mu[2])/2)
                self.omega = ((self.forwardturn_cfg.omega[2]-self.forwardturn_cfg.omega[1])/2)*action_cpg[4:8] + ((self.forwardturn_cfg.omega[1]+self.forwardturn_cfg.omega[2])/2)
                self.psi = ((self.forwardturn_cfg.psi[2]-self.forwardturn_cfg.psi[1])/2)*action_cpg[8:12] + ((self.forwardturn_cfg.psi[1]+self.forwardturn_cfg.psi[2])/2)

                
                
            elif self.command[0] < 0 :
                self.observation_cpg[61] = 0.0
                self.observation_cpg[62] = 0.0
                action_cpg = self.backward_agent.select_action(self.observation_cpg)
                self.mu = ((self.backward_cfg.mu[2]-self.backward_cfg.mu[1])/2)*action_cpg[0:4] + ((self.backward_cfg.mu[1]+self.backward_cfg.mu[2])/2)
                self.omega = ((self.backward_cfg.omega[2]-self.backward_cfg.omega[1])/2)*action_cpg[4:8] + ((self.backward_cfg.omega[1]+self.backward_cfg.omega[2])/2)
                self.psi = ((self.backward_cfg.psi[2]-self.backward_cfg.psi[1])/2)*action_cpg[8:12] + ((self.backward_cfg.psi[1]+self.backward_cfg.psi[2])/2)
                
                
                
            elif self.command[2] > 0 and self.command[0] == 0:
                self.observation_cpg[60] = 0.0
                self.observation_cpg[61] = 0.0
                action_cpg = self.turnleft_agent.select_action(self.observation_cpg)
                self.mu = ((self.turnleft_cfg.mu[2]-self.turnleft_cfg.mu[1])/2)*action_cpg[0:4] + ((self.turnleft_cfg.mu[1]+self.turnleft_cfg.mu[2])/2)
                self.omega = ((self.turnleft_cfg.omega[2]-self.turnleft_cfg.omega[1])/2)*action_cpg[4:8] + ((self.turnleft_cfg.omega[1]+self.turnleft_cfg.omega[2])/2)
                self.psi = ((self.turnleft_cfg.psi[2]-self.turnleft_cfg.psi[1])/2)*action_cpg[8:12] + ((self.turnleft_cfg.psi[1]+self.turnleft_cfg.psi[2])/2)
                
                
            
            elif self.command[2] < 0 and self.command[0] == 0:
                self.observation_cpg[60] = 0.0
                self.observation_cpg[61] = 0.0
                action_cpg = self.turnright_agent.select_action(self.observation_cpg)
                self.mu = ((self.turnright_cfg.mu[2]-self.turnright_cfg.mu[1])/2)*action_cpg[0:4] + ((self.turnright_cfg.mu[1]+self.turnright_cfg.mu[2])/2)
                self.omega = ((self.turnright_cfg.omega[2]-self.turnright_cfg.omega[1])/2)*action_cpg[4:8] + ((self.turnright_cfg.omega[1]+self.turnright_cfg.omega[2])/2)
                self.psi = ((self.turnright_cfg.psi[2]-self.turnright_cfg.psi[1])/2)*action_cpg[8:12] + ((self.turnright_cfg.psi[1]+self.turnright_cfg.psi[2])/2)
                
                
            
            
            self.omega = 2*np.pi*self.omega
            self.psi = 2*np.pi*self.psi
                
                
    
    def print_info(self,hz):
        if self.motiontime - self.last_motiontime > 1000/hz:
            self.last_motiontime = self.motiontime
            os.system('cls' if os.name == 'nt' else 'clear')
            # print(server.observation_cpg)
            print(f"mode: {self.mode}, motiontime: {self.motiontime}, command: {self.command}")
            print(f"Kp: {self.Kp}, Kd: {self.Kd}, h: {self.h}, gc: {self.gc}, gp: {self.gp}")
            print(f"r: {self.observation_cpg[36:40]}, theta: {self.observation_cpg[40:44]},phi: {self.observation_cpg[44:49]}")
            


if __name__ == "__main__":

    server = Server()

        
    while True:
        client_socket, client_address = server.server_socket.accept()
        # print(f"Connection from {client_address} has been established.")
        
        data = client_socket.recv(1024)
        
        array = np.array(struct.unpack('d' * 61, data[:61*8]))
        server.controller.check_buttons(threshold=0.2)
        server.set_command()
        server.get_observation(array)
        server.set_mode()
        server.set_param()
        server.set_gain()
        server.get_action()
        server.print_info(hz=2)
        
        response = struct.pack('d' * 23, *(server.mu.tolist() + server.omega.tolist() + server.psi.tolist() +  [server.h, server.gc, server.gp] + server.Kp.tolist() + server.Kd.tolist()))
        client_socket.send(response)
        client_socket.close()