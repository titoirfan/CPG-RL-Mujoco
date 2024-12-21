# Real

## Connecting A1 and PC with LAN Cable

<!-- markdownlint-disable-next-line MD033 -->
<img src="../img/LAN_port.svg" width="50%" alt="LAN_port">

### Start unitree controller and A1

Press the power button twice (second press is long press)
Wait until A1 is started (until it starts)
Press L2+A three times to lower the body height
Press L2+B to release

Check the IP address of Ethernet

``` cmd
ipconfig
```

Change the host in [RL_server.py](./src/RL_server.py)

``` python
def start_server():
    
    ######### 通信 #########################################
    
    host = "169.254.250.232"
    port = 12345

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Server is listening on {host}:{port}")
    
    ########################################################
```

Change the IP address in [cpg.cpp](src\unitree_legged_sdk\code\cpg.cpp)

``` cpp
void Custom::TCPClient() {
    int sock = 0;
    struct sockaddr_in serv_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "Socket creation error" << std::endl;
        return;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(12345);

    if (inet_pton(AF_INET, "169.254.250.232", &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported" << std::endl;
        return;
    }
```

## Build unitree_legged_sdk

Open two terminals

First terminal
ssh password is 123

``` cmd
ssh pi@169.254.187.189
mkdir /home/pi/cpgrl
```

Second terminal

``` cmd
scp -r Real/unitree_legged_sdk pi@169.254.187.189:/home/pi/cpgrl
```

First terminal

``` cmd
cd /home/pi/cpgrl/unitree_legged_sdk
mkdir build
cd build
cmake ..
make
```

## Start

Connect A1 and PC with LAN cable

### Start unitree controller and A1

Press the power button twice (second press is long press)
Wait until A1 is started (until it starts)
Press L2+A three times to lower the body height
Press L2+B to release

### Start custom controller

Connect xbox controller to PC

Open two terminals

First terminal

``` cmd
python Real/server.py
```

Second terminal

ssh password is 123

``` cmd
ssh pi@169.254.187.189
cd /home/pi/cpgrl/unitree_legged_sdk/build
./cpg
```

## Control
LT Button : Passive(0 Torque)
RT Button : Stand by PD Control
LB Button : Stand by PD Control
RB Button : Walk & Turn (Use CPG Policy)

Left Stick : Move forward and backward
Right Stick : Turn left and right

Left Pad up & down : Change the height of the body
Left Pad left & right : Change the gc

## Memo

``` cmd
git add .
git commit -m "update"
git push
```