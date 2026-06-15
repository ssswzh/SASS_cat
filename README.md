# Robot Cat Project
## Leiden University Robotics 2025-2026

## Teammembers
- Adolfo Miguel Martins Morgado (s4673565)
- Cao Shuai (s4851978)
- Shubham Gusain (s4846761)
- Siwen Zhang (s4683226)

```
|\---/|
| o_o |
 \_^_/
```

## Setting Up

To fully run project you will require:
- IRobot Create 3 robot on firmware version H.2.6
- Nvidia Jetson Orin Nano with Jetpack 6
    - The Jetson must be connected to the IRobot Create 3 via USB as Ethernet (see [https://iroboteducation.github.io/create3_docs/setup/jetson/](this guide))
    - The Jetson must also be connected to the appropriate servos, with pins indicated in `head.py`
- ROS 2 Humble

## Project

We contain 4 main packages
- `cat_msgs`
    - contains message definitions for passing out custom information between nodes
- `cat`
    - contains the `brain`, `speech` and `vision` nodes
- `sensors`
    - contains the `camera` and `microphone` nodes
- `actuators`
    - contains the `head` and `speaker` nodes

## Setting up & Running

First, setup and install required packages in [./dotfiles/setup.bash](setup file).

Next, clone the repository:
```bash
git clone https://github.com/AdMorgado/leiden_robo_project.git ~/ros2_ws
cd ~/ros2_ws
```

And proceed to build the project using `colcon`:
```bash
colcon build
```

Launch all of nodes with the exception of the `brain` node:
```bash
ros2 launch cat base.py
```

When you first launch the nodes, some models (like yolo's `yolo11m.pt`) will download.
Please keep the nodes alive until these dependencies finish downloading.

Assuming all went correct, you can now run the `brain` node to begin cat behaviour.
```bash
ros2 run cat brain
```

