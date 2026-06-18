# Motor-Actuated-Knee-Exoskeleton-Using-IMU-Based-Gait-Detection

Commercial knee braces and conventional knee-ankle-foot orthoses (KAFOs) provide passive stabilization but lack the capacity to deliver adaptive torque or active gait assistance, limiting rehabilitation outcomes. The goal of this project was to design and develop a motor-actuated knee exoskeleton to actively support tibiofemoral joint function through controlled flexion/extension assistance. A brushless DC outrunner motor driven via field-oriented control (FOC) provides precise torque modulation and smooth biomimetic actuation. Real-time joint position feedback from a magnetic encoder and three inertial measurement units (IMUs) embedded in the frame will acquire kinematic data, estimating angular displacement and velocity for user specific gait cycle assistance. Deterministic closed-loop control is maintained through Controller Area Network (CAN) communication between the Moteus R4.11 motor controller and a Raspberry Pi 3 microcontroller, enabling low-latency and high-accuracy actuation during lower-limb activities. Hybrid manufacturing was used for production with rigid PLA and compliant TPU filaments. Simulation of the current hardware configuration demonstrated a 7.03% reduction in knee joint work per gait cycle, with the motor operating at full rated capacity (8.50 A continuous) and an estimated system runtime of 18.5 hours. A proposed upgrade incorporating a 30:1 cycloidal reducer projects work reduction to 21.79% while reducing motor current draw to 2.14 A which is 25% of rated capacity demonstrating substantial performance headroom. This smart bionic knee brace offers active dynamic motion assistance at a fraction of the cost of commercial powered orthoses, while remaining open-source and accessible for further development.

Project Poster Below (github link included is outdated):

<img width="4608" height="3456" alt="FINAL Engineering Fair Poster 2026 (48 x 36 in) (1)" src="https://github.com/user-attachments/assets/ae1f9af5-6102-43e3-9d8b-7d50d723c341" />


Additionally, here are some images of the project in real life:

![IMG_8750](https://github.com/user-attachments/assets/bb8209d1-b03f-46b1-88cf-f1f037981cc7)

![39573592840172411](https://github.com/user-attachments/assets/d6911cbe-aa35-4e17-9706-7d49d207ac84)

The motor is effectively controlled with great precision using tuned PID control loops to rotate smoothly.

Built by Srivatsav Sura, Navin Karthik, and Srinath Ramakrishnan. Hillcrest High School 2026 (this project won 1st in Electrical and Computer Engineering at USEF -> University of Utah Science and Engineering Fair and it's prototype won 3rd in 2025)
