# Onnxruntime-Bitbot-sim2sim
Code Storage


Before this code, the onnx runtime should be install in local device.


I used pre-compiled version, "onnxruntime-linux-x64-1.22.0.tgz". 


[https://github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases)


Download this file and extract in your bitbot project dir.


Using "add_library" to link to onnx lib, which was demonstrated in CMakeLists.txt
 .

The code to achieve humanoid reinforcement learning sim2sim with an onnx policy to use in BITbot-Mujoco, with the data class template followed BipedControlGaitLab from HLQ


The contribution were:


(1) Add a new thread, use cv and mutex to manage the memory.


(2) Atomic bool trigger flag.


(3) The policy called frequency was depending on the main function every times run counter.


Due to the onnx prediction time was short, there was no calculation result buffer and IO queue consideration.


<img src="https://github.com/user-attachments/assets/1dddf983-0bc2-455d-b050-9a85ea50a9e5" width="500px"/>


<img src="https://github.com/user-attachments/assets/e3a2b23b-51c5-424e-a107-841f68187ee0" width="500px"/>


<img src="https://github.com/user-attachments/assets/b3cd72b5-5f5c-463d-8972-b79fe343d3b0" width="500px"/>

