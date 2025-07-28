# Onnxruntime-Bitbot-sim2sim
Code Storage


The code to achieve sim2sim with an onnx policy to use in BITbot-Mujoco, with the data class template followed BipedControlGaitLab from HLQ


The improve is :

(1) Add a new thread, use cv and mutex to manage the memory.

(2) Atomic bool trigger flag.

(3) The policy was called once in main function every four times run.

onnx prediction time was short, so the "if" code was used.

<img src="https://github.com/user-attachments/assets/1dddf983-0bc2-455d-b050-9a85ea50a9e5" width="500px"/>

<img src="https://github.com/user-attachments/assets/e3a2b23b-51c5-424e-a107-841f68187ee0" width="500px"/>

<img src="https://github.com/user-attachments/assets/b3cd72b5-5f5c-463d-8972-b79fe343d3b0" width="500px"/>
