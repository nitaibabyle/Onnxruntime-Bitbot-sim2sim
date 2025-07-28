#pragma once
#include <iostream>
#include <vector>
#include <numeric>
#include <Eigen/Dense>
#include <onnxruntime_cxx_api.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>

class network_process
{
public:
    network_process(); // construcion
    ~network_process();

    void RL_give_model(const std::string &model_path);             // used for model build
    void predict(Eigen::VectorXf &input, Eigen::VectorXf &output); // core calculation, give it a input, it will obtian a output

    void do_predict(Eigen::VectorXf &input);  // main function called API, everytime called the predict will run one time
    bool ask_result(Eigen::VectorXf &output); // main function called API, everytime called , it will check the memory and get the prediction result and flag
    std::atomic<bool> stop_flag{false};       // stop flag

private:
    void SetupIO();       // get input/output name and shape
    void worker_thread(); // the thread function , this function contians the predict

    // beizhu
    std::vector<const char *> input_names_robot = {"observation"}; // just a note, it will be change and correct in model constuction step
    std::vector<const char *> output_names_robot = {"action"};     // just a note, it will be change and correct in model constuction step

    std::vector<int64_t> input_shape_robot = {1, 84};  // just a note, it will be change and correct in model constuction step
    std::vector<int64_t> output_shape_robot = {1, 14}; // just a note, it will be change and correct in model constuction step

    // ONNX Runtime objects
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session_robot;
    bool initialized = false;

    bool output;

    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;

    std::atomic<bool> output_ready{false};
    std::atomic<bool> input_ready{false};
    std::atomic<bool> not_finish{false};
    Eigen::VectorXf latest_input;
    Eigen::VectorXf latest_output;
    int call_cont = 0;
};
