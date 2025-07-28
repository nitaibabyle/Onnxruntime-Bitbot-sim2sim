#include <onnx.hpp>

network_process::network_process() : env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel_wheel"),
                                     session_options()
{
    session_options.SetIntraOpNumThreads(1);
    worker = std::thread(&network_process::worker_thread, this);
    cv.notify_one();
}
network_process::~network_process()
{
    stop_flag = true;
    cv.notify_all();
    if (worker.joinable())
        worker.join();
}

void network_process::do_predict(Eigen::VectorXf &input)
{
    std::lock_guard<std::mutex> lock(mtx);
    latest_input = input;
    input_ready = true;
    call_cont += 1; // decimation
    cv.notify_one();
    // Start thread if not running
}

void network_process::worker_thread()
{
    if (stop_flag.load())
    {
        worker.join();
    }
    while (!stop_flag.load())
    {
        // auto start_start = std::chrono::high_resolution_clock::now();
        if (stop_flag.load())
        {
            break;
        }
        Eigen::VectorXf input_cal;
        Eigen::VectorXf output_cal(output_shape_robot[1]);
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this]()
                    { return !not_finish && input_ready; });
            input_cal = latest_input;
            output_ready = false;
            not_finish = true;
        }

        // Do the actual prediction
        predict(input_cal, output_cal);

        // Store the result
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this]()
                    { return call_cont > 0; }); //  internal wait. finish but need to wait,different from RLfreqtimer, this achieves wait in thread level, if call_cont=0, the result would output immediatly. without thread wait
            output_ready = true;
            latest_output = output_cal;
            input_ready = false; // Reset for next prediction
            not_finish = false;
            call_cont = 0;
        }

        // auto end_end = std::chrono::high_resolution_clock::now();
        // auto duration_dur = std::chrono::duration_cast<std::chrono::microseconds>(end_end - start_start);
        // printf("predict_duration_dur: %d\n", duration_dur);
    }
    printf("while_finish");
}

bool network_process::ask_result(Eigen::VectorXf &output)
{

    std::lock_guard<std::mutex> lock(mtx); // Still need mutex for latest_output
    // std::cout << "latest_output" << latest_output << std::endl;
    // std::cout << "output_ready" << output_ready.load() << std::endl;
    if (!output_ready)
        return false;

    output = latest_output;
    return true;
    cv.notify_one();
}

void network_process::RL_give_model(const std::string &model_path)
{
    // Load ONNX model
    session_robot = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

    // Get input/output information
    SetupIO();
    initialized = true;

    latest_input = Eigen::VectorXf::Zero(input_shape_robot[1]);
    latest_output = Eigen::VectorXf::Zero(output_shape_robot[1]);
}
void network_process::predict(Eigen::VectorXf &input, Eigen::VectorXf &output)
{
    if (!initialized)
    {
        throw std::runtime_error("Model not initialized. Call RL_wrapper() first.");
    }

    // Input validation
    if (input.size() != input_shape_robot[1])
    {
        throw std::runtime_error("Input dimension mismatch. Expected " +
                                 std::to_string(input_shape_robot[1]) +
                                 ", got " + std::to_string(input.size()));
    }

    // Prepare input tensor
    std::vector<float> input_data(input.data(), input.data() + input.size());
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape_robot.data(),
        input_shape_robot.size());

    // Run inference
    auto output_tensors = session_robot->Run(
        Ort::RunOptions{nullptr},
        input_names_robot.data(), // Use the stored names
        &input_tensor,
        1,
        output_names_robot.data(),
        1);

    // Process output
    float *output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (output_size != output_shape_robot[1])
    {
        throw std::runtime_error("Output dimension mismatch. Expected " +
                                 std::to_string(output_shape_robot[1]) +
                                 ", got " + std::to_string(output_size));
    }

    output = Eigen::Map<Eigen::VectorXf>(output_data, output_size);
}

void network_process::SetupIO()
{

    // Get input shape
    Ort::TypeInfo input_type_info = session_robot->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape_robot = input_tensor_info.GetShape();

    // Get output shape
    Ort::TypeInfo output_type_info = session_robot->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    output_shape_robot = output_tensor_info.GetShape();
}
