
#include "onnx.hpp"
wheel_model.RL_give_model("/home/bhrrobot/zjt_bitbot_simulate/20231103bitbotkuafu/bitbot-mujoco-demo_RL_wheelleg/HLQwalkingLib/onnx_model/policy_2.onnx");
int main()
	{
while (1)
{
		// angular velocity
		Eigen::Vector3d base_ang_vel;
		base_ang_vel << actRobotClass.Torso.jointAngleV[3], actRobotClass.Torso.jointAngleV[4], actRobotClass.Torso.jointAngleV[5];

		// imu
		Eigen::Matrix<double, 3, 3> torso_rotationmatrix;
		torso_rotationmatrix = StateEstimation::RPY2ROT(actRobotClass.TorsoOrign.jointAngle[3], actRobotClass.TorsoOrign.jointAngle[4], 0.0);
		torso_rotationmatrix.transposeInPlace();
		Eigen::Vector3d gravity_world;
		gravity_world << 0, 0, -1;
		// std::cout << gravity_world << std::endl;
		Eigen::Vector3d projected_gravity = torso_rotationmatrix * gravity_world;
		Eigen::Vector3d norm_gravity_vector;
		norm_gravity_vector << projected_gravity[0], projected_gravity[1], projected_gravity[2];

		// velocity
		Eigen::Vector3d base_lin_vel;
		actRobotClass.TorsoReal.jointAngleV.head(3) = torso_rotationmatrix * actRobotClass.TorsoWorldReal.jointAngleV.head(3);
		base_lin_vel << actRobotClass.TorsoReal.jointAngleV[0], actRobotClass.TorsoReal.jointAngleV[1], actRobotClass.TorsoReal.jointAngleV[2];

		// last action
		last_last_actions;
		last_actions;

		// commands
		Eigen::Vector3d cmd;
		refRobotClass.desired_speed << userCommondClass.walkingVelocityX, 0.0, userCommondClass.walkingVelocityY;
		cmd = refRobotClass.desired_speed;

		Eigen::VectorXd defult_dof(14);
		defult_dof << 0.0, -0.35, 0.87, 0.0, 0.6, 0.6, -1.0, -0.0, -0.35, 0.87, 0.0, 0.6, -0.6, -1.0;

		// dof pos
		Eigen::VectorXd dof_pos(14);
		dof_pos.setZero();
		dof_pos << actRobotClass.LLeg.jointAngle, actRobotClass.LArm.jointAngle, actRobotClass.RLeg.jointAngle, actRobotClass.RArm.jointAngle;
		dof_pos = dof_pos - defult_dof;

		// dof velo
		Eigen::VectorXd dof_vel(14);
		dof_vel.setZero();
		dof_vel << actRobotClass.LLeg.jointAngleV, actRobotClass.LArm.jointAngleV, actRobotClass.RLeg.jointAngleV, actRobotClass.RArm.jointAngleV;

		// actions
		actions;

		// commands
		float obs_scales_lin_vel = 2.0;
		float obs_scales_ang_vel = 0.25;
		float obs_scales_dof_pos = 1.0;
		float obs_scales_dof_vel = 0.05;
		float obs_scales_height_measurements = 5.0;
		Eigen::Vector3f commands_scale;
		commands_scale << obs_scales_lin_vel, obs_scales_ang_vel, obs_scales_dof_pos;
		Eigen::Vector3f commands;
		commands = cmd.cast<float>();
		commands = commands.cwiseProduct(commands_scale);

		// define input and output shape
		Eigen::VectorXf input(82);
		Eigen::VectorXf output(14);
		input.setZero();
		output.setZero();

		// input component looks like
		input << base_lin_vel.cast<float>() * obs_scales_lin_vel,
			base_ang_vel.cast<float>() * obs_scales_ang_vel,
			norm_gravity_vector.cast<float>(),
			last_actions,
			last_last_actions,
			commands.cast<float>(),
			dof_pos.cast<float>() * obs_scales_dof_pos,
			dof_vel.cast<float>() * obs_scales_dof_vel,
			actions; // prepare input vector

		bool ready = false;					   // if the prediction thread finish one calculation, the flag become true
		ready = walk_model.ask_result(output); // check predict thread lock memory, get the finish flag and the output vector

		if (RLfreqtimer > 4) // decimation influenced by the frequency, every 4 times run of main thread, the prediction thread will run 1 time
		{
			// action buffer
			last_last_actions = last_actions;
			last_actions = actions;
			actions = output;

			// input update using current action
			input << base_lin_vel.cast<float>() * obs_scales_lin_vel,
				base_ang_vel.cast<float>() * obs_scales_ang_vel,
				norm_gravity_vector.cast<float>(),
				last_actions,
				last_last_actions,
				commands.cast<float>(),
				dof_pos.cast<float>() * obs_scales_dof_pos,
				dof_vel.cast<float>() * obs_scales_dof_vel,
				actions; // prepare input vector

			// call prediction thread to do one prediciton
			walk_model.do_predict(input);
			RLfreqtimer = 0; // set zero for next time preparing
		}
		RLfreqtimer++; // conter variable

		// 输出结果
		// std::cout << "模型输出: " << output.transpose() << std::endl;

		// action rate
		double pos_action_scale = 0.5;
		double vel_action_scale = 10;

		// variable preparing
		refRobotClass.LLeg.jointAngle.setZero();
		refRobotClass.LLeg.jointAngleV.setZero();
		refRobotClass.LArm.jointAngle.setZero();
		refRobotClass.LArm.jointAngleV.setZero();
		refRobotClass.RLeg.jointAngle.setZero();
		refRobotClass.RLeg.jointAngleV.setZero();
		refRobotClass.RArm.jointAngle.setZero();
		refRobotClass.RArm.jointAngleV.setZero();

		// //if you use the defult pos to calculate torque, use this code
		// refRobotClass.LLeg.jointAngle.block(0, 0, 3, 1) = pos_action_scale * (output.block(0, 0, 3, 1).cast<double>() + defult_dof.block(0, 0, 3, 1).cast<double>());
		// refRobotClass.LLeg.jointAngleV[3] = vel_action_scale * output.cast<double>()[3];
		// refRobotClass.LArm.jointAngle.block(0, 0, 3, 1) = pos_action_scale * (output.block(4, 0, 3, 1).cast<double>() + defult_dof.block(3, 0, 3, 1).cast<double>());
		// refRobotClass.RLeg.jointAngle.block(0, 0, 3, 1) = pos_action_scale * (output.block(7, 0, 3, 1).cast<double>() + defult_dof.block(6, 0, 3, 1).cast<double>());
		// refRobotClass.RLeg.jointAngleV[3] = vel_action_scale * output.cast<double>()[10];
		// refRobotClass.RArm.jointAngle.block(0, 0, 3, 1) = pos_action_scale * (output.block(11, 0, 3, 1).cast<double>() + defult_dof.block(9, 0, 3, 1).cast<double>());

		// I use the action be the joint pos directly
		refRobotClass.LLeg.jointAngle.block(0, 0, 4, 1) = pos_action_scale * (output.block(0, 0, 4, 1).cast<double>()) + defult_dof.block(0, 0, 4, 1).cast<double>();
		refRobotClass.LArm.jointAngle.block(0, 0, 3, 1) = pos_action_scale * (output.block(4, 0, 3, 1).cast<double>()) + defult_dof.block(4, 0, 3, 1).cast<double>();
		refRobotClass.RLeg.jointAngle.block(0, 0, 4, 1) = pos_action_scale * (output.block(7, 0, 4, 1).cast<double>()) + defult_dof.block(7, 0, 4, 1).cast<double>();
		refRobotClass.RArm.jointAngle.block(0, 0, 3, 1) = pos_action_scale * (output.block(11, 0, 3, 1).cast<double>()) + defult_dof.block(11, 0, 3, 1).cast<double>();

		return 0;
}
	}
