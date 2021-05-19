#pragma once

#include <Eigen/Core>
#include "types.h"
#include "pose.h"

namespace reconstruction
{
	class View
	{
	public:
		View(const CameraId camera_id_,
			const ViewId view_id_) :
			camera_id(camera_id_),
			view_id(view_id_),
			has_pose(false),
			T_view_world(Pose(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()))
		{

		}

		const size_t &cameraId() const { return camera_id; }
		const size_t &id() const { return view_id; }

		const Pose &getPose() const { return T_view_world; }
		Pose &getMutablePose() { return T_view_world; }

		void setPose(Eigen::Matrix3d rotation_,
			Eigen::Vector3d translation_);
		
		const std::unordered_map<std::string, std::string> &getMetadata() const { return metadata; }
		std::unordered_map<std::string, std::string> &getMutableMetadata() { return metadata; }


	protected:
		const ViewId view_id;
		const CameraId camera_id;
		Pose T_view_world;
		ViewMetadata metadata;
		bool has_pose;
	};

	void View::setPose(
		Eigen::Matrix3d rotation_, 
		Eigen::Vector3d translation_)
	{
		T_view_world = Pose(rotation_, translation_);
		has_pose = true;
	}
}