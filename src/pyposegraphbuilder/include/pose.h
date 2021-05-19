#pragma once

#include <map>
#include <vector>
#include "pose_utils.h"
#include "types.h"
#include <sophus/se3.hpp>

namespace reconstruction
{
	class Pose 
	{
	protected:
		Eigen::Matrix3d essentialMatrix;
		Sophus::SE3d T_dst_src;
		double sourceScale,
			destinationScale;
	
	public:
		Pose(Eigen::Matrix3d rotation_,
			Eigen::Vector3d translation_) :
			Pose(Sophus::SE3d(Eigen::Quaterniond(rotation_), translation_),
				0.0,
				0.0)
		{

		}

		Pose(Eigen::Matrix3d rotation_,
			Eigen::Vector3d translation_,
			double sourceScale_,
			double destinationScale_) :
			Pose(Sophus::SE3d(Eigen::Quaterniond(rotation_), translation_),
				sourceScale_,
				destinationScale_)
		{

		}

		Pose(Sophus::SE3d T_dst_src_) :
			Pose(T_dst_src_, 0.0, 0.0)
		{

		}

		Pose(Sophus::SE3d T_dst_src_,
			double source_scale_,
			double destination_scale_) :
			T_dst_src(T_dst_src_),
			essentialMatrix(reconstruction::pose::getEssentialMatrixFromRelativePose(T_dst_src_)),
			sourceScale(source_scale_),
			destinationScale(destination_scale_)
		{

		}

		Pose()
		{
		}

		Pose clone() const
		{
			return Pose(T_dst_src,
				sourceScale,
				destinationScale);
		}

		inline void setPose(const Sophus::SE3d &T_dst_src_);

		inline const Sophus::SE3d &getTransform() const { return T_dst_src; }

		inline const Eigen::Matrix3d &getRotation() const { return T_dst_src.rotationMatrix(); }

		inline const Eigen::Vector3d& getTranslation() const { return T_dst_src.translation(); }

		inline const Eigen::Matrix3d& getEssentialMatrix() const { return essentialMatrix; }

		inline Eigen::Vector3d getPosition() const { return T_dst_src.rotationMatrix().transpose() * T_dst_src.translation(); }

		inline Pose getInverse() const
		{
			return Pose(T_dst_src.inverse()); // TODO: check
		}

		inline void getScales(
			double &source_scale_,
			double &destination_scale_) const
		{
			source_scale_ = sourceScale;
			destination_scale_ = destinationScale;
		}

		Pose operator*(const Pose &pose_) const
		{
			// TODO: check
			return Pose(T_dst_src * pose_.getTransform());
		}
	};

	inline void Pose::setPose(const Sophus::SE3d &T_dst_src_)
	{
		T_dst_src = T_dst_src_;
		essentialMatrix = reconstruction::pose::getEssentialMatrixFromRelativePose(T_dst_src_);
	}
}