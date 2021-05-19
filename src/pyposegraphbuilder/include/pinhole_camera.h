#pragma once

#include <Eigen/Core>
#include "types.h"

namespace reconstruction
{
	class PinholeCamera
	{
	public:
		PinholeCamera() : PinholeCamera(
			UndefinedCameraParameter,
			UndefinedCameraParameter,
			UndefinedCameraParameter,
			UndefinedCameraParameter)
		{

		}

		PinholeCamera(const double focal_length_x_,
			const double focal_length_y_,
			const double principal_point_x_,
			const double principal_point_y_)
		{
			intrinsic_parameters << focal_length_x_, 0, principal_point_x_,
				0, focal_length_y_, principal_point_y_,
				0, 0, 1;
		}

		void setIntrinsics(const double focal_length_x_,
			const double focal_length_y_,
			const double principal_point_x_,
			const double principal_point_y_)
		{
			intrinsic_parameters << focal_length_x_, 0, principal_point_x_,
				0, focal_length_y_, principal_point_y_,
				0, 0, 1;
		}

		void setDimensions(double width_,
			double height_)
		{
			setWidth(width_);
			setHeight(height_);
		}

		void setWidth(double width_)
		{
			width = width_;
		}

		void setHeight(double height_)
		{
			height = height_;
		}

		double getWidth() const
		{
			return width;
		}

		double getHeight() const
		{
			return height;
		}

		const Eigen::Matrix3d &getIntrinsics() const
		{
			return intrinsic_parameters;
		}

	protected:
		Eigen::Matrix3d intrinsic_parameters;
		Eigen::Matrix<double, 2, 1> radial_parameters;
		double width,
			height;
	};
}