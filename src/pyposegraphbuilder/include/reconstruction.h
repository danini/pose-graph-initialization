#pragma once

#include <map>
#include <vector>

#include "pinhole_camera.h"
#include "view.h"
#include "types.h"
#include <glog/logging.h>

namespace reconstruction
{
	class Reconstruction
	{
	public:
		bool addCamera(CameraId camera_id_);
		bool addCamera(CameraId camera_id_,
			double focal_length_x_,
			double focal_length_y_,
			double principal_point_x_,
			double principal_point_y_);

		bool addView(const CameraId camera_id_,
			const ViewId view_id_);

		const PinholeCamera &getCamera(CameraId camera_id_) const;
		PinholeCamera &getMutableCamera(CameraId camera_id_);
		const size_t getViewNumber() const { return views.size(); }
		const std::vector<CameraId> &getCameraIds() const { return camera_ids; }

		const View &getView(const ViewId view_id_) const;
		View &getMutableView(const ViewId view_id_);
		const std::vector<ViewId> &getViewIds() const { return view_ids; }

	protected:
		std::unordered_map<CameraId, PinholeCamera> cameras;
		std::unordered_map<ViewId, View> views;

		std::vector<CameraId> camera_ids;
		std::vector<ViewId> view_ids;
	};

	const PinholeCamera &Reconstruction::getCamera(CameraId camera_id_) const
	{
		if (cameras.find(camera_id_) == cameras.end())
		{
			static PinholeCamera undefined_camera;
			return undefined_camera;
		}
		return cameras.at(camera_id_);
	}

	PinholeCamera &Reconstruction::getMutableCamera(CameraId camera_id_)
	{
		if (cameras.find(camera_id_) == cameras.end())
		{
			static PinholeCamera undefined_camera;
			return undefined_camera;
		}
		return cameras.at(camera_id_);
	}

	const View &Reconstruction::getView(const ViewId view_id_) const
	{
		if (views.find(view_id_) == views.end())
		{
			LOG(WARNING) << "View " << view_id_ << " is not found\n";
			static View undefined_view = View(UndefinedViewParameter, UndefinedViewParameter);
			return undefined_view;
		}
		return views.at(view_id_);
	}

	View &Reconstruction::getMutableView(const ViewId view_id_)
	{
		if (views.find(view_id_) == views.end())
		{
			LOG(WARNING) << "View " << view_id_ << " is not found\n";
			static View undefined_view = View(UndefinedViewParameter, UndefinedViewParameter);
			return undefined_view;
		}
		return views.at(view_id_);
	}

	bool Reconstruction::addView(const CameraId camera_id_,
		const ViewId view_id_)
	{
		if (views.find(view_id_) != views.end())
			return false;

		view_ids.push_back(view_id_);
		views.insert({ view_id_,
			View(camera_id_, view_id_) });

		return true;
	}

	bool Reconstruction::addCamera(CameraId camera_id_)
	{
		if (cameras.find(camera_id_) != cameras.end())
			return false;

		camera_ids.push_back(camera_id_);

		cameras.insert({ camera_id_ , PinholeCamera(
			UndefinedCameraParameter,
			UndefinedCameraParameter,
			UndefinedCameraParameter,
			UndefinedCameraParameter) });

		return true;
	}

	bool Reconstruction::addCamera(CameraId camera_id_,
		double focal_length_x_,
		double focal_length_y_,
		double principal_point_x_,
		double principal_point_y_)
	{
		if (cameras.find(camera_id_) != cameras.end())
			return false;

		camera_ids.push_back(camera_id_);
		cameras.insert({ camera_id_ , PinholeCamera(
			focal_length_x_,
			focal_length_y_,
			principal_point_x_,
			principal_point_y_) });

		return true;
	}
}
