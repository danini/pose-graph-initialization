#pragma once

#include <map>
#include <string>
#include <Eigen/Core>

namespace reconstruction
{
	typedef size_t CameraId;
	typedef size_t PointId;
	typedef size_t VertexId;
	typedef VertexId ViewId;
	typedef std::unordered_map<std::string, std::string> ViewMetadata;
	typedef std::pair<ViewId, ViewId> EdgeId;

	struct EdgeIdHash
	{
		std::size_t operator() (const std::pair<ViewId, ViewId> &pair) const
		{
			return std::hash<ViewId>()(pair.first) ^ std::hash<ViewId>()(pair.second);
		}
	};

	constexpr double UndefinedCameraParameter = -1.0;
	constexpr size_t UndefinedViewParameter = std::numeric_limits<size_t>::max();
}