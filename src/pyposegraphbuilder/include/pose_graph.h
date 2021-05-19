#pragma once

#include <mutex>
#include <map>
#include <vector>
#include "types.h"
#include "pose.h"
#include <shared_mutex>

namespace reconstruction
{
	class PoseGraphVertex
	{
	public:
		PoseGraphVertex(const ViewId view_id_ = UndefinedViewParameter) :
			view_id(view_id_)
		{

		}

		const ViewId &id() const { return view_id; }
		bool isUndefined() const { return view_id == UndefinedViewParameter; }

	protected:
		ViewId view_id;
	};

	class PoseGraphEdge
	{
	public:
		PoseGraphEdge(
			const ViewId view_id_src_,
			const ViewId view_id_dst_, 
			Pose T_dst_src_,
			const double score_ = 1.0) :
			view_id_src(view_id_src_),
			view_id_dst(view_id_dst_),
			T_dst_src(T_dst_src_),
			score(score_)
		{

		}

		bool isUndefined() const { return view_id_src == UndefinedViewParameter || view_id_dst == UndefinedViewParameter; }

		const ViewId &getSourceId() const { return view_id_src; }
		const ViewId &getDestinationId() const { return view_id_dst; }

		const Pose &getValue() const { return T_dst_src; }
		Pose &getMutableValue() { return T_dst_src; }

		const double &getScore() const { return score; }

	protected:
		const ViewId view_id_src,
			view_id_dst;
		const double score;

		Pose T_dst_src;
	};

	class PoseGraph
	{
	public:
		PoseGraph() : vertex_number(0),
			edge_number(0)
		{

		}

		bool addVertex(const ViewId view_id_);
		bool hasVertex(const ViewId view_id_) const;
		const PoseGraphVertex &getVertexById(const ViewId view_id_) const;

		bool addEdge(const ViewId source_id_,
			const ViewId destination_id_,
			PoseGraphEdge edge_);

		bool hasEdge(const ViewId source_id_,
			const ViewId destination_id_) const;

		const std::vector<EdgeId> &getEdgeIds() const { return edges_ids; }
		const PoseGraphEdge &getEdgeById(const EdgeId &id_) const;
		const bool getEdgesByVertex(
			const ViewId &id_,
			std::vector<EdgeId> &edge_) const;

		const size_t getEdgeNumberByVertex(
			const ViewId &id_) const;

		size_t numVertices() const { return vertex_number; }
		size_t numEdges() const { return edge_number; }

	protected:
		typedef std::shared_mutex Lock;
		mutable Lock edge_lock,
			vertex_lock;

		std::map<ViewId, std::vector<EdgeId>> edges_of_vertices;
		std::map<ViewId, PoseGraphVertex> vertices;
		std::unordered_map<EdgeId, PoseGraphEdge, EdgeIdHash> edges;
		std::vector<EdgeId> edges_ids;

		size_t vertex_number,
			edge_number;
	};


	const size_t PoseGraph::getEdgeNumberByVertex(
		const ViewId &view_id_) const
	{
		if (!hasVertex(view_id_))
			return 0;

		std::shared_lock<Lock> lock(edge_lock);
		const auto &iterator = edges_of_vertices.find(view_id_);
		const size_t edgeNumber = 
			iterator == edges_of_vertices.end() ? 
			0 : iterator->second.size();
		lock.unlock();
		return edgeNumber;
	}

	const PoseGraphVertex &PoseGraph::getVertexById(const ViewId view_id_) const
	{
		// Reading lock
		if (!hasVertex(view_id_))
			return PoseGraphVertex();

		std::shared_lock<Lock> lock(vertex_lock);
		const auto &vertex_iterator = vertices.find(view_id_);
		lock.unlock();
		return vertex_iterator->second;
	}

	const bool PoseGraph::getEdgesByVertex(
		const ViewId &id_, 
		std::vector<EdgeId> &edge_) const
	{
		// Reading lock
		std::shared_lock<Lock> lock(edge_lock);
		const auto &iterator = edges_of_vertices.find(id_);
		lock.unlock();

		if (iterator == edges_of_vertices.end())
			return false;

		edge_ = iterator->second;
		return true;
	}

	const PoseGraphEdge &PoseGraph::getEdgeById(const EdgeId &id_) const
	{
		// Reading lock
		std::shared_lock<Lock> lock(edge_lock);
		const auto edge = edges.find(id_);
		lock.unlock();

		if (edge == edges.end())
		{
			LOG(INFO) << "Edge with id (" << id_.first << ", " << id_.second << ") does not exist.\n";
			return PoseGraphEdge(UndefinedViewParameter, UndefinedViewParameter, Pose());
		}
		return (*edge).second;
	}

	bool PoseGraph::hasVertex(const ViewId view_id_) const
	{
		// Reading lock
		std::shared_lock<Lock> lock(vertex_lock);
		const bool has_vertex = vertices.find(view_id_) != vertices.end();
		lock.unlock();
		return has_vertex;
	}

	bool PoseGraph::hasEdge(const ViewId source_id_,
		const ViewId destination_id_) const
	{
		// Reading lock
		std::shared_lock<Lock> lock(edge_lock);

		const EdgeId edge_id(source_id_, destination_id_);
		const bool has_edge = edges.find(edge_id) != edges.end();
		lock.unlock();
		return has_edge;
	}

	bool PoseGraph::addVertex(const ViewId view_id_)
	{ 
		if (hasVertex(view_id_))
			return false;

		// Writing lock
		std::unique_lock<Lock> lock(vertex_lock);
		++vertex_number;
		vertices.insert({view_id_, PoseGraphVertex(view_id_)});
		lock.unlock();
		return true;
	}

	bool PoseGraph::addEdge(
		const ViewId source_id_,
		const ViewId destination_id_,
		PoseGraphEdge edge_)
	{
		if (!hasVertex(source_id_) || 
			!hasVertex(destination_id_) || 
			hasEdge(source_id_, destination_id_))
			return false;
		
		// Writing lock
		std::unique_lock<Lock> lock(edge_lock);

		++edge_number;
		const EdgeId edge_id(source_id_, destination_id_);
		edges.insert({ edge_id, edge_ });
		edges_ids.push_back(edge_id);

		edges_of_vertices[source_id_].emplace_back(edge_id);
		edges_of_vertices[destination_id_].emplace_back(edge_id);

		lock.unlock();
		return true;
	}

}