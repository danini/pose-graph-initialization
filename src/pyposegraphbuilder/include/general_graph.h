#pragma once

#include <map>
#include <vector>
#include "types.h"
#include "pose.h"

namespace reconstruction
{
	class GraphVertex
	{
	protected:
		VertexId vertex_id;
	};

	template <typename EdgeValue_>
	class DirectedGraphEdge
	{
	protected:
		size_t source_vertex_id,
			destination_vertex_id;

		EdgeValue_ value;
	};

	template <typename EdgeValue_>
	class GeneralDirectedGraph
	{

	protected:
		std::unordered_map<VertexId, GraphVertex> vertices;
		std::unordered_map<EdgeId, DirectedGraphEdge<EdgeValue_>, EdgeIdHash> edges;
		std::vector<EdgeId> edges_ids;

		size_t vertex_number,
			edge_number;

		bool addVertex(const VertexId vertex_id_);
		bool hasVertex(const VertexId vertex_id_) const;

		bool addEdge(const VertexId source_id_,
			const VertexId destination_id_,
			DirectedGraphEdge<EdgeValue_> edge_);

		bool hasEdge(const VertexId source_id_,
			const VertexId destination_id_) const;

		const std::vector<EdgeId> &getEdgeIds() const { return edges_ids; }
		const DirectedGraphEdge<EdgeValue_> &getEdgeById(const EdgeId &id_) const;

		size_t numVertices() { return vertex_number; }
		size_t numEdges() { return edge_number; }

	};

	template <typename EdgeValue_>
	const DirectedGraphEdge<EdgeValue_> &GeneralDirectedGraph<EdgeValue_>::getEdgeById(const EdgeId &id_) const
	{
		const auto edge = edges.find(id_);
		if (edge == edges.end())
		{
			LOG(INFO) << "Edge with id (" << id_.first << ", " << id_.second << ") does not exist.\n";
			return PoseGraphEdge(UndefinedViewParameter, UndefinedViewParameter, Pose());
		}
		return (*edge).second;
	}

	template <typename EdgeValue_>
	bool GeneralDirectedGraph<EdgeValue_>::hasVertex(const VertexId view_id_) const
	{
		return vertices.find(view_id_) != vertices.end();
	}

	template <typename EdgeValue_>
	bool GeneralDirectedGraph<EdgeValue_>::hasEdge(const VertexId source_id_,
		const VertexId destination_id_) const
	{
		const EdgeId edge_id(source_id_, destination_id_);
		return edges.find(edge_id) != edges.end();
	}

	template <typename EdgeValue_>
	bool GeneralDirectedGraph<EdgeValue_>::addVertex(const VertexId view_id_)
	{
		if (hasVertex(view_id_))
			return false;

		++vertex_number;
		vertices.insert({ view_id_, PoseGraphVertex(view_id_) });
		return true;
	}

	template <typename EdgeValue_>
	bool GeneralDirectedGraph<EdgeValue_>::addEdge(
		const VertexId source_id_,
		const VertexId destination_id_,
		DirectedGraphEdge<EdgeValue_> edge_)
	{
		if (!hasVertex(source_id_) ||
			!hasVertex(destination_id_) ||
			hasEdge(source_id_, destination_id_))
			return false;

		++edge_number;
		const EdgeId edge_id(source_id_, destination_id_);
		edges.insert({ edge_id, edge_ });
		edges_ids.push_back(edge_id);
		return true;
	}
}