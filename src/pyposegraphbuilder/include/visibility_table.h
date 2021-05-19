#pragma once

#include <map>
#include <set>
#include <vector>
#include "types.h"
#include "pose.h"
#include <shared_mutex>

namespace reconstruction
{
	class VisibilityTable
	{
	protected:
		const size_t numVertices;

		typedef std::shared_mutex Lock;
		mutable Lock edge_lock;

		std::map<std::pair<ViewId, ViewId>, bool> visibility_map;
		std::unordered_map<ViewId, std::set<ViewId>> neighbors; // All views which are the neighbor of a particular view

		bool hasLinkOrdered(
			const ViewId &from_,
			const ViewId &to_) const;

	public:

		VisibilityTable(const size_t &numVertices_,
			const size_t &numEdges_) : numVertices(numVertices_)
		{
			neighbors.reserve(numVertices_);
			neighbors.max_load_factor(0.25);
		}

		bool addLink(
			const ViewId &from_,
			const ViewId &to_);

		bool hasLink(
			const ViewId &from_,
			const ViewId &to_) const;
	};

	bool VisibilityTable::addLink(
		const ViewId &from_,
		const ViewId &to_)
	{
		// There should be no link pointing from and to the same vertex
		if (from_ == to_)
			return false;

		// Writing lock
		std::unique_lock<Lock> lock(edge_lock);

		// Select the first and the second indices to put into the map
		// ordered.
		const ViewId from = MIN(from_, to_),
			to = MAX(from_, to_);
		
		// Storing the neighbor
		// TODO: replace with ordered vector and log(n) insert
		if (neighbors[from].find(to) == neighbors[from].end())
			neighbors[from].insert(to);
		if (neighbors[to].find(from) == neighbors[to].end())
			neighbors[to].insert(from);

		// Check if this link has been added already
		if (hasLinkOrdered(from, to))
		{
			lock.unlock();
			return false;
		}

		// Adding the link
		visibility_map[std::make_pair(from, to)] = true;

		// Iterating through the previous neighbors and setting the visibility map
		std::queue<ViewId> views;
		for (const ViewId &view_id : neighbors[from])
			views.emplace(view_id);
		for (const ViewId &view_id : neighbors[to])
			views.emplace(view_id);

		while (!views.empty())
		{
			const ViewId view_id =
				views.front();
			views.pop();

			if (view_id == from || view_id == to)
				continue;

			const ViewId first = MIN(view_id, from),
				second = MAX(view_id, from);

			const ViewId third = MIN(view_id, to),
				fourth = MAX(view_id, to);

			const auto pair1 = std::make_pair(first, second);
			const auto pair2 = std::make_pair(third, third);

			const bool hasPair1 =
				visibility_map.find(pair1) != visibility_map.end();
			const bool hasPair2 =
				visibility_map.find(pair2) != visibility_map.end();

			if (!hasPair1 || !hasPair1)
			{
				visibility_map[pair1] = true;
				visibility_map[pair2] = true;

				for (const ViewId &view_id : neighbors[view_id])
					views.emplace(view_id);
			}

			neighbors[view_id].insert(from);
			neighbors[view_id].insert(to);
		}

		/*for (const ViewId &view_id : neighbors[from])
		{
			const ViewId first = MIN(view_id, from),
				second = MAX(view_id, from);
			visibility_map[std::make_pair(first, second)] = true;

			const ViewId third = MIN(view_id, to),
				fourth = MAX(view_id, to);
			visibility_map[std::make_pair(third, fourth)] = true;
		}

		for (const ViewId &view_id : neighbors[to])
		{
			const ViewId first = MIN(view_id, to),
				second = MAX(view_id, to);
			visibility_map[std::make_pair(first, second)] = true;

			const ViewId third = MIN(view_id, to),
				fourth = MAX(view_id, to);
			visibility_map[std::make_pair(third, fourth)] = true;
		}*/

		lock.unlock();
		return true;
	}

	bool VisibilityTable::hasLink(
		const ViewId &from_,
		const ViewId &to_) const
	{
		// There should be no link pointing from and to the same vertex
		if (from_ == to_)
			return false;
		
		// Select the first and the second indices to put into the map
		// ordered.
		const ViewId from = MIN(from_, to_),
			to = MAX(from_, to_);

		std::shared_lock<Lock> lock(edge_lock);
		const bool has_link = hasLinkOrdered(from, to);
		lock.unlock();
		return has_link;
	}

	bool VisibilityTable::hasLinkOrdered(
		const ViewId &from_,
		const ViewId &to_) const
	{
		return visibility_map.find(std::make_pair(from_, to_)) != visibility_map.end();
	}
}