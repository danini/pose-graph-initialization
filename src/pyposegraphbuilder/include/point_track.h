#pragma once

#include <glog/logging.h>
#include <fstream>
#include <memory>
#include <string>
#include <set>
#include <map>
#include <tuple>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <stack>
#include <shared_mutex>

#include <Eigen/Core>

namespace reconstruction
{
	class PointTrack
	{
	protected:
		struct Node
		{
			ViewId view_idx;
			PointId point_idx;

			Node() :
				view_idx(0),
				point_idx(0)
			{
			}

			Node(const ViewId &view_idx_,
				const PointId &point_idx_) :
				view_idx(view_idx_),
				point_idx(point_idx_)
			{

			}

			bool operator <(const Node& node) const
			{
				return (view_idx < node.view_idx) ||
					(view_idx == node.view_idx &&
						point_idx < node.point_idx);
			}
		};

		//std::map<size_t, Node, std::less<Node>> track;
		std::map<ViewId, Node> view_to_point;
		std::map<std::pair<ViewId, ViewId>, double> inlier_probabilities;
		double combined_probability;

	public:
		PointTrack() : combined_probability(1.0)
		{

		}

		size_t getPointNumber() const
		{
			return view_to_point.size();
		}

		const std::map<ViewId, Node> &getDataAsMap() const
		{
			return view_to_point;
		}

		double getInlierProbability() const
		{
			return combined_probability;
		}

		void setCombinedProbability(const double combined_probability_)
		{
			combined_probability =
				combined_probability_;
		}

		PointId getPointId(const ViewId view_idx_) const
		{
			if (!hasView(view_idx_))
			{
				LOG(WARNING) << "View " << view_idx_ << " does not exist.";
				return std::numeric_limits<PointId>::max();
			}

			auto it = view_to_point.find(view_idx_);
			return it->second.point_idx;
		}

		void add(
			const ViewId &view_idx_,
			const PointId &point_idx_)
		{
			if (hasView(view_idx_))
			{
				LOG(WARNING) << "View " << view_idx_ << " had already been added to this track.";
				return;
			}

			view_to_point[view_idx_] = Node(view_idx_, point_idx_);
		}

		void setInlierProbability(
			const ViewId &view_idx_1_,
			const ViewId &view_idx_2_,
			const double &inlier_probability_)
		{
			if (!hasView(view_idx_1_))
			{
				LOG(WARNING) << "View " << view_idx_1_ << " does not exist.";
				return;
			}

			if (!hasView(view_idx_2_))
			{
				LOG(WARNING) << "View " << view_idx_2_ << " does not exist.";
				return;
			}

			const auto view_pair = std::make_pair(
				MIN(view_idx_1_, view_idx_2_),
				MAX(view_idx_1_, view_idx_2_));

			if (inlier_probabilities.find(view_pair) == inlier_probabilities.end())
			{
				inlier_probabilities[view_pair] = inlier_probability_;
				combined_probability += inlier_probability_;
			}
			else
			{
				combined_probability = combined_probability - inlier_probabilities[view_pair] + inlier_probability_;
				inlier_probabilities[view_pair] = inlier_probability_;
			}
		}

		bool hasView(const ViewId &view_idx_) const
		{
			if (view_to_point.size() == 0)
				return false;
			return view_to_point.find(view_idx_) != view_to_point.end();
		}

		void unite(const PointTrack &track_)
		{
			for (const auto &[view_idx, node] : track_.getDataAsMap())
			{
				if (hasView(view_idx))
					continue;

				add(view_idx,
					node.point_idx);
			}
		}

		std::tuple<bool, bool> shouldUnite(const PointTrack &track_) const
		{
			bool should_unite = false,
				corrupted = false;

			for (const auto &[view_idx, node] : view_to_point)
			{
				if (!track_.hasView(view_idx))
					continue;

				const size_t &point_idx_1 = getPointId(view_idx);
				const size_t &point_idx_2 = track_.getPointId(view_idx);

				if (point_idx_1 == point_idx_2)
				{
					should_unite = true;
					break;
				}
			}

			if (should_unite)
			{
				for (const auto &[view_idx, node] : view_to_point)
				{
					if (!track_.hasView(view_idx))
						continue;

					const size_t &point_idx_1 = getPointId(view_idx);
					const size_t &point_idx_2 = track_.getPointId(view_idx);

					if (point_idx_1 != point_idx_2)
						corrupted = true;
				}
			}

			return std::make_tuple(should_unite, corrupted);
		}
	};

	class PointTracks
	{
	protected:
		typedef std::shared_mutex Lock;
		mutable Lock reader_writer_lock;

		std::vector<PointTrack> tracks;
		std::set<ViewId> views;
		std::map<ViewId, std::vector<size_t>> view_to_tracks;
		std::map<std::pair<ViewId, PointId>, size_t> view_pair_to_tracks;
		std::mutex writing_mutex;

	public:

		size_t getTrackNumber() const
		{
			return tracks.size();
		}

		bool hasView(const ViewId &view_id_) const
		{
			std::shared_lock<Lock> lock(reader_writer_lock);
			bool answer = views.find(view_id_) != views.end();
			lock.unlock();
			return answer;
		}

		void getInlierProbabilities(
			const ViewId &view_id_,
			std::map<PointId, double> &inlier_probabilities_) const
		{
			std::shared_lock<Lock> lock(reader_writer_lock);
			if (view_to_tracks.find(view_id_) == view_to_tracks.end())
			{
				lock.unlock();
				LOG(WARNING) << "View " << view_id_ << " has not been added yet to the tracks.";
				return;
			}

			const auto &tracks_with_view = view_to_tracks.find(view_id_)->second;
			for (const size_t &track_idx : tracks_with_view)
			{
				if (track_idx > tracks.size())
				{
					printf("Happened");
					continue;
				}
				const auto &track = tracks[track_idx];

				const double &probability = 
					track.getInlierProbability();
				const PointId &point_idx =
					track.getPointId(view_id_);

				inlier_probabilities_[point_idx] = probability;
			}
			lock.unlock();
		}

		void add(const PointTrack track_)
		{
			std::unique_lock<Lock> lock(reader_writer_lock);

			const size_t track_idx = tracks.size();

			for (const auto &[viewIdx, pointIdx] : track_.getDataAsMap())
			{
				views.insert(viewIdx);
				view_to_tracks[viewIdx].emplace_back(track_idx);
			}

			tracks.emplace_back(track_);
			lock.unlock();
		}

		const PointTrack &getTrack(const size_t &track_idx_) const
		{
			static const PointTrack emptyTrack;

			if (track_idx_ >= tracks.size())
				return emptyTrack;

			return tracks[track_idx_];
		}

		PointTrack &getMutableTrack(const size_t &track_idx_)
		{
			static PointTrack emptyTrack;

			if (track_idx_ >= tracks.size())
				return emptyTrack;

			return tracks[track_idx_];
		}

		const std::vector<size_t> &getTrackIds(const ViewId &view_idx_)
		{
			static const std::vector<size_t> emptyTracks;
			
			const auto it = view_to_tracks.find(view_idx_);

			if (it == view_to_tracks.end())
				return emptyTracks;

			return it->second;
		}

		std::vector<size_t> getMutualTracks(
			const ViewId &view_idx_1_,
			const ViewId &view_idx_2_,
			const bool ordered_ = false)
		{
			std::vector<size_t> tracks,
				tracks1 = getTrackIds(view_idx_1_),
				tracks2 = getTrackIds(view_idx_2_);

			std::set<size_t> track_frequency;
		
			for (const size_t &track_idx : tracks1)
				track_frequency.insert(track_idx);

			if (ordered_)
			{
				std::priority_queue<std::pair<double, size_t>> priority_queue;

				for (const size_t &track_idx : tracks2)
					if (track_frequency.find(track_idx) != track_frequency.end())
						priority_queue.emplace(std::make_pair(this->tracks[track_idx].getInlierProbability(), track_idx));

				tracks.reserve(priority_queue.size());
				size_t idx = 0;
				while (!priority_queue.empty())
				{
					tracks.emplace_back(priority_queue.top().second);
					priority_queue.pop();
				}
	
			}
			else
			{
				tracks.reserve(
					MIN(tracks1.size(), tracks2.size()));

				for (const size_t &track_idx : tracks2)
					if (track_frequency.find(track_idx) != track_frequency.end())
						tracks.emplace_back(track_idx);
			}

			return tracks;
		}

		void add(
			const ViewId &view_idx_1_,
			const PointId &point_idx_1_,
			const ViewId &view_idx_2_,
			const PointId &point_idx_2_,
			const double &inlier_probability_)
		{
			std::unique_lock<Lock> lock(reader_writer_lock);
			views.insert(view_idx_1_);
			views.insert(view_idx_2_);

			std::pair<ViewId, PointId> view_point_pair_1 =
				std::make_pair(view_idx_1_, point_idx_1_);
			std::pair<ViewId, PointId> view_point_pair_2 =
				std::make_pair(view_idx_2_, point_idx_2_);

			const bool has_view_1 =
				hasPoint(view_point_pair_1);
			const bool has_view_2 =
				hasPoint(view_point_pair_2);

			if (has_view_1 && has_view_2)
			{
				lock.unlock();
				LOG(WARNING) << "Both view and point pairs (" << view_idx_1_ << ", " << point_idx_1_ << ") (" <<
					view_idx_2_ << ", " << point_idx_2_ << ") had been added earlier.";
				// TODO: check if they belong to the same Track
				return;
			}

			if (has_view_1 || has_view_2)
			{
				std::pair<ViewId, PointId> existing_view_point_pair,
					new_view_point_pair;

				ViewId existing_view_idx,
					new_view_idx;
				PointId existing_point_idx,
					new_point_idx;

				if (has_view_1)
				{
					existing_view_point_pair = view_point_pair_1;
					new_view_point_pair = view_point_pair_2;
				}
				else
				{
					existing_view_point_pair = view_point_pair_2;
					new_view_point_pair = view_point_pair_1;
				}
				
				const size_t &track_idx = view_pair_to_tracks[existing_view_point_pair];
				tracks[track_idx].add(new_view_point_pair.first, 
					new_view_point_pair.second);
				tracks[track_idx].setInlierProbability(
					existing_view_point_pair.first,
					new_view_point_pair.first,
					inlier_probability_);
				view_to_tracks[new_view_point_pair.first].emplace_back(track_idx);
			}
			else
			{
				addTrack(view_point_pair_1,
					view_point_pair_2,
					inlier_probability_);
			}
			lock.unlock();
		}

		void add(
			const ViewId &view_idx_1_,
			const ViewId &view_idx_2_,
			const std::vector<std::pair<PointId, PointId>> &point_indices_,
			const std::vector<double> &inlier_probabilities_)
		{
			for (size_t pair_idx = 0; pair_idx < point_indices_.size(); ++pair_idx)
				add(view_idx_1_, 
					point_indices_[pair_idx].first, 
					view_idx_2_, 
					point_indices_[pair_idx].second, 
					inlier_probabilities_[pair_idx]);
		}

		void addTrack(const std::pair<ViewId, PointId> &view_point_pair_1_,
			const std::pair<ViewId, PointId> &view_point_pair_2_,
			const double inlier_probability_)
		{
			const size_t track_idx = tracks.size();

			view_pair_to_tracks[view_point_pair_1_] = track_idx;
			view_pair_to_tracks[view_point_pair_2_] = track_idx;

			view_to_tracks[view_point_pair_1_.first].emplace_back(track_idx);
			view_to_tracks[view_point_pair_2_.first].emplace_back(track_idx);

			tracks.emplace_back(PointTrack());
			PointTrack &track = tracks.back();

			track.add(view_point_pair_1_.first,
				view_point_pair_1_.second);
			track.add(view_point_pair_2_.first,
				view_point_pair_2_.second);

			track.setInlierProbability(view_point_pair_1_.first,
				view_point_pair_2_.first,
				inlier_probability_);
		}

		void setInlierProbability(
			const ViewId &view_idx_1_,
			const PointId &point_idx_1_,
			const ViewId &view_idx_2_,
			const PointId &point_idx_2_,
			const double &probability_)
		{
			std::pair<ViewId, PointId> view_point_pair_1 =
				std::make_pair(view_idx_1_, point_idx_1_);
			std::pair<ViewId, PointId> view_point_pair_2 =
				std::make_pair(view_idx_2_, point_idx_2_);

			const bool has_view_1 =
				hasPoint(view_point_pair_1);
			const bool has_view_2 =
				hasPoint(view_point_pair_2);

			if (!has_view_1 || !has_view_2)
			{
				if (!has_view_1)
					LOG(WARNING) << "(" << view_idx_1_ << ", " << point_idx_1_ << ") view/point pair does not exist. Its inlier ratio cannot be set.";
				if (!has_view_2)
					LOG(WARNING) << "(" << view_idx_2_ << ", " << point_idx_2_ << ") view/point pair does not exist. Its inlier ratio cannot be set.";
				return;
			}

			const size_t &track_idx_1 = view_pair_to_tracks[view_point_pair_1];
			const size_t &track_idx_2 = view_pair_to_tracks[view_point_pair_2];

			if (track_idx_1 != track_idx_2)
			{
				LOG(WARNING) << "The track indices to which pairs (" << view_idx_1_ << ", " << point_idx_1_ << 
					") and (" << view_idx_2_ << ", " << point_idx_2_ << ") points to are different.";
				return;
			}

			tracks[track_idx_1].setInlierProbability(
				view_idx_1_, 
				view_idx_2_, 
				probability_);

		}

		void setInlierProbabilies(
			const ViewId &view_idx_1_,
			const ViewId &view_idx_2_,
			const std::vector<std::pair<PointId, PointId>> &point_indices_,
			const std::vector<double> &inlier_probabilities_)
		{
			if (point_indices_.size() != inlier_probabilities_.size())
			{
				LOG(WARNING) << "The sizes of the vectors are different";
				return;
 			}

			writing_mutex.lock();
			for (size_t corr_idx = 0; corr_idx < point_indices_.size(); ++corr_idx)
				setInlierProbability(view_idx_1_, 
					point_indices_[corr_idx].first, 
					view_idx_2_, 
					point_indices_[corr_idx].second,
					inlier_probabilities_[corr_idx]);
			writing_mutex.unlock();
		}
		
		bool hasPoint(const std::pair<ViewId, ViewId> &view_point_pair_) const
		{
			return view_pair_to_tracks.find(view_point_pair_) != view_pair_to_tracks.end();
		}

		bool hasPoint(const ViewId &view_idx_,
			const PointId &point_idx_) const
		{
			return hasPoint(std::make_pair(view_idx_, point_idx_));
		}
	};
	
	size_t pairHash(const std::pair<size_t, size_t> & pair_)
	{
		return pair_.first * 8001 + pair_.second;
	}

	class Tracklets
	{
	public:
		typedef std::shared_mutex Lock;
		mutable Lock reader_writer_lock;

		// Pair of (image index, point index)
		typedef std::pair<size_t, size_t> Pair;

		std::map<size_t, std::vector<size_t>> viewToPairs;
		size_t pointPairNumber;
		std::vector<size_t> parents, treeSizes;
		std::vector<std::pair<size_t, size_t>> allPairs;

		std::unordered_map<Pair, size_t, decltype(&pairHash)> pointPairs;
		std::vector<std::vector<Pair>> tmpTracks;
		std::unordered_map<size_t, std::vector<size_t>> tmpViewToTracks;
		std::unordered_map<size_t, std::vector<size_t>> tmpPairToTracks;

		Tracklets(size_t viewNumber_) : pointPairNumber(0)
		{
			tmpViewToTracks.reserve(viewNumber_);
			tmpPairToTracks.reserve(viewNumber_ * 8001);

			pointPairs = std::unordered_map<Pair, size_t, decltype(&pairHash)>(viewNumber_ * 8001, pairHash);
		}

		void getCorrespondences(
			std::vector<std::tuple<size_t, size_t, double>> &matches_,
			const size_t &viewIdSource_,
			const size_t &viewIdDestination_,
			const size_t &maximumCorrespondenceNumber_) const
		{
			std::shared_lock<Lock> lock(reader_writer_lock);

			const auto it = tmpViewToTracks.find(viewIdSource_);

			if (it == tmpViewToTracks.end())
			{
				lock.unlock();
				return;
			}

			const auto jt = tmpViewToTracks.find(viewIdDestination_);

			if (jt == tmpViewToTracks.end())
			{
				lock.unlock();
				return;
			}

			const auto &tracksSrc = it->second;
			const auto &tracksDst = jt->second;
			matches_.reserve(maximumCorrespondenceNumber_);

			std::unordered_set<size_t> trackMap;
			trackMap.reserve(tmpTracks.size());

			for (const auto &trackIdx : tracksSrc)
				trackMap.insert(trackIdx);

			for (const auto &trackIdx : tracksDst)
			{
				if (auto it = trackMap.find(trackIdx); it != trackMap.end())
				{
					std::tuple<size_t, size_t, double> pair;
					int cnt = 0;

					for (const auto &p : tmpTracks[trackIdx])
					{
						if (p.first == viewIdSource_)
						{
							std::get<0>(pair) = p.second;
							++cnt;
						} else if (p.first == viewIdDestination_)
						{
							std::get<1>(pair) = p.second;
							++cnt;
						}

						if (cnt == 2)
							break;
					}	

					matches_.emplace_back(pair);

					if (matches_.size() > maximumCorrespondenceNumber_)
						break;
				}
			}

			lock.unlock();
		}

		void add(
			const size_t &imageIdxSource_,
			const size_t &imageIdxDestination_,
			const std::vector<std::tuple<size_t, size_t, double>> &matches_,
			const std::vector<uchar> &inlierMask_)
		{
			std::unique_lock<Lock> lock(reader_writer_lock);

			Pair pairSource =
				std::make_pair(imageIdxSource_, 0);
			Pair pairDestination =
				std::make_pair(imageIdxDestination_, 0);

			//for (const auto &idx : indices_)
			for (size_t pointIdx = 0; pointIdx < matches_.size(); ++pointIdx)
			{
				if (!inlierMask_[pointIdx])
					continue;

				pairSource.second = std::get<0>(matches_[pointIdx]);
				pairDestination.second = std::get<1>(matches_[pointIdx]);

				// Converting the pairs to indices
				auto& it = pointPairs[pairSource];
				if (it == 0)
					it = pointPairNumber++;

				auto& jt = pointPairs[pairDestination];
				if (jt == 0)
					jt = pointPairNumber++;

				auto &tracksSource = tmpPairToTracks[it];
				auto &tracksDestination = tmpPairToTracks[jt];
				const size_t trackNumDestination = tracksDestination.size();
				bool added = false;

				for (const auto &trackIdx : tracksSource)
				{
					auto &track = tmpTracks[trackIdx];

					if (std::find(track.begin(), track.end(), pairDestination) != track.end())
						continue;

					tmpViewToTracks[imageIdxDestination_].emplace_back(trackIdx);
					track.emplace_back(pairDestination);
					tracksDestination.emplace_back(trackIdx);
					added = true;
				}

				for (size_t trackIdxIdx = 0; trackIdxIdx < trackNumDestination; ++trackIdxIdx)
				{
					const auto &trackIdx = tracksDestination[trackIdxIdx];
					auto &track = tmpTracks[trackIdx];

					if (std::find(track.begin(), track.end(), pairSource) != track.end())
						continue;

					tmpViewToTracks[imageIdxSource_].emplace_back(trackIdx);
					track.emplace_back(pairSource);
					tracksSource.emplace_back(trackIdx);
					added = true;
				}

				if (!added)
				{
					const size_t idx = tmpTracks.size();
					tmpTracks.emplace_back(std::vector<Pair>{ pairSource, pairDestination });
					tmpViewToTracks[imageIdxSource_].emplace_back(idx);
					tmpViewToTracks[imageIdxDestination_].emplace_back(idx);
					tracksSource.emplace_back(idx);
					tracksDestination.emplace_back(idx);
				}
			}

			lock.unlock();

			return;

			//std::unique_lock<Lock> lock(reader_writer_lock);


			//parents.reserve(parents.size() + indices_.size());
			//treeSizes.reserve(treeSizes.size() + indices_.size());

			/*for (const auto &idx : indices_)
			{
				pairSource.second = indexPairs_[idx].first;
				pairDestination.second = indexPairs_[idx].second;

				// Converting the pairs to indices
				auto& it = pointPairs[pairSource];
				if (it == 0)
				{
					it = pointPairNumber++;
					//parents.emplace_back(it);
					//treeSizes.emplace_back(1);
				}

				auto& jt = pointPairs[pairDestination];
				if (jt == 0)
				{
					jt = pointPairNumber++;
					//parents.emplace_back(jt);
					//treeSizes.emplace_back(1);
				}

				allPairs.emplace_back(std::pair<size_t, size_t>(it, jt));
			}

			parents = std::vector<size_t>(pointPairNumber);
			std::iota(parents.begin(), parents.end(), 0);

			treeSizes.clear();
			treeSizes = std::vector<size_t>(pointPairNumber, 1);

			for (const auto& pair : allPairs) {
				Union(pair.first, pair.second);
			}

			for (size_t k = 0; k < pointPairNumber; ++k) {
				Find(k);
			}

			lock.unlock();*/
		}


		/*void add(
			const size_t imageIdxSource_,
			const size_t pointIdxSource_,
			const size_t imageIdxDestination_,
			const size_t pointIdxDestination_)
		{
			std::unique_lock<Lock> lock(reader_writer_lock);

			const auto pairSource =
				std::make_pair(imageIdxSource_, pointIdxSource_);
			const auto pairDestination =
				std::make_pair(imageIdxDestination_, pointIdxDestination_);
			
			// Converting the pairs to indices
			auto& it = pointPairs[pairSource];
			if (it == 0)
			{
				it = pointPairNumber++;
				parents.emplace_back(it);
				treeSizes.emplace_back(1);
				viewToPairs[imageIdxSource_].emplace_back(it);
			}
				
			auto& jt = pointPairs[pairDestination];
			if (jt == 0)
			{
				jt = pointPairNumber++;
				parents.emplace_back(jt);
				treeSizes.emplace_back(1);
				viewToPairs[imageIdxDestination_].emplace_back(jt);
			}

			Union(it, jt);

			for (size_t k = 0; k < pointPairNumber; ++k) {
				Find(k);
			}

			lock.unlock();
		}*/
	};

	struct TracksBuilder
	{
		typedef std::pair<size_t, size_t> ImagePairs;
		typedef std::pair<size_t, size_t> PointPairs;

		void load(const std::string &filename_,
			PointTracks &tracks_)
		{
			std::ifstream file(filename_);

			if (!file.is_open())
			{
				LOG(WARNING) << "File '" << filename_ << "' cannot be opened.";
				return;
			}

			int number = 0;
			file >> number;
			
			std::string line; 
			std::getline(file, line);
			while (std::getline(file, line))
			{
				PointTrack track;
				std::istringstream iss(line);

				int pair_number;
				iss >> pair_number;
				size_t imageIdx,
					pointIdx;

				while (iss >> imageIdx >> pointIdx)
					track.add(imageIdx, pointIdx);
			
				track.setCombinedProbability(static_cast<double>(track.getPointNumber()));
				tracks_.add(track);
			}

			LOG(INFO) << tracks_.getTrackNumber() << " tracks are loaded.";

			file.close();
		}

		void build(std::vector<PointTrack> &tracks_,
			const std::vector<std::pair<ImagePairs, PointPairs>> &correspondences_)
		{
			tracks_.reserve(correspondences_.size());
			std::map<size_t, std::vector<size_t>> view_to_track,
				tmp_view_to_track;
			
			for (const auto &corr : correspondences_)
			{
				PointTrack track;
				track.add(corr.first.first,
					corr.second.first);
				track.add(corr.first.second,
					corr.second.second);

				view_to_track[corr.first.first].emplace_back(tracks_.size());
				view_to_track[corr.first.second].emplace_back(tracks_.size());

				tracks_.emplace_back(track);
			}

			std::vector<PointTrack> tmp_tracks;
			bool changed;
			do
			{
				changed = false;
				std::vector<bool> used(tracks_.size(), false);

#pragma omp parallel for num_threads(11)
				for (int i = 0; i < tracks_.size(); ++i)
				{
					if (used[i])
						continue;

					static std::mutex mm;
					
					mm.lock();
					used[i] = true;
					int new_idx = tmp_tracks.size();
					tmp_tracks.emplace_back(tracks_[i]);
					PointTrack &track_i = tmp_tracks[new_idx];
					mm.unlock();
					std::priority_queue<int> to_delete;

					for (const auto &[view_idx, node] : track_i.getDataAsMap())
					{
						mm.lock();
						tmp_view_to_track[view_idx].emplace_back(new_idx);
						mm.unlock();

						for (const auto &track_idx : view_to_track[view_idx])
						{
							if (i == track_idx ||
								used[track_idx])
								continue;

							PointTrack &track_j = tracks_[track_idx];

							const auto &[should_unite, corrupted] =
								track_i.shouldUnite(track_j);

							if (should_unite)
							{
								mm.lock();
								used[track_idx] = true;

								if (!corrupted)
								{
									track_i.unite(track_j);

									for (const auto &[loc_view_idx, loc_node] : track_j.getDataAsMap())
										if (loc_view_idx != view_idx)
											tmp_view_to_track[loc_view_idx].emplace_back(new_idx);

									LOG(INFO) << "Tracks " << i << " and " << track_idx << " are united.";
								}
								changed = true;
								mm.unlock();
							}
						}
					}
				}

				tmp_tracks.swap(tracks_);
				tmp_tracks.clear();
				tmp_view_to_track.swap(view_to_track);
				tmp_view_to_track.clear();

			} while (changed);

					/*size_t view_idx_1 = 0;


#pragma omp parallel for num_threads(11)
					for (int j = tracks_.size() - 1; j >= i + 1; --j)
					{
						PointTrack &track_j = tracks_[j];
						
						const auto &[should_unite, corrupted] =
							track_i.shouldUnite(track_j);

						if (should_unite)
						{
							static std::mutex mm;

							mm.lock();
							if (!corrupted)
							{
								track_i.unite(track_j);
								LOG(INFO) << "Tracks " << i << " and " << j << " are united.";
							}
							to_delete.emplace(j);
							changed = true;
							mm.unlock();
						}
					}

					while (!to_delete.empty())
					{
						tracks_.erase(tracks_.begin() + to_delete.top());
						to_delete.pop();
					}

				}

			} while (changed);*/
			
		}
	};

	void filterMutualTracks(
		const ViewId view_id_source_,
		const ViewId view_id_destination_,
		const cv::Mat &correspondences_,
		const std::vector<std::pair<size_t, size_t>> &index_pairs_,
		std::vector<size_t> &mutual_track_ids_,
		const reconstruction::PointTracks &tracks_,
		cv::Mat &filtered_correspondences_,
		std::vector<std::pair<size_t, size_t>> &filtered_index_pairs_,
		bool add_duplicates_ = true)
	{
		filtered_correspondences_.create(0, 4, CV_64F);
		filtered_index_pairs_.reserve(mutual_track_ids_.size());
		std::vector<size_t> toDelete;
		std::vector<bool> pointMask(correspondences_.rows, false);
		size_t localTrackIdx = 0;

		for (const auto &track_idx : mutual_track_ids_)
		{
			const auto &trackMap = tracks_.getTrack(track_idx).getDataAsMap();
			size_t point_idx_1, point_idx_2;
			int count = 0;
			for (const auto &[view_idx, track_node] : trackMap)
			{
				if (view_idx == view_id_source_)
				{
					point_idx_1 = track_node.point_idx;
					++count;

					if (count == 2)
						break;
				}

				if (view_idx == view_id_destination_)
				{
					point_idx_2 = track_node.point_idx;
					++count;

					if (count == 2)
						break;
				}
			}

			if (count == 2)
			{
				const auto it = std::find(index_pairs_.begin(), index_pairs_.end(), std::make_pair(point_idx_1, point_idx_2));

				if (it == index_pairs_.end())
					toDelete.emplace_back(localTrackIdx);
				else
				{
					const size_t idx = filtered_index_pairs_.size();
					const size_t point_idx = it - index_pairs_.begin();
					filtered_correspondences_.push_back(correspondences_.row(point_idx));
					filtered_index_pairs_.emplace_back(index_pairs_[point_idx]);
					pointMask[point_idx] = true;
				}
			}
			else
				LOG(WARNING) << "The track is wrong.";

			++localTrackIdx;
		}

		for (int i = toDelete.size() - 1; i >= 0; --i)
			mutual_track_ids_.erase(mutual_track_ids_.begin() + toDelete[i]);

		if (add_duplicates_)
		{
			for (size_t pointIdx = 0; pointIdx < pointMask.size(); ++pointIdx)
				if (!pointMask[pointIdx])
				{
					filtered_correspondences_.push_back(correspondences_.row(pointIdx));
					filtered_index_pairs_.emplace_back(index_pairs_[pointIdx]);
				}
		}
	}
}