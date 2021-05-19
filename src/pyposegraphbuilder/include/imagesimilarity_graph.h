#pragma once

#include <vector>
#include <queue>
#include "types.h"
#include "pose.h"
#include <filesystem>

#define NO_SUCH_VERTEX -99999

namespace reconstruction
{
class SimilarityTable
{
protected:

    std::vector<std::vector<double> > similarity;
    size_t size;
	const bool build_priority_queue;
	const double image_similarity_threshold;
	std::priority_queue<std::tuple<double, ViewId, ViewId>> view_pair_queue;
    std::unordered_set<ViewId> views;

public:
    SimilarityTable(
		const size_t num_imgs,
		const double image_similarity_threshold_,
		const bool build_priority_queue_ = true);

    bool setSimilarity(
            const ViewId from,
            const ViewId to,
            const double val);

    double getSimilarity(
            const ViewId from,
            const ViewId to) const;

    bool loadFromFile(const std::string fname);

	const std::priority_queue<std::tuple<double, ViewId, ViewId>> &getPrioritizedViewPairs() const { return view_pair_queue; }
	std::priority_queue<std::tuple<double, ViewId, ViewId>> &getMutablePrioritizedViewPairs() { return view_pair_queue; }

    const std::unordered_set<size_t>& getKeptViews() const { return views; }
    std::unordered_set<size_t>& getMutableKeptViews() { return views; }
	
};

SimilarityTable::SimilarityTable (
	const size_t num_imgs,
	const double image_similarity_threshold_,
	const bool build_priority_queue_) : 
	build_priority_queue(build_priority_queue_),
	image_similarity_threshold(image_similarity_threshold_)
{
    similarity.resize(num_imgs);
    for (size_t i=0; i< similarity.size(); i++)
    {
        similarity[i].resize(num_imgs);
        for (size_t j=0; j < num_imgs; j++){
            similarity[i][j] = 1.0f;
        }
    }
}
bool SimilarityTable::setSimilarity(
        const ViewId from,
        const ViewId to,
        const double val)
{
    if ((from < 0) || (from > size-1)) {
        return false;
    }
    if ((to < 0) || (to > size-1)) {
        return false;
    }

    // Put the similary to a priority queue if needed
    if (build_priority_queue &&
        from != to &&
        image_similarity_threshold <= val)
    {
        views.insert(from);
        views.insert(to);
        view_pair_queue.emplace(std::make_tuple(val, from, to));
    }

    // Adding the link
    similarity[from][to] = val;
    similarity[to][from] = val;
    return true;
}

double SimilarityTable::getSimilarity(
        const ViewId from,
        const ViewId to) const
{
    if ((from < 0) || (from > size-1)) {
        return NO_SUCH_VERTEX;
    }
    if ((to < 0) || (to > size-1)) {
        return NO_SUCH_VERTEX;
    }
    return similarity[from][to] ;
}

bool SimilarityTable::loadFromFile(const std::string fname)
{
    if (! std::filesystem::exists(fname)) {
        LOG(INFO) << "A problem occured when opening '" << fname << "'.";
        return false;
    };
    std::ifstream file(fname);

    if (!file.is_open()) {
        LOG(INFO) << "A problem occured when opening '" << fname << "'.";
        return false;
    }

    std::string line;
    std::vector<std::vector<double>> temp_arr;
    int i = 0;

    while (std::getline(file, line))
    {
        double value;
        std::stringstream ss(line);

        temp_arr.push_back(std::vector<double>());

        while (ss >> value)
        {
            temp_arr[i].push_back(value);
        }
        ++i;
    }

	size = similarity.size();

    if (temp_arr.size() == similarity.size()) {

        for (size_t i=0; i< similarity.size(); i++)
        { if (similarity[i].size() == temp_arr[i].size()) {
                for (size_t j=0; j < temp_arr[i].size(); j++){
                   similarity[i][j] = temp_arr[i][j];

				   // Put the similary to a priority queue if needed
                   if (build_priority_queue &&
                       i != j &&
                       image_similarity_threshold <= similarity[i][j] &&
                       similarity[j][i] != similarity[i][j])
                   {
                       views.insert(i);
                       views.insert(j);
                       view_pair_queue.emplace(std::make_tuple(similarity[i][j], i, j));
                   }
                }
            } else {
                LOG(INFO) << fname << " has different size than initialized similarity map";
                return false;
            }
        }
    } else {
        LOG(INFO) << fname << " has different size than initialized similarity map";
        return false;
    }

    LOG(INFO) << "Successfully loaded image similarity map from " << fname << " with size [" << similarity.size() << " x " << similarity.size() << "]";
    return true;
}


}
