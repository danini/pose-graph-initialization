#pragma once

#include <map>
#include <queue>
#include <vector>
#include "pose_graph.h"
#include "imagesimilarity_graph.h"
#include <sophus/se3.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/flann/dist.h>

namespace reconstruction
{
	template<class T>
	struct SampsonDistanceFlann
	{
		typedef cvflann::True is_kdtree_distance;
		typedef cvflann::True is_vector_space_distance;

		typedef T ElementType;
		typedef typename cvflann::Accumulator<T>::Type ResultType;
		typedef ResultType CentersType;

		const Pose pose;

		SampsonDistanceFlann(Pose pose_) : pose(pose_)
		{

		}

		/**
		 *  Compute the Sampson distance
		 */
		template <typename Iterator1, typename Iterator2>
		ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
		{			
			const double &x1 = (ResultType)*a;
			++a;
			const double &y1 = (ResultType)*a;
			++a;
			++a;
			const double &x2 = (ResultType)*b;
			++b;
			const double &y2 = (ResultType)*b;
			++b;
			++b;

			const double
				&e11 = pose.descriptor(0, 0),
				&e12 = pose.descriptor(0, 1),
				&e13 = pose.descriptor(0, 2),
				&e21 = pose.descriptor(1, 0),
				&e22 = pose.descriptor(1, 1),
				&e23 = pose.descriptor(1, 2),
				&e31 = pose.descriptor(2, 0),
				&e32 = pose.descriptor(2, 1),
				&e33 = pose.descriptor(2, 2);

			const double rxc = e11 * x2 + e21 * y2 + e31;
			const double ryc = e12 * x2 + e22 * y2 + e32;
			const double rwc = e13 * x2 + e23 * y2 + e33;
			const double r = (x1 * rxc + y1 * ryc + rwc);
			const double rx = e11 * x1 + e12 * y1 + e13;
			const double ry = e21 * x1 + e22 * y1 + e23;
			const double a1 = rxc * rxc + ryc * ryc;
			const double b1 = rx * rx + ry * ry;

			ResultType sampsonDistance = (ResultType)(r * r * (a1 + b1) / (a1 * b1));
			return sampsonDistance;
		}

		/**
		 * Partial distance, used by the kd-tree.
		 */
		template <typename U, typename V>
		inline ResultType accum_dist(const U& a, const V& b, int) const
		{
			return (a - b)*(a - b);
		}
	};

	class EssentialMatrixEvaluator
	{
	protected:
		// The sampson distance between a point_ correspondence and an essential matrix
		inline double squaredSampsonDistance(const cv::Mat& point_,
			const Eigen::Matrix3d& descriptor_) const
		{
			const double* s = reinterpret_cast<double*>(point_.data);
			const double
				& x1 = *s,
				& y1 = *(s + 1),
				& x2 = *(s + 2),
				& y2 = *(s + 3);

			const double
				& e11 = descriptor_(0, 0),
				& e12 = descriptor_(0, 1),
				& e13 = descriptor_(0, 2),
				& e21 = descriptor_(1, 0),
				& e22 = descriptor_(1, 1),
				& e23 = descriptor_(1, 2),
				& e31 = descriptor_(2, 0),
				& e32 = descriptor_(2, 1),
				& e33 = descriptor_(2, 2);

			double rxc = e11 * x2 + e21 * y2 + e31;
			double ryc = e12 * x2 + e22 * y2 + e32;
			double rwc = e13 * x2 + e23 * y2 + e33;
			double r = (x1 * rxc + y1 * ryc + rwc);
			double rx = e11 * x1 + e12 * y1 + e13;
			double ry = e21 * x1 + e22 * y1 + e23;

			return r * r /
				(rxc * rxc + ryc * ryc + rx * rx + ry * ry);
		}

	public:
		// The squared residual function used for deciding which points are inliers
		inline double squaredResidual(const cv::Mat& point_,
			const Eigen::Matrix3d& descriptor_) const
		{
			return squaredSampsonDistance(point_, descriptor_);
		}

		// The residual function used for deciding which points are inliers
		inline double residual(const cv::Mat& point_,
			const Eigen::Matrix3d& descriptor_) const
		{
			const double kSquaredResidual =
				squaredSampsonDistance(point_, descriptor_);
			return std::sqrt(kSquaredResidual);
		}

		// A function returning the set of inliers of a model
		inline void getInliers(
			// The point correspondences
			const cv::Mat &kCorrespondences_,
			// The model descriptor
			const Eigen::Matrix3d &kDescriptor_,
			// The inlier-outlier threshold
			const double &kThreshold_,
			// The indices of the found inliers
			std::vector<size_t> &inliers_) const
		{
			// The number of correspondences
			const int& kCorrespondenceNumber = kCorrespondences_.rows;
			// The squared inlier-outlier threshold
			const double kSquaredThreshold = kThreshold_ * kThreshold_;
			// The squared point-to-model residual
			double squaredResidual;
			// Occupying the maximum memory that the inlier vector can take
			inliers_.reserve(kCorrespondenceNumber);

			// Iterating over all correspondences and check if they are inliers
			for (size_t pointIdx = 0; pointIdx < kCorrespondenceNumber; ++pointIdx)
			{
				// The squared point-to-model residual
				squaredResidual = this->squaredResidual(
					kCorrespondences_.row(pointIdx), // The current correspondence
					kDescriptor_); // The model parameters

				// Check if the residual is smaller than the threshold
				if (squaredResidual < kThreshold_)
					// Store as an inlier if yes
					inliers_.emplace_back(pointIdx);
			}
		}

	};

	// A class to check a possible pose inside the graph traversal
	// when a new path is found
	template<typename _Evaluator>
	class InTraversalPoseTester
	{
	public:
		InTraversalPoseTester(
			const double kInlierOutlierThreshold_, // The inlier-outlier threshold used for determining the inliers
			const size_t kMinimumInlierNumber_, // The minimum number of inliers to accept a pose
			const _Evaluator * const kEvaluator_, // The model estimator object, it knows the residual functions
			cv::Mat * correspondences_) : // The data points
			kInlierOutlierThreshold(kInlierOutlierThreshold_),
			kSquaredInlierOutlierThreshold(kInlierOutlierThreshold_ * kInlierOutlierThreshold_),
			kMinimumInlierNumber(kMinimumInlierNumber_),
			kEvaluator(kEvaluator_),
			kPointNumber(correspondences_->rows),
			correspondences(correspondences_)
		{

		}

		// A function to test a pose if it is good enough
		bool test(const Sophus::SE3d &kPose_,
			size_t &inlierNumber_) const
		{
			// Initialize the inlier number to 0
			inlierNumber_ = 0;
			// Initialize the pose object
			Pose pose(kPose_);
			// The essential matrix
			const auto& essentialMatrix = pose.getEssentialMatrix();
			
			// Iterate through the points and count the inliers
			for (size_t pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
			{
				// Estimate the point-to-model residuals
				const double kSquaredResidual = kEvaluator->squaredResidual(
					correspondences->row(pointIdx), // The current point
					essentialMatrix); // The pose to be tested

				// If the residual is smaller than the threshold,
				// accept the point as inlier.
				if (kSquaredResidual < kSquaredInlierOutlierThreshold)
				{
					// Increase the inlier number
					++inlierNumber_;

					// If we have found enough inliers to accept the pose as a correct one
					// terminate.
					if (inlierNumber_ >= kMinimumInlierNumber)
					{
						LOG(INFO) << inlierNumber_ << " inliers found. Success.";
						return true;
					}
				}
			}

			// If the pose does not have enough inliers, consider
			// it an incorrect pose. 
			LOG(INFO) << inlierNumber_ << " inliers found. Failure.";
			return false;
		}

	protected:
		const double kInlierOutlierThreshold, // The inlier-outlier threshold used for determining the inliers
			kSquaredInlierOutlierThreshold; // The squared inlier-outlier threshold used for determining the inliers
		const size_t kMinimumInlierNumber, // The minimum number of inliers to accept a pose
			kPointNumber; // The number of data points
		cv::Mat * correspondences; // The pointer of the data point container
		const _Evaluator * const kEvaluator; // The model estimator object
	};

	/*
	* The abstract pose graph traversal class
	*/
	class PoseGraphTraversal
	{
	public:
		PoseGraphTraversal(
			// The pose-graph
			const PoseGraph * const kPoseGraph_,
			// The maxmimum number of paths to be returned when the pose tester is not specified
			// and multiple paths are to be found.
			const size_t kMaximumPathNumber_,
			// A pose tester that interrupts the traversal when a good pose is found with a reasonable number of inliers
			const InTraversalPoseTester<EssentialMatrixEvaluator> * const kPoseTester_ = nullptr) :
			kPoseGraph(kPoseGraph_),
			kPoseTester(kPoseTester_),
			kMaximumPathNumber(kMaximumPathNumber_),
			applyPoseTest(kPoseTester_ != nullptr)
		{

		}

		virtual void getPath(
			const PoseGraphVertex &kFrom_,
			const PoseGraphVertex &kTo_,
			std::vector<std::vector<const PoseGraphVertex*>> &path_,
			std::vector<Sophus::SE3d> &poses_,
			size_t &touchedNodes_,
			size_t &foundPaths_,
			bool &pathExists_) const = 0;

	protected:
		// The possible states of a node
		enum NodeState { Unseen, Seen, Open, Closed };
		// The node type consisting of the (1) view index, (2) neighboring view indices, (3) depth.
		typedef std::tuple < const PoseGraphVertex *, std::vector<const PoseGraphVertex*>, size_t > NodeType;
		// The pointer of the pose graph
		const PoseGraph * const kPoseGraph;
		// The pointer of the pose tester
		const InTraversalPoseTester<EssentialMatrixEvaluator> * const kPoseTester;
		// The maximum number of paths to be returned.
		const size_t kMaximumPathNumber;
		// A flag to decide if the pose tester should be applied
		bool applyPoseTest;

		// A function to recover the pose from a path
		void recoverPath(
			// The destination node
			const NodeType &kEnd_,
			// The path
			std::vector<const PoseGraphVertex*> &path_,
			// The returned pose
			Sophus::SE3d &pose_,
			// A flag to see if the recovery was successful
			bool &success_) const
		{
			// Recover the pose from the path
			const std::vector<const PoseGraphVertex*> &path = std::get<1>(kEnd_);

			// Initialize the pose to be an identity matrix
			pose_ = Sophus::SE3d(Sophus::Matrix3d::Identity(), Sophus::Vector3d::Zero());

			// Iterate over the path's edges
			for (size_t path_idx = 1; path_idx < path.size(); ++path_idx)
			{
				// The index of source view of the edge
				const reconstruction::ViewId &kSourceViewIdx = path[path_idx - 1]->id(),
					// The index of destination view of the edge
					&kDestinationViewIdx = path[path_idx]->id();

				// A flag to decide if the pose should be inverted.
				// This can happen if we go through the directed edge
				// in the opposite direction.
				bool inverted = false;

				// The pointer of the current edge
				const reconstruction::PoseGraphEdge *selectedEdge;

				// If the edge is used as is.
				if (kPoseGraph->hasEdge(kSourceViewIdx, kDestinationViewIdx))
					selectedEdge = &kPoseGraph->getEdgeById(std::make_pair(kSourceViewIdx, kDestinationViewIdx));
				// If the edge is inverted
				else if (kPoseGraph->hasEdge(kDestinationViewIdx, kSourceViewIdx))
				{
					selectedEdge = &kPoseGraph->getEdgeById(std::make_pair(kDestinationViewIdx, kSourceViewIdx));
					inverted = true;
				}

				// This should never happen.
				if (selectedEdge->isUndefined())
				{
					LOG(INFO) << "Even though the path was constructed, there is no existing edge in the pose graph between views (" << kSourceViewIdx << ", " << kDestinationViewIdx << ")";
					success_ = false;
					return;
				}

				// Multiply the maintained pose with the pose of the current edge or its inverse.
				if (inverted)
					pose_ = selectedEdge->getValue().getTransform().inverse() * pose_;
				else
					pose_ = selectedEdge->getValue().getTransform() * pose_;
			}

			success_ = true;
		}
	};

	/*
		Breadth-first pose-graph traversal.
		This will be quite slow when running on big graphs
	*/
	template<typename _NodeCostHeuristics>
	class BreadthFirstTraversal : public PoseGraphTraversal
	{
	public:
		BreadthFirstTraversal(
			// The pose-graph
			const PoseGraph * const kPoseGraph_, 
			// The object calculating the heuristics value
			const _NodeCostHeuristics &kHeuristicsObject_, 
			// A pose tester that interrupts the traversal when a good pose is found with a reasonable number of inliers
			const InTraversalPoseTester<EssentialMatrixEvaluator> * const kPoseTester_ = nullptr,
			// A flag to decide if multiple paths should be returned.
			// This will not do anything of a pose tester is used.
			const bool kReturnMultiplePaths_ = false,
			// The maximum depth during the traversal
			const size_t kMaximumDepth_ = std::numeric_limits<size_t>::max(),
			// The maxmimum number of paths to be returned when the pose tester is not specified
			// and multiple paths are to be found.
			const size_t kMaximumPathNumber_ = std::numeric_limits<size_t>::max()) :
			PoseGraphTraversal(kPoseGraph_, kMaximumPathNumber_, kPoseTester_),
			kReturnMultiplePaths(kReturnMultiplePaths_),
			kMaximumDepth(kMaximumDepth_)
		{

		}

		static constexpr const char *name() { return "breadth-first"; }

		void getPath(
			// The source node
			const PoseGraphVertex &kFrom_, 
			// The destination node
			const PoseGraphVertex &kTo_, 
			// The found path
			std::vector<std::vector<const PoseGraphVertex*>> &path_, 
			// The found poses
			std::vector<Sophus::SE3d> &poses_, 
			// The number of nodes touched during the traversal
			size_t &touchedNodes_,
			// The number of paths found
			size_t &foundPaths_, 
			// A flag to say if at least as single path has been found
			bool &pathExists_) const; 

	protected:
		// The maximum depth during the traversal
		const size_t kMaximumDepth;
		// A flag to decide if multiple paths should be returned.
		// This will not do anything of a pose tester is used.
		const bool kReturnMultiplePaths;
	};

	template<typename _NodeCostHeuristics>
	void BreadthFirstTraversal<_NodeCostHeuristics>::getPath(
		const PoseGraphVertex &kFrom_,
		const PoseGraphVertex &kTo_,
		std::vector<std::vector<const PoseGraphVertex*>> &path_,
		std::vector<Sophus::SE3d> &poses_,
		size_t &touchedNodes_,
		size_t &foundPaths_,
		bool &pathExists_) const
	{
		// Initialize the path exists variable to false
		pathExists_ = false;
		// The list of open nodes
		std::list<NodeType> openNodes;
		// The node where the destination is found
		std::vector<NodeType> finalNodes;
		// The states of the nodes saying if it has been used/amongst the open nodes/not used.
		std::map<reconstruction::ViewId, NodeState> nodeStates;
		// Add the first node to the open list
		openNodes.emplace_back(std::make_tuple(&kFrom_, std::vector<const PoseGraphVertex*>(), 0));
		// The id of the sought vertex
		const size_t &kDestinationViewIdx = kTo_.id();
		// The number of possible paths tested
		foundPaths_ = 0;

		// Iterate while there is at least a single open node
		while (!openNodes.empty())
		{
			// Get the next open node
			const NodeType &kNode = openNodes.front();
			// Get the linked vertex
			const PoseGraphVertex *vertex = std::get<0>(kNode);
			std::vector<const PoseGraphVertex *> parents = std::get<1>(kNode);
			const size_t &depth = std::get<2>(kNode);
			++touchedNodes_;

			// If the depth is bigger than the maximum, break the cycle
			if (depth > kMaximumDepth)
				break;

			// The id of the current vertex
			const ViewId &kVertexIdx = vertex->id();

			// If the current vertex is the one which we are looking for, return it.
			if (kVertexIdx == kDestinationViewIdx)
			{
				// Recover the pose from the current path
				bool success;
				Sophus::SE3d pose;
				std::vector<const PoseGraphVertex *> &parentsReference = std::get<1>(openNodes.front());
				parentsReference.emplace_back(&kTo_);
				path_.emplace_back(std::vector<const PoseGraphVertex*>());
				recoverPath(openNodes.front(),
					path_.back(),
					pose,
					success);

				if (success)
				{
					if (!applyPoseTest)
						poses_.emplace_back(pose);

					if (applyPoseTest)
					{
						size_t inlierNumber = 0;
						bool success = kPoseTester->test(pose, inlierNumber);

						if (inlierNumber > 5)
						{
							poses_.emplace_back(pose);
							if (success)
								break;
						}
					}

					++foundPaths_;
					if (foundPaths_ >= kMaximumPathNumber)
						break;
				}

				//final_nodes.emplace_back(node);
				if (!kReturnMultiplePaths)
					break;
				else
				{
					openNodes.pop_front();
					continue;
				}
			}

			// Add the vertex to the vector of parents which will be assigned 
			// to the next open nodes
			parents.emplace_back(vertex);

			// Set the node state to 'Open'
			nodeStates[kVertexIdx] = NodeState::Open;

			// Get the edges from the pose graph which contain the current vertex
			std::vector<reconstruction::EdgeId> edges;
			kPoseGraph->getEdgesByVertex(kVertexIdx, edges);

			// Iterate through the edges and add the implied vertices as open nodes
			for (const reconstruction::EdgeId &kEdgeIdx : edges)
			{
				const reconstruction::PoseGraphEdge &edge =
					kPoseGraph->getEdgeById(kEdgeIdx);

				const reconstruction::ViewId &kSourceIdx = edge.getSourceId(),
					&kDestinationIdx = edge.getDestinationId();

				if (kVertexIdx == kDestinationIdx)
				{
					const auto &kIteratorSourceState = nodeStates.find(kSourceIdx);
					if (kIteratorSourceState == nodeStates.end())
						openNodes.emplace_back(std::make_tuple(&kPoseGraph->getVertexById(kSourceIdx), parents, depth + 1));
				}

				if (kVertexIdx == kSourceIdx)
				{
					const auto &kIteratorDestinationState = nodeStates.find(kDestinationIdx);
					if (kIteratorDestinationState == nodeStates.end())
						openNodes.emplace_back(std::make_tuple(&kPoseGraph->getVertexById(kDestinationIdx), parents, depth + 1));
				}
			}

			nodeStates[kVertexIdx] = NodeState::Closed;
			openNodes.pop_front();
		}
		
		pathExists_ = poses_.size() > 0;
	}

	/*
		Constant heuristics for the A* algorithm to simulate breadth-first and depth-first traversals.
		If _Constant < 0, the algorithm acts as breadth-first traversal.
		If _Constant > 0, the algorithm acts as depth-first traversal.
	*/
	template<int _Constant>
	class ConstantHeuristics
	{
	public:
		ConstantHeuristics(const reconstruction::SimilarityTable& similarity_table_)
		{

		}

		// The name of the heuristics
        static  const char *name() { return "constant"; }

		// The penalty on the current edge
		const double getCost(
			const ViewId &kSourceViewIdx_, // The index of the source view
			const ViewId &kDestinationViewIdx_) const // The index of the destination view
		{
			// Returning a constant penalty
			return static_cast<double>(_Constant);
		}
	};

	/*
		Image similarity-based heuristics for the A* algorithm.
	*/
	class ImageSimilarityHeuristics
	{
	protected:
		// The reference of the image similarity table
		const reconstruction::SimilarityTable& kSimilarityTable;

	public:
		// The name of the heuristics
        static  const char *name() { return "image similarity-based"; }

		ImageSimilarityHeuristics(const reconstruction::SimilarityTable& kSimilarityTable_) :
			kSimilarityTable(kSimilarityTable_)
		{

		}

		// The penalty on the current edge
		const double getCost(
			const ViewId& kSourceViewIdx_, // The index of the source view
			const ViewId& kDestinationViewIdx_) const // The index of the destination view
		{
			// Returning the similarity from the similarity table.
			const double similarity = 
				kSimilarityTable.getSimilarity(kSourceViewIdx_, kDestinationViewIdx_);
			// Clamping in-between [0, 1]
			return std::clamp(similarity, 0.0, 1.0);
		}
	};

	template<typename _NodeCostHeuristics>
	class AStarTraversal : public PoseGraphTraversal
	{
	public:
		AStarTraversal(
			// The pose-graph
			const PoseGraph * const kPoseGraph_,
			// The object calculating the heuristics value
			const _NodeCostHeuristics &kHeuristicsObject_,
			// A pose tester that interrupts the traversal when a good pose is found with a reasonable number of inliers
			const InTraversalPoseTester<EssentialMatrixEvaluator> * const kPoseTester_ = nullptr,
			// A flag to decide if multiple paths should be returned.
			// This will not do anything of a pose tester is used.
			const bool kReturnMultiplePaths_ = false,
			// The minimum inlier ratio
			const bool kMinimumInlierRatio_ = 0.1,
			// The maximum depth during the traversal
			const size_t kMaximumDepth_ = std::numeric_limits<size_t>::max(),
			// The maxmimum number of paths to be returned when the
			const size_t kMaximumPaths_ = std::numeric_limits<size_t>::max()) :
			PoseGraphTraversal(kPoseGraph_, kMaximumPaths_, kPoseTester_),
			kReturnMultiplePaths(kReturnMultiplePaths_),
			kMaximumDepth(kMaximumDepth_),
			kReturnOnlyPose(false),
			kMinimumInlierRatio(kMinimumInlierRatio_),
			kHeuristicsObject(kHeuristicsObject_)
		{

		}

		// The name of the heuristics
		static constexpr const char *name() { return "a-star"; }

		void getPath(
			// The source node
			const PoseGraphVertex& kFrom_,
			// The destination node
			const PoseGraphVertex& kTo_,
			// The found path
			std::vector<std::vector<const PoseGraphVertex*>>& path_,
			// The found poses
			std::vector<Sophus::SE3d>& poses_,
			// The number of nodes touched during the traversal
			size_t& touchedNodes_,
			// The number of paths found
			size_t& foundPaths_,
			// A flag to say if at least as single path has been found
			bool& pathExists_) const;

	protected:
		const size_t kMaximumDepth;
		const bool kReturnMultiplePaths,
			kReturnOnlyPose;
		const double kMinimumInlierRatio;
		const _NodeCostHeuristics &kHeuristicsObject;
	};

	// A comparator object used inside a priority queue sorting the edges
	class CostComparator
	{
	public:
		typedef std::tuple<double, double, double> DoubleNodeCost;
		typedef std::tuple < const PoseGraphVertex *, std::vector<const PoseGraphVertex*>, size_t > NodeType;

		static double weight;

		static void setWeight(double weight_)
		{
			weight = weight_;
		}

		bool operator() (const std::pair<DoubleNodeCost, NodeType> &pair1_,
			const std::pair<DoubleNodeCost, NodeType> &pair2_)
		{
			return std::get<2>(pair1_.first) < std::get<2>(pair2_.first);
		}
	};

	// Initializing the static weight variable
	double CostComparator::weight{ 0.8 };

	template<typename _NodeCostHeuristics>
	void AStarTraversal<_NodeCostHeuristics>::getPath(
		// The source node
		const PoseGraphVertex &kFrom_,
		// The destination node
		const PoseGraphVertex &kTo_,
		// The found path
		std::vector<std::vector<const PoseGraphVertex*>> &path_,
		// The found poses
		std::vector<Sophus::SE3d> &poses_,
		// The number of nodes touched
		size_t &touchedNodes_,
		// The number of paths found
		size_t &foundPaths_,
		// A flag to see if a path has been found
		bool &pathExists_) const
	{
		typedef std::tuple<double, double, double> DoubleNodeCost;
		typedef std::pair<DoubleNodeCost, NodeType> CostNodePair;

		// Initialize the path exists variable to false
		pathExists_ = false;

		// The list of open nodes
		std::vector<CostNodePair> container;
		container.reserve(kPoseGraph->numEdges());
		std::priority_queue<CostNodePair,
			std::vector<CostNodePair>,
			CostComparator> openNodes(CostComparator(),
			std::move(container));

		// The states of the nodes saying if it has been used/amongst the open nodes/not used.
		// For this purpose, an unordered_map (i.e., hashmap) is used. 
		const size_t numVertices = kPoseGraph->numVertices();
		/*static const auto hashingFunction = [&numVertices](reconstruction::ViewId const& viewId_) {
			return viewId_;
		};*/

		std::unordered_map<reconstruction::ViewId, NodeState/*, hashingFunction*/> nodeStates;
		nodeStates.reserve(kPoseGraph->numVertices());

		// Add the first node to the open list
		openNodes.push(std::make_pair(std::make_tuple(1, 0, 0), 
			std::make_tuple(&kFrom_, std::vector<const PoseGraphVertex*>(), 0)));
		// The id of the sought vertex
		const size_t &kSourceViewIdx = kFrom_.id();
		const size_t &kDestinationViewIdx = kTo_.id();
		// The number of possible paths tested
		foundPaths_ = 0;
		// The weight for the heuristics
		const double &weight = CostComparator::weight;
		const double oneMinusWeight = 1.0 - weight;

		std::vector<reconstruction::EdgeId> edges;
		edges.reserve(kPoseGraph->numEdges());

		// Pre-allocate the memory of the paths
		path_.resize(1);

		// Iterate while there is at least a single open node
		while (!openNodes.empty())
		{
			// The next item in the queue
			const auto& item = openNodes.top();
			// Get the next open node
			NodeType node = item.second;
			// Get the cost of the current node
			const DoubleNodeCost kNodeCost = item.first;
			// The depth of the current node
			const size_t &kDepth = std::get<2>(node);
			// Increase the number of nodes visited
			++touchedNodes_;
			// Remove the top element from the priority queue
			openNodes.pop();

			// If the depth is bigger than the maximum, break the cycle
			if (kDepth > kMaximumDepth)
				continue;

			// Get the linked vertex
			const PoseGraphVertex *vertex = std::get<0>(node);
			// The parents of the current node
			std::vector<const PoseGraphVertex *> parents = std::get<1>(node);
			// The id of the current vertex
			const ViewId &kVertexIdx = vertex->id();

			// If the current vertex is the one which we are looking for, return it.
			if (kVertexIdx == kDestinationViewIdx)
			{
				// Adding the last destination vertex to the path
				std::get<1>(node).emplace_back(std::get<0>(node));

				// Recover the pose from the current path
				bool success;
				Sophus::SE3d pose;
				recoverPath(node,
					path_[0],
					pose,
					success);

				if (success)
				{
					LOG(INFO) << "Search depth = " << std::get<2>(node);
					++foundPaths_;

					if (!applyPoseTest)
						poses_.emplace_back(pose);

					if (applyPoseTest)
					{
						size_t inlierNumber = 0;
						bool success = kPoseTester->test(pose, inlierNumber);

						if (success)
						{
							poses_.emplace_back(pose);
							break;
						}
					}

					if (foundPaths_ >= kMaximumPathNumber)
						break;
				}

				if (!kReturnMultiplePaths)
					break;
				else
					continue;
			}

			// Add the vertex to the vector of parents which will be assigned 
			// to the next open nodes
			parents.emplace_back(vertex);

			// Set the node state to 'Open'
			nodeStates[kVertexIdx] = NodeState::Open;

			// Get the edges from the pose graph which contain the current vertex
			kPoseGraph->getEdgesByVertex(kVertexIdx, edges);
			
			// Iterate through the edges and add the implied vertices as open nodes
			if (kDepth < kMaximumDepth)
			{
				for (const reconstruction::EdgeId &kEdgeIdx : edges)
				{
					// Getting the current edge
					const reconstruction::PoseGraphEdge &kEdge =
						kPoseGraph->getEdgeById(kEdgeIdx);

					// Skip the edge if the stored inlier ratio is smaller than the specified minimum.
					// It likely does not have a reasonably accurate pose. 
					if (kEdge.getScore() < kMinimumInlierRatio)
						continue;

					// Getting the source and destination vertices' identifiers
					const reconstruction::ViewId &kSourceIdx = kEdge.getSourceId(),
						&kDestinationIdx = kEdge.getDestinationId();
							   
					// Deciding which is the next vertex
					const ViewId &kNextIdx =
						kVertexIdx == kDestinationIdx ?
						kSourceIdx : kDestinationIdx;

					// Update the cost coming from the next edge
					const double kEdgeCost =
						MIN(std::get<0>(kNodeCost), kEdge.getScore());

					// Update the cost coming from the similarity of the next vertex and the final one
					const double kNextToDestinationCost =
						MAX(std::get<1>(kNodeCost), kHeuristicsObject.getCost(kNextIdx, kDestinationViewIdx));

					// Get the linear combination of the weights
					const double kCombinedWeight =
						weight * kEdgeCost + oneMinusWeight * kNextToDestinationCost;

					// Updating the next vertex
					const auto &kIteratorSourceState = nodeStates.find(kNextIdx);
					if (kIteratorSourceState == nodeStates.end())
					{
						openNodes.push(std::make_pair(std::make_tuple(kEdgeCost, kNextToDestinationCost, kCombinedWeight),
							std::make_tuple(&kPoseGraph->getVertexById(kNextIdx), parents, kDepth + 1)));
						if (!kReturnMultiplePaths)
							nodeStates[kNextIdx] = NodeState::Seen;
					}
				}
			}

			nodeStates[kVertexIdx] = NodeState::Closed;
		}

		pathExists_ = poses_.size() > 0;
	}
}
