#pragma once

#include <algorithm>
#include <numeric>
#include <string>
#include <tuple>
#include <thread>
#include <shared_mutex>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Eigen>
#include <glog/logging.h>

#include "imagesimilarity_graph.h"
#include "feature_utils.h"
#include "utils.h"
#include "matcher.h"
#include "reconstruction.h"
#include "pose_graph.h"
#include "visibility_table.h"
#include "graph_traversal.h"
#include "point_track.h"

namespace reconstruction
{
	class PoseGraphBuilder
	{
	public:
		PoseGraphBuilder(
			const size_t kCoreNumber_,
			const size_t kMaximumTrackletNumber_,
			const size_t kMaximumSearchDepth_,
			const size_t kMaximumPathNumber_,
			const size_t kMinimumInlierNumber_,
			const size_t kMinimumPointNumber_,
			const size_t kMaximumPointNumberForEpipolarHashing_,
			const double kTraversalHeuristicsWeight_,
			const double kSimilarityThreshold_,
			const double kInlierOutlierThreshold_,
			const std::string& kImagePath_,
			const std::string& kWorkspacePath_,
			const std::string& kSimilarityGraphPath_,
			const std::string& kFocalLengthPath_,
			const bool kUsePathFinding_,
			const bool kUseGPU_,
			const bool kUseEpipolarHashing_) :
			kCoreNumber(kCoreNumber_),
			kImagePath(kImagePath_),
			kSimilarityThreshold(kSimilarityThreshold_),
			kInlierOutlierThreshold(kInlierOutlierThreshold_),
			kWorkspacePath(kWorkspacePath_),
			kFocalLengthPath(kFocalLengthPath_),
			kUseGPU(kUseGPU_),
			kSimilarityGraphPath(kSimilarityGraphPath_),
			kMaximumPathNumber(kMaximumPathNumber_),
			kMaximumSearchDepth(kMaximumSearchDepth_),
			kUsePathFinding(kUsePathFinding_),
			kTraversalHeuristicsWeight(kTraversalHeuristicsWeight_),
			kMaximumPointNumberForEpipolarHashing(kMaximumPointNumberForEpipolarHashing_),
			kMinimumPointNumber(kMinimumPointNumber_),
			kMinimumInlierNumber(kMinimumInlierNumber_),
			kMaximumTrackletNumber(kMaximumTrackletNumber_),
			kUseEpipolarHashing(kUseEpipolarHashing_)
		{

		}

		void run(
			reconstruction::Reconstruction& reconstruction_, // The reconstruction to be built
			reconstruction::PoseGraph& poseGraph_); // The pose graph

	protected:
		const size_t kCoreNumber,
			kMinimumInlierNumber,
			kMinimumPointNumber,
			kMaximumPointNumberForEpipolarHashing,
			kMaximumSearchDepth,
			kMaximumPathNumber,
			kMaximumTrackletNumber;
		const std::string kImagePath,
			kWorkspacePath,
			kSimilarityGraphPath,
			kFocalLengthPath;
		const bool kUseGPU,
			kUsePathFinding,
			kUseEpipolarHashing;
		const double kTraversalHeuristicsWeight,
			kInlierOutlierThreshold,
			kSimilarityThreshold;

		// Initializing the reconstruction and pose graph objects
		void initializeReconstruction(
			const size_t& kImageNumber_, // The number of images to be added to the reconstruction
			const std::vector<std::tuple<std::string, double, double, double>>& imageData_, // The image data
			reconstruction::Reconstruction& reconstruction_, // The reconstruction to be built
			reconstruction::PoseGraph& poseGraph_); // The pose graph

		// Loading or saving the image data
		void loadImageData(
			const size_t& kImageNumber_, // The number of images 
			const std::string& kImageDatabaseFilename_, // The path of the image database file
			std::shared_mutex& lock_, // A lock making the procedure thread-safe
			std::vector<std::tuple<std::string, double, double, double>>& imageData_); // The image data to be loaded/saved

		// Processing the dataset to build a pose graph
		void processImages(
			const reconstruction::Reconstruction& reconstruction_, // The reconstruction object
			reconstruction::PoseGraph& poseGraph_, // The pose graph object
			reconstruction::SimilarityTable& similarityTable_, // The similarity table containing the pair-wise image similarities
			const double kInlierOutlierThreshold_, // The inlier-outlier threshold
			const double kImageSimilarityThreshold_, // The image similarity threshold
			const size_t kMinimumInlierNumber_, // The minimum inlier number to accept a pose
			const std::string& kKeypointDatabaseFilename_, // The path of the keypoint database
			const std::string& kCorrespondenceDatabaseFilename_); // The path of the correspondence database

		void guidedMatching(
			const reconstruction::Reconstruction& kReconstruction_, // The reconstruction object
			const reconstruction::ViewId& kSourceViewIdx_, // The index of the source view
			const reconstruction::ViewId& kDestinationViewIdx_, // The index of the destination view
			const cv::Mat& kSourceDescriptors_, // The descriptors of the keypoints in the source image
			const cv::Mat& kDestinationDescriptors_, // The descriptors of the keypoints in destination image
			const std::vector<cv::KeyPoint>& kSourceKeypoints_, // The keypoints int source image
			const std::vector<cv::KeyPoint>& kDestinationKeypoints_, // The keypoints in destination image 
			const reconstruction::Pose& kPose_, // The relative pose
			const double& kInlierOutlierThreshold_, // The inlier-outlier threshold
			std::vector<std::tuple<size_t, size_t, double>>& matches_); // The point correspondences found

		bool findPath(
			const reconstruction::SimilarityTable& kSimilarityTable_, // The similarity table containing the pair-wise image similarities
			const reconstruction::Reconstruction& kReconstruction_, // The reconstruction object
			const reconstruction::PoseGraph& kPoseGraph_, // The pose graph
			const reconstruction::VisibilityTable& kVisibilityTable_, // The visibility table to see if there is a path between two views
			const reconstruction::View& kSourceView_, // The source view
			const reconstruction::View& kDestinationView_, // The destination view
			const double kThreshold_, // The inlier-outlier threshold
			cv::Mat& correspondences_, // The point correspondences
			size_t& nodesTouched_, // The number of touched nodes during the A* algorithm
			size_t& pathsTested_, // The number of paths tested before finding a good one
			std::vector<Sophus::SE3d>& poses_); // The found poses

		void createCorrespondenceMatrix(
			const reconstruction::Reconstruction& reconstruction_, // The reconstruction object
			const double& kInlierOutlierThreshold_, // The inlier-outlier threshold
			const size_t& kSourceViewIdx_, // The index of the source view
			const size_t& kDestinationViewIdx_, // The index of the destination view
			const std::vector<cv::KeyPoint>& kSourceKeypoints, // The keypoints in the source image
			const std::vector<cv::KeyPoint>& kDestinationKeypoints, // The keypoints in the destination image
			const std::vector<std::tuple<size_t, size_t, double>>& kMatches_, // The point matches
			cv::Mat& correspondences_, // The normalized correspondences
			double& normalizedThreshold_); // The normalized inlier-outlier threshold

		bool estimatePose(
			const reconstruction::Reconstruction& reconstruction_, // The reconstruction object
			const double& kInlierOutlierThreshold_, // The inlier-outlier threshold
			const size_t kMinimumInlierNumber_, // The minimum inlier number to accept a pose
			const size_t& kSourceViewIdx_, // The index of the source view
			const size_t& kDestinationViewIdx_, // The index of the destination view
			const cv::Mat& kCorrespondences_, // The normalized point correspondences
			const double kThreshold_, // The inlier-outlier threshold
			const std::vector<Sophus::SE3d>& poseGuesses_, // The poses from the A* traversal 
			Sophus::SE3d& estimatedPose_, // The estimated relative pose
			std::vector<uchar>& inlierMask_, // The inlier mask
			size_t& inlierNumber_); // The inlier number of the pose

		void initializeKeypointDatabase(
			const reconstruction::Reconstruction& reconstruction_, // The reconstruction object
			const std::string& kKeypointDatabaseFilename_, // The path of the keypoint database
			const reconstruction::SimilarityTable& similarityTable_, // The similarity table containing the pair-wise image similarities
			const size_t& kImageNumber_); // The number of images
	};

	void PoseGraphBuilder::run(
		reconstruction::Reconstruction& reconstruction_, // The reconstruction to be built
		reconstruction::PoseGraph& poseGraph_) // The pose graph
	{
		LOG(INFO) << "Loading image names and focal length from '" <<
			kFocalLengthPath << "'";

		// If database already available
		std::shared_mutex lock;
		const std::string kImageDatabaseFilename = kWorkspacePath + "image_data.h5";
		const std::string kKeypointDatabaseFilename = kWorkspacePath + "keypoints.h5";
		const std::string kCorrespondenceDatabaseFilename = kWorkspacePath + "correspondences.h5";

		// Check if the database file that contains the image sizes and focal
		// lengths exists.
		std::ifstream file(kImageDatabaseFilename);
		const bool kDatabaseExists = file.is_open();
		if (kDatabaseExists)
			file.close();

		// The vector containing the focal lengths and images sizes
		std::vector<std::tuple<std::string, double, double, double>> imageData;
		// The total number of images
		size_t totalImageNumber = 0;

		// Loading the image list and focal length
		load1DSfMImageList(
			kImagePath, // The path where the images are stored
			kFocalLengthPath, // The path where the focal lengths are stored
			kCoreNumber, // The number of CPU cores to be used
			!kDatabaseExists, // A flag to decide if the images should be load one-by-one to get their sizes
			totalImageNumber, // The total number of images
			imageData); // The vector containing the focal lengths and images sizes

		// The number of images with focal length
		const size_t kImageNumber = imageData.size();

		// Loading or saving the image sizes
		loadImageData(kImageNumber, // The number of images with focal length
			kImageDatabaseFilename, // The image database file's path 
			lock, // The mutex making the procedure thread-safe
			imageData); // The image data to be loaded

		// Initializing the reconstruction and the pose graph
		initializeReconstruction(
			kImageNumber, // The number of images to be added to the reconstruction
			imageData, // The image data
			reconstruction_, // The reconstruction to be built
			poseGraph_); // The pose graph

		// Loading the image similarity matrix
		reconstruction::SimilarityTable similarityTable(
			totalImageNumber,
			kSimilarityThreshold);
		similarityTable.loadFromFile(kSimilarityGraphPath);

		LOG(INFO) << "Building the pose graph.";

		processImages(reconstruction_,
			poseGraph_,
			similarityTable,
			kInlierOutlierThreshold,
			kSimilarityThreshold,
			kMinimumInlierNumber,
			kKeypointDatabaseFilename,
			kCorrespondenceDatabaseFilename);
	}

	void PoseGraphBuilder::initializeReconstruction(
		const size_t& kImageNumber_, // The number of images to be added to the reconstruction
		const std::vector<std::tuple<std::string, double, double, double>>& imageData_, // The image data
		reconstruction::Reconstruction& reconstruction_, // The reconstruction to be built
		reconstruction::PoseGraph& poseGraph_) // The pose graph
	{
		// Iterating through all images and adding them to the reconstruction
		for (int imageIdx = 0; imageIdx < kImageNumber_; ++imageIdx)
		{
			// Adding the camera to the reconstruction
			if (!reconstruction_.addCamera(imageIdx)) {
				LOG(INFO) << "A problem occured when adding camera '" << imageIdx
					<< "' to the reconstruction.";
				continue;
			}
			// Adding the view to the reconstruction
			else if (!reconstruction_.addView(imageIdx, imageIdx)) {
				LOG(INFO) << "A problem occured when adding view '" << imageIdx
					<< "' to the reconstruction.";
				continue;
			}

			// The data of the current image
			const auto& kData = imageData_[imageIdx];
			// The name of the image
			const auto& kImageName = std::get<0>(kData);
			// The focal length
			const auto& kFocalLength = std::get<1>(kData);
			// The image width
			const auto& kImageWidth = std::get<2>(kData);
			// The image height
			const auto& kImageHeight = std::get<3>(kData);

			// Reference of the newly added view
			reconstruction::View& view = reconstruction_.getMutableView(imageIdx);
			// The metadata where the image name and extension will be saved
			reconstruction::ViewMetadata& metadata = view.getMutableMetadata();
			metadata["name"] =
				kImageName.substr(0, kImageName.size() - 4); // Cut the extension
			metadata["extension"] =
				kImageName.substr(kImageName.size() - 3, kImageName.size()); // Get the extension of the image

			// Setting the size and focal length of the image to the camera
			reconstruction_.getMutableCamera(imageIdx).setWidth(kImageWidth);
			reconstruction_.getMutableCamera(imageIdx).setHeight(kImageHeight);
			reconstruction_.getMutableCamera(imageIdx).setIntrinsics(kFocalLength, kFocalLength, kImageWidth / 2.0, kImageHeight / 2.0);

			// Adding a new vertex, representing the current image, to the pose graph
			poseGraph_.addVertex(imageIdx);
		}
	}

	// Loading or saving the image data
	void PoseGraphBuilder::loadImageData(
		const size_t& kImageNumber_, // The number of images 
		const std::string& kImageDatabaseFilename_, // The path of the image database file
		std::shared_mutex& lock_, // A lock making the procedure thread-safe
		std::vector<std::tuple<std::string, double, double, double>>& imageData_) // The image data to be loaded/saved
	{
		// Opening or creating the database file that contains the image sizes and focal lengths.
		cv::Ptr<cv::hdf::HDF5> h5ImageDatabase = cv::hdf::open(kImageDatabaseFilename_);

#pragma omp parallel for num_threads(kCoreNumber)
		for (int imageIdx = 0; imageIdx < kImageNumber_; ++imageIdx)
		{
			// The properties, like focal length, of the current image
			auto& kData = imageData_[imageIdx];
			// The focal length of the current image
			const auto& kFocalLength = std::get<1>(kData);
			const auto& kImageName = std::get<0>(kData);
			cv::Mat tmpImageData(1, 3, CV_64F);

			// If the database does not exist, create it and save the image data
			if (!h5ImageDatabase->hlexists(kImageName))
			{
				// The current image data
				tmpImageData.at<double>(0) = std::get<1>(kData);
				tmpImageData.at<double>(1) = std::get<2>(kData);
				tmpImageData.at<double>(2) = std::get<3>(kData);

				// A writer lock making providing exclusive access
				std::unique_lock<std::shared_mutex> lock(lock_);
				// Creating the database for the image
				h5ImageDatabase->dscreate(1, 3, CV_64F, kImageName);
				// Saving the image data
				h5ImageDatabase->dswrite(tmpImageData, kImageName);
				// Releasing the lock
				lock.unlock();
			}
			else
			{
				// References to the image width and height
				auto& width = std::get<2>(kData);
				auto& height = std::get<3>(kData);

				// A reader lock allowing multiple threads to read simultaneously. 
				std::shared_lock<std::shared_mutex> lock(lock_);
				// Reading the image data from the database
				h5ImageDatabase->dsread(tmpImageData, kImageName);
				// Releasing the lock
				lock.unlock();

				// Setting the loaded image width and height
				width = tmpImageData.at<double>(1);
				height = tmpImageData.at<double>(2);
			}
		}
		// Closing the database
		h5ImageDatabase->close();
	}

	void PoseGraphBuilder::processImages(
		const reconstruction::Reconstruction& reconstruction_, // The reconstruction object
		reconstruction::PoseGraph& poseGraph_, // The pose graph object
		reconstruction::SimilarityTable& similarityTable_, // The similarity table containing the pair-wise image similarities
		const double kInlierOutlierThreshold_, // The inlier-outlier threshold
		const double kImageSimilarityThreshold_, // The image similarity threshold
		const size_t kMinimumInlierNumber_, // The minimum inlier number to accept a pose
		const std::string& kKeypointDatabaseFilename_, // The path of the keypoint database
		const std::string& kCorrespondenceDatabaseFilename_) // The path of the correspondence database
	{
		// Get the view pairs to process in a priority order starting from the most similar view pair
		std::priority_queue<std::tuple<double, reconstruction::ViewId, reconstruction::ViewId>>& viewPairs =
			similarityTable_.getMutablePrioritizedViewPairs();

		// A structure which knows if there is a path between two vertices
		const size_t& kImageNumber = poseGraph_.numVertices();
		reconstruction::VisibilityTable visibilityTable(kImageNumber, poseGraph_.numEdges());

		// Mutexes used for ensuring thread-safe behaviour
		std::mutex queueReadingMutex;
		std::shared_mutex featureDatabaseMutex,
			correspondenceDatabaseMutex;

		// Detecting keypoints for all images so the detection do not have to be made thread-safe to writing
		initializeKeypointDatabase(
			reconstruction_, // The reconstruction containing the views
			kKeypointDatabaseFilename_, // The filename of the database containing the keypoints
			similarityTable_, // The image similarity table
			kImageNumber); // The number of images

		// The number of processed image pairs
		size_t processedImages = 0;
		const size_t kTotalPairNumber = viewPairs.size();
		RunningStatistics statistics;

		// Initialize the point tracks
		reconstruction::Tracklets tracks(kTotalPairNumber);

		// Iterating through the image pairs which has high similarity score
#pragma omp parallel for num_threads(kCoreNumber)
		for (int coreIdx = 0; coreIdx < kCoreNumber; ++coreIdx)
		{
			std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
			std::chrono::duration<double> elapsedSeconds;

			while (!viewPairs.empty())
			{
				// Get the ids of the next most similar view pair
				queueReadingMutex.lock();
				++processedImages;
				if (processedImages % 100 == 0)
					printf("Image pair processing [%d/%d]\n", processedImages, kTotalPairNumber);
				if (viewPairs.empty())
				{
					queueReadingMutex.unlock();
					break;
				}
				// The next view pair
				const std::tuple<double, reconstruction::ViewId, reconstruction::ViewId> viewPair =
					viewPairs.top();
				viewPairs.pop();
				queueReadingMutex.unlock();

				// The indices of the next views
				const reconstruction::ViewId& kSourceViewIdx = std::get<1>(viewPair),
					& kDestinationViewIdx = std::get<2>(viewPair);

				LOG(INFO) << "Processing view pair (" << kSourceViewIdx << ", " << kDestinationViewIdx << ") with similarity " << std::get<0>(viewPair) << ".";

				// Get the reference of the source view
				const reconstruction::View& kSourceView =
					reconstruction_.getView(kSourceViewIdx);

				// If the image should not be processed, continue
				if (kSourceView.getMetadata().size() == 0)
					continue;

				// Get the reference of the destination view
				const reconstruction::View& kDestinationView =
					reconstruction_.getView(kDestinationViewIdx);

				// If the image should not be processed, continue
				if (kDestinationView.getMetadata().size() == 0)
					continue;

				// Checking if the pose graph already has such an edge
				if (poseGraph_.hasEdge(kSourceViewIdx, kDestinationViewIdx) ||
					poseGraph_.hasEdge(kDestinationViewIdx, kSourceViewIdx))
				{
					LOG(INFO) << "Edge (" << kSourceViewIdx << ", " << kDestinationViewIdx << "), somehow, already exists.";
					continue;
				}

				// Get the metadata of the source view
				const std::string
					& kSourceImageName = (*kSourceView.getMetadata().find("name")).second,
					& kSourceImageExtension = (*kSourceView.getMetadata().find("extension")).second;

				// Get the metadata of the destination view
				const std::string
					& kDestinationImageName = (*kDestinationView.getMetadata().find("name")).second,
					& kDestinationImageExtension = (*kDestinationView.getMetadata().find("extension")).second;

				// A flag to see if the views are visible or not in the pose graph
				const bool kAreViewsVisible =
					visibilityTable.hasLink(kSourceViewIdx, kDestinationViewIdx);

				// Loading or detecting keypoints in both images
				std::vector<cv::KeyPoint> keypoints_1,
					keypoints_2;
				cv::Mat descriptors_1,
					descriptors_2;

				cv::Ptr<cv::hdf::HDF5> h5FeatureDatabase = cv::hdf::open(kKeypointDatabaseFilename_);

				loadFeatures(kSourceImageName, // The name of the source image
					kSourceImageExtension, // The extension of the source image
					kImagePath, // The path where the images are stored
					kUseGPU, // A flag to decide if GPU should be used
					h5FeatureDatabase, // The feature database file
					keypoints_1, // The found keypoints
					descriptors_1, // The descriptors of the keypoints
					featureDatabaseMutex); // A mutex to make the database writing/reading thread-safe

				loadFeatures(kDestinationImageName, // The name of the destination image
					kDestinationImageExtension, // The extension of the destination image
					kImagePath, // The path where the images are stored
					kUseGPU, // A flag to decide if GPU should be used
					h5FeatureDatabase, // The feature database file
					keypoints_2, // The found keypoints
					descriptors_2, // The descriptors of the keypoints
					featureDatabaseMutex); // A mutex to make the database writing/reading thread-safe
				h5FeatureDatabase->close();

				// The feature matches
				std::vector<std::tuple<size_t, size_t, double>> matches;

				// A flag to decide if matching on demand is turned on
				bool quickMatching = kUseEpipolarHashing;

				if (kUseEpipolarHashing)
				{
					start = std::chrono::system_clock::now();
					if (kAreViewsVisible)
						tracks.getCorrespondences(
							matches, // The indices of keypoints which compose the correspondences
							kSourceViewIdx, // The identifier of the source image
							kDestinationViewIdx, // The identifier of the destination image
							kMaximumTrackletNumber); // The maximum number of tracklets which are to be used
					end = std::chrono::system_clock::now();

					// Updating the statistics
					elapsedSeconds = end - start;
					statistics.addTime("[Quick matching]", elapsedSeconds.count());

					// If there are not enough correspondences from the tracklets
					if (matches.size() < kMinimumInlierNumber)
					{
						// Turn off epipolar-hashing-based quick matching
						quickMatching = false;
						// Clearing the matches
						matches.clear();
					}
					else
					{
						// Updating the statistics
						statistics.addCount("[Quick matching] Runs", 1);
					}
				}

				if (!quickMatching)
				{
					// Loading or detecting point correspondences
					// TODO(danini): Add epipolar hashing
					cv::Ptr<cv::hdf::HDF5> h5CorrespondenceDatabase = cv::hdf::open(kCorrespondenceDatabaseFilename_);

					start = std::chrono::system_clock::now();
					matchFeatures(
						kSourceImageName, // The name of the source image
						kDestinationImageName, // The name of the destination image
						keypoints_1, // The found keypoints in the source image
						keypoints_2, // The found keypoints in the destination image
						descriptors_1, // The descriptors of the found keypoints in the source image
						descriptors_2, // The descriptors of the found keypoints in the destination image
						kUseGPU, // A flag to decide if GPU should be used
						h5CorrespondenceDatabase, // The correspondence database file
						matches, // The found matches
						correspondenceDatabaseMutex);  // A mutex to make the database writing/reading thread-safe
					end = std::chrono::system_clock::now();
					h5CorrespondenceDatabase->close();

					// Updating the statistics
					elapsedSeconds = end - start;
					statistics.addTime("[Matching]", elapsedSeconds.count());
					statistics.addCount("[Matching] Runs", 1);
				}

				// Continue if there are not enough correspondences found
				if (matches.size() < kMinimumPointNumber)
					continue;

				// Create the correspondence matrix
				double normalizedThreshold = 0.0;
				cv::Mat correspondences;
				createCorrespondenceMatrix(
					reconstruction_, // The reconstruction object containing e.g. the camera data
					kInlierOutlierThreshold_, // The manually set inlier-outlier threshold
					kSourceViewIdx, // The index of the source view
					kDestinationViewIdx, // The index of the destination view
					keypoints_1, // The keypoints in the source image
					keypoints_2, // The keypoints in the destination image
					matches, // The feature matches
					correspondences, // The correspondence matrix containing the normalizing coordinates
					normalizedThreshold); // The threshold normalized by the camera matrices

				// There is a path in the pose-graph between the current view pair
				std::vector<Sophus::SE3d> posesFromPaths; // The poses determined by the pose graph traversal
				if (kUsePathFinding &&
					kAreViewsVisible)
				{
					size_t nodesTouched,
						pathsTested;

					// The starting time of the path finding
					start = std::chrono::system_clock::now();

					// A function to apply the pose-graph traversal
					bool success = findPath(
						similarityTable_, // The matrix containing the global similarities
						reconstruction_, // The reconstruction containing the image data
						poseGraph_, // The current pose graph
						visibilityTable, // The visibility table
						kSourceView, // The source view
						kDestinationView, // The destination view
						normalizedThreshold, // The threshold normalized by the camera matrices
						correspondences, // The normalized point correspondences
						nodesTouched, // The number of nodes touched
						pathsTested, // The number of paths found
						posesFromPaths); // The poses determined by the pose graph traversal

					// The end time of the path finding
					end = std::chrono::system_clock::now();

					// Calculating the time of the path finding
					elapsedSeconds = end - start;

					// Updating the statistics file
					statistics.addTime("[A*]", elapsedSeconds.count());
					statistics.addCount("[A*] Runs", 1);
					statistics.addCount("[A*] Touched nodes", nodesTouched);
					statistics.addCount("[A*] Paths tested", pathsTested);
				}

				// The estimated relative pose
				Sophus::SE3d estimatedPose;
				// The inlier mask of the found essential matrix
				std::vector<uchar> inlierMask;
				// The number of inliers
				size_t inlierNumber;

				// The starting time of the pose estimation
				start = std::chrono::system_clock::now();

				// Estimated the pose either from a path or by MAGSAC++
				bool success = estimatePose(
					reconstruction_, // The reconstruction object containing e.g. the camera parameters
					kInlierOutlierThreshold_, // The manually set inlier-outlier threshold
					kMinimumInlierNumber_, // The minimum inlier number to accept the pose
					kSourceViewIdx, // The index of the source view
					kDestinationViewIdx, // The index of the destination view
					correspondences, // The normalized point correspondences
					normalizedThreshold, // The threshold normalized by the camera intrinsics
					posesFromPaths, // The poses determined by A*
					estimatedPose, // The estimated pose
					inlierMask, // The inlier mask of the estimated essential matrix
					inlierNumber); // The inlier number of the estimated essential matrix

				// The end time of the pose estimation
				end = std::chrono::system_clock::now();

				// Calculating the time of the pose estimation
				elapsedSeconds = end - start;

				// Updating the statistics
				statistics.addCount("[Pose estimation] Runs", 1);
				statistics.addTime("[Pose estimation]", elapsedSeconds.count());
				statistics.addCount("[Pose estimation] Inlier number", inlierNumber);

				// If the relative pose estimation failed, continue to the next image pair
				if (!success)
					continue;

				// Calculate the inlier ratio to get some similarity besides the predicted one for the edge
				const double kInlierRatio =
					static_cast<double>(inlierNumber) / static_cast<double>(matches.size());

				// Put the new edge into the pose graph
				poseGraph_.addEdge(kSourceViewIdx, // The id of the source view
					kDestinationViewIdx, // The id of the destination view
					reconstruction::PoseGraphEdge(kSourceViewIdx,
						kDestinationViewIdx,
						reconstruction::Pose(estimatedPose),
						kInlierRatio)); // The edge with the estimated pose

				// Finding additional correspondences by epipolar hashing
				if (kUseEpipolarHashing && // If epipolar hashing is requires
					quickMatching) // and quick matching was applied.
				{
					std::vector<std::tuple<size_t, size_t, double>> additionalMatches;

					// The end time of epipolar hashing
					start = std::chrono::system_clock::now();
					guidedMatching(reconstruction_,
						kSourceViewIdx,
						kDestinationViewIdx,
						descriptors_1,
						descriptors_2,
						keypoints_1,
						keypoints_2,
						estimatedPose,
						normalizedThreshold,
						additionalMatches);

					std::vector<uchar> additionalInlierMask(additionalMatches.size(), 1);

					tracks.add(kSourceViewIdx,
						kDestinationViewIdx,
						additionalMatches,
						additionalInlierMask);
					// The end time of epipolar hashing
					end = std::chrono::system_clock::now();

					statistics.addTime("[Epipolar Hashing]", elapsedSeconds.count());
					statistics.addCount("[Epipolar Hashing] Runs", 1);
					statistics.addCount("[Epipolar Hashing] Correspondences added", additionalMatches.size());
				}

				// The starting time of updating the visibility pose graph
				start = std::chrono::system_clock::now();
				// Add a new link into the visibility structure
				visibilityTable.addLink(kSourceViewIdx, kDestinationViewIdx);
				// The end time of updating the visibility pose graph
				end = std::chrono::system_clock::now();
				// Calculating the time of the visibility update
				elapsedSeconds = end - start;
				// Updating the statistics file
				statistics.addTime("[Visibility update]", elapsedSeconds.count());
				statistics.addCount("[Visibility update] Runs", 1);

				// Update the tracklets
				if (kUseEpipolarHashing &&
					!quickMatching)
					tracks.add(kSourceViewIdx,
						kDestinationViewIdx,
						matches,
						inlierMask);
			}
		}

		// Printing the statistics of the run
		statistics.print();

		printf("Edges in the pose-graph = %d\n", poseGraph_.numEdges());
	}

	void PoseGraphBuilder::guidedMatching(
		const reconstruction::Reconstruction& kReconstruction_, // The reconstruction object
		const reconstruction::ViewId& kSourceViewIdx_, // The index of the source view
		const reconstruction::ViewId& kDestinationViewIdx_, // The index of the destination view
		const cv::Mat& kSourceDescriptors_, // The descriptors of the keypoints in the source image
		const cv::Mat& kDestinationDescriptors_, // The descriptors of the keypoints in destination image
		const std::vector<cv::KeyPoint>& kSourceKeypoints_, // The keypoints int source image
		const std::vector<cv::KeyPoint>& kDestinationKeypoints_, // The keypoints in destination image 
		const reconstruction::Pose& kPose_, // The relative pose
		const double& kInlierOutlierThreshold_, // The inlier-outlier threshold
		std::vector<std::tuple<size_t, size_t, double>>& matches_) // The point correspondences found
	{
		// Getting the cameras
		const reconstruction::PinholeCamera& kSourceCamera = kReconstruction_.getCamera(kSourceViewIdx_),
			& kDestinationCamera = kReconstruction_.getCamera(kDestinationViewIdx_);

		// Initializing the matcher
		reconstruction::pointmatcher::PointMatcherWithPose* guidedMatcher;
		static cv::Mat emptyImage(0, 0, CV_64F);

		// The guided matcher object performing Epipolar Hashing
		guidedMatcher = new reconstruction::pointmatcher::HashingBasedMatcherWithPose<false, 45>(
			emptyImage, // An empty image. This parameter is for testing purposes.
			emptyImage, // An empty image. This parameter is for testing purposes.
			cv::Size(kSourceCamera.getWidth(), kSourceCamera.getHeight()), // The size of the source image
			cv::Size(kDestinationCamera.getWidth(), kDestinationCamera.getHeight()), // The size of the destination image
			kSourceDescriptors_, // The descriptors in the source image
			kDestinationDescriptors_); // The descriptors in the destination image

		std::vector<std::pair<size_t, size_t>> matches;
		std::vector<double> descriptorDistances;

		guidedMatcher->match(
			matches,
			kSourceKeypoints_,
			kDestinationKeypoints_,
			kPose_,
			kInlierOutlierThreshold_,
			descriptorDistances,
			&kSourceCamera.getIntrinsics(),
			&kDestinationCamera.getIntrinsics());

		if (matches.size() > kMaximumPointNumberForEpipolarHashing)
		{
			std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>>, std::greater<std::pair<double, size_t>> > matchQueue;
			for (size_t i = 0; i < matches.size(); ++i)
				matchQueue.emplace(std::make_pair(descriptorDistances[i], i));

			matches_.reserve(kMaximumPointNumberForEpipolarHashing);
			while (!matchQueue.empty() && matches_.size() < kMaximumPointNumberForEpipolarHashing)
			{
				const auto& match = matches[matchQueue.top().second];
				matches_.emplace_back(std::make_tuple(match.first, match.second, matchQueue.top().first));
				matchQueue.pop();
			}
		}
		else
		{
			matches_.reserve(matches.size());
			for (size_t matchIdx = 0; matchIdx < matches.size(); ++matchIdx)
			{
				const auto& match = matches[matchIdx];
				matches_.emplace_back(
					std::make_tuple(match.first, match.second, descriptorDistances[0]));
			}
		}
	}

	bool PoseGraphBuilder::findPath(
		const reconstruction::SimilarityTable& kSimilarityTable_, // The similarity table containing the pair-wise image similarities
		const reconstruction::Reconstruction& kReconstruction_, // The reconstruction object
		const reconstruction::PoseGraph& kPoseGraph_, // The pose graph
		const reconstruction::VisibilityTable& kVisibilityTable_, // The visibility table to see if there is a path between two views
		const reconstruction::View& kSourceView_, // The source view
		const reconstruction::View& kDestinationView_, // The destination view
		const double kThreshold_, // The inlier-outlier threshold
		cv::Mat& correspondences_, // The point correspondences
		size_t& nodesTouched_, // The number of touched nodes during the A* algorithm
		size_t& pathsTested_, // The number of paths tested before finding a good one
		std::vector<Sophus::SE3d>& poses_) // The found poses
	{
		constexpr double kThresholdMultiplier = 3.0 / 2.0;
		constexpr size_t kMaximumPathNumber = 1;
		const double kTruncatedThreshold = kThresholdMultiplier * kThreshold_;

		// An object knowing how to calculate the point-to-model residuals
		reconstruction::EssentialMatrixEvaluator essentialMatrixEvaluator;

		// Initialize the pose tested which is applying during the
		// traversal whenever a new candidate pose is found.
		const reconstruction::InTraversalPoseTester<reconstruction::EssentialMatrixEvaluator> kPoseTester(
			kTruncatedThreshold,
			5,
			&essentialMatrixEvaluator,
			&correspondences_);

		// The container where the recovered paths will be kept
		std::vector<std::vector<const reconstruction::PoseGraphVertex*>> recovered_paths;
		recovered_paths.reserve(kMaximumPathNumber);
		// The container where the recovered poses will be kept.
		std::vector<Sophus::SE3d> poses;
		poses.reserve(kMaximumPathNumber);
		// A flag to decide if a correct path has been found
		bool pathExists = false;
		// The number of nodes visited during the traversal
		nodesTouched_ = 0;

		LOG(INFO) << "Running the A* graph traversal algorithm";
		typedef reconstruction::AStarTraversal<reconstruction::ImageSimilarityHeuristics> AStar;

		// Update the weight of the heuristics
		reconstruction::CostComparator::weight = kTraversalHeuristicsWeight;

		// Initialize the heuristics with the similarity table
		reconstruction::ImageSimilarityHeuristics heuristics(kSimilarityTable_);

		// Initialize the traversal object
		const AStar traversal(
			&kPoseGraph_, // The pose graph
			heuristics, // The heuristics to be used
			&kPoseTester, // The pose tester applied to decide if a pose is good immediately after a path is found
			true, // A flag to decide if multiple paths should be returned
			0.0, // The minimum inlier ratio to accept an edge
			kMaximumSearchDepth, // The maximum depth
			kMaximumPathNumber); // The maximum number of paths to test

		// Recover the path and pose
		traversal.getPath(
			kPoseGraph_.getVertexById(kSourceView_.id()), // The source view
			kPoseGraph_.getVertexById(kDestinationView_.id()), // The destination view
			recovered_paths, // The found paths
			poses_, // The poses that the found paths imply
			nodesTouched_, // The number of nodes touched during the traversal
			pathsTested_, // The number of paths tested to find a good one
			pathExists); // A flag to see if a path exists

		if (pathExists)
			LOG(INFO) <<
			"Path found between views (" << kSourceView_.id() << ", " << kDestinationView_.id() << "). Statistics:" <<
			"\n\tNumber of nodes touched = " << nodesTouched_ <<
			"\n\tNumber of paths tested = " << pathsTested_;
		else
			LOG(INFO) << "No path is found between views (" << kSourceView_.id() << ", " << kDestinationView_.id() << ")";

		return false;
	}

	void PoseGraphBuilder::createCorrespondenceMatrix(
		const reconstruction::Reconstruction& reconstruction_, // The reconstruction object
		const double& kInlierOutlierThreshold_, // The inlier-outlier threshold
		const size_t& kSourceViewIdx_, // The index of the source view
		const size_t& kDestinationViewIdx_, // The index of the destination view
		const std::vector<cv::KeyPoint>& kSourceKeypoints, // The keypoints in the source image
		const std::vector<cv::KeyPoint>& kDestinationKeypoints, // The keypoints in the destination image
		const std::vector<std::tuple<size_t, size_t, double>>& kMatches_, // The point matches
		cv::Mat& correspondences_, // The normalized correspondences
		double& normalizedThreshold_) // The normalized inlier-outlier threshold
	{
		// Getting the cameras
		const reconstruction::PinholeCamera& sourceCamera = reconstruction_.getCamera(kSourceViewIdx_),
			& destinationCamera = reconstruction_.getCamera(kDestinationViewIdx_);

		// Copy the keypoints so the same index refers to the same match
		const size_t& kCorrespondenceNumber = kMatches_.size();
		correspondences_.create(kCorrespondenceNumber, 4, CV_64F);
		double* correspondencesPtr = reinterpret_cast<double*>(correspondences_.data);

		for (size_t matchIdx = 0; matchIdx < kCorrespondenceNumber; ++matchIdx)
		{
			// The (source index, destination index, SIFT ratio) tuple
			const auto& indexTuple = kMatches_[matchIdx];
			// The source index
			const auto& sourcePointIdx = std::get<0>(indexTuple);
			// The destination index
			const auto& destinationPointIdx = std::get<1>(indexTuple);

			// Copy the coordinates to the matrix
			*(correspondencesPtr++) = kSourceKeypoints[sourcePointIdx].pt.x;
			*(correspondencesPtr++) = kSourceKeypoints[sourcePointIdx].pt.y;
			*(correspondencesPtr++) = kDestinationKeypoints[destinationPointIdx].pt.x;
			*(correspondencesPtr++) = kDestinationKeypoints[destinationPointIdx].pt.y;
		}

		// Normalizing the point correspondences by the inverse camera matrices.
		// We don't use matrix multiplication since that is slower.
		const double
			& sourceFocalLengthX = sourceCamera.getIntrinsics()(0, 0),
			& sourceFocalLengthY = sourceCamera.getIntrinsics()(1, 1),
			& sourcePrincipalPointX = sourceCamera.getIntrinsics()(0, 2),
			& sourcePrincipalPointY = sourceCamera.getIntrinsics()(1, 2);

		const double
			& destinationFocalLengthX = sourceCamera.getIntrinsics()(0, 0),
			& destinationFocalLengthY = sourceCamera.getIntrinsics()(1, 1),
			& destinationPrincipalPointX = sourceCamera.getIntrinsics()(0, 2),
			& destinationPrincipalPointY = sourceCamera.getIntrinsics()(1, 2);

		// Reset the pointer to the beginning of the matrix
		correspondencesPtr = reinterpret_cast<double*>(correspondences_.data);

		for (size_t matchIdx = 0; matchIdx < kCorrespondenceNumber; ++matchIdx)
		{
			// Normalize the X coordinate in the source image
			*correspondencesPtr = (*correspondencesPtr - sourcePrincipalPointX) / sourceFocalLengthX;
			++correspondencesPtr;
			// Normalize the Y coordinate in the source image
			*correspondencesPtr = (*correspondencesPtr - sourcePrincipalPointY) / sourceFocalLengthY;
			++correspondencesPtr;
			// Normalize the X coordinate in the destination image
			*correspondencesPtr = (*correspondencesPtr - destinationPrincipalPointX) / destinationFocalLengthX;
			++correspondencesPtr;
			// Normalize the Y coordinate in the destination image
			*correspondencesPtr = (*correspondencesPtr - destinationPrincipalPointY) / destinationFocalLengthY;
			++correspondencesPtr;
		}

		// Normalizing the threshold as well
		const double kThresholdNormalizer =
			(sourceFocalLengthX + sourceFocalLengthY + destinationFocalLengthX + destinationFocalLengthY) / 4.0;
		normalizedThreshold_ =
			kInlierOutlierThreshold_ / kThresholdNormalizer;
	}

	bool PoseGraphBuilder::estimatePose(
		const reconstruction::Reconstruction& reconstruction_, // The reconstruction object
		const double& kInlierOutlierThreshold_, // The inlier-outlier threshold
		const size_t kMinimumInlierNumber_, // The minimum inlier number to accept a pose
		const size_t& kSourceViewIdx_, // The index of the source view
		const size_t& kDestinationViewIdx_, // The index of the destination view
		const cv::Mat& kCorrespondences_, // The normalized point correspondences
		const double kThreshold_, // The inlier-outlier threshold
		const std::vector<Sophus::SE3d>& poseGuesses_, // The poses from the A* traversal 
		Sophus::SE3d& estimatedPose_, // The estimated relative pose
		std::vector<uchar>& inlierMask_, // The inlier mask
		size_t& inlierNumber_) // The inlier number of the pose
	{
		// OpenCV is not able to handle if the images have different camera matrices.
		// Therefore, the points are premultiplied by the inverse intrinsic matrices
		// and we pass only an identity matrix to OpenCV.
		static const cv::Mat identityCameraMatrix =
			cv::Mat::eye(3, 3, CV_64F);

		// The number of correspondences
		const size_t kCorrespondenceNumber = kCorrespondences_.rows;

		// The truncated MSAC threshold
		constexpr double kThresholdMultiplier = 3.0 / 2.0;
		const double kTruncatedThreshold = kThresholdMultiplier * kThreshold_;

		// An object to calculate the point-to-model residuals
		reconstruction::EssentialMatrixEvaluator essentialMatrixEvaluator;
		std::vector<size_t> inliers;
		cv::Mat tempCorrespondences;
		bool success = false;
		cv::Mat cvEssentialMatrix;

		// Iterate through the pose guesses and check if they are good enough
		for (const auto& kPose : poseGuesses_)
		{
			// Clearing the inlier vector and the temp correspondence matrix
			inliers.clear();
			tempCorrespondences.release();
			// The current pose object
			reconstruction::Pose pose(kPose);
			// The essential matrix
			const auto& kEssentialMatrix = pose.getEssentialMatrix();

			// Selecting the inliers of the current essential matrix
			essentialMatrixEvaluator.getInliers(
				kCorrespondences_, // The point correspondences
				kEssentialMatrix, // The essential matrix
				kTruncatedThreshold, // The truncated MSAC threshold
				inliers); // The inliers' indices

			// The number of inliers
			const size_t& kInlierNumber =
				inliers.size();

			// Creating a correspondence matrix consisting only of the inliers
			tempCorrespondences.create(kInlierNumber, kCorrespondences_.cols, kCorrespondences_.type());

			// Copying the inliers' coordinates
			// TODO(danini): It is a waste of time but OpenCV does not solve the problem otherwise
			inlierMask_.resize(kCorrespondenceNumber, 0);
			for (size_t inlierIdx = 0; inlierIdx < kInlierNumber; ++inlierIdx)
			{
				// The index of the point
				const auto& pointIdx = inliers[inlierIdx];
				// Copying the coordinates of the inliiers
				kCorrespondences_.row(pointIdx).copyTo(tempCorrespondences.row(inlierIdx));
				// Setting the inlier mask
				inlierMask_[pointIdx] = 1;
			}

			// Applying OpenCV's RANSAC with very wide threshold so it acts as a non-minimal fitting algorithm
			std::vector<uchar> tmpInlierMask(kInlierNumber, 0);
			cvEssentialMatrix = cv::findEssentialMat(
				tempCorrespondences(cv::Rect(0, 0, 2, kInlierNumber)), // The correspondences in the source image
				tempCorrespondences(cv::Rect(2, 0, 2, kInlierNumber)), // The correspondences in the destination image
				identityCameraMatrix, // Identity camera matrix
				cv::RANSAC, // The robust estimator's flag
				0.99, // RANSAC confidence
				std::numeric_limits<double>::max(), // The normalized inlier-outlier threshold
				tmpInlierMask); // The inlier mask

			inlierNumber_ =
				std::accumulate(std::begin(tmpInlierMask), std::end(tmpInlierMask), 0);

			LOG(INFO) << "Inlier number from the pose = " << inlierNumber_;

			// A variable to see if the pose from the path has been successfully recovered
			success = inlierNumber_ >= kMinimumInlierNumber_;
		}

		if (!success)
		{
			// The inlier mask
			inlierMask_.resize(kCorrespondenceNumber, 0);

			// Applying MAGSAC++ implemented in OpenCV
			cvEssentialMatrix = cv::findEssentialMat(
				kCorrespondences_(cv::Rect(0, 0, 2, kCorrespondenceNumber)), // The correspondences in the source image
				kCorrespondences_(cv::Rect(2, 0, 2, kCorrespondenceNumber)), // The correspondences in the destination image
				identityCameraMatrix, // Identity camera matrix
				cv::USAC_MAGSAC, // The robust estimator's flag
				0.99, // RANSAC confidence
				kThreshold_, // The normalized inlier-outlier threshold
				inlierMask_); // The inlier mask

			// Counting the inliers, i.e. ones in the inlier mask 
			inlierNumber_ =
				std::accumulate(std::begin(inlierMask_), std::end(inlierMask_), 0);

			LOG(INFO) << "Inlier number = " << inlierNumber_;

			// Return if there are not enough inliers found
			if (inlierNumber_ < kMinimumInlierNumber_)
				return false;
		}

		Eigen::Map<Eigen::Matrix3d> essentialMatrixTranspose(cvEssentialMatrix.ptr<double>()); // Mapping the estimated essential matrix to Eigen format
		Eigen::Matrix3d rotation; // The estimated rotation matrix
		Eigen::Vector3d translation; // The estimated translation vector

		// Decompose the essential matrix to pose
		reconstruction::pose::getPoseFromEssentialMatrix(
			essentialMatrixTranspose.transpose(), // The essential matrix
			kCorrespondences_, // The point correspondences 
			rotation, // The estimated rotation matrix
			translation); // The estimated translation vector

		// Checking if the estimated matrices have NaN elements
		if (rotation.hasNaN() || translation.hasNaN())
			return false;

		// Outputting the estimated pose
		estimatedPose_ = Sophus::SE3d(
			Eigen::Quaterniond(rotation), // Converting the rotation matrix to quaternion
			translation); // The estimated translation vector

		return true;
	}

	void PoseGraphBuilder::initializeKeypointDatabase(
		const reconstruction::Reconstruction& reconstruction_, // The reconstruction object
		const std::string& kKeypointDatabaseFilename_, // The path of the keypoint database
		const reconstruction::SimilarityTable& similarityTable_, // The similarity table containing the pair-wise image similarities
		const size_t& kImageNumber_) // The number of images
	{
		// Collecting images that has a reasonable similarity with some other view 
		std::vector<size_t> viewIndices;
		viewIndices.reserve(kImageNumber_);
		for (const auto& viewIdx : similarityTable_.getKeptViews())
			viewIndices.emplace_back(viewIdx);

		// A mutex to make the database writing thread-safe
		std::shared_mutex featureDatabaseMutex;
		// Checking if the database file exists
		bool databaseExists = std::filesystem::exists(kKeypointDatabaseFilename_);
		// Opening or creating the database
		cv::Ptr<cv::hdf::HDF5> h5FeatureDatabase = cv::hdf::open(kKeypointDatabaseFilename_);

		// If the database file exists and it has not finished that is likely due to some interruption.
		// HDF in OpenCV has a tendency to end up in a corrupted file when interrupted.
		// So we delete the file and create it again.
		if (databaseExists && !h5FeatureDatabase->atexists("finished"))
		{
			// Closing the database file
			h5FeatureDatabase->close();
			// Deleting the file
			std::filesystem::remove(kKeypointDatabaseFilename_);
			// Re-creating an empty database
			h5FeatureDatabase = cv::hdf::open(kKeypointDatabaseFilename_);
		}

		// If the database has not been finished yet
		if (!h5FeatureDatabase->atexists("finished"))
		{
			size_t processedImages = 0;
			std::mutex printingMutex;

			// Iterate over the requires set of view indices and detect keypoints in each view
#pragma omp parallel for num_threads(kCoreNumber)
			for (int indexIdx = 0; indexIdx < viewIndices.size(); ++indexIdx)
			{
				printingMutex.lock();
				++processedImages;
				if (processedImages % 100 == 0)
					printf("Keypoint detection [%d/%d]\n", processedImages, viewIndices.size());
				printingMutex.unlock();

				// The current view index
				const auto& viewIdx = viewIndices[indexIdx];
				// The vector of detected keypoints
				std::vector<cv::KeyPoint> keypoints;
				// The descriptors
				cv::Mat descriptors;

				// The focal length of the current view
				const auto& kFocalLength =
					reconstruction_.getCamera(viewIdx).getIntrinsics()(0, 0);

				// If the focal length is ~0, it has not been read a priori.
				// We aim at estimating essential matrices, thus, these cases 
				// currently are ignored.
				// TODO(danini): consider fundamental matrix estimation in these cases
				if (kFocalLength <= std::numeric_limits<double>::epsilon())
					continue;

				// The current view
				const reconstruction::View& kView =
					reconstruction_.getView(viewIdx);

				// Get the metadata of the source view
				const std::string
					& kImageName = (*kView.getMetadata().find("name")).second, // The name of the image
					& kImageExtension = (*kView.getMetadata().find("extension")).second; // The extension of the image

				// Load or detect features
				loadFeatures(kImageName, // The name of the image
					kImageExtension, // The extension of the image
					kImagePath, // The path where the images are stored
					kUseGPU,
					h5FeatureDatabase, // The database file
					keypoints, // The detected keypoints
					descriptors, // The detected keypoints' descriptors
					featureDatabaseMutex); // A mutex making the reading and writing thread-safe
			}

			// Put a flag to the database saying the it has been created successfully
			if (!h5FeatureDatabase->atexists("finished"))
				h5FeatureDatabase->atwrite(1, "finished");
		}
		// Closing the database
		h5FeatureDatabase->close();
	}
}