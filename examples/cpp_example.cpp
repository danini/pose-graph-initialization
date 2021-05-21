#include <vector>	
#include <thread>
#include <fstream>
#include <sstream>
#include <set>
#include <unordered_set>
#include <string>
#include <algorithm>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/hdf.hpp>
#include <Eigen/Eigen>
#include <shared_mutex>
#include <stdio.h>

#include "reconstruction.h"
#include "pose_graph.h"
#include "pose_graph_builder.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#ifdef _WIN32
	#include <direct.h>
#endif 

DEFINE_int32(core_number, 20,
	"Number of CPU cores used");
DEFINE_bool(use_gpu, true,
	"A flag determining if GPU should be used where possible.");
DEFINE_bool(use_path_finding, true,
	"A flag determining if path finding should be used.");
DEFINE_bool(use_epipolar_hashing, true,
	"A flag determining if epipolar hashing should be used.");
DEFINE_string(workspace_path, "d:/Kutatas/PoseGraphRANSAC/git/data/Madrid_Metropolis/images/",
	"The path where everything will be saved.");
DEFINE_string(image_path, "d:/Kutatas/PoseGraphRANSAC/git/data/Madrid_Metropolis/images/",
	"The path where the images are stored.");
DEFINE_string(focal_length_path, "d:/Kutatas/PoseGraphRANSAC/git/data/Madrid_Metropolis/images/list_with_focals.txt",
	"The path where the file containing the focal lengths is stored.");
DEFINE_string(similarity_graph_path, "d:/Kutatas/PoseGraphRANSAC/git/data/Madrid_Metropolis/images/Madrid_Metropolis_resnet50_similarity.txt",
	"The path where the image similarity matrix is stored.");

DEFINE_double(similarity_threshold, 0.5,
	"The lower bound of image similarity between image pairs.");
DEFINE_double(inlier_outlier_threshold, 0.4,
	"The inlier-outlier threshold for essential matrix fitting.");
DEFINE_int32(minimum_inlier_number, 20,
	"The minimum number of inliers for a pose to be accepted.");
DEFINE_int32(minimum_point_number, 50,
	"The minimum number of points for the robust estimation.");
DEFINE_double(traversal_heuristics_weight, 0.8,
	"The weighting parameter in the heuristics for the A* traversal.");
DEFINE_int32(maximum_path_number, 100,
	"An upper bound for the number of paths to be tested.");
DEFINE_int32(maximum_search_depth, 5,
	"An upper bound for the search depth.");
DEFINE_int32(maximum_tracklet_number, 5000,
	"The maximum number of tracklets used for the robust estimation.");
DEFINE_int32(maximum_points_from_epipolar_hashing, 100,
	"The maximum number of points accepted when doing epipolar hashing");

bool checkSetting();

int main(int argc, char** argv)
{
	// Parsing the flags
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// Initialize Google's logging library.
	google::InitGoogleLogging(argv[0]);

	// Check if the flags have been set correctly
	if (!checkSetting())
		return 0;

	// Initialize the reconstruction
	reconstruction::Reconstruction reconstruction;
	reconstruction::PoseGraph poseGraph;

	reconstruction::PoseGraphBuilder builder(
		FLAGS_core_number,
		FLAGS_maximum_tracklet_number,
		FLAGS_maximum_search_depth,
		FLAGS_maximum_path_number,
		FLAGS_minimum_inlier_number,
		FLAGS_minimum_point_number,
		FLAGS_maximum_points_from_epipolar_hashing,
		FLAGS_traversal_heuristics_weight,
		FLAGS_similarity_threshold,
		FLAGS_inlier_outlier_threshold,
		FLAGS_image_path,
		FLAGS_workspace_path,
		FLAGS_similarity_graph_path,
		FLAGS_focal_length_path,
		FLAGS_use_path_finding,
		FLAGS_use_gpu,
		FLAGS_use_epipolar_hashing);

	builder.run(reconstruction,
		poseGraph);

	return 0;
}

bool checkSetting()
{
	if (FLAGS_image_path == "")
	{
		LOG(ERROR) << "The path of the images has not been set.";
		return false;
	}

	std::ifstream file(FLAGS_similarity_graph_path);
	if (!file.is_open())
	{
		LOG(ERROR) << "The image similarity path ('" << FLAGS_similarity_graph_path << "') is incorrect.";
		return false;
	}
	file.close();

	file.open(FLAGS_focal_length_path);
	if (!file.is_open())
	{
		LOG(ERROR) << "The focal length path ('" << FLAGS_focal_length_path << "') is incorrect.";
		return false;
	}
	file.close();
	return true;
}
