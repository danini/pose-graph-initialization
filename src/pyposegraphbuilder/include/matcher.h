#pragma once

#define _USE_MATH_DEFINES
#include <cmath> 
#include <map>
#include <vector>
#include "types.h"
#include "pose.h"
#include <opencv2/flann.hpp>
#include <opencv2/flann/dist.h>

namespace reconstruction
{
	namespace pointmatcher
	{
		// Virtual guided feature point matcher function where the relative pose is available
		class PointMatcherWithPose
		{
		public:
			virtual void match(
				std::vector<std::pair<size_t, size_t>> &matches_, // Output: The matching keypoints' indices
				const std::vector<cv::KeyPoint> &keypointsSource_, // The keypoints in the source image
				const std::vector<cv::KeyPoint> &keypointsDestination_, // The keypoints in the destination image
				const Pose &pose_, // The relative pose from the source to the destination images
				const double &threshold_, // The inlier-outlier threshold used for the pose fitting
				std::vector<double> &descriptorDistances,
				const Eigen::Matrix3d *intrinsicsSource_ = nullptr, // The intrinsic camera parameters of the source camera
				const Eigen::Matrix3d *intrinsicsDestination_ = nullptr, // The intrinsic camera parameters of the destination camera
				const std::vector<std::pair<size_t, size_t>> *indexPairs_ = nullptr, // The matching keypoints' indices if there are available ones
				const std::vector<size_t> *inliers_ = nullptr) const = 0; // The indices of inlier referring to elements in vairable "indexPairs_"
		};

		// The symmetric epipolar distance used in OpenCV's FLANN
		template<class T>
		struct SymmetricEpipolarDistanceFlann
		{
			typedef cvflann::True is_kdtree_distance;
			typedef cvflann::True is_vector_space_distance;

			typedef T ElementType;
			typedef typename cvflann::Accumulator<T>::Type ResultType;
			typedef ResultType CentersType;

			const Pose pose;

			SymmetricEpipolarDistanceFlann(Pose pose_) : pose(pose_)
			{

			}

			/**
			 *  The symmetric epipolar distance
			 */
			template <typename Iterator1, typename Iterator2>
			ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
			{
				const double &x1 = (ResultType)*a;
				++a;
				const double &y1 = (ResultType)*a;
				++a;
				const double &x2 = (ResultType)*b;
				++b;
				const double &y2 = (ResultType)*b;
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

				double squaredSymmetricEpipolarDistance =
					(ResultType)(r * r * (a1 + b1) / (a1 * b1));
				return (ResultType)squaredSymmetricEpipolarDistance;
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

		// The Sampson distance used in OpenCV's FLANN
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
			 *  Compute the chi-square distance
			 */
			template <typename Iterator1, typename Iterator2>
			ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
			{
				const double &x1 = (ResultType)*a;
				++a;
				const double &y1 = (ResultType)*a;
				++a;
				const double &x2 = (ResultType)*b;
				++b;
				const double &y2 = (ResultType)*b;
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

				double rxc = e11 * x2 + e21 * y2 + e31;
				double ryc = e12 * x2 + e22 * y2 + e32;
				double rwc = e13 * x2 + e23 * y2 + e33;
				double r = (x1 * rxc + y1 * ryc + rwc);
				double rx = e11 * x1 + e12 * y1 + e13;
				double ry = e21 * x1 + e22 * y1 + e23;

				double squaredSampsonDistance = r * r /
					(rxc * rxc + ryc * ryc + rx * rx + ry * ry);
				return (ResultType)squaredSampsonDistance;
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

		// Hashing-based guided matching with pose
		template <bool _isSampsonDistance = 1, int _binNumber = 180>
		class HashingBasedMatcherWithPose : public PointMatcherWithPose
		{
		protected:
			const cv::Mat &descriptorsSource, // The descriptors in the source image
				&descriptorsDestination; // The descriptors in the destination image
			const cv::Size &sourceImageSize,
				&destinationImageSize;
			const cv::Mat &imageSource,
				&imageDestination;

		public:

			HashingBasedMatcherWithPose(
				const cv::Mat &imageSource_,
				const cv::Mat &imageDestination_,
				const cv::Size &sourceImageSize_,
				const cv::Size &destinationImageSize_,
				const cv::Mat &descriptorsSource_, // The descriptors in the source image
				const cv::Mat &descriptorsDestination_) : // The descriptors in the destination image
				descriptorsSource(descriptorsSource_),
				descriptorsDestination(descriptorsDestination_),
				sourceImageSize(sourceImageSize_),
				destinationImageSize(destinationImageSize_),
				imageSource(imageSource_),
				imageDestination(imageDestination_),
				PointMatcherWithPose()
			{

			}

			void match(
				std::vector<std::pair<size_t, size_t>> &matches_, // Output: The matching keypoints' indices
				const std::vector<cv::KeyPoint> &keypointsSource_, // The keypoints in the source image
				const std::vector<cv::KeyPoint> &keypointsDestination_, // The keypoints in the destination image
				const Pose &pose_, // The relative pose from the source to the destination images
				const double &threshold_, // The inlier-outlier threshold used for the pose fitting
				std::vector<double> &descriptorDistances, 
				const Eigen::Matrix3d *intrinsicsSource_ = nullptr, // The intrinsic camera parameters of the source camera
				const Eigen::Matrix3d *intrinsicsDestination_ = nullptr, // The intrinsic camera parameters of the destination camera
				const std::vector<std::pair<size_t, size_t>> *indexPairs_ = nullptr, // The matching keypoints' indices if there are available ones
				const std::vector<size_t> *inliers_ = nullptr) const // The indices of inlier referring to elements in vairable "indexPairs_"
			{
				constexpr double kRadianToDegree = 180.0 / M_PI;

				// The essential matrix
				const Eigen::Matrix3d &essentialMatrix = pose_.getEssentialMatrix();
				const Eigen::Matrix3d fundamentalMatrix =
					(*intrinsicsDestination_).inverse().transpose() * essentialMatrix * (*intrinsicsSource_).inverse();

				// Decompose the essential matrix to get the epipole in the first image
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(
					fundamentalMatrix,
					Eigen::ComputeFullV);

				Eigen::Vector3d epipole =
					svd.matrixV().col(2);
				epipole /= epipole(2);

				const bool epipoleInImage =
					epipole(0) >= 0 && epipole(0) < sourceImageSize.width &&
					epipole(1) >= 0 && epipole(1) < sourceImageSize.height;

				double minAngle = 180,
					maxAngle = 0,
					angularRange;

				if (!epipoleInImage)
				{
					// Calculate the angular min/max of the epipolar lines' angle 
					const std::vector<double> cornerPoints = {
						0, 0,
						static_cast<double>(destinationImageSize.width), 0,
						static_cast<double>(destinationImageSize.width), static_cast<double>(destinationImageSize.height),
						0, static_cast<double>(destinationImageSize.height) };

					for (size_t coordinateIdx = 0; coordinateIdx < 8; coordinateIdx += 2)
					{
						const double &x = cornerPoints[coordinateIdx],
							&y = cornerPoints[coordinateIdx + 1];

						// The corresponding epipolar line in the source image.
						// The line is calculated as e = E * p2, where e is of form [nx, ny, c]^T.
						const double nx = fundamentalMatrix(0, 0) * x + fundamentalMatrix(1, 0) * y + fundamentalMatrix(2, 0);
						const double ny = fundamentalMatrix(0, 1) * x + fundamentalMatrix(1, 1) * y + fundamentalMatrix(2, 1);

						double angle = kRadianToDegree * std::atan2(ny, nx) + 180.0;
						if (angle > 180)
							angle -= 180;
						minAngle = MIN(minAngle, angle);
						maxAngle = MAX(maxAngle, angle);
					}
				}

				// Calculating the range of angle
				angularRange = maxAngle - minAngle;

				int binNumber;
				if constexpr (_binNumber <= 0) // If there is no bin number set, set it in a way that one bin will be one angle
					binNumber = static_cast<int>(angularRange);
				else
					binNumber = _binNumber;
						
				// The bins where the points are first hashed
				std::vector<std::vector<size_t>> epipolarBins(binNumber);
				// Occupy the max memory which could be needed
				for (auto &bin : epipolarBins)
					bin.reserve(keypointsDestination_.size());
							
				// Put each point in the points in the destination image to one of the bins
				for (size_t pointIdx = 0; pointIdx < keypointsDestination_.size(); ++pointIdx)
				{
					const auto &keypoint = keypointsDestination_[pointIdx];
					const auto &point = keypoint.pt;

					// The corresponding epipolar line in the source image.
					// The line is calculated as e = E * p2, where e is of form [nx, ny, c]^T.
					const double nx = fundamentalMatrix(0, 0) * point.x + fundamentalMatrix(1, 0) * point.y + fundamentalMatrix(2, 0);
					const double ny = fundamentalMatrix(0, 1) * point.x + fundamentalMatrix(1, 1) * point.y + fundamentalMatrix(2, 1);
					//const double c = fundamentalMatrix(0, 2) * point.x + fundamentalMatrix(1, 2) * point.y + fundamentalMatrix(2, 2);

					// TODO: try different hashing ideas
					// Current: hash the lines according to the angle of the line normal
					double angle = kRadianToDegree * std::atan2(ny, nx) + 180.0;
					if (angle > 180)
						angle -= 180;
					angle = (binNumber - 1) * (angle - minAngle) / angularRange;
					const int bin = MIN(MAX(0, static_cast<int>(round(angle))), binNumber - 1);
					epipolarBins[bin].emplace_back(pointIdx);
				}

				const double
					&e11 = fundamentalMatrix(0, 0),
					&e12 = fundamentalMatrix(0, 1),
					&e13 = fundamentalMatrix(0, 2),
					&e21 = fundamentalMatrix(1, 0),
					&e22 = fundamentalMatrix(1, 1),
					&e23 = fundamentalMatrix(1, 2),
					&e31 = fundamentalMatrix(2, 0),
					&e32 = fundamentalMatrix(2, 1),
					&e33 = fundamentalMatrix(2, 2);

				descriptorDistances.reserve(keypointsSource_.size());

				// Iterate through all points in the source image and get neighboring points from the destination one
				for (size_t pointIdx = 0; pointIdx < keypointsSource_.size(); ++pointIdx)
				{
					const auto &keypoint = keypointsSource_[pointIdx];
					const auto &point = keypoint.pt;

					// Calculate the direction of the epipolar line in the source image
					const double vx = point.x - epipole(0);
					const double vy = point.y - epipole(1);

					// Calculate the normal direction
					const double nx = -vy;
					const double &ny = vx;

					// Get the angle of the normal
					double angle = kRadianToDegree * atan2(ny, nx) + 180.0;
					if (angle > 180)
						angle -= 180;
					angle = (binNumber - 1) * (angle - minAngle) / angularRange;
					const int bin = MIN(MAX(0, static_cast<int>(round(angle))), binNumber - 1);

                    double secondBestDescriptorDistance  = std::numeric_limits<double>::max();
					double bestDescriptorDistance = std::numeric_limits<double>::max();
					int bestIndex = -1;

					const double &x1 = point.x;
					const double &y1 = point.y;
					
                    int count_snn = 0; // How many points we really have for Lowe check
                    double base_ratio_th_sq = 0.8*0.8; // Lowe threshold
                    double ratio_sq_correction = 1.0;


					for (const auto &neighborIdx : epipolarBins[bin])
					//for (size_t neighborIdx = 0; neighborIdx < descriptorsDestination.rows; ++neighborIdx)
					{
						const double &x2 = keypointsDestination_[neighborIdx].pt.x;
						const double &y2 = keypointsDestination_[neighborIdx].pt.y;

						const double rxc = e11 * x2 + e21 * y2 + e31;
						const double ryc = e12 * x2 + e22 * y2 + e32;
						const double rwc = e13 * x2 + e23 * y2 + e33;
						const double r = (x1 * rxc + y1 * ryc + rwc);
						const double rx = e11 * x1 + e12 * y1 + e13;
						const double ry = e21 * x1 + e22 * y1 + e23;
						const double a1 = rxc * rxc + ryc * ryc;
						const double b1 = rx * rx + ry * ry;
						
						double squaredSymmetricEpipolarDistance = r * r * (a1 + b1) / (a1 * b1);
                        
						if (squaredSymmetricEpipolarDistance >= 0.75 * 0.75) // We don't need to be so strict
                            continue;
                        
						count_snn += 1;

                        double descriptorDistance = 0, dist;

						for (size_t m = 0; m < descriptorsSource.cols; ++m)
						{
							dist = descriptorsSource.at<float>(pointIdx, m) - descriptorsDestination.at<float>(neighborIdx, m);
							descriptorDistance += dist * dist;
						}

						if (descriptorDistance < bestDescriptorDistance)
						{
							secondBestDescriptorDistance = bestDescriptorDistance;
							bestDescriptorDistance = descriptorDistance;
							bestIndex = neighborIdx;
						}
					}

                    // SNN ratio, corrected for the number of second nearest neighbors
                    if  (count_snn < 20) {
                        ratio_sq_correction = 0.65*0.65;
                    }
                    if  (count_snn < 10) {
                        ratio_sq_correction = 0.6*0.6;
                    }
                    if  (count_snn < 5) {
                        ratio_sq_correction = 0.5*0.5;
                    }
                    if  (count_snn < 3) {
                        ratio_sq_correction = 0.25*0.25;
                    }

                    const double dist_ratio_sq_adapted = (bestDescriptorDistance / secondBestDescriptorDistance) / ratio_sq_correction;
                    if (dist_ratio_sq_adapted  < 0.00001) continue; // Too good to be true :)
                    if (bestIndex > -1 && ((dist_ratio_sq_adapted < base_ratio_th_sq) || (count_snn == 1) ))
                    {
						matches_.emplace_back(std::make_pair(pointIdx, bestIndex));
                        descriptorDistances.emplace_back(dist_ratio_sq_adapted);
					}
				}
			}
		};

		// FLANN-based guided matching with pose
		template <bool _isSampsonDistance = 1>
		class FLANNBasedMatcherWithPose : public PointMatcherWithPose
		{
		protected:
			const cv::Mat &descriptorsSource, // The descriptors in the source image
				&descriptorsDestination; // The descriptors in the destination image
			const size_t nearestNeighborNumber;

		public:
			FLANNBasedMatcherWithPose(
				const cv::Mat &descriptorsSource_, // The descriptors in the source image
				const cv::Mat &descriptorsDestination_, // The descriptors in the destination image
				const size_t nearestNeighborNumber_ = 1) : // The number of nearest neighbor considered
				descriptorsSource(descriptorsSource_),
				descriptorsDestination(descriptorsDestination_),
				nearestNeighborNumber(nearestNeighborNumber_),
				PointMatcherWithPose()
			{

			}

			void match(
				std::vector<std::pair<size_t, size_t>> &matches_, // Output: The matching keypoints' indices
				const std::vector<cv::KeyPoint> &keypointsSource_, // The keypoints in the source image
				const std::vector<cv::KeyPoint> &keypointsDestination_, // The keypoints in the destination image
				const Pose &pose_, // The relative pose from the source to the destination images
				const double &threshold_, // The inlier-outlier threshold used for the pose fitting
				std::vector<double> &descriptorDistances,
				const Eigen::Matrix3d *intrinsicsSource_ = nullptr, // The intrinsic camera parameters of the source camera
				const Eigen::Matrix3d *intrinsicsDestination_ = nullptr, // The intrinsic camera parameters of the destination camera
				const std::vector<std::pair<size_t, size_t>> *indexPairs_ = nullptr, // The matching keypoints' indices if there are available ones
				const std::vector<size_t> *inliers_ = nullptr) const // The indices of inlier referring to elements in vairable "indexPairs_"
			{
				const size_t &pointNumberSource = keypointsSource_.size(), // The number of feature points in the source image
					&pointNumberDestination = keypointsDestination_.size(); // The number of feature points in the destination image

				// Copy the points to matrices
				cv::Mat keypointsSource(pointNumberSource, 2, CV_32F),
					keypointsDestination(pointNumberDestination, 2, CV_32F);

				// Copy the keypoints into matrices
				float *keypointsSourcePtr = reinterpret_cast<float *>(keypointsSource.data);
				float *keyPointsDistanationPtr = reinterpret_cast<float *>(keypointsDestination.data);

				for (size_t pointIdx = 0; pointIdx < pointNumberSource; ++pointIdx)
				{
					const auto &point = keypointsSource_[pointIdx].pt;
					*(keypointsSourcePtr++) = static_cast<float>(point.x);
					*(keypointsSourcePtr++) = static_cast<float>(point.y);
				}

				for (size_t pointIdx = 0; pointIdx < pointNumberDestination; ++pointIdx)
				{
					const auto &point = keypointsDestination_[pointIdx].pt;
					*(keyPointsDistanationPtr++) = static_cast<float>(point.x);
					*(keyPointsDistanationPtr++) = static_cast<float>(point.y);
				}

				// Normalizing the points if needed
				if (intrinsicsSource_ != nullptr)
				{
					Eigen::Matrix3d inverseIntrinsicsSource =
						intrinsicsSource_->inverse();

					for (size_t pointIdx = 0; pointIdx < pointNumberSource; ++pointIdx)
					{
						const auto &point = keypointsSource_[pointIdx].pt;
						keypointsSource.at<float>(pointIdx, 0) =
							point.x * inverseIntrinsicsSource(0, 0) +
							point.y * inverseIntrinsicsSource(0, 1) +
							inverseIntrinsicsSource(0, 2);

						keypointsSource.at<float>(pointIdx, 1) =
							point.x * inverseIntrinsicsSource(1, 0) +
							point.y * inverseIntrinsicsSource(1, 1) +
							inverseIntrinsicsSource(1, 2);
					}
				}

				if (intrinsicsDestination_ != nullptr)
				{
					Eigen::Matrix3d inverseIntrinsicsDestination =
						intrinsicsDestination_->inverse();

					for (size_t pointIdx = 0; pointIdx < pointNumberDestination; ++pointIdx)
					{
						const auto &point = keypointsDestination_[pointIdx].pt;
						keypointsDestination.at<float>(pointIdx, 0) =
							point.x * inverseIntrinsicsDestination(0, 0) +
							point.y * inverseIntrinsicsDestination(0, 1) +
							inverseIntrinsicsDestination(0, 2);

						keypointsDestination.at<float>(pointIdx, 1) =
							point.x * inverseIntrinsicsDestination(1, 0) +
							point.y * inverseIntrinsicsDestination(1, 1) +
							inverseIntrinsicsDestination(1, 2);
					}
				}
				
				cv::Mat indices(pointNumberDestination, nearestNeighborNumber, CV_32S), // The indices of the neighboring points
					dists(pointNumberDestination, nearestNeighborNumber, CV_32F); // The distances of the neighboring points

				// Initialize with Sampson distance if that is determined by the template parameter
				if constexpr (_isSampsonDistance)
				{
					SampsonDistanceFlann<float> distanceFunction(pose_);
					cv::flann::GenericIndex<SampsonDistanceFlann<float>> flannIndex(keypointsSource, cvflann::KDTreeIndexParams(), distanceFunction);
					flannIndex.knnSearch(keypointsDestination, indices, dists, nearestNeighborNumber, cvflann::SearchParams());
				}
				else // Otherwise, use Symmetric epipolar distance
				{
					SymmetricEpipolarDistanceFlann<float> distanceFunction(pose_);
					cv::flann::GenericIndex<SymmetricEpipolarDistanceFlann<float>> flannIndex(keypointsSource, cvflann::KDTreeIndexParams(), distanceFunction);
					flannIndex.knnSearch(keypointsDestination, indices, dists, nearestNeighborNumber, cvflann::SearchParams());
				}
				
				// The squared inlier-outlier threshold
				const double squaredThreshold =
					threshold_ * threshold_;

				// Occupy the memore for the matches
				matches_.reserve(pointNumberDestination);
				descriptorDistances.reserve(pointNumberDestination);

				// Check the neighbors of each point one-by-one.
				for (size_t pointIdx = 0; pointIdx < pointNumberDestination; ++pointIdx)
				{
					int bestNeighbor = -1;
					double bestDistance = std::numeric_limits<double>::max();

					// Find the neighbor which has the "closest" descriptor
					for (size_t neighborIdx = 0; neighborIdx < nearestNeighborNumber; ++neighborIdx)
					{
						// Keep the point only if the distance is smaller than the threshold
						if (dists.at<float>(pointIdx, neighborIdx) < static_cast<float>(squaredThreshold))
						{
							// The index of the point
							int index = indices.at<int>(pointIdx, neighborIdx);

							// The descriptor distance
							double descriptorDistance = 0, dist;
							for (size_t m = 0; m < descriptorsSource.cols; ++m)
							{
								dist = descriptorsSource.at<float>(index, m) - descriptorsDestination.at<float>(pointIdx, m);
								descriptorDistance += dist * dist;
							}

							// Update if a new best one is found
							if (descriptorDistance < bestDistance)
							{
								bestDistance = descriptorDistance;
								bestNeighbor = index;
							}
						}
						else // Since the distances are ordered decreasingly, we can break when a point is farther than the threshold
							break;
					}

					// If a good neighbor is found, add it to the vector.
					if (bestNeighbor > -1)
					{
						descriptorDistances.emplace_back(bestDistance);
						matches_.emplace_back(std::make_pair(bestNeighbor, pointIdx));
					}
				}

				keypointsSource.release();
				keypointsDestination.release();
				indices.release();
				dists.release();
			}

		};
	}
}
