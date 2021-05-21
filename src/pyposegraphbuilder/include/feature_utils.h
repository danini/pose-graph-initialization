#pragma once

#include <map>
#include <set>
#include <vector>
#include <shared_mutex>
#include <glog/logging.h>
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/hdf.hpp>
#include <iostream>
#include <string>

// Loading or detecting feature points in an image
inline void loadFeatures(
	const std::string& imageName_, // The name of the image
	const std::string& imageExtension_, // The extension of the image
	const std::string& imagePath_, // The path to the image folder
	const bool kUseGPU_,
	cv::Ptr<cv::hdf::HDF5> h5io_, // The hdf database file
	std::vector<cv::KeyPoint>& keypoints_, // The detected or loaded keypoints
	cv::Mat& descriptors_, // The descriptors of the feature points
	std::shared_mutex& lock_) // A readers-writer lock
{
	const std::string featuresDatabase = "feat_" + imageName_;
	const std::string descriptorsDatabase = "desc_" + imageName_;

	// Check if the feature points are stored in the database file
	if (h5io_->hlexists(featuresDatabase))
	{
		// Load both the keypoints and their descriptors
		cv::Mat keypointsMat;

		// Reader-lock that allows multiple processes reading but only a single one writing
		std::shared_lock<std::shared_mutex> lock(lock_);
		h5io_->dsread(keypointsMat, featuresDatabase);
		h5io_->dsread(descriptors_, descriptorsDatabase);
		lock.unlock();

		// Copying the loaded keypoints into the output vector
		keypoints_.resize(keypointsMat.rows);
		for (size_t pointIdx = 0; pointIdx < keypoints_.size(); ++pointIdx)
		{
			keypoints_[pointIdx].pt.x = keypointsMat.at<double>(pointIdx, 0);
			keypoints_[pointIdx].pt.y = keypointsMat.at<double>(pointIdx, 1);
			keypoints_[pointIdx].angle = keypointsMat.at<double>(pointIdx, 2);
			keypoints_[pointIdx].size = keypointsMat.at<double>(pointIdx, 3);
		}

		LOG(INFO) << "Loaded SIFT keypoint number on image '" << imageName_ << "' = " << keypoints_.size();
	}
	else // If no keypoints have been detected yet
	{
		// Note: Here, the GPU SURF implementation of OpenCV could be applied.
		// However, to my tests it leads to a significant drop in the number of 
		// matched view pairs. Therefore, we apply the CPU SIFT implementation
		// that works well. This is not much slower than the GPU implementation.
		// The matching will be done on GPU.
		// Loading the image
		cv::Mat image = cv::imread(imagePath_ + imageName_ + "." + imageExtension_,
			cv::IMREAD_GRAYSCALE);

		// Creating a SIFT detector
		cv::Ptr<cv::SiftFeatureDetector> sift = cv::SIFT::create(8000, 3, 0.0, 10000.);
		keypoints_.reserve(10001);

		//  Detect keypoints
		sift->detectAndCompute(image, cv::noArray(), keypoints_, descriptors_);

		// Normalize the descriptor vectors to get RootSIFT
		for (size_t row = 0; row < descriptors_.rows; ++row)
		{
			descriptors_.row(row) *= 1.0 / cv::norm(descriptors_.row(row), cv::NORM_L1);
			for (size_t col = 0; col < descriptors_.cols; ++col)
				descriptors_.at<float>(row, col) = std::sqrt(descriptors_.at<float>(row, col));
		}

		// Save the keypoints to the database file together with their descriptors
		cv::Mat keypointsMat(keypoints_.size(), 4, CV_64F);
		for (size_t pointIdx = 0; pointIdx < keypoints_.size(); ++pointIdx)
		{
			keypointsMat.at<double>(pointIdx, 0) = keypoints_[pointIdx].pt.x;
			keypointsMat.at<double>(pointIdx, 1) = keypoints_[pointIdx].pt.y;
			keypointsMat.at<double>(pointIdx, 2) = keypoints_[pointIdx].angle;
			keypointsMat.at<double>(pointIdx, 3) = keypoints_[pointIdx].size;
		}

		std::unique_lock<std::shared_mutex> lock(lock_);
		h5io_->dscreate(keypoints_.size(), 4, CV_64F, featuresDatabase);
		h5io_->dswrite(keypointsMat, featuresDatabase);

		h5io_->dscreate(descriptors_.rows, descriptors_.cols, descriptors_.type(), descriptorsDatabase);
		h5io_->dswrite(descriptors_, descriptorsDatabase);
		lock.unlock();

		LOG(INFO) << "Detected SIFT keypoint number on image '" << imageName_ << "' = " << keypoints_.size();
	}
}


inline void matchFeatures(
	const std::string& sourceImageName_,
	const std::string& destinationImageName,
	const std::vector<cv::KeyPoint>& keypoints1_,
	const std::vector<cv::KeyPoint>& keypoints2_,
	const cv::Mat& descriptors1_,
	const cv::Mat& descriptors2_,
	const bool useGPU_,
	cv::Ptr<cv::hdf::HDF5> h5io_,
	std::vector<std::tuple<size_t, size_t, double>>& matches_,
	std::shared_mutex& lock_)
{
	const std::string correspondenceDatabase = sourceImageName_ + "_" + destinationImageName;

	if (h5io_->hlexists(correspondenceDatabase))
	{
		cv::Mat matchesMat;
		std::shared_lock<std::shared_mutex> lock(lock_);
		h5io_->dsread(matchesMat, correspondenceDatabase);
		lock.unlock();
		matches_.reserve(matchesMat.rows);

		for (size_t pointIdx = 0; pointIdx < matchesMat.rows; ++pointIdx)
			matches_.emplace_back(std::make_tuple(
				static_cast<size_t>(matchesMat.at<double>(pointIdx, 0)),
				static_cast<size_t>(matchesMat.at<double>(pointIdx, 1)),
				matchesMat.at<double>(pointIdx, 2)));

		LOG(INFO) << "Loaded match number = " << matches_.size();
		return;
	}

	// Using GPU to match features
	std::vector<std::vector<cv::DMatch>> matches;
	std::vector<std::vector<cv::DMatch>> matches_opposite;

	if (useGPU_)
	{
		cv::cuda::GpuMat gpuDescriptors1,
			gpuDescriptors2;
		gpuDescriptors1.upload(descriptors1_);
		gpuDescriptors2.upload(descriptors2_);

		// Do brute-force matching from the source to the destination image
		cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
		matcher->knnMatch(gpuDescriptors1, gpuDescriptors2, matches, 2);

		// Do brute-force matching from the destination to the source image
		cv::Ptr<cv::cuda::DescriptorMatcher> matcher_opposite = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
		matcher->knnMatch(gpuDescriptors2, gpuDescriptors1, matches_opposite, 2);

	} else
	{
		// Do brute-force matching from the source to the destination image
		cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::MatcherType::BRUTEFORCE_SL2);
		matcher->knnMatch(descriptors1_, descriptors2_, matches, 2);

		// Do brute-force matching from the destination to the source image
		cv::Ptr<cv::DescriptorMatcher> matcher_opposite = cv::DescriptorMatcher::create(cv::DescriptorMatcher::MatcherType::BRUTEFORCE_SL2);
		matcher->knnMatch(descriptors2_, descriptors1_, matches_opposite, 2);
	}

	std::vector<std::pair<double, const std::vector<cv::DMatch>*>> good_matches;

	std::vector<cv::DMatch> ddmatches;

	// Do mutual nearest neighbor search
	for (size_t i = 0; i < matches.size(); ++i)
	{
		if (matches[i].size() < 2 ||
			matches_opposite[matches[i][0].trainIdx].size() < 2)
			continue;

		if ((matches[i][0].distance < 0.90 * matches[i][1].distance) &&
			(matches[i][0].queryIdx == matches_opposite[matches[i][0].trainIdx][0].trainIdx)) // We increased threshold for mutual snn check
		{
			ddmatches.emplace_back(matches[i][0]);
			good_matches.emplace_back(std::make_pair(matches[i][0].distance / matches[i][1].distance, &matches[i]));
		}
	}

	// Sort the correspondences according to their distance.
	// This is done for using PROSAC sampling
	std::sort(good_matches.begin(), good_matches.end());

	// Create the container for the correspondences
	matches_.reserve(good_matches.size());
	cv::Mat matchesMat(good_matches.size(), 3, CV_64F);

	// Fill the container by the selected matched
	size_t rowIdx = 0;
	for (const auto& match_ptr : good_matches)
	{
		const std::vector<cv::DMatch>& match = *match_ptr.second;
		matches_.emplace_back(std::make_tuple(match[0].queryIdx, match[0].trainIdx, match_ptr.first));
		matchesMat.at<double>(rowIdx, 0) = match[0].queryIdx;
		matchesMat.at<double>(rowIdx, 1) = match[0].trainIdx;
		matchesMat.at<double>(rowIdx, 2) = match_ptr.first;
		++rowIdx;
	}

	std::unique_lock<std::shared_mutex> lock(lock_);
	h5io_->dscreate(matches_.size(), 3, CV_64F, correspondenceDatabase);
	h5io_->dswrite(matchesMat, correspondenceDatabase);
	lock.unlock();

	LOG(INFO) << "Detected match number = " << matches_.size();
}