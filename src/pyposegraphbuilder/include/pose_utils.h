#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include "sophus/se3.hpp"

#include "pose.h"
#include "utils.h"
#include "types.h"


namespace reconstruction {
	namespace pose {
/*
	Declaration
*/
		inline bool linearTriangulation(
			const Eigen::Matrix<double, 3, 4>& projection_1_,
			const Eigen::Matrix<double, 3, 4>& projection_2_,
			const cv::Mat& point_,
			Eigen::Vector4d& triangulated_point_);

		inline void decomposeHomographyMatrix(
			const Eigen::Matrix3d& homography_,
			const Eigen::Matrix3d& intrinsics_src_,
			const Eigen::Matrix3d& intrinsics_dst_,
			std::vector<Eigen::Matrix3d>& Rs_dst_src_,
			std::vector<Eigen::Vector3d>& ts_dst_src_,
			std::vector<Eigen::Vector3d>& normals_);

		inline void poseFromHomographyMatrix(
			const Eigen::Matrix3d& homography_,
			const Eigen::Matrix3d& intrinsics_src_,
			const Eigen::Matrix3d& intrinsics_dst_,
			const cv::Mat& correspondences_,
			const std::vector<size_t>& inliers_,
			Eigen::Matrix3d& R_dst_src_,
			Eigen::Vector3d& t_dst_src_,
			Eigen::Vector3d& normal_,
			std::vector<Eigen::Vector3d>& points3D_);

		inline double calculateDepth(
			const Eigen::Matrix<double, 3, 4>& proj_matrix_,
			const Eigen::Vector3d& point3D_);

		inline bool checkCheirality(
			const Eigen::Matrix3d& R_dst_src_,
			const Eigen::Vector3d& t_dst_src_,
			const cv::Mat& correspondences_,
			const std::vector<size_t>& inliers_,
			std::vector<Eigen::Vector3d>& points3D_);

		inline double computeOppositeOfMinor(
			const Eigen::Matrix3d& matrix_,
			const size_t row_, 
			const size_t col_);

		template <typename T>
		inline int signOfNumber(const T val);

		inline Eigen::Matrix3d computeHomographyRotation(
			const Eigen::Matrix3d& H_normalized,
			const Eigen::Vector3d& tstar,
			const Eigen::Vector3d& n,
			const double v);

/*
	Implementation
*/

		//  Given two projection matrices, the function returns the implied essential matrix.
		inline Eigen::Matrix3d getEssentialMatrixFromRelativePose(
			const Sophus::SE3d& pose_) {
			// Create the Ematrix from the poses.
			const Eigen::Matrix3d R_dst_src = pose_.rotationMatrix();
			const Eigen::Vector3d t_dst_src = pose_.translation();

			// The cross product matrix of the translation vector
			Eigen::Matrix3d cross_prod_t_dst_src;
			cross_prod_t_dst_src << 0, -t_dst_src(2), t_dst_src(1), t_dst_src(2), 0, -t_dst_src(0),
				-t_dst_src(1), t_dst_src(0), 0;

			return cross_prod_t_dst_src * R_dst_src;
		}

		//  Given two projection matrices, the function returns the implied essential matrix.
		inline void getEssentialMatrixFromRelativePose(
			const Sophus::SE3d& pose_,
			Eigen::Matrix3d& essential_matrix_) {
			essential_matrix_ = getEssentialMatrixFromRelativePose(pose_);
		}

		//  Given two projection matrices, the function returns the implied essential matrix.
		inline void getEssentialMatrixFromRelativePose(
			const Eigen::Matrix3d &rotation_matrix_,
			const Eigen::Vector3d &translation_,
			Eigen::Matrix3d& essential_matrix_) {
			essential_matrix_ = getEssentialMatrixFromRelativePose(Sophus::SE3d(rotation_matrix_, translation_));
		}

		inline void convertToRotationMatrix(Eigen::Matrix3d& R_dst_src_, double* scale_) {
			const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(R_dst_src_);
			const Eigen::PermutationMatrix<3, 3> permutation_matrix(qr.colsPermutation());
			R_dst_src_ = qr.householderQ();
			const Eigen::VectorXd diag = qr.matrixQR().diagonal();
			for (int i = 0; i < 3; ++i)
				if (diag(i) < 0)
					R_dst_src_.col(i) = -R_dst_src_.col(i);
			R_dst_src_ = R_dst_src_ * permutation_matrix.inverse();

			// Recovering the scale of the input matrix
			if (scale_ != nullptr)
				*scale_ = diag.cwiseAbs().sum() / 3.0;
		}

		// Deciding if a particular point is in from of the camera represented by its rotation and
		// translation
		inline bool isTriangulatedPointInFrontOfCameras(
			const cv::Mat& correspondence_,
			const Eigen::Matrix3d& rotation_,
			const Eigen::Vector3d& position_) {

			Eigen::Vector3d dir1, dir2;
			dir1 << correspondence_.at<double>(0), correspondence_.at<double>(1), 1;
			dir2 << correspondence_.at<double>(2), correspondence_.at<double>(3), 1;

			const double dir1_sq = dir1.squaredNorm();
			const double dir2_sq = dir2.squaredNorm();
			const double dir1_dir2 = dir1.dot(dir2);
			const double dir1_pos = dir1.dot(position_);
			const double dir2_pos = dir2.dot(position_);

			return (
				dir2_sq * dir1_pos - dir1_dir2 * dir2_pos > 0 &&
				dir1_dir2 * dir1_pos - dir1_sq * dir2_pos > 0);
		}
	
// Decomposes the essential matrix into the rotation R and translation t such
// that E can be any of the four candidate solutions: [rotation1 | translation],
// [rotation1 | -translation], [rotation2 | translation], [rotation2 |
// -translation].
	inline void decomposeEssentialMatrix(
		const Eigen::Matrix3d& essential_matrix_,
		Eigen::Matrix3d& rotation_1_,
		Eigen::Matrix3d& rotation_2_,
		Eigen::Vector3d& translation_) 
	{
		Eigen::Matrix3d d;
		d << 0, 1, 0, -1, 0, 0, 0, 0, 1;

		const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
			essential_matrix_, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d V = svd.matrixV();
		if (U.determinant() < 0) {
			U.col(2) *= -1.0;
		}

		if (V.determinant() < 0) {
			V.col(2) *= -1.0;
		}

		// Possible configurations.
		rotation_1_ = U * d * V.transpose();
		rotation_2_ = U * d.transpose() * V.transpose();
		translation_ = U.col(2).normalized();
	}

	// Recovering the relative pose from the essential matrix
	inline int getPoseFromEssentialMatrix(
		const Eigen::Matrix3d& essential_matrix_,
		const cv::Mat& normalized_correspondences_,
		Eigen::Matrix3d& rotation_,
		Eigen::Vector3d& translation_) 
	{
		// Decompose the essential matrix.
		Eigen::Matrix3d rotation1, rotation2;
		Eigen::Vector3d translation;
		decomposeEssentialMatrix(essential_matrix_, rotation1, rotation2, translation);
		const std::vector<Eigen::Matrix3d> rotations = { rotation1, rotation1, rotation2, rotation2 };

		const std::vector<Eigen::Vector3d> positions = {-rotations[0].transpose() * translation,
														-rotations[1].transpose() * -translation,
														-rotations[2].transpose() * translation,
														-rotations[3].transpose() * -translation };



		// From the 4 candidate poses, find the one with the most triangulated points
		// in front of the camera.
		std::vector<size_t> points_in_front_of_cameras(4, 0);
		const Eigen::Matrix<double, 3, 4> proj_1 = Eigen::Matrix<double, 3, 4>::Identity();

		std::vector<double> best_distances(normalized_correspondences_.rows, std::numeric_limits<double>::max());
		std::vector<int> best_poses(normalized_correspondences_.rows, 5);
		for (auto i = 0; i < 4; i++) 
		{
			Eigen::Matrix<double, 3, 4> proj_2;
			proj_2 << rotations[i], ((i % 2 ? -1 : 1) * translation);
						
			for (size_t point_idx = 0; point_idx < normalized_correspondences_.rows; ++point_idx) {

				Eigen::Matrix<double, 4, 1> triangulated_point;
				linearTriangulation(proj_1, proj_2, normalized_correspondences_.row(point_idx), triangulated_point);

				Eigen::Vector3d projected_1 = proj_1 * triangulated_point; 

				if (projected_1(2) < 0)
					continue;

				Eigen::Vector3d projected_2 = proj_2 * triangulated_point;

				if (projected_2(2) < 0)
					continue;

				Eigen::Vector2d pt1(
					normalized_correspondences_.at<double>(point_idx, 0), normalized_correspondences_.at<double>(point_idx, 1));
				Eigen::Vector2d pt2(
					normalized_correspondences_.at<double>(point_idx, 2), normalized_correspondences_.at<double>(point_idx, 3));

				double error = (projected_1.hnormalized() - pt1).squaredNorm() +
					(projected_2.hnormalized() - pt2).squaredNorm();

				if (error < best_distances[point_idx])
				{
					best_distances[point_idx] = error;
					best_poses[point_idx] = i;
				}

				//if (isTriangulatedPointInFrontOfCameras(normalized_correspondences_.row(point_idx), rotations[i], positions[i])) {
				//if (triangulated_point(3) > 0) {
				//}
			}
		} 

		for (const int &vote : best_poses)
			if (vote < 5)
				++points_in_front_of_cameras[vote];
		
		// Find the pose with the most points in front of the camera.
		const auto& max_element = std::max_element(points_in_front_of_cameras.begin(),
			points_in_front_of_cameras.end());
		const int max_index =
			std::distance(points_in_front_of_cameras.begin(), max_element);

		// Set the pose.
		rotation_ = rotations[max_index];
		translation_ = (max_index % 2 ? -1 : 1) * translation; //-rotation_ * positions[max_index];
		return *max_element;
	}



	inline void poseFromHomographyMatrix(
		const Eigen::Matrix3d& homography_,
		const Eigen::Matrix3d& intrinsics_src_,
		const Eigen::Matrix3d& intrinsics_dst_,
		const cv::Mat& correspondences_,
		const std::vector<size_t>& inliers_,
		Eigen::Matrix3d& R_dst_src_,
		Eigen::Vector3d& t_dst_src_,
		Eigen::Vector3d& normal_,
		std::vector<Eigen::Vector3d>& points3D_) {
		std::vector<Eigen::Matrix3d> R_cmbs;
		std::vector<Eigen::Vector3d> t_cmbs;
		std::vector<Eigen::Vector3d> n_cmbs;

		decomposeHomographyMatrix(
			homography_,
			intrinsics_src_,
			intrinsics_dst_, 
			R_cmbs,
			t_cmbs, 
			n_cmbs);

		points3D_.clear();
		for (size_t i = 0; i < R_cmbs.size(); ++i) {
			std::vector<Eigen::Vector3d> points3D_cmb;
			checkCheirality(R_cmbs[i], t_cmbs[i], correspondences_, inliers_, points3D_cmb);
			if (points3D_cmb.size() >= points3D_.size()) {
				R_dst_src_ = R_cmbs[i];
				t_dst_src_ = t_cmbs[i];
				normal_ = n_cmbs[i];
				points3D_ = points3D_cmb;
			}
		}
	}

	// Decomposing a given homography matrix into the possible pose parameters
	inline void decomposeHomographyMatrix(
		const Eigen::Matrix3d& homography_, // The homography matrix to be decomposed
		const Eigen::Matrix3d& intrinsics_src_, // The intrinsic parameters of the first camera
		const Eigen::Matrix3d& intrinsics_dst_, // The intrinsic parameters of the second camera
		std::vector<Eigen::Matrix3d>& Rs_dst_src_, // The possible rotation matrices
		std::vector<Eigen::Vector3d>& ts_dst_src_, // The possible translation vectors
		std::vector<Eigen::Vector3d>& normals_) // The possible plane normals
	{
		// Remove calibration from homography.
		Eigen::Matrix3d H_normalized = intrinsics_dst_.inverse() * homography_ * intrinsics_src_;

		// Remove scale from normalized homography.
		Eigen::JacobiSVD<Eigen::Matrix3d> hmatrix_norm_svd(H_normalized);
		H_normalized.array() /= hmatrix_norm_svd.singularValues()[1];

		const Eigen::Matrix3d S = H_normalized.transpose() * H_normalized - Eigen::Matrix3d::Identity();

		// Check if H is rotation matrix.
		const double kMinInfinityNorm = 1e-3;
		if (S.lpNorm<Eigen::Infinity>() < kMinInfinityNorm) {
			Rs_dst_src_.emplace_back(H_normalized);
			ts_dst_src_.emplace_back(Eigen::Vector3d::Zero());
			normals_.emplace_back(Eigen::Vector3d::Zero());
			return;
		}

		const double M00 = computeOppositeOfMinor(S, 0, 0);
		const double M11 = computeOppositeOfMinor(S, 1, 1);
		const double M22 = computeOppositeOfMinor(S, 2, 2);

		const double rtM00 = std::sqrt(M00);
		const double rtM11 = std::sqrt(M11);
		const double rtM22 = std::sqrt(M22);

		const double M01 = computeOppositeOfMinor(S, 0, 1);
		const double M12 = computeOppositeOfMinor(S, 1, 2);
		const double M02 = computeOppositeOfMinor(S, 0, 2);

		const int e12 = signOfNumber(M12);
		const int e02 = signOfNumber(M02);
		const int e01 = signOfNumber(M01);

		const double nS00 = std::abs(S(0, 0));
		const double nS11 = std::abs(S(1, 1));
		const double nS22 = std::abs(S(2, 2));

		const std::array<double, 3> nS{ {nS00, nS11, nS22} };
		const size_t idx = std::distance(nS.begin(), std::max_element(nS.begin(), nS.end()));

		Eigen::Vector3d np1;
		Eigen::Vector3d np2;
		if (idx == 0) {
			np1[0] = S(0, 0);
			np2[0] = S(0, 0);
			np1[1] = S(0, 1) + rtM22;
			np2[1] = S(0, 1) - rtM22;
			np1[2] = S(0, 2) + e12 * rtM11;
			np2[2] = S(0, 2) - e12 * rtM11;
		}
		else if (idx == 1) {
			np1[0] = S(0, 1) + rtM22;
			np2[0] = S(0, 1) - rtM22;
			np1[1] = S(1, 1);
			np2[1] = S(1, 1);
			np1[2] = S(1, 2) - e02 * rtM00;
			np2[2] = S(1, 2) + e02 * rtM00;
		}
		else if (idx == 2) {
			np1[0] = S(0, 2) + e01 * rtM11;
			np2[0] = S(0, 2) - e01 * rtM11;
			np1[1] = S(1, 2) + rtM00;
			np2[1] = S(1, 2) - rtM00;
			np1[2] = S(2, 2);
			np2[2] = S(2, 2);
		}

		const double traceS = S.trace();
		const double v = 2.0 * std::sqrt(1.0 + traceS - M00 - M11 - M22);

		const double ESii = signOfNumber(S(idx, idx));
		const double r_2 = 2 + traceS + v;
		const double nt_2 = 2 + traceS - v;

		const double r = std::sqrt(r_2);
		const double n_t = std::sqrt(nt_2);

		const Eigen::Vector3d n1 = np1.normalized();
		const Eigen::Vector3d n2 = np2.normalized();

		const double half_nt = 0.5 * n_t;
		const double esii_t_r = ESii * r;

		const Eigen::Vector3d t1_star = half_nt * (esii_t_r * n2 - n_t * n1);
		const Eigen::Vector3d t2_star = half_nt * (esii_t_r * n1 - n_t * n2);

		const Eigen::Matrix3d R1 = computeHomographyRotation(H_normalized, t1_star, n1, v);
		const Eigen::Vector3d t1 = R1 * t1_star;

		const Eigen::Matrix3d R2 = computeHomographyRotation(H_normalized, t2_star, n2, v);
		const Eigen::Vector3d t2 = R2 * t2_star;

		Rs_dst_src_.emplace_back(-R1);
		Rs_dst_src_.emplace_back(R1);
		Rs_dst_src_.emplace_back(-R2);
		Rs_dst_src_.emplace_back(R2);

		ts_dst_src_.emplace_back(t1);
		ts_dst_src_.emplace_back(-t1);
		ts_dst_src_.emplace_back(t2);
		ts_dst_src_.emplace_back(-t2);

		normals_.emplace_back(-n1);
		normals_.emplace_back(n1);
		normals_.emplace_back(-n2);
		normals_.emplace_back(n2);
	}

	// Calculate the rotation matrix from a given homography, translation,
	// plane normal, and plane distance.
	inline Eigen::Matrix3d computeHomographyRotation(
		const Eigen::Matrix3d& H_normalized,
		const Eigen::Vector3d& tstar,
		const Eigen::Vector3d& n,
		const double v) {
		return H_normalized * (Eigen::Matrix3d::Identity() - (2.0 / v) * tstar * n.transpose());
	}

	inline double computeOppositeOfMinor(
		const Eigen::Matrix3d& matrix_, 
		const size_t row_, 
		const size_t col_) {
		const size_t col1 = col_ == 0 ? 1 : 0;
		const size_t col2 = col_ == 2 ? 1 : 2;
		const size_t row1 = row_ == 0 ? 1 : 0;
		const size_t row2 = row_ == 2 ? 1 : 2;
		return (matrix_(row1, col2) * matrix_(row2, col1) - matrix_(row1, col1) * matrix_(row2, col2));
	}

	// The method collection the 3D points which are in front of both cameras
	// after triangulation.
	inline bool checkCheirality(
		const Eigen::Matrix3d& R_dst_src_,
		const Eigen::Vector3d& t_dst_src_,
		const cv::Mat& correspondences_,
		const std::vector<size_t>& inliers_,
		std::vector<Eigen::Vector3d>& points3D_) 
	{
		// Initialize the first camera's projection matrix to be in the origin
		static const Eigen::Matrix<double, 3, 4> proj_matrix1 = Eigen::Matrix<double, 3, 4>::Identity();

		// Initialize the second camera's projection matrix from the given rotation and translation
		Eigen::Matrix<double, 3, 4> proj_matrix2;
		proj_matrix2.leftCols<3>() = R_dst_src_;
		proj_matrix2.rightCols<1>() = t_dst_src_;

		// Iterating through all correspondences, estimating their 3D coordinates and
		// collecting the ones which end up in front of both cameras.
		constexpr double kMinDepth = std::numeric_limits<double>::epsilon();
		const double max_depth = 1000.0f * (R_dst_src_.transpose() * t_dst_src_).norm();
		points3D_.clear();
		for (size_t i = 0; i < inliers_.size(); ++i) {

			const size_t &point_idx = inliers_[i];

			Eigen::Vector4d homogeneous_point3D;
			linearTriangulation(
				proj_matrix1, // The first cameras's projection matrix
				proj_matrix2, // The second cameras's projection matrix
				correspondences_.row(point_idx), // The point correspondence
				homogeneous_point3D); // The estimated 3D coordinate in their homogeneous form
			const Eigen::Vector3d point3D = homogeneous_point3D.head<3>();

			const double depth1 = calculateDepth(proj_matrix1, point3D); // Get the depth in the first image
			if (depth1 > kMinDepth && depth1 < max_depth) {
				const double depth2 =
					calculateDepth(proj_matrix2, point3D); // Get the depth in the second image
				if (depth2 > kMinDepth && depth2 < max_depth) {
					points3D_.push_back(point3D);
				}
			}
		}
		// The procedure was successful if there is at least a single 3D point in front of both cameras
		return !points3D_.empty();
	}

	// Returning the depth of a 3D point given a projection matrix
	inline double calculateDepth(
		const Eigen::Matrix<double, 3, 4>& proj_matrix_,
		const Eigen::Vector3d& point3D_) {
		const double proj_z = proj_matrix_.row(2).dot(point3D_.homogeneous());
		return proj_z * proj_matrix_.col(2).norm();
	}

	// Returns the sign of a number
	template <typename T>
	inline int signOfNumber(const T val) {
		return (T(0) < val) - (val < T(0));
	}

	inline bool linearTriangulation(
		const Eigen::Matrix<double, 3, 4>& projection_1_,
		const Eigen::Matrix<double, 3, 4>& projection_2_,
		const cv::Mat& point_,
		Eigen::Vector4d& triangulated_point_) 
	{
		Eigen::Matrix4d design_matrix;
		design_matrix.row(0) = point_.at<double>(0) * projection_1_.row(2) - projection_1_.row(0);
		design_matrix.row(1) = point_.at<double>(1) * projection_1_.row(2) - projection_1_.row(1);
		design_matrix.row(2) = point_.at<double>(2) * projection_2_.row(2) - projection_2_.row(0);
		design_matrix.row(3) = point_.at<double>(3) * projection_2_.row(2) - projection_2_.row(1);

		// Extract nullspace.
		triangulated_point_ = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
		return true;
	}
}
} 
