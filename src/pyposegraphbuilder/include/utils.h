#pragma once

#include <Eigen/Core>
#include <vector>
#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <sys/stat.h>
#include <mutex>

struct RunningStatistics
{
    std::mutex guard;
    std::map<std::string, std::pair<double, size_t>> timeValues;
    std::map<std::string, std::pair<size_t, size_t>> countValues;

    void addTime(const std::string& kPropertyName_,
        double value_)
    {
        guard.lock();
        auto& it = timeValues[kPropertyName_];
        it.first += value_;
        ++it.second;
        guard.unlock();
    }

    void addCount(const std::string& kPropertyName_,
        size_t value_)
    {
        guard.lock();
        auto& it = countValues[kPropertyName_];
        it.first += value_;
        ++it.second;
        guard.unlock();
    }

    const std::map<std::string, std::pair<double, size_t>>& getTimes() const
    {
        return timeValues;
    }

    const std::map<std::string, std::pair<size_t, size_t>>& getCounts() const
    {
        return countValues;
    }

    std::pair<double, bool> getAverageTime(const std::string& kPropertyName_) const
    {
        const auto it = timeValues.find(kPropertyName_);
        if (it == timeValues.end())
            return std::make_pair(0.0, false);
        return std::make_pair(it->second.first / it->second.second, true);
    }

    void print()
    {
        printf("Statistics:\n");
        for (const auto& [key, value] : getTimes())
            printf("\tAverage '%s' time = %f seconds\n", key.c_str(), getAverageTime(key).first);
        printf("\t------------------\n");

        for (const auto& [key, value] : getTimes())
            printf("\tTotal '%s' time = %f seconds\n", key.c_str(), value.first);
        printf("\t------------------\n");

        for (const auto& [key, value] : getCounts())
            printf("\tNumber of '%s' = %d\n", key.c_str(), value.first);
    }
};

// Load any matrix from file with arbitrary column types
template <typename... Args>
inline std::vector<std::tuple<Args...>> loadMatrix(std::ifstream& file_, const int num_rows_ = -1) {

    if (!file_.is_open()) {
        LOG(INFO) << "The filestream provided is not openned.";
        return std::vector<std::tuple<Args...>>();
    }

    std::vector<std::tuple<Args...>> values;
    std::tuple<Args...> row;

    while (num_rows_ == -1 || values.size() < num_rows_) {
        if (file_.eof() || values.size() == values.max_size()) {
            if (values.size() == values.max_size())
                LOG(INFO) << "We were not able to load all rows.";
            break;
        }
        readTuple(file_, row, std::index_sequence_for<Args...>{});
        values.push_back(row);
    }

    return values;
}

// Load any matrix from file with arbitrary column types
template <typename... Args>
inline std::vector<std::tuple<Args...>> loadMatrix(const std::string& filename_) {
    if (!std::filesystem::exists(filename_)) {
        LOG(INFO) << "A problem occured when opening '" << filename_ << "'.";
        return std::vector<std::tuple<Args...>>();

    };
    std::ifstream file(filename_);

    if (!file.is_open()) {
        LOG(INFO) << "A problem occured when opening '" << filename_ << "'.";
        return std::vector<std::tuple<Args...>>();
    }

    std::vector<std::tuple<Args...>> values = loadMatrix<Args...>(file);
    file.close();
    return values;
}

// Function to load the image list of the scenes from the 1DSfM dataset
// together with the focal length. Also, loading the image sizes.
inline void load1DSfMImageList(
    const std::string &kImagePath_,
    const std::string &kListPath_,
    const size_t kCoreNumber_, 
    const bool kLoadImageSizes_,
    size_t &totalImageNumber_,
    std::vector<std::tuple<std::string, double, double, double>> &results_)
{
    // Initializing the total image number to be 0
    totalImageNumber_ = 0;
    // Opening the file where the image names and focal length are stored
    std::ifstream file(kListPath_);
    std::string line;
    // Iterate through each line and check if the focal length is available
    while (std::getline(file, line))
    {
        // Increase the total image number by one
        ++totalImageNumber_;
        // Counter to see how many part a line consists of
        size_t counter = 0;
        // Splitting the line
        std::istringstream iss(line);
        std::string imageName, s;
        double focalLength;
        while (iss >> s)
        {
            switch (counter++)
            {
            case 0:
                imageName = s.substr(7, s.size() - 7);
                break;
            case 1:
                break;
            case 2:
                focalLength = std::atof(s.c_str());
                break;
            }
        }

        // Keep the image only if the focal length is known
        // Add the image with its focal length to the vector.
        // The image dimensions are currently set to 0.
        results_.emplace_back(std::make_tuple(imageName, 
            focalLength,
            0.0,
            0.0));
    }
    file.close();

    // Loading the images to get their sizes if needed
    if (kLoadImageSizes_)
#pragma omp parallel for num_threads(kCoreNumber_)
        for (int tupleIdx = 0; tupleIdx < results_.size(); ++tupleIdx)
        {
            auto& tuple = results_[tupleIdx];
            const auto& imageName = std::get<0>(tuple);
            cv::Mat image = cv::imread(kImagePath_ + imageName);
            std::get<2>(tuple) = image.cols;
            std::get<3>(tuple) = image.rows;
        }
}
