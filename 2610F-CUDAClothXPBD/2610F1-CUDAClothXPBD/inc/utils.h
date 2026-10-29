#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <chrono>

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

std::vector<float> read_binary_file(const std::string& path);

std::tuple<std::vector<float>, std::vector<float>, std::vector<unsigned int>>
ex_cloth_array(const std::vector<float>& values);

void log_execution_time(const std::string& message, const TimePoint& start);

template <typename T>
std::vector<T> read_csv(const std::string& filename) {
    std::vector<T> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return data;
    }

    // Read entire file content into a single string
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Split content by line breaks and commas
    std::vector<std::string> tokens;
    boost::split(tokens, content, boost::is_any_of(",\n"));

    // Parse each token into the specified type T, trimming whitespace
    for (auto& token : tokens) {
        boost::trim(token);
        if (!token.empty()) {
            try {
                data.push_back(boost::lexical_cast<T>(token));
            }
            catch (const boost::bad_lexical_cast&) {
                std::cerr << "Unable to parse value: " << token << std::endl;
            }
        }
    }

    return data;
}


