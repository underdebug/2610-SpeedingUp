
#include "utils.h"

std::vector<float> read_binary_file(const std::string& path) {
    std::vector<float> values;
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
        return values;
    }

    while (file) {
        float value;
        file.read(reinterpret_cast<char*>(&value), sizeof(float));
        if (file) {
            values.push_back(value);
        }
    }

    return values;
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<unsigned int>>
ex_cloth_array(const std::vector<float>& values) {

    int arr_pos_len = static_cast<int>(values[0]);
    int arr_normals_len = static_cast<int>(values[1]);
    int arr_triIds_len = static_cast<int>(values[2]);

    int st = 3;
    int en = 3 + arr_pos_len;
    std::vector<float> arr_pos(values.begin() + st, values.begin() + en);

    st = en;
    en += arr_normals_len;
    std::vector<float> arr_normals(values.begin() + st, values.begin() + en);

    st = en;
    en += arr_triIds_len;
    std::vector<float> arr_triIds_double(values.begin() + st, values.begin() + en);

    std::vector<unsigned int> arr_triIds_int;
    arr_triIds_int.reserve(arr_triIds_double.size());  // Reserve space to avoid multiple allocations
    for (const double& value : arr_triIds_double) {
        arr_triIds_int.push_back(static_cast<unsigned int>(value));
    }

    return std::make_tuple(arr_pos, arr_normals, arr_triIds_int);
}


void log_execution_time(const std::string& message, const TimePoint& start) {

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << message << ": " << duration << " microseconds" << std::endl;
};

