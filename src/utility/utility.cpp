/*
 * @Author: wujun 
 * @Date: 2019-05-20 16:46:52 
 * @Last Modified by:   wujun 
 * @Last Modified time: 2019-05-20 16:46:52 
 */

#include "utility.h"

namespace utility{

template<>
void set_env(const std::string& s, float d) {
    auto env_value = getenv(s.c_str());
    if (!env_value) {
        std::string str_env_value = s + std::to_string(d);
        putenv(const_cast<char*>(str_env_value.c_str()));
    }
    return;
}

template<>
float parse_env(const std::string& s, float d) {
    char* e = std::getenv(s.c_str());
    if (nullptr == e) {
        return d;
    } else {
        return std::stof(e);
    }
}

template<>
int parse_env(const std::string& s, int d) {
    char* e = std::getenv(s.c_str());
    if (nullptr == e) {
        return d;
    } else {
        return std::stoi(e);
    }
}

template<>
std::string parse_env(const std::string& s, std::string d) {
    char* e = std::getenv(s.c_str());
    if (nullptr == e) {
        return d;
    } else {
        return e;
    }
}
} // namespace utility