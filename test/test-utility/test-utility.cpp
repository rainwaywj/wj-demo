/*
 * @Author: wujun 
 * @Date: 2019-05-20 16:06:55 
 * @Last Modified by: wujun
 * @Last Modified time: 2019-05-20 17:30:21
 */

#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <list>
#include <set>
#include <thread>
#include <mutex>
#include <atomic>
#include <sys/timeb.h>
#include <chrono>

#include "utility/utility.h"

using namespace utility;

// #define TEST_ENV_VALUE
// #define TEST_UTC_TIME

int main(int argc, char** argv)
{
    {//test set env and parse env
#ifdef TEST_ENV_VALUE
        set_env("ENV_VALUE_TEST", float(5.5));
        auto env_value_test = parse_env("ENV_VALUE_TEST", float(0.1));
        printf("[%s,%d,%s] env_value_test[%f]\n", __FILE__, __LINE__, __FUNCTION__, env_value_test);
#endif // TEST_ENV_VALUE
    }

    {//get utc time
#ifdef TEST_UTC_TIME
        time_t cur_time = time(NULL);
        printf("The Calendar Time now is %ld\n", cur_time);
        std::string utc_time_format = "%Y-%m-%d %H:%M:%S";
        std::string utc_time_str = get_utc_time(std::to_string(cur_time), utc_time_format);
#endif // TEST_UTC_TIME
    }

    printf("$_$ Demo[%s] is Completed #_#\n", get_exe_name().c_str());

    return 0;
}
