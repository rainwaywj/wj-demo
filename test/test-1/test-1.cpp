/*
 * @Author: wujun
 * @Date: 2019-05-20 20:26:04
 * @Last Modified by:   wujun
 * @Last Modified time: 2019-05-20 20:26:04
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/timeb.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include "../../src/utility/utility.h"

// #include "opencv2/core/core.hpp"
// #include "opencv2/opencv.hpp"
// #include "opencv2/imgproc/imgproc.hpp"

// #include "cuda_runtime.h"
template <typename T>
class A
{
public:
    A() {}
    ~A() {}
    void test()
    {
        printf("[%s,%d,%s] this[%ld]\n", __FILE__, __LINE__, __FUNCTION__,
               int64_t(this));
    }
    template <int i>
    void push(T &v)
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        if (s == i)
        {
            m_list.push_back(v);
            s++;
            printf("[%s,%d,%s] successed i:%d v:%d s:%d\n", __FILE__, __LINE__,
                   __FUNCTION__, i, v, s);
        }
        else
        {
            m_map.emplace(i, v);
            printf("[%s,%d,%s] waiting i:%d v:%d s:%d m_map.size:%ld\n", __FILE__,
                   __LINE__, __FUNCTION__, i, v, s, m_map.size());
        }

        bool finished = false;
        while (!finished)
        {
            auto it = m_map.find(s);
            if (it == m_map.end())
            {
                finished = true;
            }
            else
            {
                m_list.push_back(it->second);
                m_map.erase(it);
                s++;
            }
        }
    }
    bool pop(T &v)
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        if (m_list.size() > 0)
        {
            v = m_list.front();
            m_list.pop_front();
            return true;
        }
        else
            return false;
    }

private:
    std::list<int> m_list;
    std::map<int, int> m_map;
    std::mutex m_mutex;
    int s = 0;
};
void test()
{
    A<int> a;
    std::vector<int> lt = {0, 3, 5, 1, 2};
    a.push<1>(lt[1]);
    a.push<2>(lt[2]);
    a.push<0>(lt[0]);
    // for (int k = lt.size()-1; k >=0 ; ++k)
    // {
    //     a.push<0>(lt[k]);
    // }
    int v = 0;
    while (a.pop(v))
    {
        printf("pop v:%d \n", v);
    }
}

void test_sharedptr()
{
    size_t total = 1024 * 100;
    size_t single = 1024 * 10;
    char *pdata = (char *)malloc(total);
    std::shared_ptr<char> spdata =
        std::shared_ptr<char>(pdata, [](void *p)
                              { free(p); });
    std::vector<std::shared_ptr<char>> splist(total / single);
    for (size_t i = 0; i < splist.size(); i++)
    {
        splist.at(i) = std::shared_ptr<char>(spdata.get() + i * single);
    }
    printf("[%s,%d,%s] \n", __FILE__, __LINE__, __FUNCTION__);
    splist.at(6) = nullptr; // error release
    printf("[%s,%d,%s] \n", __FILE__, __LINE__, __FUNCTION__);
    spdata = nullptr;
    printf("[%s,%d,%s] \n", __FILE__, __LINE__, __FUNCTION__);
}

void test_map()
{
    std::map<int, int> m;
    m.clear();
    m.emplace(3, 4);
    printf("[%s,%d,%s] key:%d, value:%d\n", __FILE__, __LINE__, __FUNCTION__, 3,
           m[3]);
    m.emplace(3, 100);
    printf("[%s,%d,%s] key:%d, value:%d\n", __FILE__, __LINE__, __FUNCTION__, 3,
           m[3]);
}

void remove_head()
{
    typedef enum
    {
        AVC_CODEC_TYPE_UNKNOWN = 0, ///< ????
        AVC_CODEC_TYPE_H264 = 1,    ///< H264
        AVC_CODEC_TYPE_H265 = 2,    ///< H265
        AVC_CODEC_TYPE_SVAC = 3,    ///< SVAC??????????????????
    } AVC_CodecType;
    typedef struct
    {
        //  AVC_IOT_HEADER_MARK
        int64_t mark;
        int64_t packet_id;
        int64_t timestamp;
        AVC_CodecType codec;
        int data_size;
        int width;
        int height;
        int frame_type;
    } AVCIoTHeader;

    std::string file_list[10] = {
        "/mnt/data/wujun/testdata/packet/error701/0x7fe688000cc0.dat",
        "/mnt/data/wujun/testdata/packet/error701/0x7fe698000cc0.dat",
        "/mnt/data/wujun/testdata/packet/error701/0x7fe6a0000cc0.dat",
        "/mnt/data/wujun/testdata/packet/error701/0x7fe6ac000cc0.dat",
        "/mnt/data/wujun/testdata/packet/error701/0x7fe6b8000cc0.dat",
        "/mnt/data/wujun/testdata/packet/error701/0x7fe690000cc0.dat",
        "/mnt/data/wujun/testdata/packet/error701/0x7fe69c000cc0.dat",
        "/mnt/data/wujun/testdata/packet/error701/0x7fe6a8000cc0.dat",
        "/mnt/data/wujun/testdata/packet/error701/0x7fe6a8000cc0.dat",
        "/mnt/data/wujun/testdata/packet/error701/0x7fe6b0000cc0.dat"};
    for (int k = 0; k < 10; ++k)
    {
        char *buff = nullptr;
        size_t buff_size = 0;
        {
            auto fp = fopen(file_list[k].c_str(), "rb");
            if (!fp)
            {
                printf("error: open file:%s\n", file_list[k].c_str());
                continue;
            }
            fseek(fp, 0, SEEK_END);
            auto len = ftell(fp);
            printf("file:%s length:%ld\n", file_list[k].c_str(), len);
            auto head_len = sizeof(AVCIoTHeader);
            buff_size = len - head_len;
            fseek(fp, head_len, SEEK_SET);
            buff = (char *)malloc(buff_size);
            auto sz = fread(buff, 1, buff_size, fp);
            if (sz != buff_size)
            {
                printf("error: read data file:%s sz:%ld buff_size:%ld\n", file_list[k].c_str(), sz, buff_size);
                continue;
            }
            fclose(fp);
        }

        {
            auto fn = (file_list[k] + ".h264");
            auto fp = fopen(fn.c_str(), "wb");
            if (!fp)
            {
                printf("error: open file:%s\n", fn.c_str());
                continue;
            }
            auto sz = fwrite(buff, 1, buff_size, fp);
            if (sz != buff_size)
            {
                printf("error: read data file:%s\n", file_list[k].c_str());
                continue;
            }
            fclose(fp);
        }
        if (buff)
        {
            free(buff);
            buff = nullptr;
        }
    }
}

int main(int argc, char **argv)
{
    // A *p = new A();
    // p->test();

    // test_sharedptr();
    // test_map();
    // test();
    // remove_head();
    std::string s = "eeuaabcdefkkeeee";
    std::vector<int> pos(5 * 10000, -1);
    printf("l:%d s:%d\n", s.length(), s.size());
    printf("[%s,%d,%s] info: Demo is Completed!_#_!\n", __FILE__, __LINE__,
           __FUNCTION__);

    return 0;
}
