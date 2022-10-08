/*
 * @Author: wujun 
 * @Date: 2019-05-20 16:46:45 
 * @Last Modified by:   wujun 
 * @Last Modified time: 2019-05-20 16:46:45 
 */
#pragma once

#include <stdio.h>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <thread>
#include <stdlib.h>
#include <string>
#include <memory.h>

#if defined(__unix) || defined(unix) || defined(__unix__) || \
    defined(__MACH__) || defined(__APPLE__)
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#define OS_UNIX 1
#else
#define OS_UNIX 0
#include <direct.h>
#include <io.h>
#include <windows.h>
#endif

// //代码中区分debug及no debug模式
// #ifdef NDEBUG //_DEBUG
//     printf("NO DEBUG\n");
// #else
//     printf("DEBUG \n");
// #endif

#define ERROR_EXCEPTION_BEGIN \
    try                       \
    {
#define ERROR_HANDLER_END                                          \
    }                                                              \
    catch (const std::exception &e)                                \
    {                                                              \
        std::cerr << "Exception ERROR: " << e.what() << std::endl; \
        return -1;                                                 \
    }                                                              \
    catch (...)                                                    \
    {                                                              \
        std::cerr << "Exception UNKNOWN" << std::endl;             \
        return -1;                                                 \
    }                                                              \
    return 0;

// log
#define _LOG_DEBUG_ 1
#if _LOG_DEBUG_
#define DBG_DEBUG(format, ...) printf(format, ##__VA_ARGS__)
#else
#define DEBUG(format, ...)
#endif //_LOG_DEBUG_

#define LOG_LEVEL 3

#define LOG_DEBUG 0
#define LOG_INFO 1
#define LOG_WARING 2
#define LOG_ERROR 3
#define LOG_OFF 4

#define LOG(format, ...)                                            \
    do                                                              \
    {                                                               \
        if (LOG_LEVEL >= LOG_DEBUG)                                 \
        {                                                           \
            DBG_DEBUG("\n->DEBUG   [%s,%s,%d]\n" format "\n",       \
                      __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                           \
        if (LOG_LEVEL >= LOG_INFO)                                  \
        {                                                           \
            DBG_DEBUG("\n->INFO   [%s,%s,%d]\n" format "\n",        \
                      __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                           \
        if (LOG_LEVEL <= LOG_WARING)                                \
        {                                                           \
            DBG_DEBUG("\n->WARING   [%s,%s,%d]\n" format "\n",      \
                      __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                           \
        if (LOG_LEVEL <= LOG_ERROR)                                 \
        {                                                           \
            DBG_DEBUG("\n->ERRO   [%s,%s,%d]\n" format "\n",        \
                      __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                           \
    } while (0)

const char *get_error_str(int e)
{
    //TODO
    return "nullptr";
}

#define ASSERT(e)                     \
    int r = (e);                      \
    if (r != 0)                       \
    {                                 \
        log_err(e, get_error_str(r)); \
        abort();                      \
    }

namespace utility {

int sys_ok_mkdir(const std::string &path)
{
    int ret = 0;
    if (access(path.c_str(), 0) == 0)
        return 0;
#if OS_UNIX
    ret = mkdir(path.c_str(), 0755);
#else
    ret = mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
    if (ret == -1)
    {
        printf("Fail to make debug dir.");
    }
    return ret;
}

std::string get_cur_time()
{
    char tmp[1024];
    memset(tmp, 0, sizeof(tmp));
#if OS_UNIX
    struct timeval t_val;
    gettimeofday(&t_val, NULL);
    time_t now = t_val.tv_sec;
    tm *cur_time = localtime(&now);

    snprintf(tmp, sizeof(tmp), "-%4d%02d%02d-%02d:%02d:%02d.%ld",
             1900 + cur_time->tm_year, 1 + cur_time->tm_mon, cur_time->tm_mday,
             cur_time->tm_hour, cur_time->tm_min, cur_time->tm_sec,
             long(t_val.tv_usec));
#else
    SYSTEMTIME sys;
    GetLocalTime(&sys);
    snprintf(tmp, sizeof(tmp), "%4d%02d%02d-%02d:%02d:%02d.%ld", sys.wYear,
             sys.wMonth, sys.wDay, sys.wHour, sys.wMinute, sys.wSecond,
             sys.wMilliseconds);
#endif
    std::string ret(tmp);
    return ret;
}

std::string getCurrentSystemTime()
{
	auto tt = std::chrono::system_clock::to_time_t
		(std::chrono::system_clock::now());
	struct tm* ptm = localtime(&tt);
	char date[60] = { 0 };
	sprintf(date, "%d-%02d-%02d-%02d.%02d.%02d",
		(int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
		(int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);
	return std::string(date);
}
/*
static void WriteGpuJpeg(std::string save_path, const void *pdata, int width, int height, uint64_t frame_index)
{

    cv::Mat img(cv::Size(width, height), CV_8UC3);
    // checkCudaError(cudaMemcpy(img.data, pdata, width * height * 3, cudaMemcpyDeviceToHost));
    std::string file_path = save_path + "/frame_" + std::to_string(frame_index) + ".jpg";
    printf("file_path[%s]\n", file_path.c_str());
    cv::imwrite(file_path, img);

    return;
}
*/
uint64_t get_thread_id()
{
    std::ostringstream ss;
    ss << std::this_thread::get_id();
    uint64_t id = std::stoull(ss.str());
    return id;
}

double walltime(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

std::string get_utc_time(std::string time_str, std::string str_format) {
    char utc_time[100] = {0};
    time_t ut = atoi(time_str.c_str());
    struct tm* ustm = gmtime(&ut);
    strftime(utc_time, 100, str_format.c_str(), ustm);
    printf("utc time[%s]\n", utc_time);
    return std::string(utc_time);
}
/*
void set_device_id(int device_id)
{
    int total_devices = 0;
    checkCudaError(cudaGetDeviceCount(&total_devices));
    if (device_id >= total_devices || total_devices <= 0)
    {
        printf("Error: device_id[%d] total_devices[%d]\n", device_id, total_devices);
        abort();
    }
    checkCudaError(cudaSetDevice(device_id));
    return;
}
*/
bool make_dir(std::string dir_path)
{
    int ret = access(dir_path.c_str(), F_OK); //判断文件夹是否存在
    if (0 != ret)
    {
        ret = mkdir(dir_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (0 != ret)
        {
            printf("[%s,%d,%s] Error: Creat folder[%s] failed\n", __FILE__, __LINE__, __FUNCTION__, dir_path.c_str());
            return false;
        }
        return true;
    }
    return true;
}

int get_files_name(std::string dir_path, std::vector<std::string> &file_path_list, std::string file_type)
{
    DIR *dir = opendir(dir_path.c_str());
    if (dir == NULL)
    {
        printf("[%s,%d,%s] Error: %s is not a directory or not exist\n", __FILE__, __LINE__, __FUNCTION__,
               dir_path.c_str());
        return -1;
    }

    struct dirent *d_ent = NULL;
    char fullpath[128] = {0};
    char dot[3] = ".";
    char dotdot[6] = "..";

    int file_type_len = file_type.length();

    while ((d_ent = readdir(dir)) != NULL)
    {
        if ((strcmp(d_ent->d_name, dot) != 0) && (strcmp(d_ent->d_name, dotdot) != 0)) //忽略 . 和 ..
        {
            if (d_ent->d_type == DT_DIR) // d_type可以看到当前的东西的类型,DT_DIR代表当前都到的是目录,在usr/include/dirent.h中定义的
            {
                std::string newDirectory =
                    dir_path + std::string("/") + std::string(d_ent->d_name); // d_name中存储了子目录的名字
                if (dir_path[dir_path.length() - 1] == '/')
                {
                    newDirectory = dir_path + std::string(d_ent->d_name);
                }

                if (-1 == get_files_name(newDirectory, file_path_list, file_type)) //递归子目录
                {
                    return -1;
                }
            }
            else //如果不是目录
            {
                std::string file_path = dir_path + std::string("/") + std::string(d_ent->d_name); //构建绝对路径
                if (dir_path[dir_path.length() - 1] == '/')                                       //如果传入的目录最后是/--> 例如a/b/  那么后面直接链接文件名
                {
                    file_path = dir_path + std::string(d_ent->d_name); // /a/b/1.txt
                }
                printf("file path[%s]\n", file_path.c_str());
                if (0 == strcmp(file_type.c_str(),
                                file_path.substr(file_path.length() - file_type_len, file_path.length()).c_str()))
                    file_path_list.push_back(file_path);
            }
        }
    }
    closedir(dir);

    std::sort(file_path_list.begin(), file_path_list.end());
#if 0
        for(auto &file_path: file_path_list)
        {
            printf("file path[%s]\n", file_path.c_str());
        }
#endif

    return 0;
}

int ReadDataFromFile(std::string filename, void *&pdata, size_t &datasize)
{
    FILE *fp = nullptr;
    fp = fopen(filename.c_str(), "rb");
    if (nullptr == fp)
    {
        printf("[%s,%d,%s] Error: fopen\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    auto datalen = ftell(fp);
    datasize = size_t(datalen);
    fseek(fp, 0, SEEK_SET);
    // printf("[%s,%d,%s] filename[%s]: data_length[%ld]\n", __FILE__, __LINE__, __FUNCTION__, filename.c_str(),
    // datasize);
    if (nullptr != pdata)
    {
        free(pdata);
        pdata = nullptr;
    }
    pdata = (unsigned char *)malloc(size_t(datasize));
    if (nullptr == pdata)
    {
        printf("[%s,%d,%s] Error: alloc memory\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    if (datalen != fread(pdata, 1, size_t(datasize), fp))
    {
        printf("[%s,%d,%s] Error: read file\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }
    fclose(fp);

    return 0;
}
int WriteDataToFile(std::string filename, void *pdata, size_t datasize)
{
    FILE *fp = nullptr;
    fp = fopen(filename.c_str(), "wb");
    if (nullptr == fp)
    {
        printf("[%s,%d,%s] Error: fopen\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }

    fwrite(pdata, 1, datasize, fp); 
    fclose(fp);
    fp = nullptr;
    return 0;
}
/*
int WriteDataToFile(std::string filename, void *pdata, size_t datasize, bool datadevice = false)
{
    FILE *fp = nullptr;
    fp = fopen(filename.c_str(), "wb");
    if (nullptr == fp)
    {
        printf("[%s,%d,%s] Error: fopen\n", __FILE__, __LINE__, __FUNCTION__);
        return -1;
    }
    if (!datadevice)
    {
        fwrite(pdata, 1, datasize, fp);
    }
    else
    {
        unsigned char *phost = nullptr;
        phost = (unsigned char *)malloc(datasize);
        checkCudaError(cudaMemcpy(phost, pdata, datasize, cudaMemcpyDeviceToHost));
        auto length = fwrite(phost, 1, datasize, fp);
        if (length == datasize)
        {
            free(phost);
            phost = nullptr;
        }
        else
        {
            printf("[%s,%d,%s] fwrite Error: datasize[%ld] length[%ld]\n",
                   __FILE__, __LINE__, __FUNCTION__, datasize, length);
            return -1;
        }
    }
    fclose(fp);
    fp = nullptr;
    return 0;
}
*/
/*
int realloc(void *&pmem, size_t &memsize, size_t newsize, void *pmempool = nullptr, bool datadevice = false)
{
    if (newsize <= 0)
    {
        return -1;
    }
    if (newsize > memsize || NULL == pmem)
    {
        // printf("pmem[%p] memsize[%ld] newsize[%ld]\n", pmem, memsize, newsize);
        if (nullptr != pmem)
        {
            if (nullptr == pmempool)
            {
                if (!datadevice)
                {
                    free(pmem);
                }
                else
                {
                    checkCudaError(cudaFree(pmem));
                }
            }
            else
            {
                // TODO
                // pmempool->free(pmem);
            }
            pmem = nullptr;
        }

        if (nullptr == pmempool)
        {
            if (!datadevice)
            {
                pmem = (unsigned char *)malloc(newsize);
            }
            else
            {
                checkCudaError(cudaMalloc((void **)pmem, newsize));
            }
        }
        else
        {
            // TODO
            // pmempool->free(pmem);
        }

        if (nullptr == pmem)
            return -1;
        memsize = newsize;
    }

    return 0;
}
*/
/*
int ReadJpeg(std::string filename, void *&pdata, int &step, int &width, int &height, int &channels, bool datadevice = false)
{
    assert(nullptr == pdata);

    cv::Mat image = cv::imread(filename.c_str());
    step = image.step[0];
    width = image.cols;
    height = image.rows;
    channels = image.channels();

    size_t data_size = step * height * channels;
    if (!datadevice)
    {
        pdata = (unsigned char *)malloc(data_size);
        assert(nullptr != pdata);
        memcpy(pdata, image.data, data_size);
    }
    else
    {
        checkCudaError(cudaMalloc((void **)pdata, data_size));
        checkCudaError(cudaMemcpy(pdata, image.data, data_size, cudaMemcpyHostToDevice));
    }

    return 0;
}
*/
/*
int WriteJpeg(std::string filename, void *pdata, int pitch, int width, int height, int channel, bool datadevice = false)
{
    unsigned char *phost = nullptr;
    if (!datadevice)
        phost = reinterpret_cast<unsigned char *>(pdata);
    else
    {
        size_t datasize = width * height * channel;
        phost = (unsigned char *)malloc(datasize);
        checkCudaError(cudaMemcpy2D(phost, width, pdata, pitch, width, height, cudaMemcpyDeviceToHost));
    }
    cv::Mat image(height, width, channel == 1 ? CV_8UC1 : CV_8UC3, phost, pitch);
    if (!datadevice)
        phost = nullptr;
    else if (nullptr != phost)
        free(phost);
    return 0;
}
*/
// write bmp, input - BGR, device
int writeBMPi(const char *filename, const unsigned char *d_BGR, int pitch, int width, int height)
{

    unsigned int headers[13];
    FILE *outfile;
    int extrabytes;
    int paddedsize;
    int x;
    int y;
    int n;
    int red, green, blue;

    // std::vector<unsigned char> vchanBGR(height * width * 3);
    // unsigned char* chanBGR = vchanBGR.data();
    // checkCudaError(cudaMemcpy2D(chanBGR, (size_t)width * 3, d_BGR, (size_t)pitch, width * 3, height,
    //         cudaMemcpyDeviceToHost));
    std::vector<unsigned char> vchanBGR(height * pitch * 3);
    unsigned char *chanBGR = vchanBGR.data();
    // checkCudaError(cudaMemcpy(chanBGR, d_BGR, 3 * pitch * height, cudaMemcpyDeviceToHost));

    extrabytes = 4 - ((width * 3) % 4); // How many bytes of padding to add to each
    // horizontal line - the size of which must
    // be a multiple of 4 bytes.
    if (extrabytes == 4)
        extrabytes = 0;

    paddedsize = ((width * 3) + extrabytes) * height;

    // Headers...
    // Note that the "BM" identifier in bytes 0 and 1 is NOT included in these "headers".

    headers[0] = paddedsize + 54; // bfSize (whole file size)
    headers[1] = 0;               // bfReserved (both)
    headers[2] = 54;              // bfOffbits
    headers[3] = 40;              // biSize
    headers[4] = width;           // biWidth
    headers[5] = height;          // biHeight

    // Would have biPlanes and biBitCount in position 6, but they're shorts.
    // It's easier to write them out separately (see below) than pretend
    // they're a single int, especially with endian issues...

    headers[7] = 0;          // biCompression
    headers[8] = paddedsize; // biSizeImage
    headers[9] = 0;          // biXPelsPerMeter
    headers[10] = 0;         // biYPelsPerMeter
    headers[11] = 0;         // biClrUsed
    headers[12] = 0;         // biClrImportant

    if (!(outfile = fopen(filename, "wb")))
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return 1;
    }

    //
    // Headers begin...
    // When printing ints and shorts, we write out 1 character at a time to avoid endian issues.
    //

    fprintf(outfile, "BM");

    for (n = 0; n <= 5; n++)
    {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    // These next 4 characters are for the biPlanes and biBitCount fields.

    fprintf(outfile, "%c", 1);
    fprintf(outfile, "%c", 0);
    fprintf(outfile, "%c", 24);
    fprintf(outfile, "%c", 0);

    for (n = 7; n <= 12; n++)
    {
        fprintf(outfile, "%c", headers[n] & 0x000000FF);
        fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
        fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
        fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
    }

    //
    // Headers done, now write the data...
    //

    for (y = height - 1; y >= 0; y--) // BMP image format is written from bottom to top...
    {
        for (x = 0; x <= width - 1; x++)
        {
            blue = chanBGR[(y * width + x) * 3];
            green = chanBGR[(y * width + x) * 3 + 1];
            red = chanBGR[(y * width + x) * 3 + 2];

            if (red > 255)
                red = 255;
            if (red < 0)
                red = 0;
            if (green > 255)
                green = 255;
            if (green < 0)
                green = 0;
            if (blue > 255)
                blue = 255;
            if (blue < 0)
                blue = 0;
            // Also, it's written in (b,g,r) format...

            fprintf(outfile, "%c", blue);
            fprintf(outfile, "%c", green);
            fprintf(outfile, "%c", red);
        }
        if (extrabytes) // See above - BMP lines must be of lengths divisible by 4.
        {
            for (n = 1; n <= extrabytes; n++)
            {
                fprintf(outfile, "%c", 0);
            }
        }
    }

    fclose(outfile);
    return 0;
}

std::string get_exe_name()
{
    char exe_path[PATH_MAX] = {0};
    int n = readlink("/proc/self/exe", exe_path, PATH_MAX);
    std::string ExePath = std::string(exe_path);
    int pose = ExePath.rfind("/");
    return ExePath.substr(pose + 1, ExePath.length());
}

std::string get_exe_dir()
{
    char exe_path[PATH_MAX] = {0};
    int n = readlink("/proc/self/exe", exe_path, PATH_MAX);
    std::string ExePath = std::string(exe_path);
    int pose = ExePath.rfind("/");
    return ExePath.substr(0, pose);
}

//获取以逗号为分隔符的字符串，如baobao,boy,birthday20180808,football
void get_str_by_comma(std::string full_str, std::vector<int> &sub_str_list)
{
    char *str = const_cast<char *>(full_str.c_str());
    char *ptr;
    ptr = strtok(str, ",");
    while (ptr != NULL)
    {
        sub_str_list.push_back(std::atoi(ptr));
        ptr = strtok(NULL, ",");
    }
}

static void __inline__ writetxt(FILE *fp, std::string str_name, std::string str_value)
{
    std::string str_full = str_name + str_value + "\n";
    fwrite(str_full.c_str(), 1, str_full.length(), fp);
    return;
}

void writeExcel()
{
    char chy[4] = {'x', 'a', 'h', 'w'};
    int data[4] = {1, 3, 6, 9};
    int i;
    FILE *fp = NULL;
    fp = fopen("/mnt/data/wujun/workspace/faced-in-c/build/tools/performance/performance-image-analysis/test.csv", "w");
    for (i = 0; i < 4; i++)
        fprintf(fp, "%c,%d\n", chy[i], data[i]);
    fclose(fp);

    exit(0);

    std::string filePathRead =
        "/mnt/data/wujun/workspace/faced-in-c/build/tools/performance/performance-image-analysis/120340.log";
    std::string filePahtWrite =
        "/mnt/data/wujun/workspace/faced-in-c/build/tools/performance/performance-image-analysis/120340.csv";
    FILE *fpr = fopen(filePathRead.c_str(), "r");
    FILE *fpw = fopen(filePahtWrite.c_str(), "w");
    if (!fpr)
    {
        printf("can't open file %s\n", filePathRead.c_str());
        abort();
    }
    if (!fpw)
    {
        printf("can't open file %s\n", filePahtWrite.c_str());
        abort();
    }

    float cpu_load = 0;
    float cpu_max = 0;
    int proc_mem = 0;
    int virtual_mem = 0;

    while (!feof(fpr))
    {
        fscanf(fpr, "%f%f%d%d", &cpu_load, &cpu_max, &proc_mem, &virtual_mem);
        printf("%f  %f  %d  %d\n", cpu_load, cpu_max, proc_mem, virtual_mem);
        fprintf(fpw, "%f,%f,%d,%d\n", cpu_load, cpu_max, proc_mem, virtual_mem);
    }

    fclose(fpr);
    fclose(fpw);

    printf("[%s,%d,%s] writeExcel completed\n", __FILE__, __LINE__, __FUNCTION__);
    exit(0);
}

// //设置环境变量
// template <typename T>
// void set_env(const std::string &, T);
// //解析环境变量
// template <typename T>
// T parse_env(const std::string &, T);

static std::string str_env_value; //必须是全局的

void set_env(const std::string& s, float d) {
    auto env_value = getenv(s.c_str());
    if (!env_value) {
        str_env_value = s + "=" + std::to_string(d);
        putenv(const_cast<char*>(str_env_value.c_str()));
    }
    return;
}
void set_env(const std::string& s, int d) {
    auto env_value = getenv(s.c_str());
    if (!env_value) {
        str_env_value = s + "=" + std::to_string(d);
        putenv(const_cast<char*>(str_env_value.c_str()));
    }
    return;
}
void set_env(const std::string& s, std::string d) {
    auto env_value = getenv(s.c_str());
    if (!env_value) {
        str_env_value = s + "=" + d;
        putenv(const_cast<char*>(str_env_value.c_str()));
    }
    return;
}


float parse_env(const std::string& s, float d) {
    char* e = std::getenv(s.c_str());
    if (nullptr == e) {
        return d;
    } else {
        return std::stof(e);
    }
}

int parse_env(const std::string& s, int d) {
    char* e = std::getenv(s.c_str());
    if (nullptr == e) {
        return d;
    } else {
        return std::stoi(e);
    }
}

std::string parse_env(const std::string& s, std::string d) {
    char* e = std::getenv(s.c_str());
    if (nullptr == e) {
        return d;
    } else {
        return e;
    }
}

static void save_data_format(std::string save_path, const void *pdata, int ilength)
{
    static int index = 0;
    char str_index[512] = "";
    sprintf(str_index, "video-%010d", index);
    std::string file_name = save_path + "/" + std::string(str_index) + ".packet";
    printf("[%s,%d,%s] file_name[%s]\n", __FILE__, __LINE__, __FUNCTION__, file_name.c_str());
    // WriteDataToFile(file_name, (void *)pdata, ilength);
    index++;
}

void print_bytes(unsigned char *start, int len)
{
       int i;
       for(i = 0; i < len; i++)
       {
             printf("%.2x", start[i]);
             printf("  %p\n", &start[i]);
       }
       printf("----------------------------------------\n");
}

} // namespace utility