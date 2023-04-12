#pragma once
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <stack>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <map>
#include <vector>
#include <pthread.h>
#include <unistd.h>
#include "json.hpp"
using json = nlohmann::json;

#define FILE_LOGGER 0 
#define DESTRUCTOR_LOG 0
#define DAG_LOG 0
#define KERNEL_LOG 0
#define TASKDAG_LOG 0
#define SCHEDULE_LOG 0
#define CALLBACK_LOG 0
#define CHAIN_LOG 0
#define MULTIKERNEL 1
#define ADAS 0

#define LOG _LOG
#if FILE_LOGGER
#define _LOG(...) fprintf(fp, __VA_ARGS__)
#else
#define _LOG(...)
#endif
#define STR_LENGTH 256

enum DataType { Char, Int, Float, Double };

enum Vendor { ARM_GPU, ARM_CPU };
enum GpuBufferType {RO, WO, RW};

struct MemoryTracker {
    unsigned int Allocated = 0;
    unsigned int Free = 0;
    unsigned int CurrentUsage() { return Allocated - Free; }
};

const char *get_device_type(Vendor p);
Vendor  get_vendor(std::string& platform);


extern bool profile_engine;
static MemoryTracker memory_usage;

void *operator new(size_t size);

void operator delete(void *memory, size_t size) noexcept;

static void print_memory_usage() {
    unsigned int bytes = memory_usage.CurrentUsage();
    printf("Currently allocated %zu bytes\n", bytes);
}

struct FunctionTimer {
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<float> execution_time;
    const char *name;

    FunctionTimer(const char *function) {
        start = std::chrono::steady_clock::now();
        name = function;
    }

    FunctionTimer(std::string& function) {
        start = std::chrono::steady_clock::now();
        name = (const char*)function.c_str();
    }


    ~FunctionTimer() {
        end = std::chrono::steady_clock::now();
        execution_time = end - start;
        float time_in_ms = execution_time.count() * 1000.0f;
        printf("%s:%f ms\n", name, time_in_ms);
    }
};

const char *getErrorString(cl_int error);

void check(cl_int status, const char *str);

class splitstring : public std::string {
    std::vector<std::string> flds;

  public:
    splitstring(char *s) : std::string(s){};
    std::vector<std::string> &split(char delim, int rep = 0);
};

size_t get_sizeof(DataType data_type);

struct EdgeInfo {
    int source_kernel;
    int destination_kernel;
    int source_buffer;
    int destination_buffer;

    EdgeInfo(std::string source, std::string destination) {
        splitstring source_info((char *)source.c_str());
        std::vector<std::string> source_ssublist = source_info.split(' ');
        splitstring destination_info((char *)destination.c_str());
        std::vector<std::string> destination_ssublist =
            destination_info.split(' ');
#if DAG_LOG
        std::cout << source_ssublist[0] << " " << source_ssublist[1] << "-->"
                  << destination_ssublist[0] << " " << destination_ssublist[1]
                  << "\n";
#endif
        source_kernel = std::stoi(source_ssublist[0]);
        destination_kernel = std::stoi(destination_ssublist[0]);
        source_buffer = std::stoi(source_ssublist[1]);
        destination_buffer = std::stoi(destination_ssublist[1]);
    }
    void print_edge();
};
