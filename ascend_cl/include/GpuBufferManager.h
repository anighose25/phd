#ifndef GPUBUFFERMANAGER_H
#define GPUBUFFERMANAGER_H
#ifndef CORE_H
#define CORE_H
#include "core.h"
#endif

class GpuBufferManager {

  public:
    GpuBufferManager(
        cl_context &ctx,
        const std::vector<std::pair<std::size_t, int>> &sizes_list) {
        sizes = sizes_list;

        for (int i = 0; i < 3; i++) {
            std::unordered_map<std::size_t, std::vector<cl_mem>> buf_map;
            std::unordered_map<std::size_t, std::queue<int>> buf_avail_map;
            for (int j = 0; j < sizes.size(); j++) {
                std::vector<cl_mem> buf_list;
                std::queue<int> buf_avail_queue;
                if (i == 0) {
                    for (int k = 0; k < sizes[j].second; k++) {
                        cl_int status;
                        cl_mem buf = clCreateBuffer(
                            ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                            sizes[j].first, NULL, &status);
                        buf_list.push_back(buf);

                        buf_avail_queue.push(k);
                    }
                } else if (i == 1) {
                    for (int k = 0; k < sizes[j].second; k++) {
                        cl_int status;
                        cl_mem buf = clCreateBuffer(
                            ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                            sizes[j].first, NULL, &status);
                        buf_list.push_back(buf);

                        buf_avail_queue.push(k);
                    }
                } else {
                    for (int k = 0; k < sizes[j].second; k++) {
                        cl_int status;
                        cl_mem buf = clCreateBuffer(
                            ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            sizes[j].first, NULL, &status);
                        buf_list.push_back(buf);

                        buf_avail_queue.push(k);
                    }
                }
                buf_map[sizes[j].first] = buf_list;
                buf_avail_map[sizes[j].first] = buf_avail_queue;
            }
            if (i == 0) {
                buffer_lists[RO] = buf_map;
                available_buffers[RO] = buf_avail_map;
            } else if (i == 1) {
                buffer_lists[WO] = buf_map;
                available_buffers[WO] = buf_avail_map;
            } else {
                buffer_lists[RW] = buf_map;
                available_buffers[RW] = buf_avail_map;
            }
        }
    }

    bool is_buffer_available(GpuBufferType t, std::size_t s);

    cl_mem allocate_buffer(GpuBufferType t, std::size_t s);

    void deallocate_buffer(cl_mem buffer);

    ~GpuBufferManager() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < sizes.size(); j++) {
                for (int k = 0; k < sizes[j].second; k++) {
                    if (i == 0)
                        clReleaseMemObject(buffer_lists[RO][sizes[j].first][k]);
                    else if (i == 1)
                        clReleaseMemObject(buffer_lists[WO][sizes[j].first][k]);
                    else
                        clReleaseMemObject(buffer_lists[RW][sizes[j].first][k]);
                }
            }
        }
        available_buffers.clear();
        buffer_lists.clear();
        buffer_allocation_map.clear();
    }

    mutable std::mutex mut;
    std::condition_variable cond;

    std::vector<std::pair<std::size_t, int>> sizes;

    std::unordered_map<GpuBufferType,
                       std::unordered_map<std::size_t, std::queue<int>>>
        available_buffers;
    std::unordered_map<GpuBufferType,
                       std::unordered_map<std::size_t, std::vector<cl_mem>>>
        buffer_lists;

    std::map<cl_mem, std::pair<GpuBufferType, std::pair<std::size_t, int>>>
        buffer_allocation_map;
};

#endif
