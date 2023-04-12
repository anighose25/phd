#include "GpuBufferManager.h"

bool GpuBufferManager::is_buffer_available(GpuBufferType t, std::size_t s) {
    return !(available_buffers[t][s].empty());
}

cl_mem GpuBufferManager::allocate_buffer(GpuBufferType t, std::size_t s) {
    int index = available_buffers[t][s].front();
    available_buffers[t][s].pop();

    cl_mem buffer = buffer_lists[t][s][index];

    buffer_allocation_map[buffer] = std::make_pair(t, std::make_pair(s, index));

    return buffer;
}

void GpuBufferManager::deallocate_buffer(cl_mem buffer) {
    std::pair<GpuBufferType, std::pair<std::size_t, int>> buffer_info =
        buffer_allocation_map[buffer];

    GpuBufferType t = buffer_info.first;
    std::size_t s = buffer_info.second.first;
    int index = buffer_info.second.second;

    available_buffers[t][s].push(index);

    buffer_allocation_map.erase(buffer);
}
