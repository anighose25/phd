#include "Buffer.h"

void Buffer::print_buffer_info() {
    std::cout << "pos: " << pos << " size: " << size << " write: " << write
              << " read: " << read << " host allocated?:" << allocate_host_array
              << " host associated? " << associate_host_array << "\n";
    //<< " host array iyyd: " << host_array->get_id() << "\n";
    if (host_array == NULL)
        std::cout << " Host array is null!\n";
    else
        std::cout << " host array id: " << host_array->get_id() << "\n";
}

void Buffer::print_buffer_linkage_info(Vendor p) {
    std::cout << "============================================\n";
    std::cout << "Buffer " << id << "\n";
    std::cout << "Platform oblivious info\n";
    this->print_buffer_info();
    std::cout << "Platform specific info\n";
    vendor_to_bufferflag_map[p]->print_buffer_flag_info();
    if (vendor_to_bufferflag_map[p]->link)
        std::cout << "Linked with buffer " << link->id << "\n";

    std::cout << "============================================\n";
}

cl_mem &Buffer::get_buffer(Vendor p) { return data[p]; }

int Buffer::get_pos() { return pos; }

int Buffer::get_id() { return id; }

std::string Buffer::get_pos_string() { return std::to_string(pos); }

void Buffer::initialize_buffer_flags(Vendor p) {
    vendor_to_bufferflag_map[p] = new BufferFlags(false, false, false, false);
}

void Buffer::setup_buffer_flags(Vendor p, bool create, bool write, bool read,
                                bool link) {
    vendor_to_bufferflag_map[p]->create = create;
    vendor_to_bufferflag_map[p]->write = write;
    vendor_to_bufferflag_map[p]->read = read;
    vendor_to_bufferflag_map[p]->link = link;
}

void Buffer::print_buffer_flags(Vendor p) {
    vendor_to_bufferflag_map[p]->print_buffer_flag_info();
    std::cout << "Buffer Address: " << data[p] << "\n";
}
void Buffer::set_create(Vendor p, bool value) {
    vendor_to_bufferflag_map[p]->create = value;
}

void Buffer::set_write(Vendor p, bool value) {
    vendor_to_bufferflag_map[p]->write = value;
}

void Buffer::set_read(Vendor p, bool value) {
    vendor_to_bufferflag_map[p]->read = value;
}

void Buffer::set_link(Vendor p, bool value) {
    vendor_to_bufferflag_map[p]->link = value;
}

void Buffer::link_buffer(Vendor p) {
    if (vendor_to_bufferflag_map[p]->link) {
        assert(this->link != nullptr);
        Buffer *source_buffer = this->link;
        this->placeholder = std::move(this->data[p]);
        this->data[p] = std::move(source_buffer->data[p]);
        return;
    }
}

void Buffer::reset(Vendor p) {
    if (vendor_to_bufferflag_map[p]->link) {
        vendor_to_bufferflag_map[p]->link = false;
        this->data[p] = std::move(this->placeholder);
        this->link = nullptr;
    }
}
void Buffer::create_buffer(Vendor p, cl_context &ctx,
                           GpuBufferManager *GpuBufMan) {
    cl_int status;

    if (vendor_to_bufferflag_map[p]->link) {
        assert(this->link != nullptr);
        Buffer *source_buffer = this->link;
        //        std::cout << "LINKING " << source_buffer->data[p] << "\n";
        this->data[p] = std::move(source_buffer->data[p]);
        return;
    }
    if (!vendor_to_bufferflag_map[p]->create)
        return;
    // std::cout << "Host array allocated? " << this->allocate_host_array <<
    // "\n";
    if (host_array == NULL)
        std::cout << "host array is null\n";
    switch (p) {
    case Vendor::ARM_CPU: {
        if (write && read) {
            //     std::cout << "ARM_CPU Read and Write\n";
            data[p] =
                clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                               size, host_array->get_data(), &status);
            vendor_to_bufferflag_map[p]->write = false;
            vendor_to_bufferflag_map[p]->read = false;
        }

        else if (write) {
            //   std::cout << "ARM_CPU Write only \n";
            data[p] =
                clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                               size, host_array->get_data(), &status);
            // write = false;
            vendor_to_bufferflag_map[p]->write = false;

        }

        else if (read) {
            // std::cout << "ARM_CPU Read only\n";
            data[p] =
                clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                               size, host_array->get_data(), &status);
            /* read = false; */
            vendor_to_bufferflag_map[p]->read = false;
        }

        break;
    }

    case Vendor::ARM_GPU: {

        if (write && read) {
            // std::cout << "ARM_GPU Write and Read \n";
            if (GpuBufMan == NULL) {
                // std::cout << "Size =" << size << std::endl;
                data[p] = clCreateBuffer(
                    ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL,
                    &status);
            } else {
                if (GpuBufMan->is_buffer_available(GpuBufferType::RW, size)) {
                    data[p] =
                        GpuBufMan->allocate_buffer(GpuBufferType::RW, size);
                    status = CL_SUCCESS;
                } else {
                    std::cout << "Buffer Not Available!" << std::endl;
                    exit(0);
                }
            }
        }

        else if (write) {
            // std::cout << "ARM_GPU Write only \n";
            if (GpuBufMan == NULL) {
                // std::cout << "Size =" << size << std::endl;
                data[p] = clCreateBuffer(
                    ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL,
                    &status);
            } else {
                if (GpuBufMan->is_buffer_available(GpuBufferType::WO, size)) {
                    data[p] =
                        GpuBufMan->allocate_buffer(GpuBufferType::WO, size);
                    status = CL_SUCCESS;
                } else {
                    std::cout << "Buffer Not Available!" << std::endl;
                    exit(0);
                }
            }
        }

        else if (read) {
            // std::cout << "ARM_GPU Read only \n";
            if (GpuBufMan == NULL) {
                // std::cout << "Size =" << size << std::endl;
                data[p] = clCreateBuffer(
                    ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL,
                    &status);
            } else {
                if (GpuBufMan->is_buffer_available(GpuBufferType::WO, size)) {
                    data[p] =
                        GpuBufMan->allocate_buffer(GpuBufferType::RO, size);
                    status = CL_SUCCESS;
                } else {
                    std::cout << "Buffer Not Available!" << std::endl;
                    exit(0);
                }
            }
        }

        break;
    }
    default:
        break;
    }

    check(status, "Creating Buffer");
}

void Buffer::write_buffer(Vendor p, cl_command_queue &cmd_q) {

    LOG("\t Enqueuing write buffer command\n");
    cl_int status;
    status = clEnqueueWriteBuffer(cmd_q, data[p], CL_FALSE, 0, size,
                                  host_array->get_data(), 0, NULL, NULL);
    check(status, "Writing Buffer\n");
}

void Buffer::write_buffer_synchronous(Vendor p, cl_command_queue &cmd_q) {

    LOG("\t Enqueuing write buffer blocking command\n");
    cl_int status;
    status = clEnqueueWriteBuffer(cmd_q, data[p], CL_TRUE, 0, size,
                                  host_array->get_data(), 0, NULL, NULL);

    check(status, "Writing Buffer via blocking call \n");
    cl_int finish_status;
    finish_status = clFinish(cmd_q);
    check(finish_status, "Trying to finish command queue\n");
}

void Buffer::read_buffer(Vendor p, cl_command_queue &cmd_q) {

    LOG("\t Enqueuing read buffer command\n");
    cl_int status;
    status = clEnqueueReadBuffer(cmd_q, data[p], CL_FALSE, 0, size,
                                 host_array->get_data(), 0, NULL, NULL);
    check(status, "Writing Buffer\n");
}

cl_event Buffer::write_buffer(Vendor p, cl_command_queue &cmd_q, cl_event &dep) {

    cl_event ev;
    LOG("\t Enqueuing write buffer command\n");
    cl_int status;
    if (dep != NULL) {
        status = clEnqueueWriteBuffer(cmd_q, data[p], CL_FALSE, 0, size,
                                      host_array->get_data(), 1, &dep, &ev);
        check(status, "Writing Buffer\n");
    } else {
        status = clEnqueueWriteBuffer(cmd_q, data[p], CL_FALSE, 0, size,
                                      host_array->get_data(), 0, NULL, &ev);
        check(status, "Writing Buffer\n");
    }
    return ev;
}

cl_event Buffer::read_buffer(Vendor p, cl_command_queue &cmd_q, cl_event &dep) {

    cl_event ev;
    LOG("\t Enqueuing read buffer command\n");
    cl_int status;
    status = clEnqueueReadBuffer(cmd_q, data[p], CL_FALSE, 0, size,
                                 host_array->get_data(), 1, &dep, &ev);
    check(status, "Reading Buffer\n");
    return ev;
}

void Buffer::print_host() { host_array->print(); }

/*void Buffer::allocate_host_array_for_buffer() {
    host_array = new HostArray(data_type, size_buf);
    allocate_host_array = true;
}*/

void Buffer::allocate_host_array_for_buffer() {
    if (this->file_name.size() == 0)
        host_array = new HostArray(data_type, size_buf);
    else
        host_array = new HostArray(data_type, size_buf, this->file_name);
    allocate_host_array = true;
}

void Buffer::associate_host_array_for_buffer(HostArray *h) {
    host_array = h;
    associate_host_array = true;
}
