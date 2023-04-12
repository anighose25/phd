#include "Events.h"

unsigned long long int Event::get_timestamp(std::string command) {
    unsigned long long int timestamp;
    cl_int status;
    if (command == "queued") {
        status = clGetEventProfilingInfo(this->ev, CL_PROFILING_COMMAND_QUEUED,
                                         sizeof(cl_ulong), &timestamp, NULL);

        check(status, "Getting event command queued profiling info");
    } else if (command == "submit") {
        status = clGetEventProfilingInfo(this->ev, CL_PROFILING_COMMAND_SUBMIT,
                                         sizeof(cl_ulong), &timestamp, NULL);

        check(status, "Getting event command submit profiling info");
    } else if (command == "start") {
        status = clGetEventProfilingInfo(this->ev, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), &timestamp, NULL);

        check(status, "Getting event command start profiling info");
    } else if (command == "end") {
        status = clGetEventProfilingInfo(this->ev, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), &timestamp, NULL);

        check(status, "Getting event command end profiling info");
    } else {
        printf("No such event %s exists", command.c_str());
        exit(EXIT_FAILURE);
    }
    return timestamp;
}

unsigned long long int Event::get_queue_submit() {
    return (this->get_timestamp("submit") - this->get_timestamp("queued")) /
           1000;
}

unsigned long long int Event::get_submit_start() {
    return (this->get_timestamp("start") - this->get_timestamp("submit")) /
           1000;
}

unsigned long long int Event::get_start_end() {
    return (this->get_timestamp("end") - this->get_timestamp("start")) / 1000;
}

std::string Event::dump_time() {
    std::stringstream ss;
    ss << this->type << ":" << this->get_queue_submit() << ";"
       << this->get_submit_start() << ";" << this->get_start_end() << ";";
    return ss.str();
}

cl_int Event::print_status() {
    cl_int event_status;
    cl_int status;
    status = clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS,
                            sizeof(cl_int), &event_status, NULL);
    check(status, "Probing event status\n");
    printf("Status: %d\n", event_status);
    switch (event_status) {
    case CL_QUEUED:
        printf("Queued\n");
        break;
    case CL_COMPLETE:
        printf("Complete\n");
        break;
    case CL_SUBMITTED:
        printf("Submitted\n");
        break;
    case CL_RUNNING:
        printf("Running\n");
        break;
    }
    return event_status;
}

void Event::print() {
    if (this->type == "write")
        printf("\033[0;32m");
    else if (this->type == "ndrange")
        printf("\033[1;31m");
    else if (this->type == "read")
        printf("\033[0;34m");
    printf("----------------------------------------------------\n");

    printf("%s Queued -> Submit: %llu\n", this->type.c_str(),
           this->get_queue_submit());
    printf("%s Submit -> Start: %llu\n", this->type.c_str(),
           this->get_submit_start());
    printf("%s Start -> End: %llu\n", this->type.c_str(),
           this->get_start_end());
    printf("----------------------------------------------------\n");
    printf("\033[0m");
}

void Event::dump() {
    if (this->type == "write")
        printf("\033[0;32m");
    else if (this->type == "ndrange")
        printf("\033[1;31m");
    else if (this->type == "read")
        printf("\033[0;34m");
    printf("%s Queued %llu, ", this->type.c_str(),
           this->get_timestamp("queued"));
    printf("Submitted %llu, ", this->get_timestamp("submit"));
    printf("Started %llu, ", this->get_timestamp("start"));
    printf("Ended %llu\n", this->get_timestamp("end"));
    printf("\033[0m");
}

void OpenCLEvents::associate(std::string s, cl_event ev) { events[s] = ev; }

std::string OpenCLEvents::dump_times() {

    std::string time_info = "";
    for (auto x : events) {
        if (x.first.find(write) != std::string::npos) {
            Event e(x.second, write);
            time_info += e.dump_time();
        } else if (x.first.find(ndrange) != std::string::npos) {
            Event e(x.second, ndrange);
            time_info += e.dump_time();
        } else if (x.first.find(read) != std::string::npos) {
            Event e(x.second, read);
            time_info += e.dump_time();
        }
    }

    return time_info;
}

void OpenCLEvents::print() {
    for (auto x : events) {
        std::cout << x.first << "\n";
        if (x.first.find(write) != std::string::npos) {
            Event e(x.second, write);
            e.print();
        } else if (x.first.find(ndrange) != std::string::npos) {
            Event e(x.second, ndrange);
            e.print();
        } else if (x.first.find(read) != std::string::npos) {
            Event e(x.second, read);
            e.print();
        }
    }
}

void OpenCLEvents::dump() {
    for (auto x : events) {
        if (x.first.find(write) != std::string::npos) {
            Event e(x.second, write);
            e.dump();
        } else if (x.first.find(ndrange) != std::string::npos) {
            Event e(x.second, ndrange);
            e.dump();
        } else if (x.first.find(read) != std::string::npos) {
            Event e(x.second, read);
            e.dump();
        }
    }
}

unsigned long long int OpenCLEvents::get_min_start_time() {
    unsigned long long int min_start_time = 0;
    unsigned long long int current_start_time = 0;
    for (auto x : events) {
        Event e(x.second);
        current_start_time = e.get_timestamp("start");
        if (min_start_time == 0)
            min_start_time = current_start_time;
        else
            min_start_time = std::min(min_start_time, current_start_time);
    }
    return min_start_time;
}

unsigned long long int OpenCLEvents::get_min_timestamp(const char *type,
                                                       const char *counter) {
    unsigned long long int min_start_time = 0;
    unsigned long long int current_start_time = 0;
    for (auto x : events) {

        if (x.first.find(type) != std::string::npos) {
            Event e(x.second);
            current_start_time = e.get_timestamp(counter);
            if (min_start_time == 0)
                min_start_time = current_start_time;
            else
                min_start_time = std::min(min_start_time, current_start_time);
        }
    }
    return min_start_time;
}

void OpenCLEvents::dump_json(std::string &filename) {
    json timestamps;
    std::ifstream input_file;
    input_file.open(filename);
    if (input_file) {
        input_file >> timestamps;
        input_file.close();
    }

    for (auto x : events) {
        auto event_type = x.first;
        Event e(x.second);
        std::cout << "Profiling for " << event_type << "\n";
        cl_int status = e.print_status();
        printf("Value of status in dump_json file %d \n",status); 
        if (status == CL_COMPLETE) {
            auto start_time = e.get_timestamp("start");
            auto end_time = e.get_timestamp("end");
            timestamps[event_type]["start"] = start_time;
            timestamps[event_type]["end"] = end_time;
        }
    }

    std::ofstream o(filename.c_str());
    o << timestamps << std::endl;
}
unsigned long long int OpenCLEvents::get_max_timestamp(const char *type,
                                                       const char *counter) {
    unsigned long long int max_start_time = 0;
    unsigned long long int current_start_time = 0;
    for (auto x : events) {

        if (x.first.find(type) != std::string::npos) {
            Event e(x.second);
            current_start_time = e.get_timestamp(counter);
            if (max_start_time == 0)
                max_start_time = current_start_time;
            else
                max_start_time = std::max(max_start_time, current_start_time);
        }
    }
    return max_start_time;
}

void OpenCLEvents::clear() { this->events.clear(); }

void HostEvents::record_start(const char *name) {
    start_timestamps[name] = std::chrono::system_clock::now();
}

void HostEvents::record_end(const char *name) {
    end_timestamps[name] = std::chrono::system_clock::now();
}

void HostEvents::print_timestamps() {

    for (auto x : start_timestamps) {
        std::cout << x.first << " "
                  << start_timestamps[x.first].time_since_epoch().count() << " "
                  << end_timestamps[x.first].time_since_epoch().count() << "\n";
    }
}

unsigned long long int HostEvents::get_timestamp(const char *name,
                                                 const char *counter) {

    std::string start = "start";
    std::string end = "end";
    std::chrono::time_point<std::chrono::system_clock> timestamp;
    if (start == counter)
        timestamp = start_timestamps[name];
    else if (end == counter)
        timestamp = end_timestamps[name];

    unsigned long long int host_timestamp =
        std::chrono::duration_cast<std::chrono::microseconds>(
            timestamp.time_since_epoch())
            .count();
    return host_timestamp;
}
void HostEvents::clear() {
    this->start_timestamps.clear();
    this->end_timestamps.clear();
}
