#ifndef CORE_H
#define CORE_H
#include "core.h"
#endif
#ifndef EVENTS_H
#define EVENTS_H

struct Event {
    cl_event ev;
    std::string type;

    Event(cl_event e, std::string t) {
        ev = e;
        type = t;
    }

    Event(cl_event e) { ev = e; }
    unsigned long long int get_timestamp(std::string command);

    unsigned long long int get_queue_submit();

    unsigned long long int get_submit_start();

    unsigned long long int get_start_end();

    std::string dump_time();

    void print();

    cl_int print_status();
    void dump();
};

struct OpenCLEvents {

    std::unordered_map<std::string, cl_event> events;

    std::string write{"write"};
    std::string ndrange{"ndrange"};
    std::string read{"read"};

    void associate(std::string s, cl_event ev);

    unsigned long long int get_min_start_time();

    unsigned long long int get_min_timestamp(const char *type,
                                             const char *counter);

    unsigned long long int get_max_timestamp(const char *type,
                                             const char *counter);

    void print();
    std::string dump_times();

    void dump_json(std::string &filename);

    void dump();
    void clear();
};

struct HostEvents {
    std::unordered_map<const char *,
                       std::chrono::time_point<std::chrono::system_clock>>
        start_timestamps;
    std::unordered_map<const char *,
                       std::chrono::time_point<std::chrono::system_clock>>
        end_timestamps;

    void record_start(const char *);
    void record_end(const char *);
    void print_timestamps();
    unsigned long long int get_timestamp(const char *name, const char *counter);
    void clear();
};
#endif
