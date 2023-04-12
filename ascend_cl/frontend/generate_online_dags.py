import random

def random_arrival_times_limit(lambda_rate=1.5, time_limit=10):

    intervals = [random.expovariate(lambda_rate) for i in range(1000)]
    timestamps = [0.0]
    timestamp = 0.0
    time_limit *=1000000000
    for t in intervals:
        timestamp += t*1000000000
        if timestamp>time_limit:
            break
        timestamps.append(timestamp)
    return timestamps

if __name__=="__main__":
    
    timestamps = random_arrival_times_limit()
    DAGs = ["sample"]
    for t in timestamps:
        dag = random.choice(DAGs)
        dump_str = dag+":"+str(int(t))
        print dump_str


