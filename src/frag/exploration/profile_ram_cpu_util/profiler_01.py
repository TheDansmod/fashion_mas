import os
import time
import psutil
from memory_profiler import profile

# Decorate the function to see line-by-line memory usage
@profile
def execute_workload():
    # Simulate memory allocation and CPU work
    large_list = [x ** 2 for x in range(2_000_000)]
    return sum(large_list)

if __name__ == "__main__":
    # Track the current Python process
    process = psutil.Process(os.getpid())
    
    # Initialize CPU measurement
    process.cpu_percent(interval=None)
    start_time = time.time()
    
    # Run the target code
    execute_workload()
    
    # Capture metrics after execution
    end_time = time.time()
    cpu_usage = process.cpu_percent(interval=None)
    
    # Convert bytes to Megabytes (MB)
    peak_ram_mb = process.memory_info().rss / (1024 * 1024)
    
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Overall CPU Utilization: {cpu_usage}%")
    print(f"Peak Process RAM: {peak_ram_mb:.2f} MB")
