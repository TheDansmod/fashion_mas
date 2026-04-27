# this was created with gemini
# the way this works is you start the app (uv run chainlit run app.py) in one terminal and then start this file in another terminal and it will output the used resources over time (every 2 secs)
import psutil
import time

def monitor_chainlit_app():
    target_process = None
    
    # 1. Find the root process running the Chainlit app
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = p.info['cmdline']
            if cmdline and 'chainlit' in cmdline and 'app.py' in cmdline:
                target_process = p
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not target_process:
        print("Chainlit app not found. Please start it first.")
        return

    print(f"Monitoring Process Tree rooted at PID: {target_process.pid}")
    
    # 2. Prime the CPU metrics (first call returns 0.0)
    target_process.cpu_percent(interval=None)
    
    try:
        while True:
            total_mem_bytes = 0
            total_cpu_percent = 0
            
            # 3. Retrieve the parent and all child processes recursively
            try:
                processes = [target_process] + target_process.children(recursive=True)
            except psutil.NoSuchProcess:
                print("Application has closed.")
                break
                
            # 4. Aggregate metrics across the entire tree
            for proc in processes:
                try:
                    # Sum Resident Set Size (RSS) memory
                    total_mem_bytes += proc.memory_info().rss
                    # Sum CPU utilization
                    total_cpu_percent += proc.cpu_percent(interval=None) 
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Convert bytes to Megabytes
            total_mem_mb = total_mem_bytes / (1024 * 1024)
            
            print(f"Total CPU: {total_cpu_percent:.1f}% | Total RAM: {total_mem_mb:.2f} MB")
            time.sleep(2) # Refresh every 2 seconds

    except KeyboardInterrupt:
        print("\nStopped monitoring.")

if __name__ == "__main__":
    monitor_chainlit_app()
