# this was created with gemini
# the way this works is you start the app (uv run chainlit run app.py) in one terminal and then start this file in another terminal and it will output the used resources over time (every 2 secs)
import psutil
import time
import subprocess
import json

def get_podman_stats(container_name="qdrant-server-nova"):
    try:
        # Request stats in JSON format for easy parsing
        result = subprocess.run(
            [
                "podman", "stats", "--no-stream", 
                "--format", "json", 
                container_name
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout.strip()
        if output:
            # Podman returns a JSON array containing stats for the requested container(s)
            stats_list = json.loads(output)
            if stats_list:
                container_stats = stats_list[0]
                # Podman JSON keys are lowercase (e.g., cpu_percent, mem_usage)
                cpu_str = str(container_stats.get('cpu_percent', 'N/A'))
                mem_str = str(container_stats.get('mem_usage', 'N/A'))
                
                # Append '%' if not present for consistency
                if cpu_str != 'N/A' and not cpu_str.endswith('%'):
                    cpu_str += '%'
                    
                return cpu_str, mem_str
                
        return "N/A", "N/A"
    except subprocess.CalledProcessError:
        return "Not Running", "Not Running"
    except FileNotFoundError:
        return "Podman CLI missing", "Podman CLI missing"
    except json.JSONDecodeError:
        return "Error parsing JSON", "Error parsing JSON"

def monitor_resources():
    target_process = None
    
    # 1. Find the Chainlit application process
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = p.info['cmdline']
            if cmdline and 'chainlit' in cmdline and 'app.py' in cmdline:
                target_process = p
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if target_process:
        print(f"Monitoring Chainlit Process Tree (PID: {target_process.pid}) and Qdrant Container...")
        target_process.cpu_percent(interval=None) # Prime CPU metric
    else:
        print("Chainlit app not found. Monitoring Qdrant container only...")

    try:
        while True:
            chainlit_mem_mb = 0
            chainlit_cpu = 0
            
            # 2. Gather Chainlit Python metrics
            if target_process:
                try:
                    processes = [target_process] + target_process.children(recursive=True)
                    for proc in processes:
                        chainlit_mem_mb += proc.memory_info().rss / (1024 * 1024)
                        chainlit_cpu += proc.cpu_percent(interval=None)
                except psutil.NoSuchProcess:
                    print("Chainlit application closed.")
                    target_process = None

            # 3. Gather Qdrant Docker metrics
            qdrant_cpu, qdrant_mem = get_podman_stats("qdrant-server-nova")
            
            # 4. Display combined metrics
            print("-" * 50)
            if target_process:
                print(f"Chainlit (Python): CPU: {chainlit_cpu:.1f}% | RAM: {chainlit_mem_mb:.2f} MB")
            print(f"Qdrant (Docker):   CPU: {qdrant_cpu} | RAM: {qdrant_mem}")
            
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nStopped monitoring.")

if __name__ == "__main__":
    monitor_resources()
