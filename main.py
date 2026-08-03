import subprocess
import sys
import os
import time

# Ensure we're in the project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

def main():
    print("=" * 60)
    print("           Iryax AI System Starting...")
    print("=" * 60)
    
    # Open a log file to redirect background process output so the terminal stays clean
    log_file = open("system_services.log", "w", encoding="utf-8")
    
    print("[1/4] Starting Scraper in background...")
    scraper = subprocess.Popen(
        [sys.executable, "backend/scraper.py"],
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    
    print("[2/4] Starting API Server in background...")
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    
    # Wait a moment for API to start before frontend
    time.sleep(2)
    
    print("[3/4] Starting Streamlit Frontend in background...")
    # Streamlit needs to be run from the frontend directory for its config.toml
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8502", "--server.headless", "true"],
        cwd=os.path.join(ROOT, "frontend"),
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    
    print("[4/4] Starting Terminal Chatbot...")
    print("=" * 60)
    print("Logs for API, Scraper, and Streamlit are saving to 'system_services.log'")
    print("Press Ctrl+C at any time to shut down all services.")
    print("=" * 60)
    
    try:
        # Run the terminal chatbot in the foreground
        subprocess.run([sys.executable, "backend/llm.py"])
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[System] Shutting down services...")
        scraper.terminate()
        api.terminate()
        frontend.terminate()
        log_file.close()
        
        # Ensure they are actually killed on Windows (Uvicorn and Streamlit spawn worker processes)
        subprocess.call(['taskkill', '/F', '/T', '/PID', str(scraper.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.call(['taskkill', '/F', '/T', '/PID', str(api.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.call(['taskkill', '/F', '/T', '/PID', str(frontend.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("[System] All services stopped. Goodbye!")

if __name__ == "__main__":
    main()
