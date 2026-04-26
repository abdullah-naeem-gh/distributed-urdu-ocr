import subprocess
import os

def run_hdfs_cmd(cmd_list):
    """Run an hdfs command and return stdout. Raises exception on failure."""
    try:
        result = subprocess.run(
            ["hdfs", "dfs"] + cmd_list, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"HDFS Error: {e.stderr}")
        raise

def mkdir(hdfs_path):
    """Create a directory in HDFS (like mkdir -p)."""
    # -p ensures no error if it already exists
    run_hdfs_cmd(["-mkdir", "-p", hdfs_path])

def upload_file(local_path, hdfs_path):
    """Upload a local file to HDFS."""
    run_hdfs_cmd(["-put", "-f", local_path, hdfs_path])

def download_file(hdfs_path, local_path):
    """Download a file from HDFS to local."""
    run_hdfs_cmd(["-get", "-f", hdfs_path, local_path])

def read_file(hdfs_path):
    """Read a text file from HDFS directly into memory."""
    return run_hdfs_cmd(["-cat", hdfs_path])

def list_dir(hdfs_path):
    """List contents of a directory in HDFS. Returns list of filenames."""
    try:
        output = run_hdfs_cmd(["-ls", hdfs_path])
        lines = output.strip().split('\n')
        files = []
        for line in lines:
            if not line.startswith('Found'):
                parts = line.split()
                if len(parts) > 7:
                    # The path is usually the last part
                    full_path = parts[-1]
                    files.append(os.path.basename(full_path))
        return files
    except subprocess.CalledProcessError:
        return []

def delete(hdfs_path):
    """Delete a file or directory recursively."""
    try:
        run_hdfs_cmd(["-rm", "-r", "-f", hdfs_path])
    except subprocess.CalledProcessError:
        pass # Ignore if it doesn't exist

def file_exists(hdfs_path):
    """Check if a file exists in HDFS."""
    try:
        subprocess.run(
            ["hdfs", "dfs", "-test", "-e", hdfs_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError:
        return False
