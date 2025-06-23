import subprocess
import sys


def install_ffmpeg():

    try:
        # Check if ffmpeg is already installed
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("FFmpeg is already installed.")
        return 0
    
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])

        print("FFmpeg is not installed. Installing FFmpeg...")
        # Install ffmpeg using apt-get (for Debian/Ubuntu systems)
        subprocess.run(["sudo", "apt-get", "install", "-y", "ffmpeg-python"], check=True)
        print("FFmpeg installation completed.")

        try:
            subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("FFmpeg is now installed and available.")
            return 0
        except subprocess.CalledProcessError:
            print("FFmpeg installation failed. Please check your system configuration.")
            return 1
        return 1
    
