import subprocess
import sys

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError:
        print("Error during installation of requirements.txt.")
        sys.exit(1)

def download_spacy_model():
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_trf"])
    except subprocess.CalledProcessError:
        print("Error during installation of spaCy model.")
        sys.exit(1)

def main():
    print("Installing dependencies...")
    install_requirements()

    print("Downloading SpaCy model...")
    download_spacy_model()

    print("Setup completed!")


if __name__ == "__main__":
    main()
