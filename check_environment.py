import sys
import pkg_resources

def check_environment():
    print(f"Python version: {sys.version}")
    
    packages = [
        "faiss-cpu", "sentence-transformers", "torch", "numpy",
        "transformers", "rasa-sdk"
    ]
    
    for package in packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✓ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"✗ {package}: NON INSTALLÉ")

if __name__ == "__main__":
    check_environment()