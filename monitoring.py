import datetime
import json
import os

# Sample function to simulate getting accuracy or any metric
def get_model_accuracy():
    # Replace this with actual evaluation logic
    return 0.89

def log_metrics():
    metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "accuracy": get_model_accuracy()
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/monitoring_log.json", "a") as f:
        f.write(json.dumps(metrics) + "\n")

if __name__ == "__main__":
    log_metrics()
