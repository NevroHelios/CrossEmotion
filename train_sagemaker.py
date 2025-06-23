from sagemaker.pytorch.estimator import PyTorch
from sagemaker.debugger.debugger import TensorBoardOutputConfig

#TODO: gotta update the bucket name and paths to s3

def start_training():
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path="bucket-name/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard"
    )

    estimator = PyTorch(
        entry_point="train_meld.py",
        source_dir="train",
        role="SageMakerRole",
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "epochs": "10",
            "batch-size": "32",
        },
        tensorboard_config=tensorboard_config
    )

    estimator.fit(
        {"training": "s3://bucket-name/data/train",
         "validation": "s3://bucket-name/data/val",
         "test": "s3://bucket-name/data/test"},
    )


if __name__ == "__main__":
    start_training()