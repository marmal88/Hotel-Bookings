from src.preprocess import Preprocesor
from src.mlpipe import MLpipeline

if __name__ == "__main__":
    df = Preprocesor().preprocess_df()
    MLpipeline().ml_workflow(df, "lr")
    MLpipeline().ml_workflow(df, "dt")
    MLpipeline().ml_workflow(df, "rf")
    print("Run completed please see results in output folder")
