from src.preprocess import Preprocess
from src.mlpipe import MLpipeline
from src.config.config_load import read_yaml_file

if __name__=="__main__":
    df = Preprocess().preprocess_df()
    MLpipeline().ml_workflow(df, "lr")
    MLpipeline().ml_workflow(df, "dt")
    MLpipeline().ml_workflow(df, "rf")
    print("Run completed please see results in output folder")
