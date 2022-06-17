from preprocess import Preprocess
from mlpipe import MLpipeline

df = Preprocess().preprocess_df()
MLpipeline().ml_workflow(df, "lr")
MLpipeline().ml_workflow(df, "dt")
MLpipeline().ml_workflow(df, "rf")
