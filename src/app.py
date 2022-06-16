from preprocess import Preprocess
from mlpipe import MLpipeline

df = Preprocess().preprocess_df()
MLpipeline().logregclassifier_pipeline(df)
