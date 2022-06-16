from preprocess import Preprocess
from mlpipe import MLpipeline

df = Preprocess().preprocess_df()
MLpipeline(df)