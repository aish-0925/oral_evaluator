# run_pipeline.py
import os, pandas as pd
from src import config
from src.embed_score import merge_reference_answers, add_similarity_to_df
from src.utils import ensure_dir

def run_pipeline():
    data_csv = config.DATA_CSV
    ref_csv = config.REFERENCE_CSV
    if not os.path.exists(data_csv):
        print('DATA_CSV missing:', data_csv); return
    if not os.path.exists(ref_csv):
        print('REFERENCE_CSV missing:', ref_csv); return
    df_q = pd.read_csv(data_csv, encoding='latin1')
    df_ref = pd.read_csv(ref_csv, encoding='latin1')
    df = merge_reference_answers(df_q, df_ref, ref_text_col='reference_answer_text')
    if 'transcribed_text' not in df.columns:
        df['transcribed_text'] = ''
    df_sim = add_similarity_to_df(df, hyp_col='transcribed_text')
    out_path = config.OUTPUT_CSV
    # ensure directory
    ensure_dir(out_path)
    df_sim.to_csv(out_path, index=False)
    print('Saved similarity results to', out_path)

if __name__=='__main__':
    run_pipeline()
