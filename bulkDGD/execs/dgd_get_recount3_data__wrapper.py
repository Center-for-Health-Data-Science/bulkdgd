#!/Users/fwv895/miniforge3/bin/python3.10

import os
import sys
import glob
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from functools import partial

from bulkDGD import recount3
from bulkDGD.execs import dgd_get_recount3_data as dgd_get_recount3_data__cli



# --------- parse info from pd.series to bulkDGD.execs.dgd_get_recount3_data function ----------
      

def get_recount3_data__process_row(row, output_folder):
    original_argv = sys.argv

    sys.argv = [
        'dgd_get_recount3_data.py',
        '-ip', row['input_project_name'],
        '-is', row['input_samples_category'],
        '-d', f"{output_folder}", #/{row['input_project_name']}_{row['input_samples_category']}.csv
        '-qs', row['query_string'],
        '-sm',
        '-sg',
        '-lc',
        '-v'
        # ,
        # '-vv'
    ]

    # Attempt to call the main function of the CLI module
    try:
        # print(f"\n\nAttempting download of {row['input_samples_category']}\n")
        dgd_get_recount3_data__cli.main()
        print(f"Successful download of file: {output_folder}/{row['input_project_name']}_{row['input_samples_category']}.csv\n\n\n")
    except Exception as e:
        print(f"\n\nAn error occurred: {e}\n\n\n")
    finally:
        # Restore the original sys.argv
        sys.argv = original_argv




# -------------------------- run dgd_get_recount3_data for each row in df -------------------

def get_recount3_data(data, output_folder, project_id_subset=None):
      # get data if data is path, otherwise just constrain format of DataFrame
      df = recount3.util.get_recount3_data_input_file(data=data, project_id_subset=project_id_subset)
      for n_object,row in enumerate(df.to_dict('records')):
        print(f"\n\n\nInitiating download of object number {n_object + 1}\n{row['input_samples_category']}\n\n")
        get_recount3_data__process_row(row, output_folder)




if __name__ == '__main__':
    main()