"""
This script merges multiple csv files into one dataframe, de-duplicates records
and writes it to a csv file

Data is read from the input folder and written to the output folder

Authort: Derrick Lewis
Date: 2023-01-28
"""

import glob
import json
import logging
import pandas as pd

logging.basicConfig(
    filename="./logs/data_ingestion.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


# Function for data ingestion
def merge_multiple_dataframe(
    input_path: str,
    output_path: str,
    ext: str
        ) -> pd.DataFrame:
    """Merges multiple csv files into one dataframe, de-duplicates records
    and writes it to a csv file

    Parameters
    ---
    input_folder_path: str
        Path to the folder containing the csv files

    output_folder_path: str
        Path to the folder where the output file will be written

    Returns
    ---
    final_dataframe: pd.DataFrame

    """

    final_dataframe = pd.DataFrame(
        columns=["corporation", "lastmonth_activity", "lastyear_activity",
                 "number_of_employees", "exited"])

    # check for datasets, compile them together, and write to an output file
    # check if the input folder is empty
    result = glob.glob(f'./{input_path}/*.{ext}')
    try:
        assert len(result) > 0
    except AssertionError as e:
        logging.error("Input folder %s does not contain .csv files",
                      input_path)
        raise e
    logging.info("Found (%i) files in the input folder", len(result))
    for filename in result:
        logging.info("Reading file: %s", filename)
        try:
            df = pd.read_csv(filename)
        except Exception as e:
            logging.error("Could not read file: %s due to %s", filename, e)
            raise e
        logging.info("Successfully read file: %s", filename)
        final_dataframe = pd.concat([final_dataframe, df], axis=0)
    try:
        start_len = len(final_dataframe)
        final_dataframe = final_dataframe.drop_duplicates()
        logging.info("Removed %i duplicates", start_len - len(final_dataframe))
    except Exception as e:
        logging.error("Could not remove duplicates due to %s", e)
        raise e
    logging.info("Writing output file to %s", output_folder_path)
    final_dataframe.to_csv("./" + output_path + '/finaldata.csv',
                           index=False)
    logging.info("Successfully wrote output file to %s", output_path)

    # Save ingested file names as a python list
    logging.info('Saving ingested file names as a python list')
    with open('./' + output_path + '/ingestedfiles.txt', 'w',
              encoding='utf8') as file:
        file.write(str(result))
    return final_dataframe


if __name__ == '__main__':
    # Load config.json and get input and output paths
    with open('config.json', 'r', encoding="utf8") as f:
        config = json.load(f)

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    extension = 'csv'

    merge_multiple_dataframe(input_folder_path, output_folder_path, extension)

# import ast
# with open('./' + output_folder_path + '/ingestedfiles.txt', 'r') as f:
#     r2list = ast.literal_eval(f.read())