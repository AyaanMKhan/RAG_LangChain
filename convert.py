import pyreadr
import pandas as pd
import os

def rda_to_markdown(rda_path, md_path):
    # 1. Load the .rda file
    # pyreadr returns a dictionary where keys are object names and values are pandas DataFrames
    result = pyreadr.read_r(rda_path)
    
    # 2. Extract the object name (the internal R variable name)
    object_names = list(result.keys())
    print(f"Found objects in RDA: {object_names}")
    
    # Take the first available object in the file
    first_object_name = object_names[0]
    df = result[first_object_name]
    
    # 3. Check if the extracted object is a valid DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"The object '{first_object_name}' is not a structure that can be converted to a table.")
    
    # 4. Convert the DataFrame to a Markdown table string
    # This requires the 'tabulate' library installed in your environment
    markdown_table = df.to_markdown(index=False)
    
    # 5. Write the Markdown string to an .md file
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Data Object: {first_object_name}\n\n")
        f.write(markdown_table)
        
    print(f"Successfully converted '{rda_path}' to '{md_path}'")


old_path = os.getcwd() + "/old_data"
path = os.listdir(old_path)



for i in range(len(path)):
    old_file = (old_path + "/"+path[i])

    new_file = old_file.split("/")[-1]
    new_file = new_file.split(".")[0] + ".md"
    new_path = os.getcwd() + "/new_data/"
    new = new_path+new_file
    
    rda_to_markdown(old_file, new)
    