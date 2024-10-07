import re
import pandas as pd

def extract_table_data(latex_table):
    # Remove LaTeX specific commands
    latex_table = latex_table.replace('\\textbf{', '').replace('}', '')
    lines = latex_table.strip().split('\n')
    
    # Find the lines containing headers and data
    headers = None
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith('\\begin{tabular'):
            headers = [h.strip() for h in re.split(r'&|\s*\\\\', lines[i+2]) if h.strip()]
        if line.startswith('InstructBLIP-7B') or \
            line.startswith('InstructBLIP-13B') or \
            line.startswith('LLaVA-1.5-7B') or \
            line.startswith('LLaVA-1.5-13B') or \
            line.startswith('GLaMM-FullScope') or \
            line.startswith('XComposer2') or \
            line.startswith('MiniCPM-V') or \
            line.startswith('GPT-4o'):
            data_start = i
            break
    
    # Extract data rows
    data = []
    for line in lines[data_start:]:
        row = [item.strip() for item in re.split(r'&|\s*\\\\', line) if item.strip()]
        # Ensure the row matches the length of headers
        if len(row) == len(headers):
            data.append(row)
    
    return headers, data

def convert_to_dict(headers, data):
    # Ensure headers and data are aligned correctly
    if not all(len(row) == len(headers) for row in data):
        raise ValueError("Mismatch between number of headers and data columns")
    return {header: [row[i] for row in data] for i, header in enumerate(headers)}

# latex_table_B = """
# \\begin{table}[h]
# \\centering
# \\begin{tabular}{lccccccccc}
# \\hline
# Model & F1 & $soft err_{gt}$ & clipped $err_{gt}$ & hard $err_{gt}$ & $acc$ & $err_{sym.spa}$ & $err_{sym.rr}$ & $noise$ & $p.std$ \\ 
# \\hline
# LLaVA-1.5-13B (camera3) & 100.0 & 25.7 & 38.1 & 37.6 & 55.3 & 19.3 & 24.3 & 7.0 & 9.3 \\ 
# XComposer2 (camera3) & 100.0 & 20.0 & 27.3 & \\textbf{21.1} & 92.4 & \\textbf{19.2} & \\textbf{13.1} & 9.0 & 10.5 \\ 
# GLaMM-FullScope (camera3) & 100.0 & 33.0 & 46.5 & 45.2 & \\textbf{47.2} & 29.9 & 43.9 & 10.1 & 13.7 \\ 
# MiniCPM-V (camera3) & 99.3 & \\textbf{16.8} & 28.9 & 24.9 & 89.3 & 23.4 & 16.0 & 6.6 & \\textbf{7.7} \\ 
# LLaVA-1.5-7B (camera3) & 100.0 & 20.7 & 33.0 & 33.7 & 63.2 & 25.2 & 23.0 & \\textbf{5.8} & 8.3 \\ 
# GPT-4o (camera3) & 100.0 & 27.4 & \\textbf{22.7} & 27.5 & 89.2 & 20.9 & 42.4 & 14.1 & 14.2 \\ 
# InstructBLIP-7B (camera3) & 66.7 & 43.9 & 49.6 & 57.8 & \\textbf{47.2} & 26.7 & 48.2 & 17.2 & 16.6 \\ 
# InstructBLIP-13B (camera3) & 67.3 & 43.0 & 48.5 & 55.5 & \\textbf{47.2} & 27.1 & 48.1 & 17.3 & 21.0 \\ 
# \\hline
# \\end{tabular}
# \\caption{General metrics on COSINE-Simple}
# \\label{tab:simple_metrics}
# \\end{table}
# """

# latex_table_C = """
# \\begin{table}[h]
# \\centering
# \\begin{tabular}{lccccccccc}
# \\hline
# Model & F1 & $soft err_{gt}$ & clipped $err_{gt}$ & hard $err_{gt}$ & $acc$ & $err_{sym.spa}$ & $err_{sym.rr}$ & $noise$ & $p.std$ \\ 
# \\hline
# LLaVA-1.5-13B (camera3) & 97.7 & 23.7 & 38.3 & 36.9 & 51.9 & 21.1 & 29.4 & 5.7 & 11.1 \\ 
# XComposer2 (camera3) & 94.7 & 18.9 & 33.4 & \\textbf{26.7} & 85.1 & \\textbf{15.7} & 23.2 & 6.6 & 11.8 \\ 
# GLaMM-FullScope (camera3) & 99.6 & 23.6 & 36.1 & 38.1 & \\textbf{47.2} & 23.8 & 28.5 & 9.3 & 15.0 \\ 
# MiniCPM-V (camera3) & 66.7 & 24.7 & 33.0 & 38.2 & 61.8 & 21.7 & 23.1 & 11.8 & 16.3 \\ 
# LLaVA-1.5-7B (camera3) & 88.3 & \\textbf{18.3} & \\textbf{31.3} & 32.5 & 55.1 & 20.0 & \\textbf{21.2} & \\textbf{5.3} & \\textbf{10.9} \\ 
# GPT-4o (camera3) & 95.6 & 28.3 & 33.8 & 34.9 & 78.3 & 26.8 & 38.2 & 13.1 & 16.5 \\ 
# InstructBLIP-7B (camera3) & 66.7 & 42.6 & 48.8 & 55.5 & \\textbf{47.2} & 27.3 & 47.7 & 13.3 & 20.8 \\ 
# InstructBLIP-13B (camera3) & 41.0 & 43.7 & 49.3 & 56.1 & \\textbf{47.2} & 37.4 & 53.6 & 12.7 & 18.9 \\ 
# \\hline
# \\end{tabular}
# \\caption{General metrics on COSINE-Hard}
# \\label{tab:simple_metrics}
# \\end{table}
# """

def convert_general_metrics_results(latex_table_B, latex_table_C):

    # Extract data from LaTeX tables
    headers_B, data_B = extract_table_data(latex_table_B)
    headers_C, data_C = extract_table_data(latex_table_C)

    # Convert extracted data to dictionaries
    dict_B = convert_to_dict(headers_B, data_B)
    dict_C = convert_to_dict(headers_C, data_C)

    # Create dataframes from dictionaries
    df_B = pd.DataFrame(dict_B)
    df_C = pd.DataFrame(dict_C)

    # Ensure 'Model' column has consistent naming
    df_B.columns = [col.strip() for col in df_B.columns]
    df_C.columns = [col.strip() for col in df_C.columns]

    # Remove (camera3) from Model names
    df_B['Model'] = df_B['Model'].str.replace(' (camera3)', '')
    df_C['Model'] = df_C['Model'].str.replace(' (camera3)', '')

    # Merge dataframes on Model
    df_merged = pd.merge(df_B, df_C, on='Model', suffixes=(' (B)', ' (C)'))

    # Reformat and rename columns for output table
    df_merged.columns = [
        'Model', 'F1 (B)', 'soft_err_gt (B)', 'clipped_err_gt (B)', 'hard_err_gt (B)', 'acc (B)',
        'err_sym_spa (B)', 'err_sym_rr (B)', 'noise (B)', 'p_std (B)',
        'F1 (C)', 'soft_err_gt (C)', 'clipped_err_gt (C)', 'hard_err_gt (C)', 'acc (C)',
        'err_sym_spa (C)', 'err_sym_rr (C)', 'noise (C)', 'p_std (C)'
    ]

    # Define order of columns for output
    output_columns = [
        'Model', 'F1 (B)', 'F1 (C)', 'acc (B)', 'acc (C)', 'soft_err_gt (B)', 'soft_err_gt (C)',
        'hard_err_gt (B)', 'hard_err_gt (C)', 'err_sym_spa (B)', 'err_sym_spa (C)',
        'err_sym_rr (B)', 'err_sym_rr (C)', 'noise (B)', 'noise (C)', 'p_std (B)', 'p_std (C)'
    ]

    # Reorder columns
    df_output = df_merged[output_columns]

    # Ensure the order of rows matches the specified order
    model_order = [
        "InstructBLIP-7B", "InstructBLIP-13B", "mBLIP-BLOOMZ-7B", 
        "LLaVA-1.5-7B", "LLaVA-1.5-13B", "GLaMM-FullScope", "XComposer2",
        "MiniCPM-V", "GPT-4o"
    ]
    # print("df_output:", df_output)
    df_output = df_output.set_index('Model').loc[model_order].reset_index()
    df_output.to_excel(f"workspace/comprehensive.xlsx", merge_cells=True)
    # # Convert dataframe to LaTeX table manually
    # latex_table_output = (
    #     "\\begin{table*}[ht]\n"
    #     "\\setlength{\\tabcolsep}{1.5pt}\n"
    #     "\\centering\n"
    #     "\\begin{tabular}{|c|cc|cc|cc|cc|cc|cc|cc|cc|}\n"
    #     "\\hline\n"
    #     "\\multirow{2}{*}{Model} & \\multicolumn{2}{c|}{Obj F1 ($\\uparrow$)} & \\multicolumn{2}{c|}{Acc ($\\uparrow$)} & \\multicolumn{2}{c|}{$\\varepsilon^\\textrm{cos}$ ($\\downarrow$)} & \\multicolumn{2}{c|}{$\\varepsilon^\\textrm{hemi}$ ($\\downarrow$)} & \\multicolumn{2}{c|}{$\\eta$ ($\\downarrow$)} & \\multicolumn{2}{c|}{$\sigma$ ($\\downarrow$)} & \\multicolumn{2}{c|}{$c^\\textrm{sym}$ ($\\downarrow$)} & \\multicolumn{2}{c|}{$c^\\textrm{opp}$ ($\\downarrow$)} \\\\ \\cline{2-17}\n"
    #     " & B & C & B & C & B & C & B & C & B & C & B & C & B & C & B & C\\\\ \\hline\n"
    # )

    # # Add the data rows
    # max_acc_B = df_output['acc (B)'].max()
    # max_acc_C = df_output['acc (C)'].max()

    # min_soft_err_gt_B = df_output['soft_err_gt (B)'].min()
    # min_soft_err_gt_C = df_output['soft_err_gt (C)'].min()
    # min_hard_err_gt_B = df_output['hard_err_gt (B)'].min()
    # min_hard_err_gt_C = df_output['hard_err_gt (C)'].min()
    # min_err_sym_spa_B = df_output['err_sym_spa (B)'].min()
    # min_err_sym_spa_C = df_output['err_sym_spa (C)'].min()
    # min_err_sym_rr_B = df_output['err_sym_rr (B)'].min()
    # min_err_sym_rr_C = df_output['err_sym_rr (C)'].min()
    # min_noise_B = df_output['noise (B)'].min()
    # min_noise_C = df_output['noise (C)'].min()
    # min_p_std_B = df_output['p_std (B)'].min()
    # min_p_std_C = df_output['p_std (C)'].min()

    # # Step 2: Iterate through the dataframe and format the LaTeX output
    # backslash_char = "\\"
    # for _, row in df_output.iterrows():
    #     latex_table_output += (
    #         f"{row['Model']} & "
    #         f"{row['F1 (B)']} & "
    #         f"{row['F1 (C)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['acc (B)']) + '}' if row['acc (B)'] == max_acc_B else row['acc (B)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['acc (C)']) + '}' if row['acc (C)'] == max_acc_C else row['acc (C)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['soft_err_gt (B)']) + '}' if row['soft_err_gt (B)'] == min_soft_err_gt_B else row['soft_err_gt (B)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['soft_err_gt (C)']) + '}' if row['soft_err_gt (C)'] == min_soft_err_gt_C else row['soft_err_gt (C)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['hard_err_gt (B)']) + '}' if row['hard_err_gt (B)'] == min_hard_err_gt_B else row['hard_err_gt (B)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['hard_err_gt (C)']) + '}' if row['hard_err_gt (C)'] == min_hard_err_gt_C else row['hard_err_gt (C)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['noise (B)']) + '}' if row['noise (B)'] == min_noise_B else row['noise (B)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['noise (C)']) + '}' if row['noise (C)'] == min_noise_C else row['noise (C)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['p_std (B)']) + '}' if row['p_std (B)'] == min_p_std_B else row['p_std (B)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['p_std (C)']) + '}' if row['p_std (C)'] == min_p_std_C else row['p_std (C)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['err_sym_spa (B)']) + '}' if row['err_sym_spa (B)'] == min_err_sym_spa_B else row['err_sym_spa (B)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['err_sym_spa (C)']) + '}' if row['err_sym_spa (C)'] == min_err_sym_spa_C else row['err_sym_spa (C)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['err_sym_rr (B)']) + '}' if row['err_sym_rr (B)'] == min_err_sym_rr_B else row['err_sym_rr (B)']} & "
    #         f"{backslash_char + 'textbf{' + str(row['err_sym_rr (C)']) + '}' if row['err_sym_rr (C)'] == min_err_sym_rr_C else row['err_sym_rr (C)']} \\\\\n"
    #     )

    # # Add the ending structure
    # latex_table_output += (
    #     "\\hline\n"
    #     "\\end{tabular}\n"
    #     "\\caption{\\textbf{NEWNEWNEWNEWNEW} General metrics on \\texttt{COMFORT-BALL} (denote as B) and \\texttt{COSINE-CAR} (denote as C).}\n"
    #     "\\label{tab:simple_metrics_simple_data}\n"
    #     "\\end{table*}"
    # )

    # # Print the final LaTeX table
    # print(latex_table_output)
