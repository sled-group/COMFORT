import pandas as pd
from io import StringIO

def latex_to_dataframe(latex_table):
    lines = latex_table.split('\n')
    data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('\\')]
    data_lines = [line for line in data_lines if not line.startswith('Model') and not line.startswith('\\multicolumn')]
    
    data = []
    for line in data_lines:
        values = [val.strip() for val in line.split('&')]
        if len(values) == 17:  # Ensure we have the correct number of columns
            data.append(values)
    
    columns = [
        "Model", "C_behind", "R_behind", "A_behind", "M_behind",
        "C_infrontof", "R_infrontof", "A_infrontof", "M_infrontof",
        "C_totheleft", "R_totheleft", "A_totheleft", "M_totheleft",
        "C_totheright", "R_totheright", "A_totheright", "M_totheright"
    ]
    
    df = pd.DataFrame(data, columns=columns)
    return df

def clean_latex_dataframe(df):
    df = df[~df['Model'].str.contains(r'\\', na=False)].copy()
    numeric_columns = ['C_behind', 'R_behind', 'A_behind', 'M_behind', 
                       'C_infrontof', 'R_infrontof', 'A_infrontof', 'M_infrontof',
                       'C_totheleft', 'R_totheleft', 'A_totheleft', 'M_totheleft',
                       'C_totheright', 'R_totheright', 'A_totheright', 'M_totheright']
    # print(df['M_totheright'])
    for col in numeric_columns:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    # df = df.dropna(subset=numeric_columns, how='all')
    metrics_to_average = ['C_behind', 'R_behind', 'A_behind', 'C_infrontof', 'R_infrontof', 'A_infrontof',
                          'C_totheleft', 'R_totheleft', 'A_totheleft', 'C_totheright', 'R_totheright', 'A_totheright']
    df.loc[:, 'mean'] = df[metrics_to_average].mean(axis=1)
    return df

def sort_dataframe_by_model(df, model_order):
    df = df.copy()
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
    df = df.sort_values('Model').reset_index(drop=True)
    df = df.dropna(subset=['Model'])  # Remove rows with NaN values in the 'Model' column
    return df

def dataframe_to_latex(df, cosmode):
    header = r"""\newcolumntype{M}[1]{>{\centering\arraybackslash}m{#1}}
\begin{table*}[h]
\centering
\setlength{\tabcolsep}{1pt}
\begin{tabular}{|c|ccc|ccc|ccc|ccc|c|}
\hline
\multirow{2}{*}{Model} & \multicolumn{3}{c|}{behind} & \multicolumn{3}{c|}{infrontof} & \multicolumn{3}{c|}{totheleft} & \multicolumn{3}{c|}{totheright} & \multirow{2}{*}{mean} \\ \cline{2-13}
 & C & R & A & C & R & A & C & R & A & C & R & A & \\ \hline
"""
    if cosmode == "softcos":
        footer = r"""\hline
\end{tabular}
\caption{\textbf{NEWNEWNEWNEWNEW} Results of the perspective-taking metric on \texttt{COMFORT-CAR}. All values are $\varepsilon^\textrm{cos}$. C: Camera's perspective, A: Addressee's perspective, R: Reference object's perspective}
\label{tab:perspective_taking_metrics}
\end{table*}"""
    elif cosmode == "hardcos":
        footer = r"""\hline
\end{tabular}
\caption{\textbf{NEWNEWNEWNEWNEW} Results of the perspective-taking metric on \texttt{COMFORT-CAR}. All values are $\varepsilon^\textrm{hemi}$. C: Camera's perspective, A: Addressee's perspective, R: Reference object's perspective}
\label{tab:perspective_taking_metrics}
\end{table*}"""
    elif cosmode == "acc":
        footer = r"""\hline
\end{tabular}
\caption{\textbf{NEWNEWNEWNEWNEW} Results of the perspective-taking metric on \texttt{COMFORT-CAR}. All values are Acc. C: Camera's perspective, A: Addressee's perspective, R: Reference object's perspective}
\label{tab:perspective_taking_metrics}
\end{table*}"""
    else:
        raise NotImplementedError("this cosmode not supported yet")

    backslash_char = "\\"
    body = ""
    min_mean = df['mean'].min()
    for idx, row in df.iterrows():
        line = f"{row['Model']}"
        for metric in ['C_behind', 'A_behind', 'R_behind', 'C_infrontof', 'A_infrontof', 'R_infrontof', 'C_totheleft', 'A_totheleft', 'R_totheleft', 'C_totheright', 'A_totheright', 'R_totheright']:
            line += f" & {row[metric]:.1f}" if pd.notna(row[metric]) else " & "
        if row['mean'] == min_mean:
            line += f" & {backslash_char}textbf" + '{' + f"{row['mean']:.1f}" + '}' if pd.notna(row['mean']) else " & "
        else:
            line += f" & {row['mean']:.1f}" if pd.notna(row['mean']) else " & "
        line += r" \\ "
        body += line + "\n"

    return header + body + footer


# # Define the LaTeX table
# latex_table = r"""
# \begin{table}[h]
# \centering
# \begin{tabular}{|c|cccc|cccc|cccc|cccc|}
# \hline
# \multirow{2}{*}{Model} & \multicolumn{4}{c|}{behind} & \multicolumn{4}{c|}{infrontof} & \multicolumn{4}{c|}{totheleft} & \multicolumn{4}{c|}{totheright} \\ \cline{2-17} & C & R & A & M & C & R & A & M & C & R & A & M & C & R & A & M \\ \hline
# XComposer2 & 40.4 & 59.3 & 58.0 & 52.6 & 28.2 & 58.8 & 56.3 & 47.8 & 19.2 & 66.0 & 62.5 & 49.2 & 18.9 & 63.3 & 64.0 & 48.7 \\ 
# InstructBLIP-13B & 41.5 & 51.4 & 51.1 & 48.0 & 69.2 & 63.1 & 63.8 & 65.3 & 60.6 & 69.7 & 38.6 & 56.3 & 52.9 & 40.5 & 71.1 & 54.9 \\ 
# MiniCPM-V & 41.3 & 57.7 & 56.1 & 51.7 & 43.0 & 55.8 & 54.5 & 51.1 & 35.0 & 58.0 & 54.2 & 49.1 & 33.4 & 57.7 & 55.9 & 49.0 \\ 
# LLaVA-1.5-7B & 41.2 & 60.5 & 66.5 & 56.1 & 40.3 & 59.5 & 40.1 & \textbf{46.6} & 23.5 & 66.2 & 56.1 & 48.6 & 24.9 & 61.1 & 57.1 & 47.7 \\ 
# GLaMM-FullScope & 47.3 & 55.3 & 61.6 & 54.8 & 41.2 & 59.8 & 52.4 & 51.1 & 30.1 & 61.0 & 53.5 & 48.2 & 33.7 & 63.2 & 53.2 & 50.0 \\ 
# LLaVA-1.5-13B & 37.7 & 45.1 & 51.1 & \textbf{44.6} & 49.3 & 49.4 & 41.5 & 46.7 & 29.6 & 65.2 & 53.6 & 49.5 & 31.0 & 63.5 & 58.3 & 50.9 \\ 
# InstructBLIP-7B & 51.2 & 54.1 & 54.8 & 53.4 & 59.4 & 57.7 & 56.7 & 58.0 & 56.1 & 54.3 & 54.6 & 55.0 & 55.1 & 55.8 & 54.4 & 55.1 \\ 
# GPT-4o & 43.7 & 45.8 & 46.5 & 45.3 & 51.2 & 62.3 & 56.5 & 56.6 & 23.1 & 59.2 & 59.5 & \textbf{47.3} & 21.5 & 58.1 & 60.8 & \textbf{46.8} \\ 
# \hline
# \end{tabular}
# \caption{COSINE-Hard metric: perspective taking (hardcos)}
# \label{tab:perspective_taking_metrics}
# \end{table}
# """

def convert_perspective_taking_results(latex_table, cosmode):
    # Define the desired model order
    model_order = [
        "InstructBLIP-7B", "InstructBLIP-13B", "mBLIP-BLOOMZ-7B",
        "LLaVA-1.5-7B", "LLaVA-1.5-13B", 
        "GLaMM-FullScope", "XComposer2",
        "MiniCPM-V", "GPT-4o"
    ]

    # After latex_to_dataframe
    df = latex_to_dataframe(latex_table)

    # print(df)

    # Then continue with the rest of the processing
    df_cleaned = clean_latex_dataframe(df)
    df_sorted = sort_dataframe_by_model(df_cleaned, model_order)
    df_sorted.to_excel(f"workspace/perspective_taking_metrics_{cosmode}.xlsx", merge_cells=True)
    # latex_output = dataframe_to_latex(df_sorted, cosmode)

    # print(latex_output)