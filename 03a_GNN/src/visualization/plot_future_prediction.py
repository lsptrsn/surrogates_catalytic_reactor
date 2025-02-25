from src.utils.visualization import dataframe_to_gif
import pandas as pd


def plot_future_prediction(case, gnn):
    """
    Plot future predictions.

    Parameters:
    - case (int): The case number.
    - gnn (str): The type of GNN architecture ('GCN', 'GAT', 'GGNN', 'RGGNN').

    Returns:
    None
    """
    # Define the folder for the specific case
    case_folder = 'case_' + str(case)

    # Load true conversion and temperature data
    df_c_true_train = pd.read_csv(
        f'data/interim/{case_folder}/train_conversion.csv', header=None
        )
    df_T_true_train = pd.read_csv(
        f'data/interim/{case_folder}/train_temperature.csv', header=None
        )

    # Concatenate true data from train and test sets if available
    try:
        df_c_true_test = pd.read_csv(
            f'data/interim/{case_folder}/test_conversion.csv', header=None
            )
        df_T_true_test = pd.read_csv(
            f'data/interim/{case_folder}/test_temperature.csv', header=None
            )

        df_c_true = pd.concat([df_c_true_train.iloc[:, -1],
                               df_c_true_test],
                              axis=1)
        df_T_true = pd.concat([df_T_true_train.iloc[:, -1],
                               df_T_true_test],
                              axis=1)
    except:
        df_c_true = pd.DataFrame(df_c_true_train.iloc[:, -1])
        df_T_true = pd.DataFrame(df_T_true_train.iloc[:, -1])

    # Rename columns for clarity
    new_column_names = [f'{i}' for i in range(len(df_c_true.columns))]
    df_c_true.columns = new_column_names
    df_T_true.columns = new_column_names

    # Load predicted conversion and temperature data
    df_c_pred = pd.read_csv(f'reports/predictions/{case_folder}/c_{gnn}.csv')
    df_T_pred = pd.read_csv(f'reports/predictions/{case_folder}/T_{gnn}.csv')

    # Unnormalize Temperature
    T_min = 400  # should expected lowest T
    T_max = 800  # should be the high risk T
    df_T_true = df_T_true * (T_max - T_min) + T_min
    df_T_pred = df_T_pred * (T_max - T_min) + T_min

    # Make GIFs of the true and predicted data
    dataframe_to_gif(df_T_true,
                     df_T_pred,
                     gnn,
                     case,
                     output_filepath=f'reports/predictions/{case_folder}/T_{gnn}.gif',
                     snapshots_per_second=50)
    dataframe_to_gif(df_c_true,
                     df_c_pred,
                     gnn,
                     case,
                     output_filepath=f'reports/predictions/{case_folder}/c_{gnn}.gif',
                     snapshots_per_second=50)
