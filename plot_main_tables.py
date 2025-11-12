from table_plots_scripts.plots_litellm_daily_tag_spend import plot_all_litellm_daily_tag_spend
from table_plots_scripts.plots_litellm_daily_team_spend import plot_all_litellm_daily_team_spend
from table_plots_scripts.plots_litellm_dayly_user_spend import plot_all_litellm_daily_user_spend
from table_plots_scripts.plots_litellm_end_user_table import plot_all_litellm_end_user_table
from table_plots_scripts.plots_litellm_spendlogs import plot_all_litellm_spendlogs


if __name__ == "__main__":
    DATA_DIR = "data"
    FIGS_DIR = "figs"

    plot_all_litellm_daily_tag_spend(dir_name=DATA_DIR, outdir=f"{FIGS_DIR}/litellm_daily_tag_spend")
    print("Created plots for table 'Daily Tag Spend'")
    plot_all_litellm_daily_team_spend(dir_name=DATA_DIR, outdir=f"{FIGS_DIR}/litellm_daily_team_spend")
    print("Created plots for the table 'Daily Team Spend'")
    plot_all_litellm_daily_user_spend(dir_name=DATA_DIR, outdir=f"{FIGS_DIR}/litellm_daily_user_spend")
    print("Created plots for the table 'Daily User Spend'")
    plot_all_litellm_end_user_table(dir_name=DATA_DIR, outdir=f"{FIGS_DIR}/litellm_end_user_table")
    print("Created plots for the table 'End User Table'")
    plot_all_litellm_spendlogs(dir_name=DATA_DIR, outdir=f"{FIGS_DIR}/litellm_spendlogs")
    print("Created plots for the table 'Spend Logs'")
