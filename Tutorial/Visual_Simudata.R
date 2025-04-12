rm(list=ls())

setwd("D:/WHUer/notebooks/RegChat-main/Code/Plot")
source("Visualization.R")


# -----------------------Circle Cell Signaling Interaction Network -------------------------------------#
coord_df <- data.frame(
  cell_type = c("CT1", "CT2", "CT3")
)

interaction_df = read.csv("D:/WHUer/notebooks/RegChat-main/Simudata/interaction_strength_ct-ct.csv")

plot_signaling_circle(
  signaling_df = interaction_df,
  coord_df = coord_df,
  output_file = "multiple_signaling_circle_networks.jpg",
  plot_title = "Cell Signaling Interaction Network"
)


# -----------------------Dot plot -------------------------------------#
expmatrix <- read.csv("D:/WHUer/notebooks/RegChat-main/Simudata/simulated_data_000/simulated_adata_ST_00.csv", row.names = 1)
cell_type_df <- read.csv("D:/WHUer/notebooks/RegChat-main/Simudata/simulated_data_000/spot_CT1_CT3_label_str.txt", sep = '\t')
result <- calculate_signaling_expression(interaction_df, expmatrix, cell_type_df)

plot_signaling_dotplot(
    data = result,
    output_file = "signaling_dot_plot.jpg",
    plot_title = "Cell-Cell Communication Strength",
    width = 12,
    height = 9
  )


# -----------------------Circos plot -------------------------------------#
result <- extract_ligand_receptor_pairs(interaction_df, source_cell = "CT1", min_strength = 0.01)
plot_cell_communication_circos(
  data = result,
  output_file = "Circos_plot.jpg",
  width = 12,
  height = 12,
  dpi = 300
)
