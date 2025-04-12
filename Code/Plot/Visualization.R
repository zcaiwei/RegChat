library(igraph)
library(ggplot2)
library(dplyr)
library(scales)
library(circlize)


plot_signaling_circle <- function(signaling_df, coord_df, output_file = NULL, 
                                             plot_title = "Comparison of 10 Signaling Pathways") {
  # Define a palette with more colors
  color_palette <- c('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
  
  # Create color mapping based on unique cell types in coord_df
  unique_cell_types <- unique(coord_df$cell_type)
  color_map <- setNames(color_palette[1:length(unique_cell_types)], unique_cell_types)
  
  # Get all unique signaling pathways
  signaling_pathways <- unique(signaling_df$Signaling)
  if (length(signaling_pathways) > 10) {
    warning("Only plotting the first 10 signaling pathways")
    signaling_pathways <- signaling_pathways[1:10]
  }
  
  # Set up output file with larger canvas
  if (!is.null(output_file)) {
    file_ext <- tools::file_ext(output_file)
    if (file_ext == "png") {
      png(output_file, width = 24, height = 16, units = "in", res = 300)  # Increased size
    } else if (file_ext == "pdf") {
      pdf(output_file, width = 24, height = 16)  # Increased size
    } else if (file_ext %in% c("jpg", "jpeg")) {
      jpeg(output_file, width = 24, height = 16, units = "in", res = 300)  # Increased size
    } else {
      stop("Unsupported file format. Use png, pdf, or jpg.")
    }
  }
  
  # Set up multi-panel plot layout with proper margins
  n_cols <- min(ceiling(sqrt(length(signaling_pathways))), 4)  # Max 4 columns
  n_rows <- ceiling(length(signaling_pathways) / n_cols)
  
  # Adjust graphical parameters
  par(mfrow = c(n_rows, n_cols), 
      mar = c(2, 2, 4, 2),  # Adjusted margins
      oma = c(0, 0, 4, 0))  # Outer margin for title
  
  # Plot each signaling network
  for (pathway in signaling_pathways) {
    signaling_subset <- signaling_df %>% filter(Signaling == pathway)
    
    # Create igraph object
    g <- graph.empty(directed = TRUE)
    
    # Add nodes
    for (cell_type in unique_cell_types) {
      g <- add_vertices(g, 1, name = cell_type)
    }
    
    # Add edges and weights
    for (i in 1:nrow(signaling_subset)) {
      sender <- as.character(signaling_subset$Sender[i])
      receiver <- as.character(signaling_subset$Receiver[i])
      weight <- signaling_subset$Interaction_Strength[i]
      if (sender %in% unique_cell_types && receiver %in% unique_cell_types) {
        g <- add_edges(g, c(sender, receiver), weight = weight)
      }
    }
    
    # Set node colors and larger size
    V(g)$color <- color_map[V(g)$name]
    vertex.weight <- rep(25, length(V(g)))  # Increased node size
    
    # Set edge properties
    if (length(E(g)) > 0) {
      edge.start <- ends(g, es = E(g), names = FALSE)
      E(g)$color <- V(g)$color[edge.start[, 1]]
      edge.width = rescale(log(E(g)$weight + 1), to = c(1, 10))  # Slightly thicker
    } else {
      edge.width <- 1
      E(g)$color <- "black"
    }
    
    # Draw the plot with adjusted parameters
    plot(g, 
         vertex.size = vertex.weight, 
         vertex.color = V(g)$color, 
         vertex.label.cex = 2,  # Larger labels
         vertex.label.color = "black", 
         vertex.frame.color = NA,
         edge.width = edge.width,
         edge.color = E(g)$color,
         edge.arrow.size = 1.5,  # Larger arrows
         edge.curved = 0.2,
         layout = layout_in_circle(g)
         # main = pathway,
         # cex.main = 2  # Larger title for each subplot
    )
    title(main = pathway, cex.main = 2)
  }
  
  # Add overall title with proper spacing
  mtext(plot_title, outer = TRUE, cex = 1.8, line = 1, font = 3)
  
  # Add legend in empty space or last plot
  if (length(signaling_pathways) < n_rows * n_cols) {
    plot.new()
    legend("center", 
           legend = names(color_map),
           col = color_map,
           pch = 19,
           pt.cex = 2,  # Larger legend symbols
           title = "Cell Types",
           cex = 2,   # Larger text
           title.adj = 0.5)
  } else {
    # If no empty space, add compact legend to last plot
    legend("bottomright", 
           legend = names(color_map),
           col = color_map,
           pch = 19,
           pt.cex = 1.2,
           title = "Cell Types",
           cex = 1,
           bg = "white")
  }
  
  if (!is.null(output_file)) {
    dev.off()
  }
}


calculate_signaling_expression <- function(signaling_df, expmatrix, cell_type_df) {
  result_df <- data.frame(
    signaling = character(),
    condition = character(),
    expvalue = numeric(),
    commstrength = numeric(),
    stringsAsFactors = FALSE
  )
  
  cell_to_type <- setNames(cell_type_df$Label, cell_type_df$`Sample.Name`)
  
  for (i in 1:nrow(signaling_df)) {
    signaling <- signaling_df$Signaling[i]
    sender_type <- signaling_df$Sender[i]
    receiver_type <- signaling_df$Receiver[i]
    strength <- signaling_df$Interaction_Strength[i]
    
    components <- unlist(strsplit(signaling, "_"))
    ligand <- components[1]
    receptor_components <- components[-1]  
    
    sender_cells <- names(cell_to_type)[cell_to_type == sender_type]
    receiver_cells <- names(cell_to_type)[cell_to_type == receiver_type]
    
    ligand_expr <- 0
    if (ligand %in% rownames(expmatrix) && length(sender_cells) > 0) {
      ligand_expr <- mean(as.numeric(expmatrix[ligand, sender_cells]))
    }
    
    receptor_expr <- 0
    if (length(receptor_components) > 0 && length(receiver_cells) > 0) {
      receptor_expr_values <- numeric(0)
      valid_receptor <- TRUE
      
      for (comp in receptor_components) {
        if (comp %in% rownames(expmatrix)) {
          comp_expr <- mean(as.numeric(expmatrix[comp, receiver_cells]))
          if (comp_expr == 0) {
            valid_receptor <- FALSE
            break
          }
          receptor_expr_values <- c(receptor_expr_values, comp_expr)
        } else {
          valid_receptor <- FALSE
          break
        }
      }
      
      if (valid_receptor && length(receptor_expr_values) > 0) {
        receptor_expr <- mean(receptor_expr_values)
      }
    }
    
    expvalue <- 0
    if (ligand_expr > 0 && receptor_expr > 0) {
      expvalue <- mean(c(ligand_expr, receptor_expr))
    }
    
    condition <- paste0(sender_type, "->", receiver_type)
    
    result_df <- rbind(result_df, data.frame(
      signaling = signaling,
      condition = condition,
      expvalue = expvalue,
      commstrength = strength,
      stringsAsFactors = FALSE
    ))
  }
  
  return(result_df)
}


plot_signaling_dotplot <- function(data, output_file = NULL, 
                                   plot_title = "Signaling between Cell Types",
                                   width = 10, height = 8, dpi = 300) {
  required_cols <- c("signaling", "condition", "expvalue", "commstrength")
  if (!all(required_cols %in% colnames(data))) {
    stop("The input data must include the following columns: ", paste(required_cols, collapse = ", "))
  }
  
  plot_data <- data %>%
    filter(expvalue != 0) %>%  
    mutate(
      comm_group = case_when(
        commstrength > 0.1 ~ "> 0.1",
        commstrength > 0.01 ~ "0.01-0.1",
        TRUE ~ "< 0.01"  
      ),
      comm_group = factor(comm_group, levels = c("> 0.1", "0.01-0.1", "< 0.01"))
    )
  

  p <- ggplot(plot_data, aes(x = condition, y = signaling)) +
    geom_point(
      aes(color = expvalue, size = comm_group),
      alpha = 0.8,  
      stroke = 0.3  
    ) +
    scale_color_gradientn(
      colors = c("#3288bd", "#99d594", "#e6f598", "#fee08b", "#fc8d59", "#d53e4f"),
      name = "Expression\nStrength"
    ) +
    scale_size_manual(
      values = c("> 0.1" = 10, "0.01-0.1" = 7, "< 0.01" = 2),
      name = "Communication\nStrength"
    ) +
    labs(title = plot_title) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      panel.border = element_rect(fill = NA, color = "black"),
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
  
  if (!is.null(output_file)) {
    ggsave(
      filename = output_file,
      plot = p,
      width = width,
      height = height,
      dpi = dpi,
      bg = "white"
    )
  }
  
  return(p)
}



extract_ligand_receptor_pairs <- function(signaling_df, source_cell = "CT1", min_strength = 0.1) {
  required_cols <- c("Signaling", "Sender", "Receiver", "Interaction_Strength")
  if (!all(required_cols %in% colnames(signaling_df))) {
    stop("The input data frame must include the following columns: ", 
         paste(required_cols, collapse = ", "))
  }
  
  sub_df <- signaling_df %>% 
    filter(Sender == source_cell, 
           Interaction_Strength > min_strength,
           Sender != Receiver)  
  
  if (nrow(sub_df) == 0) {
    warning("No signal data found that meets the criteria (Sender = ", source_cell, 
            " and Interaction_Strength > ", min_strength, " and Sender != Receiver)")
    return(data.frame())
  }
  
  components <- strsplit(as.character(sub_df$Signaling), "_")
  sub_df$Source_Gene <- sapply(components, `[`, 1)
  sub_df$Target_Gene <- sapply(components, `[`, 2)
  
  result <- sub_df %>%
    mutate(Communication_Strength = 1) %>%
    select(
      Source_Cell = Sender,
      Source_Gene,
      Target_Cell = Receiver,
      Target_Gene,
      Communication_Strength
    ) %>%
    distinct()  
  
  return(result)
}


plot_cell_communication_circos <- function(data, 
                                           output_file, 
                                           color_palette = NULL,
                                           width = 10, 
                                           height = 10, 
                                           dpi = 300) {
  
  required_cols <- c("Source_Cell", "Target_Cell", "Source_Gene", "Target_Gene", "Communication_Strength")
  if (!all(required_cols %in% colnames(data))) {
    stop("The input data must include the following columns: ", paste(required_cols, collapse = ", "))
  }
  
  if (is.null(color_palette)) {
    color_palette <- c('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
  }
  
  all_celltypes <- unique(c(data$Source_Cell, data$Target_Cell))
  cell_colors <- setNames(
    color_palette[1:length(all_celltypes)],
    all_celltypes
  )
  
  jpeg(output_file, width = width, height = height, units = "in", res = dpi)
  circos.clear()
  
  circos.par(
    gap.after = rep(5, length(all_celltypes)),
    start.degree = 90,
    track.margin = c(0.01, 0.01)
  )
  
  gene_counts <- table(c(data$Source_Cell, data$Target_Cell))[all_celltypes]
  
  padding <- 0.5
  xlim <- cbind(rep(-padding, length(gene_counts)), gene_counts + padding)
  circos.initialize(factors = names(gene_counts), xlim = xlim)
  
  circos.trackPlotRegion(
    factors = names(gene_counts),
    ylim = c(0, 1),
    bg.col = cell_colors[names(gene_counts)],
    bg.border = "black",
    panel.fun = function(x, y) {
      circos.text(
        CELL_META$xcenter,
        CELL_META$ylim[2] + 0.6,
        CELL_META$sector.index,
        facing = "bending.inside",
        adj = c(0.5, 0.5),
        cex = 0.8,
        col = "white"
      )
    }
  )
  
  gene_positions_dict <- list()
  for (sector in names(gene_counts)) {
    genes <- unique(c(
      data$Source_Gene[data$Source_Cell == sector],
      data$Target_Gene[data$Target_Cell == sector]
    ))
    
    gene_positions <- seq(
      from = -padding + 0.5,
      to = gene_counts[sector] + padding - 0.5,
      length.out = length(genes)
    )
    
    for (i in seq_along(genes)) {
      gene_positions_dict[[paste(sector, genes[i], sep = "_")]] <- gene_positions[i]
    }
    
    for (i in seq_along(genes)) {
      circos.text(
        sector.index = sector,
        x = gene_positions[i],
        y = 1.5,
        labels = genes[i],
        facing = "clockwise",
        cex = 2,
        niceFacing = TRUE
      )
      
      circos.segments(
        sector.index = sector,
        x0 = gene_positions[i],
        y0 = 1.05,
        x1 = gene_positions[i],
        y1 = 1.25,
        col = "gray50",
        lwd = 0.8
      )
    }
  }
  
  for (i in 1:nrow(data)) {
    source_key <- paste(data$Source_Cell[i], data$Source_Gene[i], sep = "_")
    target_key <- paste(data$Target_Cell[i], data$Target_Gene[i], sep = "_")
    
    circos.link(
      sector.index1 = data$Source_Cell[i],
      point1 = gene_positions_dict[[source_key]],
      sector.index2 = data$Target_Cell[i],
      point2 = gene_positions_dict[[target_key]],
      col = adjustcolor(cell_colors[data$Source_Cell[i]], alpha.f = 0.7),
      lwd = data$Communication_Strength[i] * 2,
      directional = 1,
      arr.type = "triangle",
      arr.width = 1,
      arr.length = 0.65,
      arr.col = cell_colors[data$Source_Cell[i]]
    )
  }
  
  legend("bottomright", 
         legend = names(cell_colors),
         fill = cell_colors,
         title = "Cell Types",
         cex = 2,
         border = NA)
  
  dev.off()
}