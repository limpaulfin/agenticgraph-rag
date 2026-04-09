#!/usr/bin/env Rscript
# Abbildung: Knowledge Graph Construction Statistics
# Datenquelle: output/kg_construction_stats_100.json
# Farben: CUD Okabe-Ito palette (colorblind-friendly)
# Ausgabe: fig-kg-stats.pdf

library(ggplot2)
library(jsonlite)
library(gridExtra)

# --- CUD Okabe-Ito Farben ---
CUD_BLUE      <- "#0072B2"
CUD_ORANGE    <- "#E69F00"
CUD_GREEN     <- "#009E73"
CUD_VERMILION <- "#D55E00"

# --- Load real data ---
data_path <- "./output/"
raw <- fromJSON(data_path)
df <- as.data.frame(raw$per_question)

# --- Panel A: Node count distribution ---
p1 <- ggplot(df, aes(x = nodes)) +
  geom_histogram(binwidth = 10, fill = CUD_BLUE, color = "black",
                 linewidth = 0.3, alpha = 0.85) +
  geom_vline(xintercept = mean(df$nodes), linetype = "dashed",
             color = CUD_VERMILION, linewidth = 0.7) +
  annotate("text", x = mean(df$nodes) + 5, y = Inf,
           label = sprintf("mean = %.1f", mean(df$nodes)),
           vjust = 1.5, hjust = 0, size = 3, color = CUD_VERMILION) +
  labs(x = "Number of Nodes", y = "Count",
       subtitle = "(a) Node distribution") +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    plot.subtitle = element_text(face = "bold", size = 9)
  )

# --- Panel B: Edge count distribution ---
p2 <- ggplot(df, aes(x = edges)) +
  geom_histogram(binwidth = 50, fill = CUD_ORANGE, color = "black",
                 linewidth = 0.3, alpha = 0.85) +
  geom_vline(xintercept = mean(df$edges), linetype = "dashed",
             color = CUD_VERMILION, linewidth = 0.7) +
  annotate("text", x = mean(df$edges) + 20, y = Inf,
           label = sprintf("mean = %.1f", mean(df$edges)),
           vjust = 1.5, hjust = 0, size = 3, color = CUD_VERMILION) +
  labs(x = "Number of Edges", y = "Count",
       subtitle = "(b) Edge distribution") +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    plot.subtitle = element_text(face = "bold", size = 9)
  )

# --- Panel C: Nodes vs Edges scatter ---
p3 <- ggplot(df, aes(x = nodes, y = edges)) +
  geom_point(color = CUD_GREEN, alpha = 0.7, size = 1.8) +
  geom_smooth(method = "lm", se = TRUE, color = CUD_VERMILION,
              fill = CUD_VERMILION, alpha = 0.15, linewidth = 0.7) +
  labs(x = "Nodes", y = "Edges",
       subtitle = "(c) Nodes vs. Edges (n = 100)") +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    plot.subtitle = element_text(face = "bold", size = 9)
  )

# --- Panel D: Average degree distribution ---
p4 <- ggplot(df, aes(x = avg_degree)) +
  geom_histogram(binwidth = 1, fill = CUD_GREEN, color = "black",
                 linewidth = 0.3, alpha = 0.85) +
  geom_vline(xintercept = mean(df$avg_degree), linetype = "dashed",
             color = CUD_VERMILION, linewidth = 0.7) +
  annotate("text", x = mean(df$avg_degree) + 0.5, y = Inf,
           label = sprintf("mean = %.1f", mean(df$avg_degree)),
           vjust = 1.5, hjust = 0, size = 3, color = CUD_VERMILION) +
  labs(x = "Average Degree", y = "Count",
       subtitle = "(d) Degree distribution") +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    plot.subtitle = element_text(face = "bold", size = 9)
  )

# --- Combine ---
combined <- arrangeGrob(p1, p2, p3, p4, ncol = 2)

# --- Output ---
output_path <- "./output/"
ggsave(output_path, combined, width = 6.5, height = 5, device = cairo_pdf)
cat("Saved:", output_path, "\n")
