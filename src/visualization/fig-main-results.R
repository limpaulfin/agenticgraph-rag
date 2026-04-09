#!/usr/bin/env Rscript
# Abbildung: Main Results - Grouped Bar Chart (EM + F1)
# Daten: 3-layer pipeline results, n=1000, seed=42
# Farben: CUD Okabe-Ito palette (colorblind-friendly)
# Ausgabe: fig-main-results.pdf

library(ggplot2)

# --- CUD Okabe-Ito Farben ---
CUD_SKY_BLUE   <- "#56B4E9"
CUD_BLUE       <- "#0072B2"
CUD_ORANGE     <- "#E69F00"
CUD_GREEN      <- "#009E73"

# --- Experimentelle Daten (n=1000, seed=42) ---
# AgenticGraph-RAG: EM@ALL/F1@ALL from 3-layer pipeline
# HotpotQA: top_k=10, prompt v2 | MuSiQue: top_k=20, prompt v1
results <- data.frame(
  Dataset = rep(c("HotpotQA", "MuSiQue"), each = 4),
  Method  = rep(c("Naive RAG", "GraphRAG-Local", "GraphRAG-Global",
                   "AgenticGraph-RAG"), 2),
  EM      = c(0.732, 0.781, 0.793, 0.868,
              0.473, 0.565, 0.569, 0.690),
  F1      = c(0.734, 0.783, 0.794, 0.912,
              0.480, 0.570, 0.575, 0.781)
)

# Faktor-Reihenfolge
results$Method <- factor(results$Method,
  levels = c("Naive RAG", "GraphRAG-Local", "GraphRAG-Global",
             "AgenticGraph-RAG"))
results$Dataset <- factor(results$Dataset,
  levels = c("HotpotQA", "MuSiQue"))

# --- Reshape to long format ---
library(tidyr)
results_long <- pivot_longer(results, cols = c(EM, F1),
                             names_to = "Metric", values_to = "Score")
results_long$Metric <- factor(results_long$Metric, levels = c("EM", "F1"))

# --- Plot ---
method_colors <- c(
  "Naive RAG"              = CUD_SKY_BLUE,
  "GraphRAG-Local"         = CUD_BLUE,
  "GraphRAG-Global"        = CUD_ORANGE,
  "AgenticGraph-RAG"         = CUD_GREEN
)

p <- ggplot(results_long, aes(x = Metric, y = Score, fill = Method)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.75),
           width = 0.65, color = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.3f", Score)),
            position = position_dodge(width = 0.75),
            vjust = -0.5, size = 2.5) +
  facet_wrap(~ Dataset, scales = "free_y") +
  scale_fill_manual(values = method_colors) +
  scale_y_continuous(limits = c(0, 1.0), breaks = seq(0, 1, 0.2),
                     expand = expansion(mult = c(0, 0.08))) +
  labs(x = NULL, y = "Score", fill = "Method") +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 9),
    legend.text = element_text(size = 8),
    strip.text = element_text(face = "bold", size = 11),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text = element_text(size = 9),
    axis.title.y = element_text(size = 10),
    plot.margin = margin(5, 10, 5, 5)
  )

# --- Output ---
output_path <- "./output/"

ggsave(output_path, p, width = 6.5, height = 3.5, device = cairo_pdf)
cat("Saved:", output_path, "\n")
