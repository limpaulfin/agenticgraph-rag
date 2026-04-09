#!/usr/bin/env Rscript
# Abbildung: Latency Decomposition - pie/bar chart of time breakdown
# Ausgabe: fig-latency.pdf

library(ggplot2)

CUD_SKY_BLUE   <- "#56B4E9"
CUD_BLUE       <- "#0072B2"
CUD_ORANGE     <- "#E69F00"
CUD_GREEN      <- "#009E73"
CUD_VERMILION  <- "#D55E00"

results <- data.frame(
  Component = c("KG Construction", "Retrieval + W-RRF", "Re-ranking",
                "Community Summ.", "LLM Generation"),
  Time = c(0.43, 0.50, 1.20, 12.10, 15.12),
  Pct  = c(1.5, 1.7, 4.1, 41.2, 51.5)
)
results$Component <- factor(results$Component,
  levels = rev(results$Component))

colors <- rev(c(CUD_SKY_BLUE, CUD_BLUE, CUD_GREEN, CUD_ORANGE, CUD_VERMILION))

p <- ggplot(results, aes(x = Component, y = Time, fill = Component)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.3, width = 0.6) +
  geom_text(aes(label = sprintf("%.1fs (%.0f%%)", Time, Pct)),
            hjust = -0.05, size = 3) +
  coord_flip(ylim = c(0, 20)) +
  scale_fill_manual(values = colors) +
  labs(x = NULL, y = "Time (seconds/query)") +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none",
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size = 9))

output_path <- "./output/"
ggsave(output_path, p, width = 5, height = 2.5, device = cairo_pdf)
cat("Saved:", output_path, "\n")
