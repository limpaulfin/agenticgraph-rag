#!/usr/bin/env Rscript
# Abbildung: Layer Progression - Cumulative EM improvement L1→L2→L3
# Daten: computed from checkpoint JSONL (n=1000, seed=42)
# HotpotQA: top_k=10, prompt v2 | MuSiQue: top_k=20, prompt v1
# Ausgabe: fig-layer-progression.pdf

library(ggplot2)

CUD_SKY_BLUE   <- "#56B4E9"
CUD_BLUE       <- "#0072B2"
CUD_GREEN      <- "#009E73"

results <- data.frame(
  Dataset = rep(c("HotpotQA", "MuSiQue"), each = 3),
  Layer   = rep(c("L1: CoT", "L1+L2: Verify", "L1+L2+L3: Search"), 2),
  EM      = c(0.603, 0.816, 0.868,
              0.475, 0.625, 0.690)
)
results$Layer <- factor(results$Layer,
  levels = c("L1: CoT", "L1+L2: Verify", "L1+L2+L3: Search"))
results$Dataset <- factor(results$Dataset, levels = c("HotpotQA", "MuSiQue"))

layer_colors <- c(CUD_SKY_BLUE, CUD_BLUE, CUD_GREEN)

p <- ggplot(results, aes(x = Layer, y = EM, fill = Layer, group = 1)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.3, width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", EM)), vjust = -0.5, size = 3) +
  facet_wrap(~ Dataset) +
  scale_fill_manual(values = layer_colors) +
  scale_y_continuous(limits = c(0, 1.0), breaks = seq(0, 1, 0.2),
                     expand = expansion(mult = c(0, 0.08))) +
  labs(x = NULL, y = "Exact Match (EM)") +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 25, hjust = 1, size = 8),
        strip.text = element_text(face = "bold", size = 11),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank())

output_path <- "./output/"
ggsave(output_path, p, width = 6, height = 3, device = cairo_pdf)
cat("Saved:", output_path, "\n")
