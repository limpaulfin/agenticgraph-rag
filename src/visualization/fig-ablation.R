#!/usr/bin/env Rscript
# Abbildung: Ablation Study - Component Impact
# Daten: ablation n=200, seed=42
# Ausgabe: fig-ablation.pdf

library(ggplot2)

CUD_GREEN      <- "#009E73"
CUD_ORANGE     <- "#E69F00"
CUD_BLUE       <- "#0072B2"
CUD_VERMILION  <- "#D55E00"
CUD_SKY_BLUE   <- "#56B4E9"

results <- data.frame(
  Variant = c("Full model", "w/o KG", "w/o Community", "w/o Vector", "w/o Fusion"),
  EM  = c(0.550, 0.535, 0.480, 0.460, 0.480),
  R5  = c(0.800, 0.790, 0.765, 0.730, 0.730),
  sig = c("", "", "*", "**", "*")
)
results$Variant <- factor(results$Variant,
  levels = rev(c("Full model", "w/o KG", "w/o Community", "w/o Vector", "w/o Fusion")))

colors <- rev(c(CUD_GREEN, CUD_SKY_BLUE, CUD_ORANGE, CUD_VERMILION, CUD_BLUE))

p <- ggplot(results, aes(x = Variant, y = R5, fill = Variant)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.3, width = 0.65) +
  geom_text(aes(label = sprintf("%.3f", R5)), hjust = -0.15, size = 3.2) +
  coord_flip(ylim = c(0, 0.9)) +
  scale_fill_manual(values = colors) +
  labs(x = NULL, y = "Recall@5") +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none",
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text = element_text(size = 9))

output_path <- "./output/"
ggsave(output_path, p, width = 5, height = 2.5, device = cairo_pdf)
cat("Saved:", output_path, "\n")
