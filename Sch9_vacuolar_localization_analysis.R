library(tidyverse)
library(ggbeeswarm)
library(RColorBrewer)
library(ggeasy)
library(lawstat)

data_directory = "/path/to/your/data/directory"
strain_list = c("strain_01", "strain_02")
control_strain = "strain_01"

# Data loading function
load_csv_data = function(data_directory, strain, treatment) {
	csv_files = list.files(file.path(data_directory, strain, treatment, "measure"), 
                         pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
	 map_df(csv_files, ~ read_csv(.x, show_col_types = FALSE) %>%
           mutate(strain = strain, treatment = treatment))
}

# Background values calculation function
calculate_background_medians = function(filtered_data_by_size, control_strain, treatments) {
	filtered_data_by_size %>%
	filter(strain == control_strain) %>% 
	group_by(treatment) %>% 
	summarize(
		background_cytoplasm = median(cytoplasm), 
		background_vacuolar_membrane = median(vacuolar_membrane),
		.groups = "drop"
		)
}

# Background correction
correct_background = function(df, background_values) {
	df %>%
	left_join(background_values, by = "treatment") %>%
	mutate(
		cytoplasm = cytoplasm - background_cytoplasm,
		vacuolar_membrane = vacuolar_membrane - background_vacuolar_membrane
		) %>%
    select(-background_cytoplasm, -background_vacuolar_membrane)
}


# Boxplot generation function
generate_boxplot = function(df, strain, output_path, plot_title = "") {
	g = ggplot(df, aes(x = treatment, y = ratio, fill = treatment)) + geom_boxplot(outlier.colour = NA, 
		position = position_dodge(width = 0.8)) + geom_quasirandom(aes(group = treatment, 
		alpha = 0.5), dodge.width = 0.8, size = 1) + stat_summary(fun = mean, geom = "point", 
		colour = "red", size = 3, stroke = 2, shape = 4, position = position_dodge(width = 0.8)) + 
		scale_y_continuous(breaks = seq(0, 5, 0.1)) + scale_fill_manual(values = c(brewer.pal(9, 
		"Greens")[3], brewer.pal(9, "Reds")[3])) + theme_bw() + ggtitle(plot_title)
	g
	# ggsave(output_path, plot = g)
}

# Main processing
all_strain_data = map(strain_list, function(strain) {
	treatments = dir(file.path(data_directory, strain))
	map_dfr(treatments, function(treatment) {
		load_csv_data(data_directory, strain, treatment)
	})
})

combined_data = bind_rows(all_strain_data) %>% drop_na()

# Size filtering
filtered_data_by_size = combined_data %>% filter(cell_size >= 1100, vm_size >= 300)

# Background correction
background_values = calculate_background_medians(filtered_data_by_size, control_strain, unique(filtered_data_by_size$treatment))
background_corrected_data = correct_background(filtered_data_by_size, background_values)

# Final dataset
positive_data = background_corrected_data %>%
  filter(cytoplasm > 20, vacuolar_membrane > 10) %>%
  mutate(
    ratio = vacuolar_membrane / cytoplasm,
    treatment = factor(treatment, levels = c("treatment_01", "treatment_02"))
)


# Generate and save plot
generate_boxplot(filter(positive_data, strain == "strain_02"), "/path/to/save/plot/plot.pdf")


# Brunner-Munzel test
control_ratios = filter(positive_data, strain == "strain_02" & treatment == "treatment_01")$ratio
treatment_ratios = filter(positive_data, strain == "strain_02" & treatment == "treatment_02")$ratio
brunner.munzel.test(control_ratios, treatment_ratios)
