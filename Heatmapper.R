library(plyr)
library(reshape)
library(ggplot2)
library(scales)
library(comprehenr)


# Create our results table
#table <- read.table("milestone_results/loss_caml1_mnist_rotations.pt_2020_11_02_11_32_34_bc277e34d8684c0ea57e0ed85cb9beb3_data.csv", sep=",")
table <- read.table("milestone_results/online_mnist_rotations.pt_2020_11_02_11_36_08_e6a6ab3192a347be849dba44b2632f9e_data.csv", sep=",")
colnames(table) <- gsub(x = colnames(table), pattern = "V", replacement = "T")
colnames(table)
lt <- lower.tri(table, diag = TRUE)

# Data frame with just the lower triang. + diag
table_lower <- data.frame(row = rownames(table)[row(table)[lt]],
                          col = colnames(table)[col(table)[lt]],
                          Acc = table[lt],
                          stringsAsFactors = FALSE)
# Try with binding!
# Get the right order
# Assign the columns to by T1, ..., TN
table_lower$row1 <- factor(table_lower$row, levels = rev(rownames(table)))
#table_lower$col1 <- factor(table_lower$col, levels = colnames(table))
table_lower$col1 <- factor(table_lower$col, levels = colnames(table))
#rescale_m
#table_lower <- ddply(table_lower, .(col), transform, rescale = rescale_max(vals, to=c(0, 100)))
#table_lower <- ddply(table_lower, .(col), transform, rescale = rescale(vals))

p <- ggplot(table_lower, aes(x=col1, y=row1)) + 
  geom_tile(aes(fill = Acc), colour = "white") + 
  scale_fill_gradient(low = "white", high = "steelblue",
                      limits=c(0, 0.66)) 

# Add the theme formatting
base_size <- 9
p + theme_grey(base_size = base_size) + 
  labs(x = "Test Task Accuracy", y = "Time (tasks seen)") + scale_x_discrete(expand = c(0, 0)) + 
  scale_y_discrete(expand = c(0, 0)) + 
  ggtitle("Online Learning Continual Learning Heat Map") +
  theme(legend.position = "right", axis.ticks = element_blank(), 
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(size = base_size * 0.8, 
                                   angle = 0, hjust = 0, colour = "grey50"))



# Old shit
ids <- rownames(table)
table <- cbind(id=ids, table)
table$id
table$id <- factor(table$id, levels = rev(table$id))


table.m <- melt(table)
table.m <- ddply(table.m, .(variable), transform, rescale = rescale(value))

p <- ggplot(table.m, aes(x=variable, y = id)) + 
  geom_tile(aes(fill = rescale), colour = "white") + 
  scale_fill_gradient(low = "white", high = "steelblue") #+
  #geom_text(aes(label=value))

# Add the theme formatting
base_size <- 9
p + theme_grey(base_size = base_size) + 
  labs(x = "", y = "") + scale_x_discrete(expand = c(0, 0)) + 
  scale_y_discrete(expand = c(0, 0)) + 
  theme(legend.position = "none", axis.ticks = element_blank(), 
        axis.text.x = element_text(size = base_size * 0.8, 
                                   angle = 0, hjust = 0, colour = "grey50"))
