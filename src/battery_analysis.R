rm(list = ls())
library(tidyverse)

setwd("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\src")
directory = "..\\data\\Output-2023-05-20"


read_loss_files <- function(directory, model_type) {
  # Create an empty list to store the dataframes
  dfs <- list()
  
  # Get the list of subdirectories
  subdirectories <- list.files(directory, full.names = TRUE)
  
  # Loop through the subdirectories
  for (subdir in subdirectories) {
    model_dirs <- list.dirs(subdir, full.names = TRUE, recursive = FALSE)
    
    for (model in model_dirs){
    # Check if the subdirectory matches the specified model_type
      if (grepl(paste0("^", model_type, "$"), basename(model))) {
      # Get the path to the loss file
        loss_file <- file.path(model, paste0(model_type, "_loss.csv"))
        
        # Read the loss file as a dataframe
        df <- read.csv(loss_file) %>% mutate(lang = tools::file_path_sans_ext(basename(subdir)))
        
        
        # Add the dataframe to the list
        dfs <- c(dfs, list(df))
      }
    }
  }
  # Combine all dataframes into a single dataframe
  combined_df <- do.call(rbind, dfs)
  
  # Return the combined dataframe
  return(combined_df)

}


pareto <- function(data, x, y, convex=TRUE){
  
  pareto.1 <- logical(length(data[,x]))
  x.sort <- sort(data[,x])
  y.sort <- data[,y][order(data[,x])]
  
  for(i in 1:length(x.sort)){
    pareto.1[i] <- all(y.sort[1:i] >= y.sort[i])
  }
  
  if(convex){
    chull.df <- data.frame(x=x.sort,y=y.sort,pareto=pareto.1)
    chull.filtered = chull.df[chull.df$pareto & is.finite(chull.df$x) & is.finite(chull.df$y),]
    indices <- chull(chull.filtered$x, chull.filtered$y)
    hull_points <- rownames(chull.filtered)[indices]
    chull.df$convex <- rownames(chull.df) %in% hull_points
    pareto.1 = chull.df$convex
  }
  
  return(pareto.1[match(1:length(data[,x]), order(data[,x]))])
}



add_pareto <- function(data, group, x, y, max_x, max_y){
  groups = unique(data[,group])
  edited_data = list()
  for(g in groups){
    data_subset = data %>% filter(data[,group] == g)
    data_subset$is.pareto <- pareto(data_subset, x, y)
    end_point = data_subset[nrow(data_subset),]
    end_point[,group] = g
    end_point[,x] = max_x
    end_point[,y] = 0
    end_point$is.pareto=TRUE
    edited_data[[g]] = bind_rows(data_subset, end_point)
  }
  return(bind_rows(edited_data))
}


rect_under_pareto <- function(data, group, factor, x, y, max_x, max_y){
  data_subset = data %>% filter(data[,group] == factor)
  data_subset$is.pareto <- pareto(data_subset, x, y)
  pareto_points = filter(data_subset, is.pareto)
  pareto_points = pareto_points[order(pareto_points[,x]),]
  start = c(0, max_y)
  area = 0
  
  old_p = start
  
  for(row in  1:nrow(pareto_points)){
    p = c(pareto_points[row, x], pareto_points[row, y])

    dx = (p-old_p)[[1]]
    trap_a = (dx * old_p[[2]])
    area = area + trap_a
    old_p = p
    
  }
  
  p = c(max_x, 0)
  dx = (p-old_p)[[1]]
  trap_a = (dx * old_p[[2]])
  area = area + trap_a
  return(area)
}

trap_under_pareto <- function(data, group, factor, x, y, max_x, max_y, min_x, min_y){
  data_subset = data %>% filter(data[,group] == factor)
  data_subset$is.pareto <- pareto(data_subset, x, y)
  pareto_points = filter(data_subset, is.pareto)
  pareto_points = pareto_points[order(pareto_points[,x]),]
  start = c(pareto_points[1, x], pareto_points[1, y])
  area = 0
  
  old_p = start
  
  for(row in  2:nrow(pareto_points)){
    p = c(pareto_points[row, x], pareto_points[row, y])
    if(p[[2]] < min_y){
      p[[2]] = min_y
    }
    dx = (p-old_p)[[1]]
    trap_a = (dx * (p+old_p)[[2]])/2
    area = area + trap_a
    old_p = p
    
  }
  
  p = c(max_x, old_p[[2]])
  dx = (p-old_p)[[1]]
  trap_a = (dx * (p+old_p)[[2]])/2
  area = area + trap_a
  return(area)
}


language_folders = list.files(directory, full.names=TRUE)
languages = list.files(directory, full.names=FALSE)

lstm_data = read_loss_files(directory, "lstm") %>% mutate(fsa.group = substr(lang,1,1)) %>% mutate(lang.type = substr(lang,2,2)) %>% mutate(kld = test.loss - best)
lstm_data = lstm_data %>% group_by(lam, num, lang) %>% slice(n())
lstm_groups = lstm_data %>% group_by(fsa.group) %>% group_split()

for(lstm_group in lstm_groups){
  lstm_group = as.data.frame(lstm_group)
  languages = unique(lstm_group["lang.type"])
  max_x = max(lstm_group[,"l0"])
  max_y = max(lstm_group[,"kld"])
  min_x = min(lstm_group[,"l0"])
  min_y = 0
  sl_area <- trap_under_pareto(lstm_group, "lang.type", "1", "l0", "kld", max_x, max_y, min_x, min_y)
  sp_area <- trap_under_pareto(lstm_group, "lang.type", "2", "l0", "kld", max_x, max_y, min_x, min_y)
  reg_area <- trap_under_pareto(lstm_group, "lang.type", "3", "l0", "kld", max_x, max_y, min_x, min_y) 
  sp_sl_diff = sp_area - sl_area
  reg_sl_diff = reg_area - sl_area
  reg_sp_diff = reg_area - sp_area
  
  print(sp_sl_diff)
  print(reg_sl_diff)
  print(reg_sp_diff)
  
  sp_sl_diff_bigger <- rep(NA, 1000)
  sp_sl_df = lstm_group %>% filter(lang.type == "1" | lang.type == "2")
  for (i in 1:1000){
    sample = sp_sl_df
    sampled_lang = sample(sp_sl_df$lang.type)
    sample$lang.type <- sampled_lang
    sample_sl_area <- trap_under_pareto(sample, "lang.type", "1", "l0", "kld", max_x, max_y, min_x, min_y)
    
    sample_sp_area <- trap_under_pareto(sample, "lang.type", "2", "l0", "kld", max_x, max_y, min_x, min_y)
    
    sp_sl_diff_bigger[i] = abs(sample_sp_area-sample_sl_area) > abs(sp_sl_diff)
    
  }
  print(sum(sp_sl_diff_bigger,na.rm=TRUE) / 1000)
  
  reg_sl_diff_bigger <- rep(NA, 1000)
  reg_sl_df = lstm_group %>% filter(lang.type == "1" | lang.type == "3")
  for (i in 1:1000){
    sample = reg_sl_df
    sampled_lang = sample(reg_sl_df$lang.type)
    sample$lang.type <- sampled_lang
    sample_sl_area <- trap_under_pareto(sample, "lang.type", "1", "l0", "kld", max_x, max_y, min_x, min_y)
    
    sample_reg_area <- trap_under_pareto(sample, "lang.type", "3", "l0", "kld", max_x, max_y, min_x, min_y)
    
    reg_sl_diff_bigger[i] = abs(sample_reg_area-sample_sl_area) > abs(reg_sl_diff)
    
  }
  print(sum(reg_sl_diff_bigger,na.rm=TRUE) / 1000)
  
  reg_sp_diff_bigger <- rep(NA, 1000)
  reg_sp_df = lstm_group %>% filter(lang.type == "2" | lang.type == "3")
  for (i in 1:1000){
    sample = reg_sp_df
    sampled_lang = sample(reg_sp_df$lang.type)
    sample$lang.type <- sampled_lang
    sample_reg_area <- trap_under_pareto(sample, "lang.type", "3", "l0", "kld", max_x, max_y, min_x, min_y)
    
    sample_sp_area <- trap_under_pareto(sample, "lang.type", "2", "l0", "kld", max_x, max_y, min_x, min_y)
    
    reg_sp_diff_bigger[i] = abs(sample_reg_area-sample_sp_area) > abs(reg_sp_diff)
    
  }
  print(sum(reg_sp_diff_bigger,na.rm=TRUE) / 1000)
}



for(lstm_group in lstm_groups){
  lstm_group = as.data.frame(lstm_group)
  max_x = max(lstm_group[,"l0"])
  min_x = min(lstm_group[,"l0"])
  plot_lstm = add_pareto(lstm_group, "lang", "l0", "kld", max_x, max_y) %>% mutate(kld = ifelse(kld < 0, 0, kld))
  
  plot = ggplot(lstm_group, aes(x=l0, y=kld, color=lang, group=lang)) +geom_point() + facet_wrap(~lang.type, labeller = as_labeller(c("1"="Strictly Local", "2" = "Strictly Piecewise", "3" = "Strictly Regular")))+
    geom_ribbon(data = plot_lstm %>% filter(is.pareto) %>% arrange(lang, l0), aes(fill=lang, ymax = kld, ymin=0), alpha=.25)+
    coord_cartesian(ylim=c(0,.5), xlim=c(min_x,75)) + theme_bw() + scale_color_manual(labels=c("Strictly Local", "Strictly Piecewise", "Strictly Regular"), values=c("#E69F00", "#56B4E9","#D55E00")) + scale_fill_manual(labels=c("Strictly Local", "Strictly Piecewise", "Strictly Regular"), values=c("#E69F00", "#56B4E9","#D55E00"))+
    xlab("Number of parameters (expected value)")+ylab("Kullback-Liebler Divergence") +labs(color="Language", fill="Language") + theme(text = element_text(size=17))
  print(plot)
}

plot_data = list()
for(lstm_group in lstm_groups){
  lstm_group = as.data.frame(lstm_group)
  max_x = max(lstm_group[,"l0"])
  min_x = min(lstm_group[,"l0"])
  plot_lstm = add_pareto(lstm_group, "lang", "l0", "kld", max_x, max_y) %>% mutate(kld = ifelse(kld < 0, 0, kld))
  plot_data = c(plot_data, list(plot_lstm))
}
plot_lstm = do.call(rbind, plot_data)

plot = ggplot(lstm_data, aes(x=l0, y=kld, color=lang.type, group=lang)) +geom_point() + facet_grid(fsa.group~lang.type, labeller = as_labeller(c("1"="Strictly Local", "2" = "Strictly Piecewise", "3" = "Strictly Regular", 'g'="G","a"="A","b"="B")))+
  geom_ribbon(data = plot_lstm %>% filter(is.pareto) %>% arrange(lang, l0), aes(fill=lang.type, ymax = kld, ymin=0), alpha=.25)+
  coord_cartesian(ylim=c(0,.5), xlim=c(min_x,75)) + theme_bw() + scale_color_manual(labels=c("Strictly Local", "Strictly Piecewise", "Strictly Regular"), values=c("#E69F00", "#56B4E9","#D55E00")) + scale_fill_manual(labels=c("Strictly Local", "Strictly Piecewise", "Strictly Regular"), values=c("#E69F00", "#56B4E9","#D55E00"))+
  xlab("Number of parameters (expected value)")+ylab("Kullback-Liebler Divergence") +labs(color="Language", fill="Language") + theme(text = element_text(size=17),legend.position="none")
print(plot)

eps_data = lstm_data %>% filter(abs(kld)<10e-3)%>%mutate(fsa.group=as.factor(fsa.group))%>%mutate(lang.type=as.factor(lang.type))
result = aov(data=eps_data,l0~lang.type*fsa.group)
posthoc <- TukeyHSD(result)
print(posthoc)


ggplot(eps_data, aes(y=l0, x=lang.type, color=lang.type))+facet_wrap(~fsa.group, labeller = as_labeller(c("a"="A", "b" = "B", "g" = "G")))+geom_boxplot(lwd=1.2)+scale_x_discrete(labels = c("SL", "SP", "Reg"))+
  xlab("Regular Language Type") + ylab("Number of parameters (expected value)")+theme_bw()+theme(legend.position = "none", text = element_text(size=17))+scale_color_manual(values=c("#E69F00", "#56B4E9","#D55E00"))

plot = ggplot(lstm_data, aes(x=l0, y=kld, color=lang.type, group=lang)) +geom_point(color="gray") + facet_grid(fsa.group~lang.type, labeller = as_labeller(c("1"="Strictly Local", "2" = "Strictly Piecewise", "3" = "Strictly Regular", 'g'="G","a"="A","b"="B")))+
  geom_ribbon(data = plot_lstm %>% filter(is.pareto) %>% arrange(lang, l0), aes(fill=lang.type, ymax = kld, ymin=0), alpha=.25)+geom_point(data=lstm_data %>% filter(abs(kld)<10e-3))+
  coord_cartesian(ylim=c(0,.5), xlim=c(min_x,75)) + theme_bw() + scale_color_manual(labels=c("Strictly Local", "Strictly Piecewise", "Strictly Regular"), values=c("#E69F00", "#56B4E9","#D55E00")) + scale_fill_manual(labels=c("Strictly Local", "Strictly Piecewise", "Strictly Regular"), values=c("#E69F00", "#56B4E9","#D55E00"))+
  xlab("Number of parameters (expected value)")+ylab("Kullback-Liebler Divergence") +labs(color="Language", fill="Language") + theme(text = element_text(size=17),legend.position="none")
print(plot)


lstm_data = lstm_data %>% group_by(lam, num, lang) %>% slice(n())
