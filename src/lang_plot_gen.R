rm(list = ls())
library(tidyverse)


fl_lstm_data = read.csv("F:\\PycharmProjects\\complex_hierarchy\\data\\Output-2023-05-19\\fl_small_lstm_loss.csv") %>% mutate(lang = "fl")
sh_lstm_data = read.csv("F:\\PycharmProjects\\complex_hierarchy\\data\\Output-2023-05-19\\sh_small_lstm_loss.csv") %>% mutate(lang = "sh")

lstm_data = rbind(fl_lstm_data, sh_lstm_data)


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
    start_point = data_subset[1,]
    end_point = data_subset[nrow(data_subset),]
    start_point[,group] = g
    start_point[,x] = -10
    start_point[,y] = max_y
    start_point$is.pareto = TRUE
    end_point[,group] = g
    end_point[,x] = max_x
    end_point[,y] = 0
    end_point$is.pareto=TRUE
    edited_data[[g]] = bind_rows(start_point, data_subset, end_point)
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
  start = c(min_x, max_y)
  area = 0
  
  old_p = start
  
  for(row in  1:nrow(pareto_points)){
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




# permutation test

max_y = max(lstm_data[,"loss"])
max_x = max(lstm_data[,"l0"])
min_x = min(lstm_data[,"l0"])
min_y = 0

sh_area <- trap_under_pareto(lstm_data, "lang", "sh", "l0", "loss", max_x, max_y, min_x, min_y)

fl_area <- trap_under_pareto(lstm_data, "lang", "fl", "l0", "loss", max_x, max_y, min_x, min_y)

area_diff <- fl_area-sh_area

area_diff_bigger <- rep(NA, 1000)

for (i in 1:1000){
  sample_harmony = lstm_data
  sampled_lang = sample(lstm_data$lang)
  sample_harmony$lang <- sampled_lang
  sh_area <- trap_under_pareto(sample_harmony, "lang", "sh", "l0", "loss", max_x, max_y, min_x, min_y)
  
  fl_area <- trap_under_pareto(sample_harmony, "lang", "fl", "l0", "loss", max_x, max_y, min_x, min_y)
  
  area_diff_bigger[i] = fl_area-sh_area > area_diff
  if(i %% 100 == 0){print(sum(area_diff_bigger,na.rm=TRUE) / i)}
  
}

print(sum(area_diff_bigger,na.rm=TRUE) / 1000)

plot_lstm = add_pareto(lstm_data, "lang", "l0", "loss", max_x, max_y) %>% mutate(loss = ifelse(loss < 0, 0, loss))

ggplot(plot_lstm, aes(x=l0, y=loss)) + geom_point() + facet_wrap(~lang)

ggplot(lstm_data, aes(x=l0, y=loss, color=lang, group=lang)) +geom_point() + facet_wrap(~lang)+
  geom_rect(data = plot_lstm %>% filter(is.pareto) %>% arrange(lang, l0), aes(fill=lang, ymax = loss, ymin=0, xmin = l0, xmax = lead(l0)), alpha=.25, color=NA)+
  geom_step(data=plot_lstm%>% filter(is.pareto)) + coord_cartesian(ylim=c(0,.7), xlim=c(0,150)) + theme_bw()


ggplot(lstm_data, aes(x=l0, y=loss, color=lang, group=lang)) +geom_point() + facet_wrap(~lang)+
  geom_ribbon(data = plot_lstm %>% filter(is.pareto) %>% arrange(lang, l0), aes(fill=lang, ymax = loss, ymin=0), alpha=.25)+
  coord_cartesian(ylim=c(0,max_y), xlim=c(min_x,75)) + theme_bw()
