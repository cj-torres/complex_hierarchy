rm(list = ls())
library(tidyverse)
broom::tidy(ad_aov)
library(broom)

frontiers1 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-06\\dyck1_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="dyck")
frontiers2 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-06\\abn_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="abn")
frontiers3 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-06\\anbn_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="anbn")
frontiers4 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-06\\a2nb2m_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="a2nb2m")
frontiers5 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-08\\dyck1_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="dyck")
frontiers6 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-08\\abn_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="abn")
frontiers7 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-08\\anbn_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="anbn")
frontiers8 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-08\\a2nb2m_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="a2nb2m")

frontierssh1 = read.csv("F:\\PycharmProjects\\complex_hierarchy\\Output-2022-12-02\\sh_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="sh")
frontierssh2 = read.csv("F:\\PycharmProjects\\complex_hierarchy\\Output-2023-01-11\\sh_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="sh")

frontiersfl1 = read.csv("F:\\PycharmProjects\\complex_hierarchy\\Output-2022-12-02\\fl_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="fl")
frontiersfl2 = read.csv("F:\\PycharmProjects\\complex_hierarchy\\Output-2023-01-11\\fl_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="fl")


#frontiers = rbind(frontiers1, frontiers2)
#frontiers = rbind(frontiers, frontiers3)
frontiers = rbind(frontiers1, frontiers2, frontiers3, frontiers4, frontiers5, frontiers6, frontiers7, frontiers8)%>% mutate(is_regular = ((lang == "a2nb2m")|(lang=="abn")))

frontiersfl = rbind(frontiersfl1, frontiersfl2)
frontierssh = rbind(frontierssh1, frontierssh2)

frontiers$model_num <- as.factor(frontiers$model_num)
frontiers$lambda <- as.factor(frontiers$lambda)
frontiers_filtered = frontiers %>% filter(((epoch>1000)|((lang=="abn") & (epoch>500)))&(accuracy>.995))%>% mutate(log_loss = log(loss)) %>% filter(is.finite(log_loss)) %>% mutate(lang = as.factor(lang))
frontiers_hulls = frontiers_filtered%>%select(lang,l0,loss)%>% group_by(lang) %>% slice(chull(l0,loss))
plot = ggplot(frontiers_filtered, aes(x = l0, y = loss)) + geom_point() +xlab("Number of parameters")+ylab("Bernoulli Loss (log scale)") + theme_bw() + facet_wrap(~lang)
plot + scale_y_continuous(breaks = c(10e-2, 10e-4, 10e-6, 10e-8), limits = c(10e-9,10e-1), trans="log2")

analysis = manova(cbind(l0, loss) ~ lang, data=frontiers_filtered)
summary(analysis)


frontiers_modified = frontiers_filtered %>% mutate(is_regular = ((lang == "a2nb2m")|(lang=="abn")))
frontiers_modified = frontiers_modified %>% mutate(lang = as.factor(lang)) %>% mutate(log_loss_inv = 1/log_loss)

harmony = rbind(frontierssh, frontiersfl) %>% mutate(lang = as.factor(lang)) %>% mutate(log_loss = log(loss)) %>% filter(is.finite(log_loss)) %>% filter(epoch>750)



pareto <- function(data, x, y){

  pareto.1 <- logical(length(data[,x]))
  x.sort <- sort(data[,x])
  y.sort <- data[,y][order(data[,x])]
  
  for(i in 1:length(x.sort)){
    pareto.1[i] <- all(y.sort[1:i] >= y.sort[i])
  }
  
  return(pareto.1[match(1:length(data[,x]), order(data[,x]))])
}


get_pareto <- function(data, group, x, y){
  new_data = c()
  for (factor in levels(data[,group])){
    data_subset = data %>% filter(data[,group] == factor)
    data_subset$is.pareto = pareto(data_subset, x, y)
    new_data = rbind(new_data, data_subset)
    
  }
  return(new_data)
}


linear_pareto_fit <- function(data, group, factor, x, y){
  data_subset = data %>% filter(data[,group] == factor)
  data_subset$is.pareto <- pareto(data_subset, x, y)
  #return(data_subset)
  model = lm(get(y) ~ get(x), data = data_subset %>% filter(is.pareto))
  return(model)
}



numerical_under_pareto <- function(data, group, factor, x, y){
  data_subset = data %>% filter(data[,group] == factor)
  data_subset$is.pareto <- pareto(data_subset, x, y)
  pareto_points = filter(data_subset, is.pareto)
  pareto_points = pareto_points[order(pareto_points[,x]),]
  start = c(0, max(data_subset[,y]))
  area = 0
  
  old_p = start
  
  for(row in  1:nrow(pareto_points)){
    p = c(pareto_points[row, x], pareto_points[row, y])
    dx = (p-old_p)[[1]]
    trap_a = (dx * (p+old_p)[[2]])/2
    area = area + trap_a
    old_p = p
    
  }
  
  p = c(max(data_subset[,x]), old_p[[2]])
  dx = (p-old_p)[[1]]
  trap_a = (dx * (p+old_p)[[2]])/2
  area = area + trap_a
  return(area)
}

# pareto permutation test

sh_area <- numerical_under_pareto(harmony, "lang", "sh", "l0", "loss")

fl_area <- numerical_under_pareto(harmony, "lang", "fl", "l0", "loss")

area_diff <- abs(sh_area-fl_area)

area_diff_bigger <- rep(NA, 1000)

for (i in 1:1000){
  sample_harmony = harmony
  sampled_lang = sample(harmony$lang)
  sample_harmony$lang <- sampled_lang
  sh_area <- numerical_under_pareto(sample_harmony, "lang", "sh", "l0", "loss")
  
  fl_area <- numerical_under_pareto(sample_harmony, "lang", "fl", "l0", "loss")
  
  is_bigger = abs(sh_area-fl_area) > area_diff
  
  area_diff_bigger[i] = abs(sh_area-fl_area) > area_diff
  if(i %% 100 == 0){print(sum(area_diff_bigger,na.rm=TRUE) / i)}
  
}

print(sum(area_diff_bigger,na.rm=TRUE) / 1000)

plot = ggplot(harmony %>% filter(accuracy==1), aes(x = l0, y = loss)) + geom_point(aes(color = epoch)) +xlab("Number of parameters")+ylab("Bernoulli Loss (log scale)") + theme_classic() + facet_wrap(~lang)
plot + scale_y_continuous(breaks = c(10e-2, 10e-4, 10e-6, 10e-8), limits = c(10e-9,10e-1), trans="log2")

reg_area <- numerical_under_pareto(frontiers_filtered, "is_regular", TRUE, "l0", "loss")

nonreg_area <- numerical_under_pareto(frontiers_filtered, "is_regular", FALSE, "l0", "loss")

area_diff <- abs(reg_area-nonreg_area)
area_diff_bigger <- rep(NA, 1000)

for (i in 1:1000){
  sample_frontiers = frontiers_filtered
  sampled_lang = sample(frontiers_filtered$is_regular)
  sample_frontiers$is_regular <- sampled_lang
  
  reg_area <- numerical_under_pareto(sample_frontiers, "is_regular", TRUE, "l0", "loss")
  nonreg_area <- numerical_under_pareto(sample_frontiers, "is_regular", FALSE, "l0", "loss")
  
  is_bigger = abs(reg_area-nonreg_area) > area_diff
  
  area_diff_bigger[i] = abs(reg_area-nonreg_area) > area_diff
  if(i %% 100 == 0){print(sum(area_diff_bigger,na.rm=TRUE) / i)}
  
}



#

abn_area = numerical_under_pareto(frontiers_filtered, "lang", "abn", "l0", "loss")
a2nb2m_area = numerical_under_pareto(frontiers_filtered, "lang", "a2nb2m", "l0", "loss")
dyck_area = numerical_under_pareto(frontiers_filtered, "lang", "dyck", "l0", "loss")
anbn_area = numerical_under_pareto(frontiers_filtered, "lang", "anbn", "l0", "loss")

area_diff_bigger = matrix(data=NA,nrow=1000,ncol=6)

areas = c(abn_area, a2nb2m_area, dyck_area, anbn_area)

area_diffs = apply(combn(areas,2), 2, function(x) abs(x[[1]]-x[[2]]))

for (i in 1:1000){
  sample_frontiers = frontiers_filtered
  sampled_lang = sample(frontiers_filtered$lang)
  sample_frontiers$lang <- sampled_lang
  
  abn_area = numerical_under_pareto(frontiers_filtered, "lang", "abn", "l0", "loss")
  a2nb2m_area = numerical_under_pareto(frontiers_filtered, "lang", "a2nb2m", "l0", "loss")
  dyck_area = numerical_under_pareto(frontiers_filtered, "lang", "dyck", "l0", "loss")
  anbn_area = numerical_under_pareto(frontiers_filtered, "lang", "anbn", "l0", "loss")
  
  areas = c(abn_area, a2nb2m_area, dyck_area, anbn_area)
  
  sampled_diffs = apply(combn(areas,2), 2, function(x) abs(x[[1]]-x[[2]]))
  
  area_diff_bigger[i,] = sampled_diffs > area_diffs
  if(i %% 100 == 0){print(colSums(area_diff_bigger,na.rm=TRUE,dims=1) / i)}
  
}

front_copy <- new_frontiers %>% filter(is.pareto) %>% filter(loss>0) %>% select(l0, loss, lang)
for (factor in levels(front_copy$lang)){
  front_copy = rbind(front_copy, data.frame(loss=c(.1,0), l0=c(0,100), lang=c(factor,factor)))
  
}

