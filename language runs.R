#rm(list = ls())
library(tidyverse)
broom::tidy(ad_aov)
#library(broom)

frontiers1 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-06\\dyck1_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="dyck")
frontiers2 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-06\\abn_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="abn")
frontiers3 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-06\\anbn_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="anbn")
frontiers4 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-06\\a2nb2m_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="a2nb2m")
frontiers5 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-08\\dyck1_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="dyck")
frontiers6 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-08\\abn_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="abn")
frontiers7 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-08\\anbn_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="anbn")
frontiers8 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-08\\a2nb2m_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="a2nb2m")




#frontiers = rbind(frontiers1, frontiers2)
#frontiers = rbind(frontiers, frontiers3)
frontiers = rbind(frontiers1, frontiers2, frontiers3, frontiers4, frontiers5, frontiers6, frontiers7, frontiers8)%>% mutate(is_regular = ((lang == "a2nb2m")|(lang=="abn")))

frontiers$model_num <- as.factor(frontiers$model_num)
frontiers$lambda <- as.factor(frontiers$lambda)
frontiers_filtered = frontiers %>% filter(((epoch>1000)|((lang=="abn") & (epoch>500)))&(accuracy>.995))%>% mutate(log_loss = log(loss)) %>% filter(is.finite(log_loss))
frontiers_hulls = frontiers_filtered%>%select(lang,l0,loss)%>% group_by(lang) %>% slice(chull(l0,loss))
plot = ggplot(frontiers_filtered, aes(x = l0, y = loss)) + geom_point(aes(color = "black")) +geom_point(data=frontiers_hulls, aes(color="red"))+xlab("Number of parameters")+ylab("Bernoulli Loss (log scale)") + theme_classic() + facet_wrap(~lang)
plot + scale_y_continuous(breaks = c(10e-2, 10e-4, 10e-6, 10e-8), limits = c(10e-9,10e-1), trans="log2")

analysis = manova(cbind(l0, loss) ~ lang, data=frontiers_filtered)
summary(analysis)


frontiers_modified = frontiers_filtered %>% mutate(is_regular = ((lang == "a2nb2m")|(lang=="abn")))
frontiers_modified = frontiers_modified %>% mutate(lang = as.factor(lang)) %>% mutate(log_loss_inv = 1/log_loss)
frontiers_modified_D = frontiers_modified[order(frontiers_modified$l0,frontiers_modified$loss,decreasing=FALSE),] 
front = D[which(!duplicated(cummin(frontiers_modified_D$loss))),]


pareto <- function(data, x, y){

  pareto <- logical(length(data[,x]))
  x.sort <- sort(data[,x])
  y.sort <- data[,y][order(data[,x])]
  
  for(i in 1:length(x.sort)){
    pareto[i] <- all(y.sort[1:i] >= y.sort[i])
  }
  
  return(re_ordered.pareto = pareto.2[match(1:length(data[,x]), order(data[,x]))])
}

frontiers_modified$is_pareto <- re_ordered.pareto

plot = ggplot(frontiers_modified, aes(x=l0, y=log_loss, color=is_pareto)) + geom_point() #+ facet_wrap(~lang)
plot

model = lm(log_loss_inv ~ l0+epoch+lang, data = frontiers_modified)
summary(model)



t.test(l0~is_regular, data=frontiers%>%filter(epoch>1500))







p_slope_diffs = permutation_test_slopes(frontiers_filtered, "l0", "log_loss", "lang" )