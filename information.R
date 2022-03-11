library(tidyverse)

get_kld_growth = function(weights, accuracy_table, weight_interval_sequence, accuracy_sequence){
  accuracy_groupings = accuracy_table %>%mutate(acc_group=cut(accuracy,accuracy_sequence,ordered_result=TRUE))
  dataframe = merge(weights, accuracy_groupings, by = 'row.names')%>% gather("parameter","weight",weight_names)
  percentages_df = dataframe %>% mutate(interval = cut(weight,weight_interval_sequence,ordered_result=TRUE)) %>% drop_na(interval, acc_group)%>% select(interval, acc_group) %>% group_by(acc_group, interval) %>% summarize(n=n()) %>% group_by(acc_group) %>% mutate(percentage = n/sum(n))%>%ungroup()
  entropy_df = percentages_df %>% mutate(log_p = log(percentage)) %>% mutate(point_entropy = -percentage*log_p)
  cross_entropy_df = entropy_df %>% group_by(interval)%>%mutate(point_crosswise = -percentage*lag(log_p,order_by=acc_group))%>%ungroup()%>%replace(is.na(.),0)
  
  kld_df= cross_entropy_df %>% group_by(acc_group) %>% summarize(kld = sum(point_crosswise-point_entropy))
  
  return(kld_df)
}


get_kld_growth_w_ci = function(weights, accuracy_table, weight_interval_sequence, accuracy_sequence, batches){
  
  accuracy_groupings = accuracy_table %>%mutate(acc_group=cut(accuracy,accuracy_sequence,ordered_result=TRUE))
  dataframe = merge(weights, accuracy_groupings, by = 'row.names')%>% gather("parameter","weight",weight_names)%>% drop_na(acc_group)
  
  groups = group_split(dataframe, acc_group)
  
  kld_df = data.frame(matrix(ncol = 2, nrow = 0))
  colnames(kld_df) = c('acc_group', 'kld') 
  for (i in seq_along(groups[1:length(groups)-1])){
    for (j in 1:batches){
      pres_acc_group = groups[[i+1]][1,]$acc_group
      n_i = nrow(groups[[i+1]])
      n_prev = nrow(groups[[i]])
      present_group = groups[[i+1]][sample(n_i,size=n_i,replace=TRUE),]
      prev_group = groups[[i]] #[sample(n_prev,size=n_prev,replace=TRUE),]
      sampled = rbind(present_group, prev_group)
      percentages_df = sampled %>% mutate(interval = cut(weight,weight_interval_sequence,ordered_result=TRUE)) %>% drop_na(interval, acc_group)%>% select(interval, acc_group) %>% group_by(acc_group, interval) %>% summarize(n=n()) %>% group_by(acc_group) %>% mutate(percentage = n/sum(n))%>%ungroup()
      entropy_df = percentages_df %>% mutate(log_p = log(percentage)) %>% mutate(point_entropy = -percentage*log_p)
      cross_entropy_df = entropy_df %>% group_by(interval)%>%mutate(point_crosswise = -percentage*lag(log_p,order_by=acc_group))%>%ungroup()%>%replace(is.na(.),0)
      kld = cross_entropy_df %>% filter(acc_group == pres_acc_group) %>% group_by(acc_group) %>% summarize(kld = sum(point_crosswise-point_entropy))
      kld_df = rbind(kld_df, kld)
    }
    print(paste(pres_acc_group, " finished"))
  }
  
  
  return(kld_df)
}

anbn_weights = read.csv("C:\\Users\\Charlie\\PycharmProjects\\complex_hierarchy\\anbn_model_c_lstm_seq1_rand.csv", header=FALSE)
dyck_weights = read.csv("C:\\Users\\Charlie\\PycharmProjects\\complex_hierarchy\\dyck1_model_c_lstm_seq1_rand.csv", header=FALSE)
anbn_acc = read.csv("C:\\Users\\Charlie\\PycharmProjects\\complex_hierarchy\\anbn_model_c_lstm_loss_seq1_rand.csv", header=TRUE)
dyck_acc = read.csv("C:\\Users\\Charlie\\PycharmProjects\\complex_hierarchy\\dyck1_model_c_lstm_loss_seq1_rand.csv", header=TRUE)
abn_weights = read.csv("C:\\Users\\Charlie\\PycharmProjects\\complex_hierarchy\\abn_model_c_lstm_seq1_rand.csv", header=FALSE)
anbm_weights = read.csv("C:\\Users\\Charlie\\PycharmProjects\\complex_hierarchy\\anbm_model_c_lstm_seq1_rand.csv", header=FALSE)
abn_acc = read.csv("C:\\Users\\Charlie\\PycharmProjects\\complex_hierarchy\\abn_model_c_lstm_loss_seq1_rand.csv", header=TRUE)
anbm_acc = read.csv("C:\\Users\\Charlie\\PycharmProjects\\complex_hierarchy\\anbm_model_c_lstm_loss_seq1_rand.csv", header=TRUE)
#anbm_weights = read.csv("C:\\Users\\Charlie\\PycharmProjects\\complex_hierarchy\\anbm_model_r.csv", header=FALSE)
#abn_weights = read.csv("C:\\Users\\Charlie\\PycharmProjects\\complex_hierarchy\\abn_model_r.csv", header=FALSE)
weight_names = colnames(anbn_weights)
#anbn_acc = anbn_acc %>%mutate(acc_group=cut(accuracy,seq(.5,1,.05),ordered_result=TRUE))
#dyck_acc = dyck_acc %>%mutate(acc_group=cut(accuracy,seq(.5,1,.05),ordered_result=TRUE))

dyck_kld = get_kld_growth(dyck_weights, dyck_acc, seq(-2,2,.02), seq(.5,1,.025)) %>% mutate(language = "dyck") %>% filter(acc_group != "(0.5,0.525]")
anbn_kld = get_kld_growth(anbn_weights, anbn_acc, seq(-2,2,.02), seq(.5,1,.025)) %>% mutate(language = "anbn") %>% filter(acc_group != "(0.5,0.525]")
abn_kld = get_kld_growth(abn_weights, abn_acc, seq(-2,2,.02), seq(.5,1,.025)) %>% mutate(language = "abn") %>% filter(acc_group != "(0.5,0.525]")
anbm_kld = get_kld_growth(anbm_weights, anbm_acc, seq(-2,2,.02), seq(.5,1,.025)) %>% mutate(language = "anbm") %>% filter(acc_group != "(0.5,0.525]")

klds = rbind(dyck_kld,anbn_kld,abn_kld,anbm_kld)

plot = ggplot(klds, aes(x=acc_group,y=kld, group=language)) + geom_line(aes(color=language))
plot

#anbn = merge(anbn_weights, anbn_acc, by = 'row.names')%>% gather("parameter","weight",weight_names)#%>% filter(acc_group!="(0.975,1.03]")
#dyck = merge(dyck_weights, dyck_acc, by = 'row.names')%>% gather("parameter","weight",weight_names)#%>% filter(acc_group!="(0.975,1.03]")

#percentages_dyck = dyck %>% mutate(interval = cut(weight,seq(-2,2,.02),ordered_result=TRUE)) %>% drop_na(interval, acc_group)%>% select(interval, acc_group) %>% group_by(acc_group, interval) %>% summarize(n=n()) %>% group_by(acc_group) %>% mutate(percentage = n/sum(n))%>%ungroup()
#percentages_anbn = anbn %>% mutate(interval = cut(weight,seq(-2,2,.02),ordered_result=TRUE)) %>% drop_na(interval, acc_group)%>% select(interval, acc_group) %>% group_by(acc_group, interval) %>% summarize(n=n()) %>% group_by(acc_group) %>% mutate(percentage = n/sum(n))%>%ungroup()

#entropy_dyck = percentages_dyck %>% mutate(log_p = log(percentage)) %>% mutate(point_entropy = -percentage*log_p)
#cross_entropy_dyck = entropy_dyck %>% group_by(interval)%>%mutate(point_crosswise = -percentage*lag(log_p,order_by=acc_group))%>%ungroup()%>%replace(is.na(.),0)

#kld_dyck= cross_entropy_dyck %>% group_by(acc_group) %>% summarize(kld = sum(point_crosswise-point_entropy))




get_kld_growth_w_ci(dyck_weights, dyck_acc, seq(-2,2,.02), seq(.5,1,.025), 1000)

dyck_all_weights <- data.frame(unlist(dyck_weights))
anbn_all_weights <- data.frame(unlist(anbn_weights))
#anbm_all_weights <- data.frame(unlist(anbm_weights))
#abn_all_weights <- data.frame(unlist(abn_weights))

plot = ggplot(dyck, aes(x = weight)) + geom_histogram(binwidth = .02) + xlim(-2.5, 2.5)+ ylim(0,50000)+ theme_classic() + xlab("") + ylab("") + facet_wrap(~acc_group)
plot
plot = ggplot(anbn, aes(x = weight)) + geom_histogram(binwidth = .02) + xlim(-2.5, 2.5)+ ylim(0,50000)+ theme_classic() + xlab("") + ylab("") + facet_wrap(~acc_group)
plot
#plot = ggplot(anbm_all_weights, aes(x = unlist.anbm_weights.)) + geom_histogram(binwidth = .02) + xlim(-2.5, 2.5)+ ylim(0,1100)+ theme_classic() + xlab("") + ylab("")
#plot
#plot = ggplot(abn_all_weights, aes(x = unlist.abn_weights.)) + geom_histogram(binwidth = .02) + xlim(-2.5, 2.5)+ ylim(0,1100)+ theme_classic() + xlab("") + ylab("")
#plot

best_trained = dyck %>% filter(acc_group=="(0.975,1.03]")

plot = ggplot(best_trained, aes(x=weight))  + geom_histogram(binwidth = .02)
plot

best_trained = anbn %>% filter(acc_group=="(0.975,1.03]")

plot = ggplot(best_trained, aes(x=weight))  + geom_histogram(binwidth = .02)
plot




dyck_all_weights <- data.frame(unlist(dyck_weights))
anbn_all_weights <- data.frame(unlist(anbn_weights))
anbm_all_weights <- data.frame(unlist(anbm_weights[7:31]))
abn_all_weights <- data.frame(unlist(abn_weights[7:31]))

plot = ggplot(dyck_all_weights, aes(x = unlist.dyck_weights.7.31..)) + geom_histogram(binwidth = .02) + theme_classic()
  theme(plot.title = element_text(hjust = 0.5))
plot
plot = ggplot(anbn_all_weights, aes(x = unlist.anbn_weights.27.31..)) + geom_histogram(binwidth = .02) + theme_classic() + ggtitle("anbn RNN Weight Distribution") + xlab("weight values") +
  theme(plot.title = element_text(hjust = 0.5))
plot
plot = ggplot(anbm_all_weights, aes(x = unlist.anbm_weights.7.38..)) + geom_histogram(binwidth = .02) + theme_classic() + ggtitle("anbm RNN Weight Distribution") + xlab("weight values") +
  theme(plot.title = element_text(hjust = 0.5))
plot
plot = ggplot(abn_all_weights, aes(x = unlist.abn_weights.7.38..)) + geom_histogram(binwidth = .02) + theme_classic() + ggtitle("abn RNN Weight Distribution") + xlab("weight values") +
  theme(plot.title = element_text(hjust = 0.5))
plot

plot= ggplot(dyck_weights, aes(x = V63)) + geom_histogram(binwidth = .02) + theme_classic() + theme_classic() + xlab("") + ylab("")
  theme(plot.title = element_text(hjust = 0.5))
plot


dyck_all_weights <- data.frame(unlist(dyck_weights[39:63]))
anbn_all_weights <- data.frame(unlist(anbn_weights[39:63]))
anbm_all_weights <- data.frame(unlist(anbm_weights[39:63]))
abn_all_weights <- data.frame(unlist(abn_weights[39:63]))

plot = ggplot(dyck_all_weights, aes(x = unlist.dyck_weights.39.63..)) + geom_histogram(binwidth = .02) + theme_classic() + ggtitle("Dyck-1 Classifier Head Weight Distribution") + xlab("weight values") +
  theme(plot.title = element_text(hjust = 0.5))
plot
plot = ggplot(anbn_all_weights, aes(x = unlist.anbn_weights.39.63..)) + geom_histogram(binwidth = .02) + theme_classic() + ggtitle("anbn Classifier Head Weight Distribution") + xlab("weight values") +
  theme(plot.title = element_text(hjust = 0.5))
plot
plot = ggplot(anbm_all_weights, aes(x = unlist.anbm_weights.39.63..)) + geom_histogram(binwidth = .02) + theme_classic() + ggtitle("anbm Classifier Head Weight Distribution") + xlab("weight values") +
  theme(plot.title = element_text(hjust = 0.5))
plot
plot = ggplot(abn_all_weights, aes(x = unlist.abn_weights.39.63..)) + geom_histogram(binwidth = .02) + theme_classic() + ggtitle("abn Classifier Head Weight Distribution") + xlab("weight values") +
  theme(plot.title = element_text(hjust = 0.5))
plot

for (data in c(dyck_weights, anbn_weights, anbm_weights, abn_weights)){
  all_weights <- data.frame(unlist(data))
  plot = ggplot(all_weights, aes(x = unlist.data.)) + geom_histogram(binwidth = .02)
  print(plot)
  Sys.sleep(2)
}

all_weights <- data.frame(unlist(anbn_weights))

plot = ggplot(all_weights, aes(x = unlist.anbn_weights.)) + geom_histogram(binwidth = .02)

plot


all_weights$unlist.weights.

plot = ggplot(dyck_weights, aes(x = V40)) + geom_histogram(binwidth = .1)

plot

madm = function(x){
  mu = median(x)
  return(mean(abs(anbm_weights$V1 - mu)))
}

laplace_fit = function(x){
  b = madm(x)
  m = median(x)
  return(c(m,b))
}

gauss_fit = function(x){
  mu = mean(x)
  sigma2 = sd(x)^2
  return(mu, sigma2)
}

std_laplace = c(0,1)
std_gauss = c(0,1)

install.packages("VGAM")
library(VGAM)
x <- seq(-4, 4, length=100)
y <- dnorm(x)
z <- dlaplace(x)
data = data.frame(x,y,z)

plot = ggplot(data = data, aes(x=x)) + geom_line(aes(y=y))+ geom_line(aes(y=z))
plot
