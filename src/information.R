rm(list=ls())
library(tidyverse)

bias_vectors = c("V105","V106","V107","V108","V109","V110","V111","V112","V113","V114","V115","V116","V117","V118","V119","V120","V121","V122","V123","V124","V125","V126","V127","V128","V129","V130","V131","V132","V133","V134","V135","V136","V153","V154","V155","V156","V169","V170","V171")

get_kld_growth = function(weights, accuracy_table, weight_interval_sequence, accuracy_sequence){
  accuracy_groupings = accuracy_table %>%mutate(acc_group=cut(accuracy,accuracy_sequence,ordered_result=TRUE))
  weight_names = colnames(weights)
  #matrices = weights %>% select(!bias_vectors)
  #biases = weights %>% select(bias_vectors)
  #matrices_labeled = matrices %>% mutate(ID = row_number())
  #biases_labeled = biases %>% mutate(ID = row_number())
  weights_labeled = weights %>% mutate(ID = row_number())

  dataframe = merge(weights_labeled, accuracy_groupings, by = 'row.names')%>% gather("parameter","weight",colnames(weights)) %>% mutate(interval = cut(weight,weight_interval_sequence,ordered_result=TRUE)) %>% drop_na(interval, acc_group)%>%
    separate(interval, c("lower.bound", "upper.bound"), sep=",",remove=FALSE) %>% mutate(lower.bound = as.numeric(substr(lower.bound, 2, nchar(lower.bound))))%>% mutate(upper.bound = as.numeric(substr(upper.bound, 1, nchar(upper.bound)-1)))%>% mutate(log_q = log(pnorm(upper.bound)-pnorm(lower.bound)))
  percentages_df_accuracy = dataframe  %>% group_by(acc_group, interval) %>% summarise(n=n()) %>% group_by(acc_group) %>% mutate(p_acc = n/sum(n))
  percentages_df_model = dataframe  %>% group_by(ID, interval) %>% summarise(n=n()) %>% group_by(ID) %>% mutate(p_model= n/sum(n))
  percentages_df = inner_join(dataframe,percentages_df_accuracy %>% select(!c(n)))
  percentages_df = inner_join(percentages_df,percentages_df_model %>% select(!c(n)))
  log_p_df = percentages_df %>% mutate(log_p = log(p_acc)) %>% mutate(log_p_model = log(p_model)) #%>% mutate(point_entropy = -percentage*log_p)
  model_wise_kld = log_p_df %>% group_by(ID, acc_group)  %>%summarize(model_kld=sum(p_model*(log_p_model-log_q)))
  accuracy_wise_kld = log_p_df %>% group_by(acc_group)  %>%summarize(model_cross_entropy=sum(p_acc*(log_p-log_q)))
  #cross_entropy_df = entropy_df %>% group_by(interval)%>%mutate(point_crosswise = -percentage*lag(log_p,order_by=acc_group))%>%ungroup()%>%replace(is.na(.),0)
  # select(interval, ID) %>% 
  #kld_df= cross_entropy_df %>% group_by(acc_group) %>% summarize(kld = sum(point_crosswise-point_entropy))
  
  return(list(dataframe, model_wise_kld, accuracy_wise_kld))
}


get_kld_growth_loss = function(weights, accuracy_table, weight_interval_sequence, accuracy_sequence){
  accuracy_groupings = accuracy_table %>%mutate(acc_group=cut(best_loss,accuracy_sequence,ordered_result=TRUE))
  weight_names = colnames(weights)
  matrices = weights %>% select(!bias_vectors)
  biases = weights %>% select(bias_vectors)
  matrices_labeled = matrices %>% mutate(ID = row_number())
  biases_labeled = biases %>% mutate(ID = row_number())
  weights_labeled = weights %>% mutate(ID = row_number())
  
  dataframe = merge(weights_labeled, accuracy_groupings, by = 'row.names')%>% gather("parameter","weight",colnames(weights)) %>% mutate(interval = cut(weight,weight_interval_sequence,ordered_result=TRUE)) %>% drop_na(interval, acc_group)%>%
    separate(interval, c("lower.bound", "upper.bound"), sep=",",remove=FALSE) %>% mutate(lower.bound = as.numeric(substr(lower.bound, 2, nchar(lower.bound))))%>% mutate(upper.bound = as.numeric(substr(upper.bound, 1, nchar(upper.bound)-1)))%>% mutate(log_q = log(pnorm(upper.bound)-pnorm(lower.bound)))
  percentages_df_accuracy = dataframe  %>% group_by(acc_group, interval) %>% summarise(n=n()) %>% group_by(acc_group) %>% mutate(p_acc = n/sum(n))
  percentages_df_model = dataframe  %>% group_by(ID, interval) %>% summarise(n=n()) %>% group_by(ID) %>% mutate(p_model= n/sum(n))
  percentages_df = inner_join(dataframe,percentages_df_accuracy %>% select(!c(n)))
  percentages_df = inner_join(percentages_df,percentages_df_model %>% select(!c(n)))
  log_p_df = percentages_df %>% mutate(log_p = log(p_acc)) %>% mutate(log_p_model = log(p_model)) #%>% mutate(point_entropy = -percentage*log_p)
  model_wise_kld = log_p_df %>% group_by(ID, interval)%>%summarize_all(mean)%>%group_by(ID)  %>%summarize(model_kld=sum(p_model*(log_p_model-log_q)))
  accuracy_wise_kld = log_p_df %>% group_by(acc_group)%>%summarize_all(mean)%>%group_by(acc_group)   %>%summarize(model_cross_entropy=sum(p_acc*(log_p-log_q)))
  #cross_entropy_df = entropy_df %>% group_by(interval)%>%mutate(point_crosswise = -percentage*lag(log_p,order_by=acc_group))%>%ungroup()%>%replace(is.na(.),0)
  # select(interval, ID) %>% 
  #kld_df= cross_entropy_df %>% group_by(acc_group) %>% summarize(kld = sum(point_crosswise-point_entropy))
  
  return(list(percentages_df, model_wise_kld, accuracy_wise_kld))
}



anbn_weights = read.csv("~/complex_hierarchy/anbn_model_c_lstm_seq1_rand2.csv", header=FALSE)
dyck_weights = read.csv("~/complex_hierarchy/dyck1_model_c_lstm_seq1_rand2.csv", header=FALSE)
anbn_acc = read.csv("~/complex_hierarchy/anbn_model_c_lstm_loss_seq1_rand2.csv", header=TRUE)%>% mutate(language = "anbn")
dyck_acc = read.csv("~/complex_hierarchy/dyck1_model_c_lstm_loss_seq1_rand2.csv", header=TRUE)%>% mutate(language = "dyck")
abn_weights = read.csv("~/complex_hierarchy/abn_model_c_lstm_seq1_rand2.csv", header=FALSE)
anbm_weights = read.csv("~/complex_hierarchy/anbm_model_c_lstm_seq1_rand2.csv", header=FALSE)
abn_acc = read.csv("~/complex_hierarchy/abn_model_c_lstm_loss_seq1_rand2.csv", header=TRUE)%>% mutate(language = "abn")
anbm_acc = read.csv("~/complex_hierarchy/anbm_model_c_lstm_loss_seq1_rand2.csv", header=TRUE)%>% mutate(language = "anbm")
#anbm_weights = read.csv("C:\\Users\\Charlie\\PycharmProjects\\complex_hierarchy\\anbm_model_r.csv", header=FALSE)
#abn_weights = read.csv("C:\\Users\\Charlie\\PycharmProjects\\complex_hierarchy\\abn_model_r.csv", header=FALSE)
#anbn_acc = anbn_acc %>%mutate(acc_group=cut(accuracy,seq(.5,1,.05),ordered_result=TRUE))
#dyck_acc = dyck_acc %>%mutate(acc_group=cut(accuracy,seq(.5,1,.05),ordered_result=TRUE))

dyck_kld = get_kld_growth(dyck_weights, dyck_acc, seq(-2,2,.02), seq(.5,1,.025)) #%>% mutate(language = "dyck") %>% filter(acc_group != "(0.5,0.525]")
anbn_kld = get_kld_growth(anbn_weights, anbn_acc, seq(-2,2,.02), seq(.5,1,.025)) #%>% mutate(language = "anbn") %>% filter(acc_group != "(0.5,0.525]")
abn_kld = get_kld_growth(abn_weights, abn_acc, seq(-2,2,.02), seq(.5,1,.025)) #%>% mutate(language = "abn") %>% filter(acc_group != "(0.5,0.525]")
anbm_kld = get_kld_growth(anbm_weights, anbm_acc, seq(-2,2,.02), seq(.5,1,.025)) #%>% mutate(language = "anbm") %>% filter(acc_group != "(0.5,0.525]")

klds = rbind(dyck_kld[[2]]%>% mutate(language = "dyck"),anbn_kld[[2]]%>% mutate(language = "anbn"),abn_kld[[2]]%>% mutate(language = "abn"), anbm_kld[[2]]%>% mutate(language = "anbm"))

plot = ggplot(klds, aes(x=acc_group,y=model_kld, group=language)) + geom_line(aes(color=language))
plot



dyck_kld_loss = get_kld_growth_loss(dyck_weights, dyck_acc, seq(-2,2,.02), seq(1,.25,-.25)) #%>% mutate(language = "dyck") %>% filter(acc_group != "(0.5,0.525]")
anbn_kld_loss = get_kld_growth_loss(anbn_weights, anbn_acc, seq(-2,2,.02), seq(1,.25,-.25)) #%>% mutate(language = "anbn") %>% filter(acc_group != "(0.5,0.525]")
abn_kld_loss = get_kld_growth_loss(abn_weights, abn_acc, seq(-2,2,.02), seq(1,.25,-.25)) #%>% mutate(language = "abn") %>% filter(acc_group != "(0.5,0.525]")
anbm_kld_loss = get_kld_growth_loss(anbm_weights, anbm_acc, seq(-2,2,.02), seq(1,.25,-.25)) #%>% mutate(language = "anbm") %>% filter(acc_group != "(0.5,0.525]")

klds_loss = rbind(dyck_kld_loss[[2]]%>% mutate(language = "dyck")%>%inner_join(dyck_acc%>%mutate(ID = row_number())),
                  anbn_kld_loss[[2]]%>% mutate(language = "anbn")%>%inner_join(anbn_acc%>%mutate(ID = row_number())),
                  abn_kld_loss[[2]]%>% mutate(language = "abn")%>%inner_join(abn_acc%>%mutate(ID = row_number())), 
                  anbm_kld_loss[[2]]%>% mutate(language = "anbm")%>%inner_join(anbm_acc%>%mutate(ID = row_number())))

plot = ggplot(klds_loss, aes(x=accuracy,y=model_kld, group=language)) + geom_point(alpha=.1, aes(color=language))+geom_smooth()
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
